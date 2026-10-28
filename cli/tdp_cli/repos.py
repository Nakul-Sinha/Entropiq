"""Turn what the user typed (a folder or a GitHub URL) into a repository folder to analyse.

Local paths must be inside a git work tree or be a bare repository; a subfolder analyses its whole
repository. URLs are checked with :func:`tdp_core.security.urls.validate_repo_url` (https on github.com or
gitlab.com only) and kept as lean bare mirrors under ``<home>/mirrors``; an existing mirror is reused when
the network is down.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from rich.console import Console

from tdp_cli.errors import CliError
from tdp_cli.home import EntropiqHome
from tdp_cli.theme import activity, clean, notice
from tdp_core.security.urls import (
    DEFAULT_ALLOWED_HOSTS,
    HostLookupError,
    RepoRef,
    UnsafeRepoError,
    validate_ref,
    validate_repo_url,
)

if TYPE_CHECKING:
    from tdp_core.mining.git import GitError

_URL_PREFIX: Final = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*://|git@|(?:www\.)?(?:github|gitlab)\.com/)")
_SLUG_JUNK: Final = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class Target:
    """A repository ready to analyse.

    ``repo_dir`` is the work tree, the bare repository or the mirror; ``display`` is a short name
    (``requests`` or ``psf/requests``); ``slug`` names its folder of saved analyses; ``ref`` is the revision
    the user asked for (``HEAD`` by default).
    """

    kind: Literal["local", "url"]
    source: str
    repo_dir: Path
    display: str
    slug: str
    ref: str


def looks_like_url(text: str) -> bool:
    """True for ``scheme://...``, ``git@...`` or ``github.com/...`` input that is not an existing path."""
    text = text.strip()
    if not text or Path(text).expanduser().exists():
        return False
    return bool(_URL_PREFIX.match(text))


def _slugify(text: str) -> str:
    slug = _SLUG_JUNK.sub("-", text.lower()).strip("-.")[:60].strip("-.")
    return slug or "repo"


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return clean(line.strip())[:300]
    return ""


def _git_hint(exc: GitError) -> str | None:
    return _first_line(exc.stderr) or None


def resolve_local(text: str, ref: str | None = None) -> Target:
    """A local repository from a folder path (absolute or relative to the current folder).

    Raises :class:`CliError` (exit code 2) for a missing path, a file, a folder outside any git repository,
    or an invalid revision.
    """
    from tdp_core.mining.git import GitError, check_rev, run_git

    raw = text.strip() or "."
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else Path.cwd() / path).resolve()
    if not path.exists():
        raise CliError(f"{path} does not exist", exit_code=2)
    if not path.is_dir():
        raise CliError(f"{path} is not a folder", exit_code=2, hint="Pass the folder of a git repository.")
    try:
        out = run_git(
            ["rev-parse", "--is-bare-repository", "--is-inside-git-dir", "--absolute-git-dir"],
            path,
            timeout=30,
        )
        bare, inside_git_dir, git_dir = out.decode("utf-8", "replace").splitlines()[:3]
        if bare.strip() == "true":
            root = Path(git_dir.strip())
        elif inside_git_dir.strip() == "true":
            root = Path(git_dir.strip()).parent
        else:
            top = run_git(["rev-parse", "--show-toplevel"], path, timeout=30)
            root = Path(top.decode("utf-8", "replace").strip())
    except (GitError, ValueError) as exc:
        hint = _git_hint(exc) if isinstance(exc, GitError) else None
        raise CliError(f"{path} is not a git repository", exit_code=2, hint=hint) from exc
    root = root.resolve()
    rev = (ref or "HEAD").strip()
    try:
        check_rev(rev)
    except ValueError as exc:
        raise CliError(f"{rev!r} is not a valid revision", exit_code=2) from exc
    name = root.name[:-4] if root.name.lower().endswith(".git") else root.name
    display = name or "repo"
    digest = hashlib.sha1(os.path.normcase(str(root)).encode("utf-8")).hexdigest()[:8]
    return Target("local", str(root), root, display, f"{_slugify(display)}-{digest}", rev)


def local_target(path: Path, ref: str = "HEAD") -> Target:
    """Convenience wrapper of :func:`resolve_local` for a :class:`~pathlib.Path`."""
    return resolve_local(str(path), ref)


def resolve_url(text: str, ref: str | None, home: EntropiqHome) -> tuple[Target, RepoRef]:
    """Validate a repository URL (no DNS lookup yet) and work out its mirror folder.

    ``github.com/owner/name`` without a scheme gets ``https://``. Raises :class:`CliError` (exit code 2)
    for URLs or refs that :mod:`tdp_core.security.urls` refuses (http, ssh, other hosts, odd paths).
    """
    url = text.strip()
    if "://" not in url and not url.startswith("git@"):
        url = f"https://{url}"
    try:
        repo = validate_repo_url(url, DEFAULT_ALLOWED_HOSTS, resolve_dns=False)
    except UnsafeRepoError as exc:
        raise CliError(
            f"Cannot analyse this URL: {exc}",
            exit_code=2,
            hint="Use https://github.com/<owner>/<name> or https://gitlab.com/<owner>/<name>.",
        ) from exc
    rev = "HEAD"
    if ref is not None and ref.strip() and ref.strip() != "HEAD":
        try:
            rev = validate_ref(ref.strip())
        except UnsafeRepoError as exc:
            raise CliError(f"Cannot use this ref: {exc}", exit_code=2) from exc
    if repo.host == "github.com":
        slug = f"{repo.owner}__{repo.name}".lower()
    else:
        slug = _slugify(f"{repo.host}__{repo.owner}__{repo.name}")
    target = Target(
        "url",
        repo.canonical_url,
        home.mirror_path(repo.host, repo.owner, repo.name),
        f"{repo.owner}/{repo.name}",
        slug,
        rev,
    )
    return target, repo


def _mirror_age(repo_dir: Path) -> str:
    try:
        stamp = max(
            (repo_dir / name).stat().st_mtime for name in ("FETCH_HEAD", "HEAD") if (repo_dir / name).exists()
        )
    except (OSError, ValueError):
        return "earlier"
    return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


def prepare_mirror(target: Target, repo: RepoRef, console: Console) -> None:
    """Clone (first time) or update the mirror of ``target``; reuse an old copy when offline.

    Raises :class:`CliError`: exit code 2 when the host resolves to a non-public address, 1 when a first
    clone fails or the host cannot be resolved and there is no copy yet. A failed first clone leaves no
    partial folder behind, also when it is cancelled with Ctrl+C.
    """
    from tdp_core.mining import mirror as mirror_module
    from tdp_core.mining.git import GitError

    existed = (target.repo_dir / "HEAD").is_file()
    try:
        validate_repo_url(repo.canonical_url, DEFAULT_ALLOWED_HOSTS)
    except HostLookupError as exc:
        if existed:
            notice(console, f"Offline ({exc}): analysing the copy fetched {_mirror_age(target.repo_dir)}.")
            return
        raise CliError(f"Cannot reach {repo.host}: {exc}", hint="Check the network and try again.") from exc
    except UnsafeRepoError as exc:
        raise CliError(f"Cannot analyse this URL: {exc}", exit_code=2) from exc
    label = (
        f"Updating the mirror of {target.display}"
        if existed
        else f"Cloning {repo.canonical_url} (first time only)"
    )
    try:
        target.repo_dir.parent.mkdir(parents=True, exist_ok=True)
        with activity(console, label):
            mirror_module.ensure_mirror(repo.clone_url, target.repo_dir)
    except (GitError, OSError) as exc:
        detail = _git_hint(exc) if isinstance(exc, GitError) else clean(str(exc))
        if existed:
            notice(
                console,
                f"Could not update the mirror ({detail or 'git failed'}); "
                f"analysing the copy fetched {_mirror_age(target.repo_dir)}.",
            )
            return
        shutil.rmtree(target.repo_dir, ignore_errors=True)
        raise CliError(f"Cannot clone {repo.canonical_url}", hint=detail or None) from exc
    except KeyboardInterrupt:
        if not existed:
            shutil.rmtree(target.repo_dir, ignore_errors=True)
        raise


def resolve_target(text: str | None, ref: str | None, *, home: EntropiqHome, console: Console) -> Target:
    """The repository named by ``text`` (``None`` or empty means the current folder), mirrored if a URL."""
    raw = (text or "").strip() or "."
    if looks_like_url(raw):
        target, repo = resolve_url(raw, ref, home)
        prepare_mirror(target, repo, console)
        return target
    return resolve_local(raw, ref)


def resolve_commit(target: Target) -> str:
    """The full sha of ``target.ref``. Raises :class:`CliError` (exit code 2) when it names no commit."""
    from tdp_core.mining.git import GitError, check_rev, run_git

    try:
        check_rev(target.ref)
        out = run_git(
            ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{target.ref}^{{commit}}"],
            target.repo_dir,
            timeout=60,
        )
    except (GitError, ValueError) as exc:
        raise CliError(
            f"{target.ref!r} does not name a commit in {target.display}",
            exit_code=2,
            hint="Pass --ref with a branch, tag or commit that exists.",
        ) from exc
    sha = out.decode("ascii", "replace").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
        raise CliError(f"{target.ref!r} does not name a commit in {target.display}", exit_code=2)
    return sha
