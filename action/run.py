"""Helper of the Entropiq GitHub Action (``action/action.yml``).

Each subcommand is one step of the composite action:

- ``resolve`` checks the inputs and works out which model release to use, its cache folder and cache key;
- ``fetch`` downloads that release asset with ``gh``, checks its sha256 and unpacks it, or falls back to the
  demo model (and says why) when any of that fails;
- ``report`` turns the ``tdp diff --format json`` document into one pull request comment, created or updated
  in place through the hidden marker ``<!-- tdp-report -->``, adds it to the job summary and applies
  ``fail-on`` through the exit code.

Every value comes from environment variables set by ``action.yml`` (``INPUT_*``, ``ENTROPIQ_*`` and the
runner's ``GITHUB_*``), never from the command line or a shell string, so text from a pull request cannot
inject shell code. Only the standard library is used, except that ``report`` imports the Markdown escaping
helpers of ``tdp_core``, which is installed in the same environment as the CLI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

MARKER: Final = "<!-- tdp-report -->"
DIFF_SCHEMA: Final = "entropiq.diff.v1"
FAIL_ON_CHOICES: Final = ("none", "high", "critical")
FAILING_BANDS: Final[dict[str, tuple[str, ...]]] = {
    "none": (),
    "high": ("Critical", "High"),
    "critical": ("Critical",),
}
THRESHOLD_WORDS: Final = {"high": "High or Critical", "critical": "Critical"}
BAND_ORDER: Final = ("Critical", "High", "Medium", "Low")
MAX_ROWS: Final = 20
MAX_BODY_CHARS: Final = 60_000
DOWNLOAD_ATTEMPTS: Final = 3
RETRY_WAITS_S: Final = (5.0, 15.0)
GH_TIMEOUT_S: Final = 300.0
TRUE_WORDS: Final = frozenset({"true", "yes", "1", "on"})
FALSE_WORDS: Final = frozenset({"false", "no", "0", "off"})
VERSION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
REPO_RE: Final = re.compile(r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
RELEASE_URL_RE: Final = re.compile(
    r"^https://github\.com/(?P<repo>[A-Za-z0-9-]+/[A-Za-z0-9._-]+)/releases/download/(?P<tag>[^/]+)/(?P<asset>[^/]+)$"
)
HEX_RE: Final = re.compile(r"^[0-9a-fA-F]{7,40}$")
NO_MODEL: Final = "no production model has been published yet"
NO_POINTER: Final = "models/production.json could not be read at the action ref"
USAGE: Final = "usage: run.py resolve|fetch|report (inputs come from environment variables)"


class ActionError(Exception):
    """A failure the action reports as one error annotation; ``exit_code`` 2 means bad input."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class Pointer:
    """The fields of ``models/production.json``; a missing, blank or non-string field is ``None``."""

    version: str | None
    uri: str | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Which release asset holds the model; ``sha256`` ``None`` means use the release's checksum file."""

    version: str
    repo: str
    tag: str
    asset: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    """The model the analysis uses: the verified release bundle folder, or the demo model and why."""

    source: Literal["release", "dev"]
    model_dir: Path | None
    reason: str
    downloaded: bool = False


@dataclass(frozen=True, slots=True)
class Comment:
    """A pull request comment that starts with the marker; ``login`` is ``""`` for a deleted user."""

    id: int
    login: str


type Runner = Callable[[Sequence[str], str | None], str]
type Downloader = Callable[[ModelSpec, Path, bool], None]


# Inputs


def parse_bool(value: str, name: str) -> bool:
    """``true``/``yes``/``1``/``on`` or ``false``/``no``/``0``/``off`` (any case, spaces ignored).

    Anything else, the empty string included, raises ``ActionError`` with exit code 2 naming the input.
    """
    word = value.strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ActionError(f"the {name} input must be true or false, not {value.strip()!r}", exit_code=2)


def parse_fail_on(value: str) -> str:
    """The ``fail-on`` input in lower case; empty means ``none``. Other values raise (exit code 2)."""
    word = value.strip().lower() or "none"
    if word not in FAIL_ON_CHOICES:
        message = f"the fail-on input must be none, high or critical, not {value.strip()!r}"
        raise ActionError(message, exit_code=2)
    return word


def check_inputs(env: Mapping[str, str]) -> None:
    """Check ``base``, ``comment``, ``fail-on`` and ``sarif``; a bad value raises with exit code 2."""
    if not env.get("INPUT_BASE", "").strip():
        raise ActionError(
            "The base input is required, for example the pull request base sha "
            "(github.event.pull_request.base.sha).",
            exit_code=2,
        )
    parse_bool(env.get("INPUT_COMMENT", "true"), "comment")
    parse_fail_on(env.get("INPUT_FAIL_ON", "none"))
    if env.get("INPUT_SARIF", "").strip() == "-":
        raise ActionError("the sarif input must be a file path, not -", exit_code=2)


def _internal(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ActionError(f"internal: {name} is not set", exit_code=2)
    return value


# Model choice


def read_pointer(path: Path) -> Pointer | None:
    """``models/production.json``, or ``None`` when it is missing, not JSON or not a JSON object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def field(key: str) -> str | None:
        value = data.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    return Pointer(field("version"), field("uri"), field("sha256"))


def model_spec(pointer: Pointer | None, requested: str, default_repo: str) -> ModelSpec | None:
    """The release asset of the requested model version, or of the pointer's version when none is asked.

    ``None`` when neither gives a version. By convention the release is ``model-<version>`` of
    ``default_repo`` with the asset ``<version>.tar.gz``. For the pointer's own version the sha256 comes from
    the pointer, and a release asset URL in ``uri`` overrides the repository, tag and asset. Any other version
    is verified later with the release's ``<asset>.sha256`` file. A bad version, repository, tag or asset name
    (an empty ``default_repo`` that is needed included) raises ``ActionError`` with exit code 2.
    """
    version = requested.strip() or (pointer.version if pointer else None)
    if not version:
        return None
    if not VERSION_RE.fullmatch(version):
        raise ActionError(f"the model version {version!r} is not a valid release name", exit_code=2)
    repo, tag, asset, sha = default_repo.strip(), f"model-{version}", f"{version}.tar.gz", None
    if pointer is not None and version == pointer.version:
        if pointer.sha256 and SHA256_RE.fullmatch(pointer.sha256.lower()):
            sha = pointer.sha256.lower()
        match = RELEASE_URL_RE.fullmatch(pointer.uri or "")
        if match:
            repo = match["repo"]
            tag = urllib.parse.unquote(match["tag"])
            asset = urllib.parse.unquote(match["asset"])
    if not REPO_RE.fullmatch(repo):
        raise ActionError(f"the model repository {repo!r} is not owner/name", exit_code=2)
    for label, name in (("tag", tag), ("asset", asset)):
        if not NAME_RE.fullmatch(name):
            raise ActionError(f"the model release {label} {name!r} is not a valid name", exit_code=2)
    return ModelSpec(version, repo, tag, asset, sha)


def no_model_reason(pointer: Pointer | None) -> str:
    """Why no release is used when ``model_spec`` gives ``None``."""
    return NO_POINTER if pointer is None else NO_MODEL


def cache_key(spec: ModelSpec) -> str:
    """The ``actions/cache`` key of a model version."""
    return f"tdp-model-{spec.version}"


def cache_dir(work_dir: Path, spec: ModelSpec) -> Path:
    """The folder that keeps the verified tarball (and its checksum file) between runs."""
    return work_dir / "model-cache" / spec.version


def sha256_file(path: Path) -> str:
    """The lowercase hex sha256 of a file, read in 1 MiB chunks."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksum(path: Path) -> str | None:
    """The sha256 in a ``sha256sum`` file (``hex  name`` or ``hex *name``), lowercased.

    ``None`` when the file is missing or unreadable, or its first non-blank line does not start with 64 hex
    digits.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    for line in lines:
        if line.strip():
            token = line.split()[0].lower()
            return token if SHA256_RE.fullmatch(token) else None
    return None


def extract_bundle(tarball: Path, dest: Path, version: str) -> Path:
    """Unpack a model tarball into ``dest`` and return ``dest/<version>``.

    Every member must be a file or folder named ``<version>`` or ``<version>/...`` without ``..``; anything
    else, an empty archive, or a ``metadata.json`` whose ``version`` differs raises ``ActionError``. Archive
    errors (``tarfile.TarError``, ``OSError``, ``EOFError``, ``zlib.error``) propagate.
    """
    with tarfile.open(tarball, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ActionError("the model archive is empty")
        for member in members:
            name = member.name
            inside = name == version or name.startswith(version + "/")
            if not inside or ".." in PurePosixPath(name).parts or not (member.isfile() or member.isdir()):
                raise ActionError(f"unexpected archive member {name!r}")
        archive.extractall(dest, filter="data")
    folder = dest / version
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ActionError(f"the bundle has no readable metadata.json ({error})") from error
    found = meta.get("version") if isinstance(meta, dict) else None
    if found != version:
        raise ActionError(f"the bundle's metadata.json names version {found!r}, not {version!r}")
    return folder


# GitHub CLI


def _first_line(text: str, limit: int = 300) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return "no error message"


def run_gh(args: Sequence[str], stdin: str | None = None, *, timeout: float = 60.0) -> str:
    """Run ``gh <args>`` with ``stdin`` and return its stdout; the token comes from ``GH_TOKEN``.

    ``timeout`` is in seconds. A missing ``gh``, a timeout or a non-zero exit raises ``ActionError`` with the
    first line of stderr (at most 300 characters).
    """
    exe = shutil.which("gh")
    if exe is None:
        raise ActionError("the GitHub CLI (gh) is not installed on this runner")
    label = " ".join(args[:2])
    try:
        done = subprocess.run(
            [exe, *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ActionError(f"gh {label} timed out after {timeout:.0f} s") from error
    except OSError as error:
        raise ActionError(f"gh {label} could not start ({error})") from error
    if done.returncode != 0:
        raise ActionError(f"gh {label} failed: {_first_line(done.stderr or '')}")
    return done.stdout or ""


def _slow_gh(args: Sequence[str], stdin: str | None) -> str:
    return run_gh(args, stdin, timeout=GH_TIMEOUT_S)


def gh_download(spec: ModelSpec, dest: Path, with_checksum: bool, *, run: Runner = _slow_gh) -> None:
    """Download the release's model asset (plus ``<asset>.sha256`` when ``with_checksum``) into ``dest``."""
    args = ["release", "download", spec.tag, "--repo", spec.repo, "--dir", str(dest), "--clobber"]
    args += ["--pattern", spec.asset]
    if with_checksum:
        args += ["--pattern", f"{spec.asset}.sha256"]
    run(args, None)


def fetch_model(
    spec: ModelSpec | None,
    work_dir: Path,
    *,
    no_model: str = NO_MODEL,
    download: Downloader = gh_download,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelOutcome:
    """Get the verified model bundle folder for ``spec``, or the demo model with the reason.

    A tarball already in the cache folder whose sha256 matches is used without downloading. Otherwise the
    asset is downloaded into ``work_dir/download`` (up to ``DOWNLOAD_ATTEMPTS`` tries, waiting
    ``RETRY_WAITS_S`` seconds between them), checked against the pointer's sha256 or the release's
    ``.sha256`` file, and only then moved into the cache. The bundle is unpacked fresh into
    ``work_dir/model`` on every run. Every failure gives ``source="dev"`` and a reason sentence fragment.
    """
    if spec is None:
        return ModelOutcome("dev", None, no_model)
    cache = cache_dir(work_dir, spec)
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / spec.asset
    checksum_file = cache / f"{spec.asset}.sha256"
    expected = spec.sha256 or read_checksum(checksum_file)
    downloaded = False
    if not (tarball.is_file() and expected and sha256_file(tarball) == expected):
        outcome = _download(spec, work_dir, cache, download=download, sleep=sleep)
        if isinstance(outcome, ModelOutcome):
            return outcome
        expected, downloaded = outcome, True
    model_root = work_dir / "model"
    shutil.rmtree(model_root, ignore_errors=True)
    try:
        model_root.mkdir(parents=True)
        folder = extract_bundle(tarball, model_root, spec.version)
    except ActionError as error:
        return ModelOutcome("dev", None, f"the model archive could not be unpacked ({error.message})")
    except (tarfile.TarError, OSError, ValueError, EOFError, zlib.error) as error:
        return ModelOutcome("dev", None, f"the model archive could not be unpacked ({error})")
    return ModelOutcome("release", folder, "", downloaded)


def _download(
    spec: ModelSpec, work_dir: Path, cache: Path, *, download: Downloader, sleep: Callable[[float], None]
) -> ModelOutcome | str:
    """Download and verify into the cache; the verified sha256, or the demo outcome on failure."""
    staging = work_dir / "download"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        error: ActionError | None = None
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                download(spec, staging, spec.sha256 is None)
            except ActionError as caught:
                error = caught
                if attempt < DOWNLOAD_ATTEMPTS - 1:
                    sleep(RETRY_WAITS_S[min(attempt, len(RETRY_WAITS_S) - 1)])
                continue
            error = None
            break
        if error is not None:
            why = f"the model release {spec.tag} could not be downloaded ({error.message})"
            return ModelOutcome("dev", None, why)
        fetched = staging / spec.asset
        if not fetched.is_file():
            return ModelOutcome("dev", None, f"the release {spec.tag} has no asset {spec.asset}")
        expected = spec.sha256 or read_checksum(staging / f"{spec.asset}.sha256")
        if expected is None:
            return ModelOutcome("dev", None, f"the release {spec.tag} has no sha256 file to verify the model")
        digest = sha256_file(fetched)
        if digest != expected:
            return ModelOutcome(
                "dev",
                None,
                f"the downloaded model failed its sha256 check (expected {expected[:12]}, got {digest[:12]})",
            )
        os.replace(fetched, cache / spec.asset)
        if spec.sha256 is None:
            (cache / f"{spec.asset}.sha256").write_text(f"{expected}  {spec.asset}\n", encoding="utf-8")
        return expected
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# Report


def _int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    return value if isinstance(value, str) else ""


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _is_are(n: int) -> str:
    return "is" if n == 1 else "are"


def _summary(diff: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = diff.get("summary")
    return summary if isinstance(summary, Mapping) else {}


def highest_band(summary: Mapping[str, Any]) -> str:
    """The highest band with at least one changed ranked file (``Critical`` first), else ``none``."""
    for band in BAND_ORDER:
        if _int(summary, band.lower()) > 0:
            return band
    return "none"


def gate(diff: Mapping[str, Any], fail_on: str) -> tuple[bool, str]:
    """Whether ``fail-on`` fails the job, and the sentence that explains it in the comment footer.

    It never fails with ``none`` or when the demo model scored the files, because demo rankings are not real
    risk.
    """
    if fail_on == "none":
        return False, "Fail-on is off."
    if diff.get("demo_model") is True:
        return False, f"Fail-on {fail_on} is not applied because the demo model is in use."
    summary = _summary(diff)
    n = sum(_int(summary, band.lower()) for band in FAILING_BANDS[fail_on])
    words = THRESHOLD_WORDS[fail_on]
    if n > 0:
        files = f"{_count(n, 'changed file')} {_is_are(n)}"
        return True, f"Fail-on {fail_on}: {files} {words}, so this check fails."
    return False, f"Fail-on {fail_on}: no changed file is {words}."


def render_comment(
    diff: Mapping[str, Any], *, fail_on: str, reason: str = "", max_rows: int = MAX_ROWS
) -> str:
    """The pull request comment for a ``tdp diff --format json`` document; ends with exactly one newline.

    It starts with the marker, then a one-line summary, a table of at most ``max_rows`` changed ranked files
    (band, priority, risk and top reason), the unranked count, a warning when the demo model is in use (with
    ``reason``), and a footer with the commits, the model and the ``fail-on`` result. Paths go into code
    spans and free text is escaped. Missing values show as ``n/a`` (or ``-`` for no reason). Rows are dropped
    one at a time while the body is longer than ``MAX_BODY_CHARS``.
    """
    rows = max(0, max_rows)
    while True:
        body = _render(diff, fail_on=fail_on, reason=reason, rows=rows)
        if len(body) <= MAX_BODY_CHARS or rows == 0:
            return body
        rows -= 1


def _render(diff: Mapping[str, Any], *, fail_on: str, reason: str, rows: int) -> str:
    from tdp_core.report.common import format_percent
    from tdp_core.report.markdown import code_span, escape_text

    def code(value: object, limit: int | None = None) -> str:
        text = value if isinstance(value, str) else ""
        return code_span(text[:limit] if limit else text, in_table=False) or "n/a"

    summary = _summary(diff)
    changed, ranked = _int(summary, "changed"), _int(summary, "ranked")
    files = [item for item in diff.get("files") or [] if isinstance(item, Mapping)]
    unranked = [item for item in diff.get("unranked") or [] if isinstance(item, Mapping)]
    base = diff.get("base") if isinstance(diff.get("base"), str) else ""
    base_sha = diff.get("base_sha") if isinstance(diff.get("base_sha"), str) else ""
    base_display = code(base_sha, 12) if HEX_RE.fullmatch(base or "") and base_sha else code(base, 100)

    if changed == 0:
        sentence = f"No files changed compared with {base_display}."
    elif ranked == 0:
        sentence = "The changed file is not ranked."
        if changed > 1:
            sentence = f"None of the {changed} changed files are ranked."
    else:
        counts = ", ".join(f"{_int(summary, band.lower())} {band}" for band in BAND_ORDER)
        sentence = f"{ranked} of {_count(changed, 'changed file')} {_is_are(ranked)} ranked: {counts}."
    lines = [MARKER, "### Entropiq report", "", sentence, ""]

    shown = files[:rows]
    if shown:
        lines += ["| File | Band | Priority | Risk | Top reason |", "| --- | --- | ---: | ---: | --- |"]
        for item in shown:
            priority, risk = item.get("priority"), item.get("risk")
            cells = [
                code_span(_text(item, "path")) or "n/a",
                escape_text(_text(item, "band")) or "n/a",
                str(priority) if _is_number(priority) else "n/a",
                format_percent(float(risk)) if isinstance(risk, int | float) and _is_number(risk) else "n/a",
                escape_text(_text(item, "top_reason")) or "-",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        total = max(ranked, len(files))
        if len(shown) < total:
            lines += [f"Showing the top {len(shown)} of {total} ranked files.", ""]
    if ranked > 0 and unranked:
        n = len(unranked)
        why = "(deleted, tests, skipped or not Python)"
        lines += [f"{_count(n, 'changed file')} {_is_are(n)} not ranked {why}.", ""]
    if diff.get("demo_model") is True:
        why = reason.strip().rstrip(".")
        lines += [
            "> [!WARNING]",
            f"> Demo model in use: {escape_text(why)}. These rankings show how Entropiq works, not real risk."
            if why
            else "> Demo model in use. These rankings show how Entropiq works, not real risk.",
            "",
        ]
    _, gate_sentence = gate(diff, fail_on)
    lines.append(
        f"Commit {code(diff.get('head_sha'), 12)} compared with {base_display} "
        f"(merge base {code(diff.get('merge_base'), 12)}), model {code(diff.get('model_version'))}. "
        + gate_sentence
    )
    return "\n".join(lines).rstrip("\n") + "\n"


# Pull request comment


def pr_number(event: Mapping[str, Any]) -> int | None:
    """The pull request number of a workflow event payload, or ``None`` for other events."""
    pull = event.get("pull_request")
    number = pull.get("number") if isinstance(pull, Mapping) else None
    return number if isinstance(number, int) and not isinstance(number, bool) else None


class GhApi:
    """The GitHub REST calls of the comment, made through ``gh api`` (``GH_TOKEN`` is the token)."""

    def __init__(self, run: Runner = run_gh) -> None:
        self.run = run

    def viewer_login(self) -> str | None:
        """The login of the token's user, or ``None`` (``GITHUB_TOKEN`` cannot read ``/user``)."""
        try:
            login = self.run(["api", "user", "--jq", ".login"], None).strip()
        except ActionError:
            return None
        return login or None

    def report_comments(self, repo: str, number: int) -> list[Comment]:
        """Every comment on the pull request whose body starts with the marker, in API order."""
        jq = f'.[] | select((.body // "") | startswith({json.dumps(MARKER)})) | [.id, .user.login] | @json'
        url = f"repos/{repo}/issues/{number}/comments?per_page=100"
        out = self.run(["api", "--paginate", url, "--jq", jq], None)
        comments = []
        for line in out.splitlines():
            if line.strip():
                comment_id, login = json.loads(line)
                comments.append(Comment(int(comment_id), login if isinstance(login, str) else ""))
        return comments

    def create_comment(self, repo: str, number: int, body: str) -> int:
        """Post a new comment and return its id."""
        out = self.run(
            ["api", "--method", "POST", f"repos/{repo}/issues/{number}/comments", "--input", "-"],
            json.dumps({"body": body}),
        )
        return int(json.loads(out)["id"])

    def update_comment(self, repo: str, comment_id: int, body: str) -> int:
        """Replace the body of a comment and return its id."""
        out = self.run(
            ["api", "--method", "PATCH", f"repos/{repo}/issues/comments/{comment_id}", "--input", "-"],
            json.dumps({"body": body}),
        )
        return int(json.loads(out)["id"])


def is_own(comment: Comment, viewer: str | None) -> bool:
    """Whether the action wrote this marker comment: a bot author, or the token's own user."""
    return comment.login.endswith("[bot]") or (bool(viewer) and comment.login == viewer)


def upsert_comment(
    api: GhApi, repo: str, number: int, body: str
) -> tuple[Literal["created", "updated"], int]:
    """Update the oldest marker comment the action wrote, or create one; human comments are never edited."""
    comments = api.report_comments(repo, number)
    viewer = api.viewer_login() if any(not c.login.endswith("[bot]") for c in comments) else None
    own = sorted((c for c in comments if is_own(c, viewer)), key=lambda c: c.id)
    if own:
        try:
            return "updated", api.update_comment(repo, own[0].id, body)
        except ActionError as error:
            annotate(
                "warning", f"Could not update comment {own[0].id} ({error.message}), so a new one is posted."
            )
    return "created", api.create_comment(repo, number, body)


# Workflow commands


def annotate(level: str, message: str) -> None:
    """Print a workflow command (``::warning::...``) with ``%``, CR and LF escaped."""
    text = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level}::{text}", flush=True)


def write_outputs(env: Mapping[str, str], values: Mapping[str, str]) -> None:
    """Append ``key=value`` lines to ``GITHUB_OUTPUT`` (CR and LF in values become spaces), or print them."""
    text = "".join(f"{key}={value.replace('\r', ' ').replace('\n', ' ')}\n" for key, value in values.items())
    target = env.get("GITHUB_OUTPUT", "")
    if not target:
        print(text, end="", flush=True)
        return
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def append_summary(env: Mapping[str, str], text: str) -> None:
    """Append Markdown to the job summary (``GITHUB_STEP_SUMMARY``), or print it when that is unset."""
    target = env.get("GITHUB_STEP_SUMMARY", "")
    if not target:
        print(text, end="", flush=True)
        return
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


# Subcommands


def _spec_from(env: Mapping[str, str]) -> tuple[Pointer | None, ModelSpec | None]:
    pointer = read_pointer(Path(_internal(env, "ENTROPIQ_POINTER")))
    spec = model_spec(pointer, env.get("INPUT_MODEL_VERSION", ""), env.get("ENTROPIQ_DEFAULT_REPO", ""))
    return pointer, spec


def cmd_resolve(env: Mapping[str, str]) -> int:
    """Check the inputs and write the ``version``, ``cache-dir`` and ``cache-key`` outputs."""
    check_inputs(env)
    work_dir = Path(_internal(env, "ENTROPIQ_WORK_DIR"))
    pointer, spec = _spec_from(env)
    if spec is None:
        write_outputs(env, {"version": "", "cache-dir": "", "cache-key": ""})
        print(f"No model release to use ({no_model_reason(pointer)}), so the demo model will be used.")
        return 0
    write_outputs(
        env,
        {"version": spec.version, "cache-dir": str(cache_dir(work_dir, spec)), "cache-key": cache_key(spec)},
    )
    print(f"The model {spec.version} comes from release {spec.tag} of {spec.repo}.")
    return 0


def cmd_fetch(
    env: Mapping[str, str], *, download: Downloader = gh_download, sleep: Callable[[float], None] = time.sleep
) -> int:
    """Fetch the model and write the ``source``, ``model-dir``, ``reason`` and ``downloaded`` outputs."""
    work_dir = Path(_internal(env, "ENTROPIQ_WORK_DIR"))
    pointer, spec = _spec_from(env)
    outcome = fetch_model(spec, work_dir, no_model=no_model_reason(pointer), download=download, sleep=sleep)
    write_outputs(
        env,
        {
            "source": outcome.source,
            "model-dir": str(outcome.model_dir) if outcome.model_dir else "",
            "reason": outcome.reason,
            "downloaded": "true" if outcome.downloaded else "false",
        },
    )
    if outcome.source == "dev" or spec is None:
        annotate("warning", f"Falling back to the demo model: {outcome.reason}")
    else:
        print(f"Using model {spec.version} from release {spec.tag} of {spec.repo} (sha256 verified).")
    return 0


def _load_diff(env: Mapping[str, str]) -> dict[str, Any]:
    path = Path(_internal(env, "ENTROPIQ_DIFF_JSON"))
    try:
        diff = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ActionError(f"the tdp diff report {path} could not be read ({error})") from error
    if not isinstance(diff, dict) or diff.get("schema") != DIFF_SCHEMA:
        raise ActionError(f"the tdp diff report {path} is not an {DIFF_SCHEMA} document")
    return diff


def _event(env: Mapping[str, str]) -> Mapping[str, Any]:
    try:
        event = json.loads(Path(env.get("GITHUB_EVENT_PATH", "")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return event if isinstance(event, dict) else {}


def cmd_report(env: Mapping[str, str], *, api: GhApi | None = None) -> int:
    """Write the comment and the job summary, set the outputs, and return 1 when ``fail-on`` trips."""
    diff = _load_diff(env)
    fail_on = parse_fail_on(env.get("INPUT_FAIL_ON", "none"))
    comment = parse_bool(env.get("INPUT_COMMENT", "true"), "comment")
    body = render_comment(diff, fail_on=fail_on, reason=env.get("ENTROPIQ_MODEL_REASON", ""))
    append_summary(env, body)
    comment_id = ""
    if comment:
        number = pr_number(_event(env)) if env.get("GITHUB_EVENT_PATH") else None
        repo = env.get("GITHUB_REPOSITORY", "").strip()
        if number is None or not repo:
            annotate("notice", "No pull request in this event, so the report is only in the job summary.")
        else:
            try:
                action, ident = upsert_comment(api or GhApi(), repo, number, body)
            except (ActionError, ValueError, KeyError, TypeError) as error:
                message = error.message if isinstance(error, ActionError) else f"unexpected reply ({error})"
                annotate("warning", f"Could not write the pull request comment: {message}")
            else:
                comment_id = str(ident)
                print(f"{action.capitalize()} comment {ident}.")
    model_version = diff.get("model_version")
    write_outputs(
        env,
        {
            "highest-band": highest_band(_summary(diff)),
            "comment-id": comment_id,
            "model-version": model_version if isinstance(model_version, str) else "",
        },
    )
    failed, sentence = gate(diff, fail_on)
    if fail_on != "none" and diff.get("demo_model") is True:
        annotate("warning", sentence)
    if failed:
        annotate("error", sentence)
        return 1
    return 0


COMMANDS: Final[dict[str, Callable[[Mapping[str, str]], int]]] = {
    "resolve": cmd_resolve,
    "fetch": cmd_fetch,
    "report": cmd_report,
}


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """Run one subcommand; bad usage gives 2, an ``ActionError`` becomes an error annotation and its code."""
    args = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if len(args) != 1 or args[0] not in COMMANDS:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        return COMMANDS[args[0]](env)
    except ActionError as error:
        annotate("error", error.message)
        return error.exit_code


if __name__ == "__main__":
    sys.exit(main())
