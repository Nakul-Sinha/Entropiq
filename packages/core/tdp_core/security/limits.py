"""Size and visibility limits for repositories that users ask the platform to analyse.

:func:`check_repo_size` asks the GitHub API for a repository's size before anything is cloned (github.com
only; other hosts have no size API here). :func:`check_mirror_size` measures a mirror after it was cloned or
fetched, which covers every host. Both fail closed: when the size cannot be established the check raises
instead of letting an unbounded clone through. Error messages never contain a token.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import quote

from tdp_core.labeling.github_issues import HttpResponse, Transport, urllib_transport
from tdp_core.mining.git import run_git
from tdp_core.security.urls import RepoRef, UnsafeRepoError

logger = logging.getLogger(__name__)

GITHUB_HOST = "github.com"
GITHUB_API_URL = "https://api.github.com"
DEFAULT_MAX_REPO_KB = 1_000_000
"""Default size limit in KiB (about 1 GB), the same default the platform settings use for ``max_repo_kb``."""

_NOT_PUBLIC_VISIBILITY = frozenset({"private", "internal"})


class RepoTooLargeError(UnsafeRepoError):
    """The repository is larger than the limit. ``size_kb`` and ``max_kb`` are in KiB."""

    def __init__(self, size_kb: int, max_kb: int) -> None:
        super().__init__(size_kb, max_kb)
        self.size_kb = size_kb
        self.max_kb = max_kb

    def __str__(self) -> str:
        return f"repository is {self.size_kb:,} KB, over the {self.max_kb:,} KB limit"


class RepoNotFoundError(UnsafeRepoError):
    """Missing, private, internal or legally blocked repository (one message for all)."""


class RepoCheckUnavailableError(RuntimeError):
    """The size could not be checked right now (network failure, rate limit, unexpected API answer).

    The request may succeed later; an API should answer 503. ``status`` is the HTTP status of the GitHub
    response, or ``None`` when no response arrived.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _is_rate_limited(resp: HttpResponse) -> bool:
    if resp.status == 429:
        return True
    return resp.status == 403 and (
        resp.headers.get("x-ratelimit-remaining") == "0" or b"rate limit" in resp.body.lower()
    )


def check_repo_size(
    ref: RepoRef,
    token: str | None,
    max_kb: int,
    *,
    transport: Transport | None = None,
    timeout: float = 10.0,
) -> int:
    """Return a public GitHub repository's size in KiB (the API's ``size`` field), or -1 for other hosts.

    For ``github.com`` this makes exactly one ``GET /repos/{owner}/{name}`` request with no retries (it is
    meant to run inside an API request), sending ``Authorization: Bearer <token>`` only when ``token`` is a
    non-empty string. ``transport`` defaults to ``github_issues.urllib_transport``, which follows redirects
    only on the API host (renamed repositories still resolve), so the token never goes anywhere else.
    ``timeout`` is in seconds. Hosts other than ``github.com`` return -1 without any request; measure their
    mirrors with :func:`check_mirror_size` after cloning. A size equal to ``max_kb`` passes. GitHub may
    report 0 for a freshly pushed repository.

    Raises :class:`RepoTooLargeError` when the size is over ``max_kb``; :class:`RepoNotFoundError` for 404,
    410 and 451 answers and for private or internal repositories (the same message for all, so the API
    never confirms that a private repository exists); :class:`RepoCheckUnavailableError` when the network
    fails (``status=None``), GitHub rate-limits the request, answers any other status, or returns a body
    without a valid ``size``; and ``ValueError`` for a negative ``max_kb`` (before any request).
    """
    if max_kb < 0:
        raise ValueError(f"max_kb must be >= 0, got {max_kb}")
    if ref.host.lower() != GITHUB_HOST:
        logger.debug("no size check for host %r", ref.host)
        return -1
    slug = f"{ref.owner}/{ref.name}"
    url = f"{GITHUB_API_URL}/repos/{quote(ref.owner, safe='')}/{quote(ref.name, safe='')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "entropiq-tdp",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    get = transport if transport is not None else urllib_transport
    try:
        resp = get(url, headers, timeout)
    except OSError as exc:
        logger.info("GitHub API unreachable for %s: %s", slug, type(exc).__name__)
        raise RepoCheckUnavailableError(
            f"could not reach the GitHub API to check {slug} ({type(exc).__name__})"
        ) from None

    not_found = f"repository {slug} was not found on github.com or is not public"
    if resp.status in (404, 410, 451):
        raise RepoNotFoundError(not_found)
    if _is_rate_limited(resp):
        raise RepoCheckUnavailableError(
            f"GitHub API rate limit reached while checking {slug}; try again later", status=resp.status
        )
    if resp.status != 200:
        raise RepoCheckUnavailableError(
            f"GitHub API returned HTTP {resp.status} while checking {slug}", status=resp.status
        )

    unexpected = f"unexpected GitHub API response while checking {slug}"
    try:
        data = json.loads(resp.body)
    except (ValueError, RecursionError):
        raise RepoCheckUnavailableError(unexpected, status=200) from None
    if not isinstance(data, dict):
        raise RepoCheckUnavailableError(unexpected, status=200)
    if data.get("private") is True or data.get("visibility") in _NOT_PUBLIC_VISIBILITY:
        raise RepoNotFoundError(not_found)
    size = data.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RepoCheckUnavailableError(unexpected, status=200)
    if size > max_kb:
        raise RepoTooLargeError(size, max_kb)
    return size


def check_mirror_size(repo_dir: Path, max_kb: int) -> int:
    """Return the on-disk object size of the bare repository ``repo_dir`` in KiB, raising if over ``max_kb``.

    ``repo_dir`` must be a bare repository (a mirror made by ``ensure_mirror``). The size is
    ``size + size-pack + size-garbage`` from ``git count-objects -v``, so loose objects left by a fetch count
    too. ``--git-dir`` is passed so git never walks up into an enclosing checkout. A relative ``repo_dir`` is
    made absolute against the current directory first, because git runs inside ``repo_dir`` and would
    otherwise read ``--git-dir`` relative to it. A size equal to ``max_kb`` passes.

    Raises :class:`RepoTooLargeError` when the size is over ``max_kb``, ``ValueError`` for a negative
    ``max_kb`` (before git runs), and ``GitError`` from ``run_git`` when ``repo_dir`` is missing or is not a
    git repository.
    """
    if max_kb < 0:
        raise ValueError(f"max_kb must be >= 0, got {max_kb}")
    git_dir = Path(repo_dir).absolute()
    out = run_git([f"--git-dir={git_dir}", "count-objects", "-v"], git_dir)
    stats: dict[str, int] = {}
    for line in out.decode("utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        try:
            stats[key.strip()] = int(value.strip())
        except ValueError:
            continue
    size = stats.get("size", 0) + stats.get("size-pack", 0) + stats.get("size-garbage", 0)
    if size > max_kb:
        raise RepoTooLargeError(size, max_kb)
    return size
