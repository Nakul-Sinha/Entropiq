"""Download a repository's GitHub issues (pull requests excluded) and flag bug reports.

The crawl uses the REST endpoint ``GET /repos/{owner}/{name}/issues?state=all`` and follows the cursor
links in the ``Link`` header (GitHub answers 422 to page numbers from page 100 on). Only the standard
library talks to the network: the HTTP call is an injectable ``Transport`` so tests run offline, and waits
go through an injectable ``sleep``. The token is sent only to the API host and never appears in a URL, a log
line or an error message.

The result is the ``data/issues/<slug>.parquet`` layout of plan section 3.4 (``ISSUES_SCHEMA``).
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import IO, Any
from urllib.parse import quote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

API_URL = "https://api.github.com"
PER_PAGE = 100

ISSUES_SCHEMA = pa.schema(
    [
        ("number", pa.int32()),
        ("created_at", pa.timestamp("s", tz="UTC")),
        ("closed_at", pa.timestamp("s", tz="UTC")),  # null while the issue is open
        ("labels", pa.list_(pa.string())),
        ("is_bug", pa.bool_()),
    ]
)

# Label rules. The first two are the plan's; the third catches triage verdicts such as "Status: Not a Bug",
# "Question/Not a bug" or "non-bug", which the first one would otherwise count as bug labels.
_BUG_LABEL = re.compile(r"(^|[^a-z])(bug|defect|regression|crash)([^a-z]|$)", re.IGNORECASE)
_BUG_TYPE_LABEL = re.compile(r"^(type|kind)[:/ ]+bug", re.IGNORECASE)
_NOT_BUG_LABEL = re.compile(r"(^|[^a-z])(not|non|no)[\s_-]*(a[\s_-]+)?bug([^a-z]|$)", re.IGNORECASE)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
_INT32_MAX = 2**31 - 1
_MAX_LIMIT_WAITS = 3  # primary rate-limit waits allowed for one request
_MAX_WAIT_S = 3900.0  # a primary rate limit resets within an hour
_DEFAULT_WAIT_S = 60.0
_MAX_BACKOFF_S = 120.0
_ISSUE_COLUMNS = ("number", "created_at", "closed_at", "labels")


class GitHubError(RuntimeError):
    """The GitHub API refused a request or kept failing. ``status`` is the last HTTP status, if any."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """One HTTP response. ``headers`` maps lower-cased header names to values."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]
"""``(url, headers, timeout_s) -> HttpResponse``. Raises ``OSError`` when the network fails.

HTTP error statuses are returned as responses, not raised."""


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Follows redirects only within the same scheme and host, so the token never leaves the API host.

    GitHub redirects renamed repositories to ``/repositories/<id>/...`` on the same host. A redirect anywhere
    else is refused and surfaces as the 3xx status.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameHostRedirect())


def _lower_headers(headers: Message | None) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(k).lower(): str(v) for k, v in headers.items()}


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """GET ``url`` with ``urllib`` (redirects only on the same host); the default ``Transport``.

    HTTP error statuses come back as ``HttpResponse`` (body empty if it cannot be read). A broken response
    (for example a truncated body) raises ``ConnectionError``; other network failures raise ``OSError``.
    """
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with _OPENER.open(request, timeout=timeout) as resp:
            return HttpResponse(int(resp.status), _lower_headers(resp.headers), resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except (OSError, http.client.HTTPException):
            body = b""
        return HttpResponse(int(exc.code), _lower_headers(exc.headers), body)
    except http.client.HTTPException as exc:
        raise ConnectionError(f"{type(exc).__name__}: {exc}") from exc


def is_not_bug_label(label: str) -> bool:
    """True for a triage verdict that an issue is not a bug ("Status: Not a Bug", "non-bug", "nobug")."""
    return bool(_NOT_BUG_LABEL.search(label))


def is_bug_label(label: str) -> bool:
    """True when ``label`` marks a bug report.

    Matches ``bug``, ``defect``, ``regression`` or ``crash`` as a whole word (case insensitive, so "debug" and
    "bugfix" do not count) or ``type: bug`` / ``kind/bug``, unless the label is a not-a-bug verdict.
    """
    if is_not_bug_label(label):
        return False
    return bool(_BUG_LABEL.search(label) or _BUG_TYPE_LABEL.search(label))


def is_bug_issue(labels: Sequence[str]) -> bool:
    """True when some label is a bug label and no label says "not a bug" (a triage verdict overrides)."""
    return any(is_bug_label(lb) for lb in labels) and not any(is_not_bug_label(lb) for lb in labels)


def _parse_time(value: Any) -> datetime | None:
    """An ISO 8601 string as an aware UTC datetime in whole seconds; None when missing or malformed."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def _label_names(value: Any) -> list[str]:
    """Label names in GitHub's order: dicts give their ``name``, strings are kept, the rest is dropped."""
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = item.get("name") if isinstance(item, Mapping) else item
        if isinstance(name, str) and name:
            names.append(name)
    return names


def issues_table(items: Iterable[Mapping[str, Any]]) -> pa.Table:
    """Turn raw GitHub issue objects into an ``ISSUES_SCHEMA`` table sorted by issue number.

    Items with a ``pull_request`` key are pull requests and are skipped. Items without an integer ``number``
    or a parseable ``created_at`` are skipped with one warning. ``closed_at`` is null while an issue is open.
    When a number appears twice (a crawl that overlapped), the last item wins. ``is_bug`` is
    ``is_bug_issue(labels)``. No items give an empty table with the schema.
    """
    rows: dict[int, tuple[datetime, datetime | None, list[str]]] = {}
    skipped = 0
    for item in items:
        if not isinstance(item, Mapping):
            skipped += 1
            continue
        if "pull_request" in item:
            continue
        number = item.get("number")
        created = _parse_time(item.get("created_at"))
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 0 < number <= _INT32_MAX
            or created is None
        ):
            skipped += 1
            continue
        rows[number] = (created, _parse_time(item.get("closed_at")), _label_names(item.get("labels")))
    if skipped:
        logger.warning("Skipped %d malformed issue records", skipped)
    numbers = sorted(rows)
    labels = [rows[n][2] for n in numbers]
    return pa.Table.from_pydict(
        {
            "number": numbers,
            "created_at": [rows[n][0] for n in numbers],
            "closed_at": [rows[n][1] for n in numbers],
            "labels": labels,
            "is_bug": [is_bug_issue(lb) for lb in labels],
        },
        schema=ISSUES_SCHEMA,
    )


def read_issues(path: Path) -> pa.Table:
    """Read an issues Parquet file as an ``ISSUES_SCHEMA`` table sorted by number.

    ``is_bug`` is recomputed from the labels with the current rule, so a cache never carries a stale flag.
    Raises ``ValueError`` when a required column (number, created_at, closed_at, labels) is missing, and
    ``OSError`` or ``pyarrow.ArrowException`` for unreadable files.
    """
    table = pq.read_table(path)
    missing = [c for c in _ISSUE_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"{path} lacks issue columns: {', '.join(missing)}")
    table = table.select(list(_ISSUE_COLUMNS)).combine_chunks()
    labels = table.column("labels").cast(ISSUES_SCHEMA.field("labels").type)
    is_bug = pa.chunked_array(
        [pa.array([is_bug_issue(lb or []) for lb in labels.to_pylist()], pa.bool_())], pa.bool_()
    )
    out = pa.Table.from_arrays(
        [
            table.column("number").cast(pa.int32()),
            table.column("created_at").cast(ISSUES_SCHEMA.field("created_at").type),
            table.column("closed_at").cast(ISSUES_SCHEMA.field("closed_at").type),
            labels,
            is_bug,
        ],
        schema=ISSUES_SCHEMA,
    )
    return out.sort_by("number")


def _check_name(value: str, what: str) -> None:
    if not _NAME.match(value) or value in {".", ".."}:
        raise ValueError(f"invalid GitHub repository {what}: {value!r}")


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme.lower(), parts.netloc.lower()


def _reset_wait(headers: Mapping[str, str], clock: Callable[[], float]) -> float:
    """Seconds until the primary rate limit resets (``x-ratelimit-reset``) plus one, within 1 s..65 min."""
    try:
        reset = int(headers.get("x-ratelimit-reset", ""))
    except ValueError:
        return _DEFAULT_WAIT_S
    return min(_MAX_WAIT_S, max(1.0, reset - clock() + 1.0))


def _retry_after(headers: Mapping[str, str]) -> float:
    """Seconds asked for by ``retry-after`` (secondary rate limit); 60 s when missing or malformed."""
    try:
        wait = float(headers.get("retry-after", ""))
    except ValueError:
        return _DEFAULT_WAIT_S
    if math.isnan(wait) or wait < 0:
        return _DEFAULT_WAIT_S
    return min(_MAX_WAIT_S, wait)


def _api_message(body: bytes) -> str:
    """The ``message`` field of a GitHub error body (shortened), or ``""``."""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    message = data.get("message") if isinstance(data, dict) else None
    return message[:200] if isinstance(message, str) else ""


def _get_page(
    url: str,
    headers: Mapping[str, str],
    *,
    repo: str,
    transport: Transport,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    max_retries: int,
    timeout: float,
) -> tuple[list[Any], str | None, Mapping[str, str]]:
    """GET one page with retries. Returns (items, next-page URL or None, response headers)."""
    attempt = 0  # retries consumed
    limit_waits = 0
    while True:
        status: int | None = None
        try:
            resp = transport(url, headers, timeout)
        except OSError as exc:
            reason = f"network error ({type(exc).__name__}: {exc})"
        else:
            status = resp.status
            remaining = resp.headers.get("x-ratelimit-remaining")
            if status == 200:
                try:
                    data = json.loads(resp.body)
                except ValueError:
                    data = None
                if isinstance(data, list):
                    match = _NEXT_LINK.search(resp.headers.get("link", ""))
                    return data, (match.group(1) if match else None), resp.headers
                reason = "response body is not a JSON list"
            elif status in (403, 429) and remaining == "0":
                if limit_waits >= _MAX_LIMIT_WAITS:
                    raise GitHubError(
                        f"GitHub rate limit still exhausted for {repo} after {limit_waits} waits",
                        status=status,
                    )
                limit_waits += 1
                wait = _reset_wait(resp.headers, clock)
                logger.warning("GitHub rate limit exhausted; waiting %.0f s for the reset", wait)
                sleep(wait)
                continue
            elif status == 429 or (status == 403 and b"rate limit" in resp.body.lower()):
                if attempt >= max_retries:
                    raise GitHubError(
                        f"GitHub secondary rate limit for {repo} persisted after {attempt + 1} attempts",
                        status=status,
                    )
                wait = _retry_after(resp.headers)
                logger.warning("GitHub secondary rate limit; waiting %.0f s", wait)
                sleep(wait)
                attempt += 1
                continue
            elif status >= 500:
                reason = f"HTTP {status}"
            elif status == 410:
                logger.warning("Issues are disabled for %s; returning no issues", repo)
                return [], None, resp.headers
            else:
                message = _api_message(resp.body)
                raise GitHubError(
                    f"GitHub API returned {status} for {repo}" + (f": {message}" if message else ""),
                    status=status,
                )
        if attempt >= max_retries:
            raise GitHubError(
                f"GitHub request for {repo} failed after {attempt + 1} attempts: {reason}", status=status
            )
        wait = min(_MAX_BACKOFF_S, 5.0 * 2**attempt)
        logger.warning("GitHub request for %s failed (%s); retrying in %.0f s", repo, reason, wait)
        sleep(wait)
        attempt += 1


def _write_cache(table: pa.Table, cache_path: Path) -> None:
    """Write atomically: a ``.tmp`` sibling, then ``os.replace``."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, cache_path)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_issues(
    owner: str,
    name: str,
    token: str | None,
    cache_path: Path | None = None,
    *,
    refresh: bool = False,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    max_retries: int = 5,
    timeout: float = 60.0,
    max_pages: int | None = None,
    api_url: str = API_URL,
) -> pa.Table:
    """All issues of ``owner/name`` (pull requests excluded) as an ``ISSUES_SCHEMA`` table sorted by number.

    ``token`` (``None`` or ``""`` for anonymous access, 60 requests per hour) is sent as a Bearer token to the
    API host only. With ``cache_path``, an existing readable cache is returned without any request unless
    ``refresh``; an unreadable cache is fetched again and overwritten. A complete crawl is written to the
    cache atomically. ``max_pages`` caps the pages fetched (100 issues and pull requests each); a capped
    crawl is partial and is never cached. Repositories with issues disabled (HTTP 410) give an empty table.

    Waits: when the primary rate limit is used up the crawl sleeps until ``x-ratelimit-reset``; secondary
    limits honour ``retry-after``; network errors, 5xx answers and truncated bodies are retried up to
    ``max_retries`` times with 5, 10, 20, 40, 80 s backoff. ``clock`` (Unix seconds) and ``sleep`` are
    injectable for tests, as is ``transport`` (default ``urllib_transport``). ``timeout`` is per request in
    seconds.

    Raises ``ValueError`` for an invalid owner or name, ``GitHubError`` when the API refuses the request
    (401, 404, 422, ...), retries run out, or a pagination link points to another host.
    """
    _check_name(owner, "owner")
    _check_name(name, "name")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if max_pages is not None and max_pages < 1:
        raise ValueError(f"max_pages must be >= 1, got {max_pages}")
    base = _origin(api_url)
    if base[0] not in ("https", "http") or not base[1]:
        raise ValueError(f"api_url must be an http(s) URL, got {api_url!r}")
    repo = f"{owner}/{name}"

    if cache_path is not None and not refresh and cache_path.exists():
        try:
            return read_issues(cache_path)
        except (OSError, ValueError, pa.ArrowException) as exc:
            logger.warning("Issues cache %s is unreadable (%s); fetching again", cache_path, exc)

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "entropiq-tdp",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    get = transport if transport is not None else urllib_transport

    url: str | None = (
        f"{api_url.rstrip('/')}/repos/{quote(owner, safe='')}/{quote(name, safe='')}/issues"
        f"?state=all&per_page={PER_PAGE}&sort=created&direction=asc"
    )
    items: list[Any] = []
    pages = 0
    complete = True
    started = time.perf_counter()
    while url is not None:
        page_items, next_url, resp_headers = _get_page(
            url,
            headers,
            repo=repo,
            transport=get,
            sleep=sleep,
            clock=clock,
            max_retries=max_retries,
            timeout=timeout,
        )
        pages += 1
        items.extend(page_items)
        if pages % 10 == 0:
            logger.info("%s: %d pages, %d items so far", repo, pages, len(items))
        url = next_url
        if url is None:
            break
        if _origin(url) != base:
            raise GitHubError(f"GitHub pagination for {repo} pointed to another host; refusing to follow it")
        if max_pages is not None and pages >= max_pages:
            logger.warning("%s: stopped after max_pages=%d; partial result, not cached", repo, pages)
            complete = False
            break
        if resp_headers.get("x-ratelimit-remaining") == "0":
            wait = _reset_wait(resp_headers, clock)
            logger.info("GitHub rate limit used up; waiting %.0f s before the next page", wait)
            sleep(wait)

    table = issues_table(items)
    n_bug = sum(1 for flag in table.column("is_bug").to_pylist() if flag)
    logger.info(
        "%s: %d pages, %d issues, %d bug issues in %.1f s",
        repo,
        pages,
        table.num_rows,
        n_bug,
        time.perf_counter() - started,
    )
    if cache_path is not None and complete:
        _write_cache(table, cache_path)
    return table
