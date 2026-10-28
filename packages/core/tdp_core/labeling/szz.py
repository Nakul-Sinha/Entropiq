"""SZZ: find the commits that introduced the lines a bug-fix commit deleted or modified.

For a bug-fix commit F the tracing rules are:

1. **Diff.** One ``git diff-tree -r -p -M -U0 -w`` of F against its first parent, with every option that
   changes hunks pinned (Myers algorithm, quoted paths, ``a/`` and ``b/`` prefixes, no inter-hunk context,
   no external diff or textconv), so the user's git configuration cannot change the result. From each hunk
   header ``@@ -a,b +c,d @@`` the parent-side range ``(a, b)`` is kept when ``b > 0``. A root commit and a
   merge commit print no diff, so they yield nothing.
2. **Files.** Only modified files are traced: added files, deleted files, binary entries, pure renames,
   mode-only changes, symlinks (mode 120000) and submodules (mode 160000) give no lines. A file pair is
   traced when ``is_source`` holds for its old or its new name (the same rule as bug-fix detection). A
   renamed and modified file is traced under its old name, the name in F's parent.
3. **Blame.** One ``git blame --line-porcelain -w <F>^ -L a,+b ... -- <path>`` per file (at most
   ``_MAX_RANGES_PER_BLAME`` ranges per call; ``blame.ignoreRevsFile`` is reset). Blame follows earlier
   renames, so an inducing commit may have written the line under another name.
4. **Content filter.** Parent lines that are blank, comment-only (``#`` after stripping) or import lines
   (``import x`` or ``from x import y``) are dropped, using blame's own copy of each line.
5. **Bulk and date filters.** Lines whose inducing commit is in ``bulk_shas`` are dropped, and so are lines
   whose inducing author time is after ``not_after`` (the creation time of the linked bug issue). A line
   exactly at ``not_after`` is kept.

Lines are reported with the path in F's parent (repo-relative POSIX, decoded like ``mine_history`` decodes
paths) and the 1-based line number in that parent revision. Every git call goes through
:func:`tdp_core.mining.git.run_git`, and revisions are checked with ``check_rev``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.compute as pc

from tdp_core.labeling.bugfix import default_is_source
from tdp_core.mining.git import GitError, check_rev, run_git
from tdp_core.mining.gitlog import MinedHistory, _unix_seconds

logger = logging.getLogger(__name__)

__all__ = ["SZZ_SCHEMA", "SZZ_VERSION", "InducingLine", "inducing_lines", "szz_table"]

# Bump when the tracing rules change; it is part of the cache file names, so old caches are ignored.
SZZ_VERSION: str = "1"

# The data/szz/<slug>.parquet layout of plan section 3.4, plus the additive ``n_lines`` column.
SZZ_SCHEMA = pa.schema(
    [
        ("fix_sha", pa.string()),
        ("fix_time", pa.timestamp("s", tz="UTC")),  # author time of the fix
        ("path", pa.string()),  # path in the fix's parent revision
        ("inducing_sha", pa.string()),
        ("inducing_time", pa.timestamp("s", tz="UTC")),  # author time of the inducing commit (from blame)
        ("n_lines", pa.int32()),  # inducing lines behind the row
    ]
)

# Ranges per blame call; module level so tests can lower it.
_MAX_RANGES_PER_BLAME = 200

_DIFF_ARGS = (
    "-c",
    "diff.algorithm=myers",
    "-c",
    "core.quotePath=true",
    "diff-tree",
    "-r",
    "-p",
    "-M",
    "-U0",
    "-w",
    "--inter-hunk-context=0",
    "--indent-heuristic",
    "--no-commit-id",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--no-relative",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)
_BLAME_ARGS = (
    "-c",
    "blame.ignoreRevsFile=",
    "-c",
    "diff.algorithm=myers",
    "-c",
    "core.quotePath=true",
    "--literal-pathspecs",
    "blame",
    "--line-porcelain",
    "-w",
)

_HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_BLAME_HEADER = re.compile(rb"^([0-9a-f]{40}(?:[0-9a-f]{24})?) (\d+) (\d+)(?: \d+)?$")
_IMPORT_LINE = re.compile(r"(?:import\s|from\s+\S+\s+import\b)")
_SPECIAL_MODES = frozenset({b"120000", b"160000"})  # symlinks and submodules
_C_ESCAPES = {
    ord("a"): 0x07,
    ord("b"): 0x08,
    ord("t"): 0x09,
    ord("n"): 0x0A,
    ord("v"): 0x0B,
    ord("f"): 0x0C,
    ord("r"): 0x0D,
    ord('"'): 0x22,
    ord("\\"): 0x5C,
}
_OCTAL = frozenset(b"01234567")


@dataclass(frozen=True, slots=True)
class InducingLine:
    """One line a fix deleted or modified, with the commit that last wrote it before the fix."""

    path: str  # path in the fix's parent revision (repo-relative POSIX)
    line: int  # 1-based line number in the parent revision
    inducing_sha: str
    inducing_time: datetime  # author time of the inducing commit, aware UTC


class _RawLine(NamedTuple):
    """A traced line before the bulk and date filters (what the cache stores)."""

    path: str
    line: int
    sha: str
    author_time: int  # unix seconds


class _Fix(NamedTuple):
    sha: str
    fix_time: int  # author time, unix seconds
    not_after: int | None  # issue creation time, unix seconds


# ------------------------------------------------------------------------------------------------ parsing
def _c_unquote(body: bytes) -> bytes:
    """Undo git's C-style quoting of a path (the text between the double quotes)."""
    out = bytearray()
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch != 0x5C or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in _C_ESCAPES:
            out.append(_C_ESCAPES[nxt])
            i += 2
        elif i + 4 <= n and all(c in _OCTAL for c in body[i + 1 : i + 4]):
            out.append(int(body[i + 1 : i + 4], 8) & 0xFF)
            i += 4
        else:  # unknown escape: keep it verbatim
            out.append(ch)
            i += 1
    return bytes(out)


def _unquote_label(raw: bytes) -> str | None:
    """The file name in a ``---`` or ``+++`` label of a patch, still carrying its ``a/`` or ``b/`` prefix.

    Git appends a TAB to labels whose name contains a space (one trailing TAB is stripped; a real TAB in a
    name is always quoted as ``\\t``). ``/dev/null`` gives None. A label in double quotes is C-unquoted
    (``\\a \\b \\t \\n \\v \\f \\r \\" \\\\`` and ``\\ooo`` octal bytes). Bytes are decoded as UTF-8 with
    replacement, the same way ``mine_history`` decodes paths, so the names equal the mined ones.
    """
    label = raw[:-1] if raw.endswith(b"\t") else raw
    if label == b"/dev/null":
        return None
    if len(label) >= 2 and label.startswith(b'"') and label.endswith(b'"'):
        label = _c_unquote(label[1:-1])
    return label.decode("utf-8", errors="replace")


def _strip_prefix(name: str | None, prefix: str) -> str | None:
    return None if name is None else name.removeprefix(prefix)


def _parse_diff(raw: bytes, is_source: Callable[[str], bool]) -> dict[str, list[tuple[int, int]]]:
    """Parent-side line ranges ``(start, count)`` per traced parent path, in diff order (see rule 2).

    ``raw`` is the output of ``git diff-tree -p -U0`` with ``a/`` and ``b/`` prefixes. In a hunk every line
    is body until the next ``@@`` or ``diff --git`` line, so a deleted line that reads ``-- x`` (printed as
    ``--- x``) is never taken for a header. ``is_source`` is called at most once per distinct name.
    """
    memo: dict[str, bool] = {}

    def source(name: str) -> bool:
        hit = memo.get(name)
        if hit is None:
            hit = memo[name] = bool(is_source(name))
        return hit

    ranges: dict[str, list[tuple[int, int]]] = {}
    old: str | None = None
    new: str | None = None
    old_mode: bytes | None = None
    in_header = False
    target: list[tuple[int, int]] | None = None
    for line in raw.split(b"\n"):
        if line.startswith(b"diff --git "):
            old = new = None
            old_mode = None
            in_header = True
            target = None
            continue
        if line.startswith(b"@@ "):
            if in_header:
                in_header = False
                keep = (
                    old is not None
                    and new is not None
                    and old_mode not in _SPECIAL_MODES
                    and (source(old) or source(new))
                )
                target = ranges.setdefault(old, []) if keep and old is not None else None
            if target is None:
                continue
            match = _HUNK.match(line)
            if match is None:
                raise GitError(f"unexpected hunk header in git diff-tree output: {line[:120]!r}")
            count = 1 if match[2] is None else int(match[2])
            if count > 0:
                target.append((int(match[1]), count))
            continue
        if not in_header:
            continue  # hunk body, or text before the first entry
        if line.startswith(b"--- "):
            old = _strip_prefix(_unquote_label(line[4:]), "a/")
        elif line.startswith(b"+++ "):
            new = _strip_prefix(_unquote_label(line[4:]), "b/")
        elif line.startswith((b"old mode ", b"deleted file mode ")):
            old_mode = line.rsplit(b" ", 1)[-1]
        elif line.startswith(b"index "):
            parts = line.split(b" ")
            if len(parts) >= 3:
                old_mode = parts[2]
    return {path: spans for path, spans in ranges.items() if spans}


def _parse_blame(raw: bytes) -> list[tuple[int, str, int, bytes]]:
    """``(final_line, sha, author_time, content)`` per line of ``git blame --line-porcelain`` output.

    Every entry is a header ``<sha> <orig_line> <final_line> [<group_size>]``, key lines, and the content
    line, which starts with a TAB (removed here) and ends the entry. Only ``author-time`` is read; other keys
    (``boundary``, ``previous``, ``filename``, ...) are ignored. Empty lines between entries are skipped.
    Raises ``GitError`` for a malformed entry (content without a header or without ``author-time``, or a
    truncated last entry).
    """
    out: list[tuple[int, str, int, bytes]] = []
    sha: str | None = None
    final_line = 0
    author_time: int | None = None
    for line in raw.split(b"\n"):
        if sha is None:
            if not line:
                continue
            match = _BLAME_HEADER.match(line)
            if match is None:
                raise GitError(f"unexpected git blame output: {line[:120]!r}")
            sha = match[1].decode("ascii")
            final_line = int(match[3])
            author_time = None
            continue
        if line.startswith(b"\t"):
            if author_time is None:
                raise GitError(f"git blame entry for {sha} line {final_line} has no author-time")
            out.append((final_line, sha, author_time, line[1:]))
            sha = None
        elif line.startswith(b"author-time "):
            try:
                author_time = int(line[len(b"author-time ") :])
            except ValueError as exc:
                raise GitError(f"bad author-time in git blame output: {line[:120]!r}") from exc
    if sha is not None:
        raise GitError(f"truncated git blame output after {sha} line {final_line}")
    return out


def _is_ignored_line(content: bytes) -> bool:
    """True for a blank, comment-only or import line (rule 4); the strip also removes a CRLF ``\\r``."""
    text = content.decode("utf-8", errors="replace").strip()
    return not text or text.startswith("#") or _IMPORT_LINE.match(text) is not None


# ------------------------------------------------------------------------------------------------ tracing
def _trace(repo_dir: Path, fix_sha: str, is_source: Callable[[str], bool], timeout: float) -> list[_RawLine]:
    """Rules 1 to 4 for one fix: traced lines sorted by (path, line), before the bulk and date filters."""
    check_rev(fix_sha)
    diff = run_git([*_DIFF_ARGS, fix_sha, "--"], repo_dir, timeout=timeout)
    by_path = _parse_diff(diff, is_source)
    lines: list[_RawLine] = []
    for path in sorted(by_path):
        if "\ufffd" in path:
            logger.debug("SZZ: %s skips %r (the name is not valid UTF-8)", fix_sha, path)
            continue
        spans = by_path[path]
        for start in range(0, len(spans), _MAX_RANGES_PER_BLAME):
            args = list(_BLAME_ARGS)
            for first, count in spans[start : start + _MAX_RANGES_PER_BLAME]:
                args += ["-L", f"{first},+{count}"]
            args += [f"{fix_sha}^", "--", path]
            blamed = _parse_blame(run_git(args, repo_dir, timeout=timeout))
            for final_line, sha, author_time, content in blamed:
                if not _is_ignored_line(content):
                    lines.append(_RawLine(path, final_line, sha, author_time))
    lines.sort()
    return lines


def _keep(lines: Sequence[_RawLine], bulk_shas: AbstractSet[str], not_after: int | None) -> list[_RawLine]:
    """Rule 5: drop lines from bulk commits and lines authored after ``not_after`` (unix seconds)."""
    return [
        line
        for line in lines
        if line.sha not in bulk_shas and (not_after is None or line.author_time <= not_after)
    ]


def inducing_lines(
    repo_dir: Path,
    fix_sha: str,
    *,
    is_source: Callable[[str], bool],
    bulk_shas: AbstractSet[str] = frozenset(),
    not_after: datetime | None = None,
    timeout: float = 300.0,
) -> list[InducingLine]:
    """The lines ``fix_sha`` deleted or modified in source files, with the commit that introduced each.

    Applies rules 1 to 5 of the module docstring. ``is_source`` decides which files are traced (pass
    ``default_is_source`` to match bug-fix detection). ``bulk_shas`` are commits whose lines never count
    (bulk commits); ``not_after`` (aware datetime, usually the linked issue's creation time) drops inducing
    commits authored strictly later. ``timeout`` is in seconds per git call. Results are sorted by
    ``(path, line)``; a fix that only adds lines, only adds or deletes whole files, or is a root commit gives
    ``[]``.

    Raises ``ValueError`` for an option-like ``fix_sha`` or a naive ``not_after``, and ``GitError``
    (``TransientGitError`` on a timeout) when git fails, for example for an unknown commit.
    """
    check_rev(fix_sha)
    cutoff = None if not_after is None else _unix_seconds(not_after)
    kept = _keep(_trace(repo_dir, fix_sha, is_source, timeout), bulk_shas, cutoff)
    return [
        InducingLine(line.path, line.line, line.sha, datetime.fromtimestamp(line.author_time, tz=UTC))
        for line in kept
    ]


# ------------------------------------------------------------------------------------------------ table
def _int_seconds(column: pa.ChunkedArray) -> list[int | None]:
    """A timestamp column (any unit) as whole unix seconds, nulls kept."""
    seconds = pc.cast(column, pa.timestamp("s", tz="UTC"), safe=False)
    values: list[int | None] = pc.cast(seconds, pa.int64()).to_pylist()
    return values


def _select_fixes(
    hist: MinedHistory, bugfix: pa.Table, max_fixes: int | None
) -> tuple[list[_Fix], frozenset[str]]:
    """The fixes to trace, newest first, and the bulk shas of ``hist``."""
    shas = hist.commits.column("sha").to_pylist()
    times = _int_seconds(hist.commits.column("author_time"))
    bulk_flags = hist.commits.column("is_bulk").to_pylist()
    info = {sha: (t, bool(bulk)) for sha, t, bulk in zip(shas, times, bulk_flags, strict=True)}
    bulk_shas = frozenset(sha for sha, bulk in zip(shas, bulk_flags, strict=True) if bulk)

    fix_shas = bugfix.column("sha").to_pylist()
    flags = bugfix.column("is_bugfix").to_pylist()
    if "issue_created_at" in bugfix.column_names:
        created = _int_seconds(bugfix.column("issue_created_at"))
    else:
        created = [None] * len(fix_shas)

    chosen: list[_Fix] = []
    seen: set[str] = set()
    missing = skipped_bulk = 0
    for sha, flag, issue_time in zip(fix_shas, flags, created, strict=True):
        if not flag or sha is None or sha in seen:
            continue
        seen.add(sha)
        commit = info.get(sha)
        if commit is None or commit[0] is None:
            missing += 1
            continue
        if commit[1]:
            skipped_bulk += 1
            continue
        chosen.append(_Fix(sha, commit[0], issue_time))
    chosen.sort(key=lambda fix: (-fix.fix_time, fix.sha))
    if max_fixes is not None:
        chosen = chosen[:max_fixes]
    if missing or skipped_bulk:
        logger.debug("SZZ: skipped %d fixes missing from the history, %d bulk fixes", missing, skipped_bulk)
    return chosen, bulk_shas


def _check_commits(repo_dir: Path, shas: Sequence[str], timeout: float) -> None:
    """Raise ``GitError`` unless every sha names a commit in ``repo_dir`` (one ``cat-file`` process)."""
    out = run_git(
        ["cat-file", "--batch-check=%(objectname) %(objecttype)"],
        repo_dir,
        timeout=timeout,
        input=("\n".join(shas) + "\n").encode("ascii", errors="replace"),
    )
    answers = out.decode("ascii", errors="replace").splitlines()
    bad = [sha for sha, answer in zip(shas, answers, strict=False) if answer.split(" ")[-1:] != ["commit"]]
    bad += list(shas[len(answers) :])
    if bad:
        raise GitError(
            f"{len(bad)} bug-fix commits are not commits of {repo_dir} (first: {', '.join(bad[:3])}); "
            "check repo_dir and that the history was mined from it"
        )


def _cache_file(cache_dir: Path, sha: str) -> Path:
    return cache_dir / sha[:2] / f"{sha}.v{SZZ_VERSION}.json"


def _cache_load(cache_dir: Path, sha: str) -> list[_RawLine] | None:
    """The cached raw lines of ``sha``, or None when the file is missing, unreadable or malformed."""
    path = _cache_file(cache_dir, sha)
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.debug("SZZ: ignoring unreadable cache file %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("fix_sha") != sha or data.get("version") != SZZ_VERSION:
        logger.debug("SZZ: ignoring cache file %s with an unexpected header", path)
        return None
    rows = data.get("lines")
    if not isinstance(rows, list):
        return None
    lines: list[_RawLine] = []
    for row in rows:
        if not (
            isinstance(row, list)
            and len(row) == 4
            and isinstance(row[0], str)
            and isinstance(row[1], int)
            and isinstance(row[2], str)
            and isinstance(row[3], int)
        ):
            logger.debug("SZZ: ignoring malformed cache file %s", path)
            return None
        lines.append(_RawLine(row[0], row[1], row[2], row[3]))
    return lines


def _cache_store(cache_dir: Path, sha: str, lines: Sequence[_RawLine]) -> None:
    """Write the raw lines of ``sha`` atomically; failures are logged at debug level and ignored."""
    path = _cache_file(cache_dir, sha)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    payload = {"fix_sha": sha, "version": SZZ_VERSION, "lines": [list(line) for line in lines]}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("SZZ: could not write cache file %s: %s", path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def szz_table(
    repo_dir: Path,
    hist: MinedHistory,
    bugfix: pa.Table,
    *,
    max_fixes: int | None = None,
    jobs: int = 4,
    is_source: Callable[[str], bool] = default_is_source,
    cache_dir: Path | None = None,
    timeout: float = 300.0,
) -> pa.Table:
    """Run SZZ for the bug fixes of ``bugfix`` and return a :data:`SZZ_SCHEMA` table.

    Fixes are the rows with ``is_bugfix`` true (null counts as false; a repeated sha uses its first such
    row) whose sha is a non-bulk commit of ``hist``; other rows are ignored. They are traced newest first
    (author time descending, ties by sha) and ``max_fixes`` keeps the newest N. Each fix's
    ``issue_created_at`` (optional column) is its ``not_after``, and the bulk commits of ``hist`` are the
    ``bulk_shas`` (see :func:`inducing_lines`).

    The output has one row per distinct (``fix_sha``, ``path``, ``inducing_sha``): ``fix_time`` is the fix's
    author time from ``hist``, ``inducing_time`` the inducing commit's author time from blame, ``n_lines``
    the number of traced lines behind the row. Rows are sorted by (``fix_time``, ``fix_sha``, ``path``,
    ``inducing_sha``), so the result is the same for any ``jobs`` and with or without the cache.

    Work runs in ``jobs`` threads (git does the work in child processes). All fix shas are checked first in
    one ``git cat-file`` call, which raises ``GitError`` when any is not a commit of ``repo_dir`` (a wrong
    repository or a stale table). After that, a fix whose git call fails (``GitError``, including a
    ``TransientGitError`` timeout after ``timeout`` seconds) is logged as a warning and skipped.

    ``cache_dir`` (optional) keeps one JSON file per fix (``<cache_dir>/<sha[:2]>/<sha>.v<SZZ_VERSION>.json``)
    holding its traced lines before the bulk and date filters, so one cache serves any bug-fix table (with
    or without issue dates) and an interrupted run resumes. The cache assumes the same ``is_source``: use one
    ``cache_dir`` per predicate. Unreadable or malformed cache files are recomputed.

    Raises ``ValueError`` for ``jobs < 1``, a negative ``max_fixes``, or a ``bugfix`` table without the
    ``sha`` and ``is_bugfix`` columns. No git runs when no fix is selected.
    """
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if max_fixes is not None and max_fixes < 0:
        raise ValueError(f"max_fixes must be >= 0, got {max_fixes}")
    missing = [c for c in ("sha", "is_bugfix") if c not in bugfix.column_names]
    if missing:
        raise ValueError(f"bugfix table lacks columns {missing}")

    fixes, bulk_shas = _select_fixes(hist, bugfix, max_fixes)
    if not fixes:
        return SZZ_SCHEMA.empty_table()
    started = time.perf_counter()
    _check_commits(repo_dir, [fix.sha for fix in fixes], timeout)

    def work(fix: _Fix) -> tuple[list[_RawLine] | None, bool]:
        if cache_dir is not None:
            cached = _cache_load(cache_dir, fix.sha)
            if cached is not None:
                return cached, True
        try:
            lines = _trace(repo_dir, fix.sha, is_source, timeout)
        except GitError as exc:
            logger.warning("SZZ skipped fix %s: %s", fix.sha, exc)
            return None, False
        if cache_dir is not None:
            _cache_store(cache_dir, fix.sha, lines)
        return lines, False

    if jobs == 1 or len(fixes) == 1:
        results = [work(fix) for fix in fixes]
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(fixes)), thread_name_prefix="szz") as pool:
            results = list(pool.map(work, fixes))

    groups: dict[tuple[int, str, str, str], list[int]] = {}
    for fix, (lines, _) in zip(fixes, results, strict=True):
        if lines is None:
            continue
        for line in _keep(lines, bulk_shas, fix.not_after):
            entry = groups.setdefault((fix.fix_time, fix.sha, line.path, line.sha), [0, line.author_time])
            entry[0] += 1
    keys = sorted(groups)
    table = pa.Table.from_pydict(
        {
            "fix_sha": [k[1] for k in keys],
            "fix_time": [k[0] for k in keys],
            "path": [k[2] for k in keys],
            "inducing_sha": [k[3] for k in keys],
            "inducing_time": [groups[k][1] for k in keys],
            "n_lines": [groups[k][0] for k in keys],
        },
        schema=SZZ_SCHEMA,
    )
    cached = sum(1 for lines, hit in results if hit)
    failed = sum(1 for lines, _ in results if lines is None)
    logger.info(
        "SZZ: %d fixes (%d traced, %d from cache, %d failed), %d rows in %.1f s",
        len(fixes),
        len(fixes) - cached - failed,
        cached,
        failed,
        table.num_rows,
        time.perf_counter() - started,
    )
    return table
