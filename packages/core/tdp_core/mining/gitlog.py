"""Mine a repository's history into the ``commits`` and ``file_changes`` Arrow tables with one ``git log``."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA
from tdp_core.mining.git import GitError, check_rev, run_git

# Each record: \x1e sha \x1f name \x1f email \x1f unix-time \x1f parents \x1f body \x1f, then (with -z) a NUL,
# then, if the commit touched files, a newline and NUL-terminated numstat entries.
LOG_FORMAT = "%x1e%H%x1f%aN%x1f%aE%x1f%at%x1f%P%x1f%B%x1f"
_RECORD_START = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\x1f")

# Pinned so the user's global git config cannot change what we parse (hidden root diffs, signature lines,
# a different diff algorithm giving different line counts).
_LOG_CONFIG = ("-c", "log.showRoot=true", "-c", "log.showSignature=false", "-c", "diff.algorithm=myers")


@dataclass(frozen=True)
class MinedHistory:
    commits: pa.Table  # COMMITS_SCHEMA, sorted by author_time ascending (ties by sha)
    file_changes: pa.Table  # FILE_CHANGES_SCHEMA, rows in commit order, numstat order within a commit


class _Change(NamedTuple):
    path: str
    old_path: str | None
    added: int
    deleted: int
    is_binary: bool


def _unix_seconds(t: datetime) -> int:
    """Whole seconds since the epoch for an aware datetime (floored, so ``<= t`` comparisons stay exact)."""
    if t.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware: {t!r}")
    return math.floor(t.timestamp())


def _commit_times(commits: pa.Table) -> list[tuple[int, str]]:
    """``(author_time_unix, sha)`` for every commit, sorted ascending (ties by sha)."""
    times = commits.column("author_time").cast(pa.int64()).to_pylist()
    shas = commits.column("sha").to_pylist()
    return sorted(zip(times, shas, strict=True))


def mine_history(
    repo_dir: Path,
    rev: str = "HEAD",
    *,
    since: datetime | None = None,
    max_commits: int | None = None,
    settings: Settings = DEFAULT_SETTINGS,
) -> MinedHistory:
    """Mine all non-merge commits reachable from ``rev`` in one ``git log -M --numstat -z`` process.

    ``since`` (aware datetime) maps to ``--since``, which git applies to the committer date. ``max_commits``
    keeps the newest N non-merge commits (git's default newest-first walk). Author names and emails go through
    ``.mailmap``; emails are lowercased. Renames are detected with git's default 50% similarity. Binary
    changes have ``is_binary=True`` and ``added == deleted == 0``. A commit is bulk when it touches more than
    ``settings.bulk_commit_files`` files. Non-UTF-8 bytes in messages or paths are replaced with U+FFFD.

    Raises ``ValueError`` for a revision that starts with ``-`` or a negative ``max_commits``, and
    ``GitError`` when git fails (for example an unknown revision or a repository without commits).
    """
    check_rev(rev)
    args = [
        *_LOG_CONFIG,
        "log",
        "--no-merges",
        "-M",
        "--numstat",
        "-z",
        "--date=unix",
        "--encoding=UTF-8",
        "--no-color",
        f"--format={LOG_FORMAT}",
    ]
    if since is not None:
        args.append(f"--since=@{_unix_seconds(since)}")
    if max_commits is not None:
        if max_commits < 0:
            raise ValueError(f"max_commits must be >= 0, got {max_commits}")
        args += ["-n", str(max_commits)]
    args += ["--end-of-options", rev, "--"]
    return parse_git_log(run_git(args, repo_dir), settings)


def parse_git_log(raw: bytes, settings: Settings = DEFAULT_SETTINGS) -> MinedHistory:
    """Parse the output of ``git log -z --numstat --format=LOG_FORMAT`` into a :class:`MinedHistory`.

    Tolerates ``\\x1e`` and ``\\x1f`` inside commit messages (a record only starts at ``\\x1e`` followed by a
    full sha and ``\\x1f``, and the message ends at the first ``\\x1f`` followed by NUL). Raises ``GitError``
    on output it cannot parse.
    """
    text = raw.decode("utf-8", errors="replace")
    records: list[str] = []
    for chunk in text.split("\x1e")[1:]:
        if records and not _RECORD_START.match(chunk):
            records[-1] += "\x1e" + chunk  # a literal \x1e inside a message or path
        else:
            records.append(chunk)

    rows: list[tuple[int, str, str, str, list[str], str, list[_Change]]] = []
    for record in records:
        try:
            sha, name, email, at, parents, rest = record.split("\x1f", 5)
            end = rest.index("\x1f\x00")
            message = rest[:end].strip()
            changes = _parse_numstat(rest[end + 2 :])
            rows.append((int(at), sha, name, email.lower(), parents.split(), message, changes))
        except (ValueError, IndexError) as exc:
            raise GitError(f"could not parse git log record starting {record[:60]!r}") from exc
    rows.sort(key=lambda r: (r[0], r[1]))

    commit_cols: dict[str, list[Any]] = {name: [] for name in COMMITS_SCHEMA.names}
    change_cols: dict[str, list[Any]] = {name: [] for name in FILE_CHANGES_SCHEMA.names}
    for at_unix, sha, name, email, parent_list, message, changes in rows:
        commit_cols["sha"].append(sha)
        commit_cols["author_name"].append(name)
        commit_cols["author_email"].append(email)
        commit_cols["author_time"].append(at_unix)
        commit_cols["parents"].append(parent_list)
        commit_cols["subject"].append(message.split("\n", 1)[0].rstrip("\r"))
        commit_cols["message"].append(message)
        commit_cols["files_touched"].append(len(changes))
        commit_cols["is_bulk"].append(len(changes) > settings.bulk_commit_files)
        for ch in changes:
            change_cols["sha"].append(sha)
            change_cols["path"].append(ch.path)
            change_cols["old_path"].append(ch.old_path)
            change_cols["added"].append(ch.added)
            change_cols["deleted"].append(ch.deleted)
            change_cols["is_binary"].append(ch.is_binary)
    return MinedHistory(
        commits=pa.Table.from_pydict(commit_cols, schema=COMMITS_SCHEMA),
        file_changes=pa.Table.from_pydict(change_cols, schema=FILE_CHANGES_SCHEMA),
    )


def _parse_numstat(tail: str) -> list[_Change]:
    """Parse NUL-separated numstat entries.

    Normal entry ``added\\tdeleted\\tpath\\0``; rename ``added\\tdeleted\\t\\0old\\0new\\0``; binary uses
    ``-`` for both counts. Paths are verbatim under ``-z`` (no quoting), so tabs and newlines inside paths
    survive.
    """
    if tail.startswith("\n"):
        tail = tail[1:]
    tokens = tail.split("\x00")
    changes: list[_Change] = []
    i = 0
    while i < len(tokens):
        entry = tokens[i]
        i += 1
        if not entry:
            continue
        added_s, deleted_s, path = entry.split("\t", 2)
        old_path: str | None = None
        if not path:
            old_path, path = tokens[i], tokens[i + 1]
            i += 2
        binary = added_s == "-" or deleted_s == "-"
        changes.append(
            _Change(
                path=path,
                old_path=old_path,
                added=0 if binary else int(added_s),
                deleted=0 if binary else int(deleted_s),
                is_binary=binary,
            )
        )
    return changes
