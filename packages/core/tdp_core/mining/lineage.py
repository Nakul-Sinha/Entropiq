"""File lineage: every historical name of each file, following renames."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pyarrow as pa

from tdp_core.mining.gitlog import _commit_times, _unix_seconds


def path_lineage(file_changes: pa.Table, commits: pa.Table, until: datetime) -> dict[str, frozenset[str]]:
    """Map each path alive at ``until`` to the set of every name it had (including itself).

    Walks the changes of commits with ``author_time <= until`` (aware datetime) in time order (ties by sha);
    changes whose sha is not in ``commits`` are ignored. A rename row (``old_path`` set) moves the lineage of
    ``old_path`` to ``path``. Renames inside one commit are applied together, so swaps and chains cannot
    clobber each other.

    Deletions: ``FILE_CHANGES_SCHEMA`` has no change-type column, so a deletion is inferred from line counts.
    Each path keeps a running count (added minus deleted, carried through renames, starting at 0 when the path
    first appears); a text change with ``added == 0``, ``deleted > 0`` that brings the count to exactly 0
    deletes the path. A later change to that path starts a new lineage, so names from before the deletion are
    dropped. The count becomes unknown (and deletion is never inferred) after a binary change or if it would
    go negative, which happens for files that predate a ``since``-limited history. Consequences: deleting an
    empty or binary file is not detected (the path stays in the result), and a text file emptied to zero lines
    is treated as deleted (it is missing from the result). Callers should look paths up with
    ``lineage.get(path, frozenset({path}))``.

    Returns a dict sorted by path.
    """
    cutoff = _unix_seconds(until)
    order = [sha for t, sha in _commit_times(commits) if t <= cutoff]

    shas = file_changes.column("sha").to_pylist()
    paths = file_changes.column("path").to_pylist()
    old_paths = file_changes.column("old_path").to_pylist()
    added = file_changes.column("added").to_pylist()
    deleted = file_changes.column("deleted").to_pylist()
    binary = file_changes.column("is_binary").to_pylist()
    rows_by_sha: dict[str, list[int]] = defaultdict(list)
    for i, sha in enumerate(shas):
        rows_by_sha[sha].append(i)

    names: dict[str, set[str]] = {}
    lines: dict[str, int | None] = {}  # running line count per alive path; None = unknown
    for sha in order:
        rows = rows_by_sha.get(sha)
        if not rows:
            continue
        # Detach every rename source first.
        carried: dict[str, tuple[set[str], int | None]] = {}
        for i in rows:
            old = old_paths[i]
            if old is not None:
                carried[paths[i]] = (names.pop(old, {old}), lines.pop(old, None))
        for i in rows:
            path, old = paths[i], old_paths[i]
            count: int | None
            if old is not None:
                lineage, count = carried[path]
                lineage.add(path)
            elif path in names:
                lineage, count = names[path], lines[path]
            else:
                lineage, count = {path}, 0

            if binary[i] or count is None:
                count = None
            else:
                count += added[i] - deleted[i]
                if old is None and added[i] == 0 and deleted[i] > 0 and count == 0:
                    names.pop(path, None)
                    lines.pop(path, None)
                    continue
                if count < 0:
                    count = None
            names[path] = lineage
            lines[path] = count
    return {path: frozenset(names[path]) for path in sorted(names)}
