"""Snapshot times T and the commit that represents the repository at each T.

Two rules pick the commit. Without ``repo_dir`` (the original rule) it is the last commit of the table with
``author_time <= T``. With ``repo_dir`` it is the leakage-safe state from :mod:`tdp_core.mining.reach`: the
newest first-parent commit of the default branch with committer date ``<= T``.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import Snapshot
from tdp_core.mining.gitlog import _commit_times, _unix_seconds
from tdp_core.mining.reach import _head_ref, snapshot_sha

logger = logging.getLogger(__name__)

_DAY = 86_400


def _schedule(order: list[tuple[int, str]], settings: Settings) -> list[int]:
    """Snapshot times (unix seconds) for commits ``order`` (``(author_time, sha)`` sorted ascending).

    The first T is the first author time plus ``min_history_days``; later ones follow every
    ``snapshot_step_days`` while ``T + horizon_days <= last author time``. An empty ``order`` gives ``[]``.
    """
    if not order:
        return []
    last = order[-1][0]
    t = order[0][0] + settings.min_history_days * _DAY
    times: list[int] = []
    while t + settings.horizon_days * _DAY <= last:
        times.append(t)
        t += settings.snapshot_step_days * _DAY
    return times


def make_snapshots(
    repo: str,
    commits: pa.Table,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    repo_dir: Path | None = None,
) -> list[Snapshot]:
    """Training snapshots for ``repo`` (a slug) from a ``COMMITS_SCHEMA`` table.

    The first T is the first commit's author time plus ``settings.min_history_days``; later ones follow every
    ``settings.snapshot_step_days``. Only snapshots with ``T + settings.horizon_days <= last author time`` are
    kept, so every label window is fully observed. Both rules below use this same schedule. Indexes are
    0-based and contiguous. An empty table gives ``[]``.

    Without ``repo_dir`` each snapshot's ``sha`` is the last commit (by author time, ties by sha) with
    ``author_time <= T``.

    With ``repo_dir`` (a work tree or bare mirror; ``commits`` should come from ``mine_history(repo_dir)``,
    which mines ``HEAD``, the default branch) each ``sha`` is :func:`~tdp_core.mining.reach.snapshot_sha` at
    T: the newest first-parent commit of the default branch with committer date ``<= T``. It can be a merge
    commit, which is not in ``commits``. A T with no such commit is skipped (logged at debug level), so the
    first snapshot can start later than the schedule says. ``HEAD`` is resolved once and git runs once per T;
    no git runs when the schedule is empty.

    Raises ``ValueError`` if ``snapshot_step_days < 1`` (in both modes), and ``GitError`` with ``repo_dir``
    when ``HEAD`` is detached or git fails.
    """
    if settings.snapshot_step_days < 1:
        raise ValueError(f"snapshot_step_days must be >= 1, got {settings.snapshot_step_days}")
    order = _commit_times(commits)
    times = _schedule(order, settings)
    snapshots: list[Snapshot] = []
    if repo_dir is None:
        for t in times:
            i = bisect_right(order, t, key=lambda row: row[0]) - 1
            when = datetime.fromtimestamp(t, UTC)
            snapshots.append(Snapshot(repo=repo, index=len(snapshots), time=when, sha=order[i][1]))
        return snapshots
    if not times:
        return []
    ref = _head_ref(repo_dir)
    for t in times:
        when = datetime.fromtimestamp(t, UTC)
        sha = snapshot_sha(repo_dir, when, ref)
        if sha is None:
            logger.debug("%s has no first-parent commit with committer date <= %s", repo, when.isoformat())
            continue
        snapshots.append(Snapshot(repo=repo, index=len(snapshots), time=when, sha=sha))
    return snapshots


def snapshot_at(repo: str, commits: pa.Table, t: datetime, *, repo_dir: Path | None = None) -> Snapshot:
    """The snapshot of ``repo`` at time ``t`` (aware datetime; used at inference with ``t`` = commit time).

    ``index`` is 0 and ``time`` is ``t`` converted to UTC. Without ``repo_dir``, ``sha`` is the last commit
    (by author time, ties by sha) with ``author_time <= t``. With ``repo_dir``, ``sha`` is
    :func:`~tdp_core.mining.reach.snapshot_sha` at ``t`` on the default branch (it can be a merge commit) and
    ``commits`` is not consulted.

    Raises ``ValueError`` if ``t`` is naive or no commit qualifies (every commit is newer than ``t``), and
    ``GitError`` with ``repo_dir`` when ``HEAD`` is detached or git fails.
    """
    if repo_dir is not None:
        sha = snapshot_sha(repo_dir, t)
        if sha is None:
            raise ValueError(f"no first-parent commit of {repo} has committer date <= {t.isoformat()}")
        return Snapshot(repo=repo, index=0, time=t.astimezone(UTC), sha=sha)
    cutoff = _unix_seconds(t)
    order = _commit_times(commits)
    i = bisect_right(order, cutoff, key=lambda row: row[0]) - 1
    if i < 0:
        raise ValueError(f"no commit of {repo} has author_time <= {t.isoformat()}")
    return Snapshot(repo=repo, index=0, time=t.astimezone(UTC), sha=order[i][1])
