"""Snapshot times T and the commit that represents the repository at each T."""

from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, datetime

import pyarrow as pa

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import Snapshot
from tdp_core.mining.gitlog import _commit_times, _unix_seconds

_DAY = 86_400


def make_snapshots(repo: str, commits: pa.Table, settings: Settings = DEFAULT_SETTINGS) -> list[Snapshot]:
    """Training snapshots for ``repo`` (a slug) from a ``COMMITS_SCHEMA`` table.

    The first T is the first commit's author time plus ``settings.min_history_days``; later ones follow every
    ``settings.snapshot_step_days``. Only snapshots with ``T + settings.horizon_days <= last author time`` are
    kept, so every label window is fully observed. Each snapshot's ``sha`` is the last commit (by author time,
    ties by sha) with ``author_time <= T``. Indexes are 0-based. An empty table gives ``[]``.

    Raises ``ValueError`` if ``snapshot_step_days < 1``.
    """
    if settings.snapshot_step_days < 1:
        raise ValueError(f"snapshot_step_days must be >= 1, got {settings.snapshot_step_days}")
    order = _commit_times(commits)
    if not order:
        return []
    last = order[-1][0]
    t = order[0][0] + settings.min_history_days * _DAY
    snapshots: list[Snapshot] = []
    while t + settings.horizon_days * _DAY <= last:
        i = bisect_right(order, t, key=lambda row: row[0]) - 1
        snapshots.append(
            Snapshot(repo=repo, index=len(snapshots), time=datetime.fromtimestamp(t, UTC), sha=order[i][1])
        )
        t += settings.snapshot_step_days * _DAY
    return snapshots


def snapshot_at(repo: str, commits: pa.Table, t: datetime) -> Snapshot:
    """The snapshot of ``repo`` at time ``t`` (aware datetime; used at inference with ``t`` = commit time).

    ``sha`` is the last commit (by author time, ties by sha) with ``author_time <= t``; ``index`` is 0 and
    ``time`` is ``t`` converted to UTC. Raises ``ValueError`` if ``t`` is naive or precedes every commit.
    """
    cutoff = _unix_seconds(t)
    order = _commit_times(commits)
    i = bisect_right(order, cutoff, key=lambda row: row[0]) - 1
    if i < 0:
        raise ValueError(f"no commit of {repo} has author_time <= {t.isoformat()}")
    return Snapshot(repo=repo, index=0, time=t.astimezone(UTC), sha=order[i][1])
