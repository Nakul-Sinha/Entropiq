from datetime import UTC, datetime, timedelta, timezone

import pytest

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import Snapshot
from tdp_core.mining import make_snapshots, mine_history, snapshot_at
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)


def test_default_settings_respect_min_history_and_horizon(repo_builder: RepoBuilder, t0: datetime) -> None:
    c0 = repo_builder.commit({"a.py": "0\n"}, "start", t0)
    for d in (400, 600, 900):
        repo_builder.commit({"a.py": f"{d}\n"}, f"day {d}", t0 + d * DAY)
    commits = mine_history(repo_builder.path).commits

    snaps = make_snapshots("acme__widget", commits)

    # T0 = t0 + 365 d; T1 = t0 + 547 d is dropped because T1 + 365 d is after the last commit (t0 + 900 d).
    assert snaps == [Snapshot(repo="acme__widget", index=0, time=t0 + 365 * DAY, sha=c0)]


def test_snapshot_schedule_and_shas(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas = {d: repo_builder.commit({"a.py": f"{d}\n"}, f"day {d}", t0 + d * DAY) for d in range(0, 101, 10)}
    settings = with_overrides(DEFAULT_SETTINGS, min_history_days=30, snapshot_step_days=20, horizon_days=40)
    commits = mine_history(repo_builder.path).commits

    snaps = make_snapshots("acme__widget", commits, settings)

    # T in {30, 50}: T = 70 is dropped since 70 + 40 > 100. A commit exactly at T is included.
    assert [(s.index, s.time, s.sha) for s in snaps] == [
        (0, t0 + 30 * DAY, shas[30]),
        (1, t0 + 50 * DAY, shas[50]),
    ]
    for s in snaps:
        assert s.time.tzinfo is not None
        assert s.time >= t0 + settings.min_history_days * DAY
        assert s.time + settings.horizon_days * DAY <= t0 + 100 * DAY


def test_short_or_empty_history_has_no_snapshots(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    repo_builder.commit({"a.py": "2\n"}, "two", t0 + 500 * DAY)
    h = mine_history(repo_builder.path)

    assert make_snapshots("r", h.commits) == []
    assert make_snapshots("r", mine_history(repo_builder.path, max_commits=0).commits) == []
    with pytest.raises(ValueError):
        make_snapshots("r", h.commits, with_overrides(DEFAULT_SETTINGS, snapshot_step_days=0))


def test_snapshot_at(repo_builder: RepoBuilder, t0: datetime) -> None:
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    second = repo_builder.commit({"a.py": "2\n"}, "two", t0 + 10 * DAY)
    commits = mine_history(repo_builder.path).commits

    assert snapshot_at("r", commits, t0 + 5 * DAY) == Snapshot("r", 0, t0 + 5 * DAY, first)
    assert snapshot_at("r", commits, t0 + 10 * DAY).sha == second
    ist = timezone(timedelta(hours=5, minutes=30))
    local = snapshot_at("r", commits, (t0 + 20 * DAY).astimezone(ist))
    assert local.time.utcoffset() == timedelta(0)
    assert local.time == t0 + 20 * DAY
    with pytest.raises(ValueError):
        snapshot_at("r", commits, t0 - DAY)
    with pytest.raises(ValueError):
        snapshot_at("r", commits, datetime(2022, 6, 1))


def test_ties_pick_the_largest_sha(repo_builder: RepoBuilder, t0: datetime) -> None:
    a = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    b = repo_builder.commit({"a.py": "2\n"}, "two", t0)
    commits = mine_history(repo_builder.path).commits

    assert snapshot_at("r", commits, t0).sha == max(a, b)
    assert snapshot_at("r", commits, datetime(2030, 1, 1, tzinfo=UTC)).sha == max(a, b)
