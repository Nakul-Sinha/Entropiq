from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

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


def _rows(snaps: list[Snapshot]) -> list[tuple[int, datetime, str]]:
    return [(s.index, s.time, s.sha) for s in snaps]


def test_repo_dir_rule_on_leaky_history(
    repo_builder: RepoBuilder, t0: datetime, leaky_history: dict[str, str]
) -> None:
    h = leaky_history
    path = repo_builder.path
    commits = mine_history(path).commits
    settings = with_overrides(DEFAULT_SETTINGS, min_history_days=1, snapshot_step_days=4, horizon_days=4)

    old = make_snapshots("r", commits, settings)
    new = make_snapshots("r", commits, settings, repo_dir=path)

    times = [t0 + d * DAY for d in (1, 5, 9, 13, 17, 21)]
    assert [s.time for s in old] == [s.time for s in new] == times
    # The author-date rule already sees "late" (authored on day 4, landed on day 20) from day 5 on.
    assert [s.sha for s in old] == [h["base"]] + [h["late"]] * 5
    names = ("base", "main2", "main2", "merge", "merge", "late")
    assert _rows(new) == [(i, t, h[n]) for i, (t, n) in enumerate(zip(times, names, strict=True))]

    # From day 0 the first T has no state yet (base landed on day 1): it is skipped, indexes stay contiguous.
    settings = with_overrides(settings, min_history_days=0)
    assert _rows(make_snapshots("r", commits, settings, repo_dir=path)) == [
        (0, t0 + 4 * DAY, h["main2"]),
        (1, t0 + 8 * DAY, h["main2"]),
        (2, t0 + 12 * DAY, h["merge"]),
        (3, t0 + 16 * DAY, h["merge"]),
        (4, t0 + 20 * DAY, h["late"]),
    ]
    assert len(make_snapshots("r", commits, settings)) == 6


def test_repo_dir_matches_author_rule_on_clean_linear_history(
    repo_builder: RepoBuilder, t0: datetime
) -> None:
    for d in range(0, 101, 10):
        repo_builder.commit({"a.py": f"{d}\n"}, f"day {d}", t0 + d * DAY)
    settings = with_overrides(DEFAULT_SETTINGS, min_history_days=30, snapshot_step_days=20, horizon_days=40)
    commits = mine_history(repo_builder.path).commits

    new = make_snapshots("acme__widget", commits, settings, repo_dir=repo_builder.path)

    assert len(new) == 2
    assert new == make_snapshots("acme__widget", commits, settings)


def test_repo_dir_ties_follow_branch_order(repo_builder: RepoBuilder, t0: datetime) -> None:
    a = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    b = repo_builder.commit({"a.py": "2\n"}, "two", t0)
    commits = mine_history(repo_builder.path).commits

    # Equal dates: the state is the branch tip, not the larger sha the author-date rule picks.
    assert snapshot_at("r", commits, t0, repo_dir=repo_builder.path).sha == b
    assert snapshot_at("r", commits, t0).sha == max(a, b)


def test_repo_dir_without_snapshots_runs_no_git(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    missing = tmp_path / "missing"  # any git call here would fail
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    repo_builder.commit({"a.py": "2\n"}, "two", t0 + 500 * DAY)
    h = mine_history(repo_builder.path)

    assert make_snapshots("r", mine_history(repo_builder.path, max_commits=0).commits, repo_dir=missing) == []
    assert make_snapshots("r", h.commits, repo_dir=missing) == []
    with pytest.raises(ValueError):
        make_snapshots(
            "r", h.commits, with_overrides(DEFAULT_SETTINGS, snapshot_step_days=0), repo_dir=missing
        )


def test_snapshot_at_with_repo_dir(
    repo_builder: RepoBuilder, t0: datetime, leaky_history: dict[str, str]
) -> None:
    h = leaky_history
    path = repo_builder.path
    commits = mine_history(path).commits

    assert snapshot_at("r", commits, t0 + 2.5 * DAY).sha == h["feat"]
    assert snapshot_at("r", commits, t0 + 2.5 * DAY, repo_dir=path).sha == h["base"]
    assert snapshot_at("r", commits, t0 + 5 * DAY).sha == h["late"]
    assert snapshot_at("r", commits, t0 + 5 * DAY, repo_dir=path) == Snapshot(
        "r", 0, t0 + 5 * DAY, h["main2"]
    )

    at_merge = snapshot_at("r", commits, t0 + 12 * DAY, repo_dir=path)
    assert at_merge.sha == h["merge"]
    assert h["merge"] not in commits.column("sha").to_pylist()

    ist = timezone(timedelta(hours=5, minutes=30))
    local = snapshot_at("r", commits, (t0 + 12 * DAY).astimezone(ist), repo_dir=path)
    assert local.time.utcoffset() == timedelta(0)
    assert local == at_merge

    assert snapshot_at("r", commits, t0 + 0.5 * DAY).sha == h["base"]
    with pytest.raises(ValueError):
        snapshot_at("r", commits, t0 + 0.5 * DAY, repo_dir=path)
    with pytest.raises(ValueError):
        snapshot_at("r", commits, datetime(2022, 6, 1), repo_dir=path)
