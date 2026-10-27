import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.testing import assert_frame_equal

from tdp_core.config import DEFAULT_SETTINGS, Settings, with_overrides
from tdp_core.contracts import COMMITS_SCHEMA, FEATURE_FAMILIES, FILE_CHANGES_SCHEMA, Snapshot
from tdp_core.features.evolution import (
    EVOLUTION_COLUMNS,
    OWNERSHIP_COLUMNS,
    TOUCH_COLUMNS,
    commit_activity,
    evolution_features,
    file_touches,
    ownership_features,
)
from tdp_core.mining import MinedHistory, mine_history, path_lineage, snapshot_at
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)
ANA = ("Ana", "ana@example.com")
BOB = ("Bob", "bob@example.com")
CAROL = ("Carol", "carol@example.com")

Commit = tuple[str, str, datetime, bool]  # sha, email, author time, is_bulk
Change = tuple[str, str, str | None, int, int]  # sha, path, old_path, added, deleted


def make_hist(commits: Sequence[Commit], changes: Sequence[Change]) -> MinedHistory:
    """A MinedHistory built directly from tuples (short shas are fine)."""
    ordered = sorted(commits, key=lambda c: (c[2], c[0]))
    touched = Counter(ch[0] for ch in changes)
    commit_table = pa.Table.from_pydict(
        {
            "sha": [c[0] for c in ordered],
            "author_name": [c[1].split("@")[0] for c in ordered],
            "author_email": [c[1] for c in ordered],
            "author_time": [c[2] for c in ordered],
            "parents": [[] for _ in ordered],
            "subject": ["m" for _ in ordered],
            "message": ["m" for _ in ordered],
            "files_touched": [touched[c[0]] for c in ordered],
            "is_bulk": [c[3] for c in ordered],
        },
        schema=COMMITS_SCHEMA,
    )
    change_table = pa.Table.from_pydict(
        {
            "sha": [ch[0] for ch in changes],
            "path": [ch[1] for ch in changes],
            "old_path": [ch[2] for ch in changes],
            "added": [ch[3] for ch in changes],
            "deleted": [ch[4] for ch in changes],
            "is_binary": [False for _ in changes],
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    return MinedHistory(commits=commit_table, file_changes=change_table)


def snap(t: datetime) -> Snapshot:
    return Snapshot("r", 0, t, "x")


def mined(
    repo: RepoBuilder,
    t: datetime,
    settings: Settings = DEFAULT_SETTINGS,
    *,
    since: datetime | None = None,
    rev: str = "HEAD",
) -> tuple[MinedHistory, dict[str, frozenset[str]]]:
    hist = mine_history(repo.path, rev, since=since, settings=settings)
    return hist, path_lineage(hist.file_changes, hist.commits, t)


def build_four_edits(repo: RepoBuilder, t0: datetime) -> None:
    repo.commit({"a.py": "1\n2\n3\n", "b.py": "x\n"}, "add", t0)
    repo.commit({"a.py": "1\n2\n3\n4\n"}, "grow", t0 + 50 * DAY)
    repo.commit({"a.py": "1\n2\n"}, "shrink", t0 + 110 * DAY)
    repo.commit({"a.py": "1\n2\nz\n"}, "edit", t0 + 190 * DAY)


def w(age: float, half_life: float = 90.0) -> float:
    return float(2 ** (-age / half_life))


def values(frame: pd.DataFrame, path: str, columns: Sequence[str] | None = None) -> list[float]:
    """One row of a feature frame as floats, in column order."""
    selected = frame.loc[[path], list(frame.columns) if columns is None else list(columns)]
    return [float(v) for v in selected.to_numpy(dtype=np.float64)[0]]


# ------------------------------------------------------------------------------------- RepoBuilder tests
def test_counts_decay_age_and_burstiness(repo_builder: RepoBuilder, t0: datetime) -> None:
    build_four_edits(repo_builder, t0)
    t = t0 + 200 * DAY
    hist, lineage = mined(repo_builder, t)

    evo = evolution_features(hist, snap(t), lineage, ["b.py", "a.py"])

    a = evo.loc["a.py"]
    assert a["commits_total"] == 4
    assert a["commits_90d"] == 2  # ages 90 (inclusive) and 10
    assert a["commits_180d"] == 3
    assert a["commits_decayed"] == pytest.approx(sum(w(x) for x in (200, 150, 90, 10)))
    assert a["churn_total"] == 7
    assert a["churn_decayed"] == pytest.approx(3 * w(200) + w(150) + 2 * 0.5 + w(10))
    assert a["age_days"] == 200.0
    assert a["staleness_days"] == 10.0
    assert a["burstiness"] == pytest.approx(np.std([50, 60, 80]) / np.mean([50, 60, 80]))
    assert a["is_new"] == 0
    assert values(evo, "b.py") == pytest.approx([1, 0, 0, w(200), 1, w(200), 200.0, 200.0, 0.0, 0])

    own = ownership_features(hist, snap(t), lineage, ["a.py", "b.py"])
    for path in ("a.py", "b.py"):
        assert own.loc[path, "n_authors"] == 1
        assert own.loc[path, "owner_share"] == 1.0
        assert own.loc[path, "minor_authors"] == 0
        assert own.loc[path, "author_entropy"] == 0.0
        assert math.copysign(1.0, values(own, path, ["author_entropy"])[0]) == 1.0
        # b.py: Ana last committed (to a.py) 10 days before T.
        assert own.loc[path, "owner_active"] == 1

    activity = commit_activity(hist, snap(t), lineage, ["a.py", "b.py"])
    expected_months = [str(p) for p in pd.period_range(end=t.strftime("%Y-%m"), periods=24, freq="M")]
    assert [m for m, _ in activity["a.py"]] == expected_months
    counts = dict(activity["a.py"])
    for when in (t0, t0 + 50 * DAY, t0 + 110 * DAY, t0 + 190 * DAY):
        assert counts[when.strftime("%Y-%m")] == 1
    assert sum(counts.values()) == 4
    assert dict(activity["b.py"]) == {m: int(m == t0.strftime("%Y-%m")) for m in expected_months}
    assert all(type(c) is int for _, c in activity["a.py"])


def test_decay_at_one_half_life(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "".join(f"{i}\n" for i in range(10))}, "add", t0)
    t = t0 + 90 * DAY
    hist, lineage = mined(repo_builder, t)

    evo = evolution_features(hist, snap(t), lineage, ["a.py"])
    assert evo.loc["a.py", "commits_decayed"] == 0.5
    assert evo.loc["a.py", "churn_decayed"] == 5.0
    assert evo.loc["a.py", "commits_90d"] == 1
    assert evo.loc["a.py", "is_new"] == 0  # 90 is not < 90

    fast = evolution_features(
        hist, snap(t), lineage, ["a.py"], with_overrides(DEFAULT_SETTINGS, half_life_days=45.0)
    )
    assert fast.loc["a.py", "commits_decayed"] == 0.25
    assert fast.loc["a.py", "churn_decayed"] == 2.5


def test_bulk_commits_are_ignored_except_for_age(repo_builder: RepoBuilder, t0: datetime) -> None:
    settings = with_overrides(DEFAULT_SETTINGS, bulk_commit_files=2)
    repo_builder.commit({"a.py": "1\n"}, "a", t0, author=ANA)
    repo_builder.commit({"a.py": "1\n2\n", "b.py": "b\n", "c.py": "c\n"}, "format", t0 + 10 * DAY, author=BOB)
    repo_builder.commit({"a.py": "1\n2\n3\n"}, "a again", t0 + 20 * DAY, author=ANA)
    t = t0 + 30 * DAY
    hist, lineage = mined(repo_builder, t, settings)
    assert hist.commits.column("is_bulk").to_pylist() == [False, True, False]
    paths = ["a.py", "b.py", "c.py"]

    evo = evolution_features(hist, snap(t), lineage, paths, settings)
    own = ownership_features(hist, snap(t), lineage, paths, settings)
    activity = commit_activity(hist, snap(t), lineage, paths)

    assert values(evo, "a.py", ["commits_total", "churn_total", "age_days", "staleness_days"]) == [
        2,
        2,
        30.0,
        10.0,
    ]
    assert own.loc["a.py", "n_authors"] == 1
    assert own.loc["a.py", "owner_share"] == 1.0
    for path in ("b.py", "c.py"):
        assert values(evo, path, ["commits_total", "commits_90d", "commits_180d", "churn_total"]) == [
            0,
            0,
            0,
            0,
        ]
        assert values(evo, path, ["commits_decayed", "churn_decayed", "burstiness"]) == [0.0, 0.0, 0.0]
        # The bulk commit proves the file existed 20 days before T.
        assert values(evo, path, ["age_days", "staleness_days", "is_new"]) == [20.0, 20.0, 1]
        assert values(own, path) == [0, 0.0, 0, 0.0, 0]
        assert all(c == 0 for _, c in activity[path])

    assert file_touches(hist, snap(t), lineage, ["b.py"]).empty
    with_bulk = file_touches(hist, snap(t), lineage, ["b.py"], include_bulk=True)
    assert with_bulk["is_bulk"].tolist() == [True]
    assert with_bulk["author_email"].tolist() == ["bob@example.com"]


def test_rename_lineage_is_followed(repo_builder: RepoBuilder, t0: datetime) -> None:
    body = "".join(f"line {i}\n" for i in range(10))
    repo_builder.commit({"old/a.py": body}, "add", t0, author=ANA)
    repo_builder.commit({"old/a.py": body + "x\n"}, "grow", t0 + 10 * DAY, author=ANA)
    repo_builder.rename(
        "old/a.py", "pkg/b.py", "move", t0 + 20 * DAY, author=BOB, new_content=body + "x\ny\n"
    )
    repo_builder.commit({"pkg/b.py": body + "x\ny\nz\n"}, "edit", t0 + 30 * DAY, author=BOB)
    t = t0 + 40 * DAY
    hist, lineage = mined(repo_builder, t)
    assert lineage["pkg/b.py"] == frozenset({"old/a.py", "pkg/b.py"})

    evo = evolution_features(hist, snap(t), lineage, ["pkg/b.py"])
    own = ownership_features(hist, snap(t), lineage, ["pkg/b.py"])

    row = evo.loc["pkg/b.py"]
    assert row["commits_total"] == 4
    assert row["churn_total"] == 13
    assert row["age_days"] == 40.0
    assert row["staleness_days"] == 10.0
    assert row["burstiness"] == 0.0  # evenly spaced
    assert values(own, "pkg/b.py") == [2, 0.5, 0, 1.0, 1]

    without = evolution_features(hist, snap(t), {}, ["pkg/b.py"])
    assert values(without, "pkg/b.py", ["commits_total", "churn_total", "age_days"]) == [2, 2, 20.0]


def test_untouched_files_and_windowed_history(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "a\n", "b.py": "b\n"}, "add", t0)
    repo_builder.commit({"b.py": "b\nc\n"}, "grow", t0 + 200 * DAY)
    t = t0 + 300 * DAY
    paths = ["a.py", "b.py", "ghost.py"]

    hist, lineage = mined(repo_builder, t)
    full = evolution_features(hist, snap(t), lineage, paths)
    assert values(full, "a.py", ["age_days", "staleness_days"]) == [300.0, 300.0]

    # Only the last 100 days are mined: a.py exists but has no touching commit in the window.
    window, window_lineage = mined(repo_builder, t, since=t0 + 100 * DAY)
    evo = evolution_features(window, snap(t), window_lineage, paths)
    own = ownership_features(window, snap(t), window_lineage, paths)
    for path in ("a.py", "ghost.py"):
        assert evo.loc[path, "commits_total"] == 0
        assert evo.loc[path, "age_days"] == 100.0  # the observed history span, a lower bound
        assert evo.loc[path, "staleness_days"] == 100.0
        assert evo.loc[path, "is_new"] == 0
        assert values(own, path) == [0, 0.0, 0, 0.0, 0]
    assert evo.loc["b.py", "age_days"] == 100.0
    assert evo.loc["b.py", "commits_total"] == 1

    # Before the first commit there is no history at all.
    early = t0 - DAY
    hist_e, lineage_e = mined(repo_builder, early)
    evo_e = evolution_features(hist_e, snap(early), lineage_e, paths)
    assert (evo_e.drop(columns="is_new") == 0).all().all()
    assert evo_e["is_new"].tolist() == [1, 1, 1]
    assert (ownership_features(hist_e, snap(early), lineage_e, paths) == 0).all().all()
    activity = commit_activity(hist_e, snap(early), lineage_e, paths)
    assert all(c == 0 for series in activity.values() for _, c in series)


def test_no_leakage_from_commits_after_t(repo_builder: RepoBuilder, t0: datetime) -> None:
    settings = with_overrides(DEFAULT_SETTINGS, bulk_commit_files=2)
    repo_builder.commit({"a.py": "a\n", "b.py": "b\n"}, "add", t0, author=ANA)
    repo_builder.commit({"a.py": "a\na\n"}, "edit a", t0 + 10 * DAY, author=BOB)
    repo_builder.commit({"b.py": "b\nb\n"}, "edit b", t0 + 20 * DAY, author=ANA)
    t = t0 + 25 * DAY
    repo_builder.commit({"a.py": "a\na\na\n"}, "new author", t0 + 30 * DAY, author=CAROL)
    repo_builder.rename("a.py", "c.py", "move", t0 + 40 * DAY, author=CAROL)
    repo_builder.commit({"b.py": "z\n", "c.py": "z\n", "d.py": "z\n"}, "bulk", t0 + 50 * DAY, author=CAROL)
    paths = ["a.py", "b.py"]

    full, full_lineage = mined(repo_builder, t, settings)
    sha_at_t = snapshot_at("r", full.commits, t).sha
    past, past_lineage = mined(repo_builder, t, settings, rev=sha_at_t)
    assert past.commits.num_rows == 3 and full.commits.num_rows == 6

    for fn in (evolution_features, ownership_features):
        assert_frame_equal(
            fn(full, snap(t), full_lineage, paths, settings), fn(past, snap(t), past_lineage, paths, settings)
        )
    assert commit_activity(full, snap(t), full_lineage, paths) == commit_activity(
        past, snap(t), past_lineage, paths
    )


def test_reachable_set_matching_the_time_rule_gives_identical_results(
    repo_builder: RepoBuilder, t0: datetime
) -> None:
    build_four_edits(repo_builder, t0)
    t = t0 + 100 * DAY
    hist, lineage = mined(repo_builder, t)
    times = hist.commits.column("author_time").cast(pa.int64()).to_pylist()
    shas = hist.commits.column("sha").to_pylist()
    reachable = frozenset(s for s, at in zip(shas, times, strict=True) if at <= t.timestamp())
    assert len(reachable) == 2
    paths = ["a.py", "b.py"]

    for fn in (evolution_features, ownership_features):
        assert_frame_equal(
            fn(hist, snap(t), lineage, paths), fn(hist, snap(t), lineage, paths, reachable=reachable)
        )
    assert commit_activity(hist, snap(t), lineage, paths) == commit_activity(
        hist, snap(t), lineage, paths, reachable=reachable
    )
    assert_frame_equal(
        file_touches(hist, snap(t), lineage, paths),
        file_touches(hist, snap(t), lineage, paths, reachable=reachable),
    )


# --------------------------------------------------------------------------------------- synthetic tests
T = datetime(2024, 3, 15, 12, tzinfo=UTC)


def test_minor_authors_and_entropy() -> None:
    commits: list[Commit] = [(f"a{i:02d}", "ana@x.org", T - (30 - i) * DAY, False) for i in range(20)]
    commits.append(("b00", "bob@x.org", T - 5 * DAY, False))
    hist = make_hist(commits, [(c[0], "f.py", None, 1, 0) for c in commits])

    own = ownership_features(hist, snap(T), {}, ["f.py"])
    row = own.loc["f.py"]
    assert row["n_authors"] == 2
    assert row["owner_share"] == pytest.approx(20 / 21)
    assert row["minor_authors"] == 1  # 1/21 < 0.05
    expected = -(20 / 21 * math.log2(20 / 21) + 1 / 21 * math.log2(1 / 21))
    assert row["author_entropy"] == pytest.approx(expected)
    assert row["owner_active"] == 1

    loose = ownership_features(
        hist, snap(T), {}, ["f.py"], with_overrides(DEFAULT_SETTINGS, minor_author_share=0.04)
    )
    assert loose.loc["f.py", "minor_authors"] == 0


def test_owner_ties_and_owner_activity() -> None:
    base: list[Commit] = [
        ("f1", "ana@x.org", T - 300 * DAY, False),
        ("f2", "bob@x.org", T - 250 * DAY, False),
        ("f3", "ana@x.org", T - 200 * DAY, False),
        ("f4", "bob@x.org", T - 150 * DAY, False),
    ]
    base_changes: list[Change] = [(c[0], "f.py", None, 1, 0) for c in base]

    # Tie on two commits each: bob touched f.py more recently, so bob owns it. The window is inclusive.
    hist = make_hist(
        [*base, ("g1", "bob@x.org", T - 90 * DAY, False)], [*base_changes, ("g1", "g.py", None, 1, 0)]
    )
    own = ownership_features(hist, snap(T), {}, ["f.py"])
    assert values(own, "f.py", ["n_authors", "owner_share", "author_entropy"]) == [2, 0.5, 1.0]
    assert own.loc["f.py", "owner_active"] == 1

    # Bob's last counted commit is 91 days old; his bulk commit and Ana's recent commit do not help.
    later: list[Commit] = [
        ("g1", "bob@x.org", T - 91 * DAY, False),
        ("g2", "ana@x.org", T - 10 * DAY, False),
        ("g3", "bob@x.org", T - 5 * DAY, True),
    ]
    later_changes: list[Change] = [(c[0], "g.py", None, 1, 0) for c in later]
    hist = make_hist([*base, *later], [*base_changes, *later_changes])
    assert ownership_features(hist, snap(T), {}, ["f.py"]).loc["f.py", "owner_active"] == 0

    # One more commit makes Ana the owner, and she is active.
    extra: Commit = ("f0", "ana@x.org", T - 400 * DAY, False)
    hist = make_hist([extra, *base, *later], [("f0", "f.py", None, 1, 0), *base_changes, *later_changes])
    own = ownership_features(hist, snap(T), {}, ["f.py"])
    assert own.loc["f.py", "owner_share"] == pytest.approx(0.6)
    assert own.loc["f.py", "owner_active"] == 1


def test_owner_tie_on_recency_falls_back_to_email() -> None:
    commits: list[Commit] = [
        ("c1", "zed@x.org", T - 200 * DAY, False),
        ("c2", "amy@x.org", T - 200 * DAY, False),
    ]
    commits.append(("c3", "zed@x.org", T - 10 * DAY, False))  # zed is active, amy is not
    hist = make_hist(
        commits, [("c1", "f.py", None, 1, 0), ("c2", "f.py", None, 1, 0), ("c3", "g.py", None, 1, 0)]
    )
    assert ownership_features(hist, snap(T), {}, ["f.py"]).loc["f.py", "owner_active"] == 0


@pytest.mark.parametrize(
    ("ages", "expected"),
    [
        ((30, 20, 10), 0.0),
        ((10, 9, 6), 0.5),  # gaps 1 and 3: std 1 / mean 2
        ((10, 5), 0.0),  # fewer than 3 commits
        ((7, 7, 7), 0.0),  # same second: mean gap 0
    ],
)
def test_burstiness(ages: tuple[int, ...], expected: float) -> None:
    commits: list[Commit] = [(f"c{i}", "ana@x.org", T - age * DAY, False) for i, age in enumerate(ages)]
    hist = make_hist(commits, [(c[0], "f.py", None, 1, 0) for c in commits])
    assert evolution_features(hist, snap(T), {}, ["f.py"]).loc["f.py", "burstiness"] == pytest.approx(
        expected
    )


def test_reachable_set_replaces_the_time_filter() -> None:
    commits: list[Commit] = [
        ("c1", "ana@x.org", T - 7 * DAY, False),
        ("c2", "ana@x.org", T + 3 * DAY, False),  # authored after T but reachable (rebased)
        ("c3", "ana@x.org", T - 2 * DAY, False),  # authored before T but merged later
    ]
    changes: list[Change] = [
        ("c1", "f.py", None, 1, 0),
        ("c2", "f.py", None, 2, 0),
        ("c3", "f.py", None, 3, 1),
    ]
    hist = make_hist(commits, changes)

    default = evolution_features(hist, snap(T), {}, ["f.py"])
    assert values(default, "f.py", ["commits_total", "churn_total", "staleness_days"]) == [2, 5, 2.0]

    reach = {"c1", "c2", "unknown-sha"}
    row = evolution_features(hist, snap(T), {}, ["f.py"], reachable=reach).loc["f.py"]
    assert row["commits_total"] == 2
    assert row["churn_total"] == 3
    assert row["staleness_days"] == 0.0  # clipped at T
    assert row["age_days"] == 7.0
    assert row["commits_decayed"] == pytest.approx(w(7) + 1.0)
    activity = dict(commit_activity(hist, snap(T), {}, ["f.py"], reachable=reach)["f.py"])
    assert activity["2024-03"] == 2  # c2 counts in T's month

    nothing = evolution_features(hist, snap(T), {}, ["f.py"], reachable=set()).loc["f.py"]
    assert nothing["age_days"] == 0.0
    assert nothing["is_new"] == 1
    assert nothing["commits_total"] == 0

    touches = file_touches(hist, snap(T), {}, ["f.py"], reachable=reach)
    assert touches["sha"].tolist() == ["c1", "c2"]
    assert touches["author_time"].tolist()[1] == int((T + 3 * DAY).timestamp())
    assert touches["age_days"].tolist() == [7.0, 0.0]


def test_commit_activity_months() -> None:
    commits: list[Commit] = [
        ("c1", "ana@x.org", datetime(2024, 3, 1, tzinfo=UTC), False),
        ("c2", "ana@x.org", T, False),  # exactly T: included
        ("c3", "ana@x.org", datetime(2024, 2, 10, tzinfo=UTC), False),
        ("c4", "ana@x.org", datetime(2022, 4, 1, tzinfo=UTC), False),  # first month of the window
        ("c5", "ana@x.org", datetime(2022, 3, 31, 23, 59, 59, tzinfo=UTC), False),  # before the window
        ("c6", "ana@x.org", datetime(2024, 3, 20, tzinfo=UTC), False),  # after T
        ("c7", "ana@x.org", datetime(2024, 1, 5, tzinfo=UTC), True),  # bulk
    ]
    hist = make_hist(commits, [(c[0], "f.py", None, 1, 0) for c in commits])

    activity = commit_activity(hist, snap(T), {}, ["untouched.py", "f.py"])
    assert list(activity) == ["f.py", "untouched.py"]
    labels = [m for m, _ in activity["f.py"]]
    assert labels[0] == "2022-04" and labels[-1] == "2024-03" and len(labels) == 24
    counts = dict(activity["f.py"])
    assert counts == {m: {"2024-03": 2, "2024-02": 1, "2022-04": 1}.get(m, 0) for m in labels}
    assert [c for _, c in activity["untouched.py"]] == [0] * 24

    assert commit_activity(hist, snap(T), {}, ["f.py"], months=1) == {"f.py": (("2024-03", 2),)}
    with pytest.raises(ValueError, match="months"):
        commit_activity(hist, snap(T), {}, ["f.py"], months=0)


def test_file_touches_rows_and_churn() -> None:
    commits: list[Commit] = [
        ("r0", "ana@x.org", T - 30 * DAY, False),
        ("r1", "bob@x.org", T - 20 * DAY, False),
        ("r2", "ana@x.org", T - 10 * DAY, False),
    ]
    changes: list[Change] = [
        ("r0", "y", None, 2, 0),
        ("r0", "z", None, 4, 0),
        ("r1", "x", "y", 1, 2),  # rename: both names match, counted once
        ("r2", "x", None, 1, 1),
        ("r2", "y", None, 5, 0),  # a new file reusing the old name still matches the lineage
    ]
    hist = make_hist(commits, changes)
    lineage = {"x": frozenset({"x", "y"})}

    touches = file_touches(hist, snap(T), lineage, ["z", "x", "x"])
    assert tuple(touches.columns) == TOUCH_COLUMNS
    assert isinstance(touches.index, pd.RangeIndex)
    assert touches[["path", "sha", "churn"]].to_dict("records") == [
        {"path": "x", "sha": "r0", "churn": 2},
        {"path": "x", "sha": "r1", "churn": 3},
        {"path": "x", "sha": "r2", "churn": 7},
        {"path": "z", "sha": "r0", "churn": 4},
    ]
    assert touches["author_email"].tolist() == ["ana@x.org", "bob@x.org", "ana@x.org", "ana@x.org"]
    assert touches["age_days"].tolist() == [30.0, 20.0, 10.0, 30.0]
    assert touches["is_bulk"].tolist() == [False] * 4
    assert {c: str(touches[c].dtype) for c in ("author_time", "age_days", "churn", "is_bulk")} == {
        "author_time": "int64",
        "age_days": "float64",
        "churn": "int64",
        "is_bulk": "bool",
    }
    assert all(pd.api.types.is_string_dtype(touches[c]) for c in ("path", "sha", "author_email"))

    empty = file_touches(hist, snap(T), lineage, [])
    assert tuple(empty.columns) == TOUCH_COLUMNS and empty.empty


def test_output_conventions() -> None:
    commits: list[Commit] = [
        ("c1", "ana@x.org", T - 10 * DAY, False),
        ("c2", "bob@x.org", T - 5 * DAY, False),
    ]
    hist = make_hist(commits, [("c1", "b.py", None, 1, 0), ("c2", "a.py", None, 1, 1)])
    paths = ["b.py", "a.py", "zz.py", "a.py"]

    evo = evolution_features(hist, snap(T), {}, paths)
    own = ownership_features(hist, snap(T), {}, paths)
    assert tuple(evo.columns) == EVOLUTION_COLUMNS
    assert tuple(own.columns) == OWNERSHIP_COLUMNS
    assert set(EVOLUTION_COLUMNS) == set(FEATURE_FAMILIES["E"]) - {"rel_churn_decayed", "cc_trend"}
    assert FEATURE_FAMILIES["O"] == OWNERSHIP_COLUMNS
    int_columns = {"commits_total", "commits_90d", "commits_180d", "churn_total", "is_new"}
    int_columns |= {"n_authors", "minor_authors", "owner_active"}
    for frame in (evo, own):
        assert frame.index.tolist() == ["a.py", "b.py", "zz.py"]
        assert frame.index.name == "path"
        assert pd.api.types.is_string_dtype(frame.index)
        assert not frame.isna().any().any()
        for col in frame.columns:
            assert frame[col].dtype == (np.int64 if col in int_columns else np.float64), col

    for fn in (evolution_features, ownership_features):
        empty = fn(hist, snap(T), {}, [])
        assert empty.empty and empty.index.name == "path"
        assert empty.dtypes.to_dict() == (evo if fn is evolution_features else own).dtypes.to_dict()
    assert commit_activity(hist, snap(T), {}, []) == {}


def test_validation() -> None:
    hist = make_hist([("c1", "ana@x.org", T - DAY, False)], [("c1", "f.py", None, 1, 0)])
    naive = snap(datetime(2024, 3, 15, 12))
    with pytest.raises(ValueError, match="timezone-aware"):
        evolution_features(hist, naive, {}, ["f.py"])
    with pytest.raises(ValueError, match="timezone-aware"):
        ownership_features(hist, naive, {}, ["f.py"])
    with pytest.raises(ValueError, match="timezone-aware"):
        commit_activity(hist, naive, {}, ["f.py"])
    with pytest.raises(ValueError, match="timezone-aware"):
        file_touches(hist, naive, {}, ["f.py"])
    with pytest.raises(ValueError, match="half_life_days"):
        evolution_features(hist, snap(T), {}, ["f.py"], with_overrides(DEFAULT_SETTINGS, half_life_days=0.0))


# ------------------------------------------------------------------------- reference and performance
def random_history(
    rng: np.random.Generator, n_commits: int, n_files: int, n_renamed: int, span_days: int, end: datetime
) -> tuple[MinedHistory, dict[str, frozenset[str]], list[str], list[Commit], list[Change]]:
    """Random history: skewed authors, 1 to 5 files per commit, 1% bulk, some files renamed halfway."""
    end_s = int(end.timestamp())
    times = np.sort(rng.integers(end_s - span_days * 86_400, end_s, n_commits))
    author_p = 1.0 / np.arange(1, 61)
    authors = rng.choice(60, size=n_commits, p=author_p / author_p.sum())
    bulk = rng.random(n_commits) < 0.01
    commits: list[Commit] = [
        (f"{i:040x}", f"dev{authors[i]}@x.org", datetime.fromtimestamp(int(times[i]), UTC), bool(bulk[i]))
        for i in range(n_commits)
    ]
    half = n_commits // 2
    rename_at = {j: half + j for j in range(n_renamed)}

    def name(j: int, i: int) -> str:
        return f"old/m{j}.py" if j in rename_at and i < rename_at[j] else f"pkg/m{j}.py"

    per_commit = rng.integers(1, 6, n_commits)
    commit_of_row = np.repeat(np.arange(n_commits), per_commit)
    file_of_row = rng.integers(0, n_files, len(commit_of_row))
    pairs = np.unique(np.stack([commit_of_row, file_of_row], axis=1), axis=0)
    lines = rng.integers(0, 40, (len(pairs), 2))
    changes: list[Change] = []
    renames = {i: j for j, i in rename_at.items()}
    for (i, j), (added, deleted) in zip(pairs.tolist(), lines.tolist(), strict=True):
        if renames.get(i) == j:
            continue
        changes.append((commits[i][0], name(j, i), None, added, deleted))
    for j, i in rename_at.items():
        changes.append((commits[i][0], f"pkg/m{j}.py", f"old/m{j}.py", 1, 0))
    lineage = {f"pkg/m{j}.py": frozenset({f"old/m{j}.py", f"pkg/m{j}.py"}) for j in rename_at}
    paths = [f"pkg/m{j}.py" for j in range(n_files)]
    return make_hist(commits, changes), lineage, paths, commits, changes


def reference_features(
    commits: Sequence[Commit],
    changes: Sequence[Change],
    t: datetime,
    lineage: Mapping[str, frozenset[str]],
    paths: Sequence[str],
    settings: Settings,
) -> pd.DataFrame:
    """Slow, loop-based reference for the evolution and ownership rules."""
    ts = t.timestamp()
    history = sorted((c for c in commits if c[2].timestamp() <= ts), key=lambda c: (c[2], c[0]))
    rows_by_sha: dict[str, list[Change]] = defaultdict(list)
    for ch in changes:
        rows_by_sha[ch[0]].append(ch)
    last_counted: dict[str, float] = {}
    for _sha, email, when, bulk in history:
        if not bulk:
            last_counted[email] = (ts - when.timestamp()) / 86_400
    span = (ts - history[0][2].timestamp()) / 86_400 if history else 0.0
    out = {}
    for p in sorted(set(paths)):
        names = lineage.get(p, frozenset()) | {p}
        touches = []
        for sha, email, when, bulk in history:
            rows = [r for r in rows_by_sha[sha] if r[1] in names or (r[2] is not None and r[2] in names)]
            if rows:
                touches.append(
                    (email, (ts - when.timestamp()) / 86_400, sum(r[3] + r[4] for r in rows), bulk)
                )
        counted = [x for x in touches if not x[3]]
        ages = [x[1] for x in counted]
        age_days = max(x[1] for x in touches) if touches else span
        gaps = [ages[i] - ages[i + 1] for i in range(len(ages) - 1)]
        burst = float(np.std(gaps) / np.mean(gaps)) if len(gaps) >= 2 and np.mean(gaps) > 0 else 0.0
        by_author = Counter(x[0] for x in counted)
        shares = {a: k / len(counted) for a, k in by_author.items()}
        owner_active = 0
        if counted:
            newest = {a: min(x[1] for x in counted if x[0] == a) for a in by_author}
            owner = min(by_author, key=lambda a: (-by_author[a], newest[a], a))
            owner_active = int(last_counted[owner] <= settings.owner_active_days)
        weights = [2 ** (-a / settings.half_life_days) for a in ages]
        out[p] = {
            "commits_total": len(counted),
            "commits_90d": sum(a <= 90 for a in ages),
            "commits_180d": sum(a <= 180 for a in ages),
            "commits_decayed": sum(weights),
            "churn_total": sum(x[2] for x in counted),
            "churn_decayed": sum(x[2] * wt for x, wt in zip(counted, weights, strict=True)),
            "age_days": age_days,
            "staleness_days": min(ages) if ages else age_days,
            "burstiness": burst,
            "is_new": int(age_days < settings.new_file_days),
            "n_authors": len(by_author),
            "owner_share": max(shares.values(), default=0.0),
            "minor_authors": sum(s < settings.minor_author_share for s in shares.values()),
            "author_entropy": -sum(s * math.log2(s) for s in shares.values()),
            "owner_active": owner_active,
        }
    return pd.DataFrame.from_dict(out, orient="index")


def test_matches_loop_reference_on_random_history() -> None:
    rng = np.random.default_rng(DEFAULT_SETTINGS.seed)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    hist, lineage, paths, commits, changes = random_history(rng, 600, 40, 8, 900, end)
    t = end - 200 * DAY

    got = pd.concat(
        [
            evolution_features(hist, snap(t), lineage, paths),
            ownership_features(hist, snap(t), lineage, paths),
        ],
        axis=1,
    )
    expected = reference_features(commits, changes, t, lineage, paths, DEFAULT_SETTINGS)
    assert got["commits_total"].sum() > 100
    assert got.index.tolist() == expected.index.tolist()
    for col in got.columns:
        np.testing.assert_allclose(
            got[col].to_numpy(float), expected[col].to_numpy(float), rtol=1e-9, err_msg=col
        )


def test_large_history_is_fast() -> None:
    rng = np.random.default_rng(DEFAULT_SETTINGS.seed)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    hist, lineage, paths, _, _ = random_history(rng, 20_000, 1_000, 100, 8 * 365, end)
    assert hist.commits.num_rows == 20_000

    start = time.perf_counter()
    evo = evolution_features(hist, snap(end), lineage, paths)
    own = ownership_features(hist, snap(end), lineage, paths)
    activity = commit_activity(hist, snap(end), lineage, paths)
    elapsed = time.perf_counter() - start

    assert elapsed < 15.0
    assert len(evo) == len(own) == len(activity) == 1_000
    assert int(evo["commits_total"].sum()) == len(file_touches(hist, snap(end), lineage, paths))
    renamed = evo.loc["pkg/m0.py"]
    assert renamed["age_days"] > 5 * 365  # history from before the rename is followed
