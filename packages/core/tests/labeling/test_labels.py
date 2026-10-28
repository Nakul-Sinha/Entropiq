import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pandas.testing import assert_frame_equal

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA, INDEX_COLUMNS, LABEL_COLUMNS, Snapshot
from tdp_core.labeling.bugfix import BUGFIX_SCHEMA, classify_bugfix, default_is_source
from tdp_core.labeling.labels import (
    DEFECT_HISTORY_COLUMNS,
    GRADE_CUTS,
    LABEL_FRAME_COLUMNS,
    defect_history_features,
    grades_from_pain,
    index_frame,
    make_labels,
    pain_from_counts,
    reachable_window,
)
from tdp_core.labeling.szz import SZZ_SCHEMA, szz_table
from tdp_core.mining import MinedHistory, list_files, mine_history, path_lineage, snapshot_at
from tdp_core.mining.reach import reachable_shas
from tdp_core.testing import RepoBuilder

T0 = datetime(2020, 1, 1, tzinfo=UTC)

Commit = tuple[str, float, str, str, bool]  # sha, day, subject, email, is_bulk
Change = tuple[str, str, str | None, int, int]  # sha, path, old_path, added, deleted
SzzRow = tuple[str, float, str, str, float]  # fix_sha, fix day, path, inducing_sha, inducing day


def _at(day: float) -> datetime:
    return T0 + timedelta(days=day)


def _c(
    sha: str, day: float, *, subject: str | None = None, email: str = "ana@example.com", bulk: bool = False
) -> Commit:
    return (sha, day, subject if subject is not None else f"commit {sha}", email, bulk)


def _hist(commits: Sequence[Commit], changes: Sequence[Change] = ()) -> MinedHistory:
    ordered = sorted(commits, key=lambda c: (c[1], c[0]))
    n = len(ordered)
    commit_table = pa.Table.from_pydict(
        {
            "sha": [c[0] for c in ordered],
            "author_name": ["Ana"] * n,
            "author_email": [c[3] for c in ordered],
            "author_time": [_at(c[1]) for c in ordered],
            "parents": [[] for _ in ordered],
            "subject": [c[2] for c in ordered],
            "message": [c[2] for c in ordered],
            "files_touched": [1] * n,
            "is_bulk": [c[4] for c in ordered],
        },
        schema=COMMITS_SCHEMA,
    )
    change_table = pa.Table.from_pydict(
        {
            "sha": [x[0] for x in changes],
            "path": [x[1] for x in changes],
            "old_path": [x[2] for x in changes],
            "added": [x[3] for x in changes],
            "deleted": [x[4] for x in changes],
            "is_binary": [False] * len(changes),
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    return MinedHistory(commit_table, change_table)


def _szz(rows: Sequence[SzzRow]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "fix_sha": [r[0] for r in rows],
            "fix_time": [_at(r[1]) for r in rows],
            "path": [r[2] for r in rows],
            "inducing_sha": [r[3] for r in rows],
            "inducing_time": [_at(r[4]) for r in rows],
            "n_lines": [1] * len(rows),
        },
        schema=SZZ_SCHEMA,
    )


def _bugfix(fixes: Sequence[str], others: Sequence[str] = ()) -> pa.Table:
    rows = [{"sha": s, "is_bugfix": True, "source": "keyword"} for s in fixes]
    rows += [{"sha": s, "is_bugfix": False, "source": "none"} for s in others]
    return pa.Table.from_pylist(rows, schema=BUGFIX_SCHEMA)


def _snap(day: float, index: int = 0) -> Snapshot:
    return Snapshot("o__r", index, _at(day), "0" * 40)


def _lineage(*paths: str) -> dict[str, frozenset[str]]:
    """Identity lineage for ``paths`` (each file only ever had its own name); see ``_with`` for renames."""
    return {p: frozenset({p}) for p in paths}


def _with(lineage: Mapping[str, frozenset[str]], path: str, *names: str) -> dict[str, frozenset[str]]:
    out = dict(lineage)
    out[path] = frozenset({path, *names})
    return out


def _col(frame: pd.DataFrame, column: str) -> dict[str, object]:
    return dict(zip(frame["path"].tolist(), frame[column].tolist(), strict=True))


def _empty_szz() -> pa.Table:
    return SZZ_SCHEMA.empty_table()


# ------------------------------------------------------------------------------------------- pain and grades
def test_pain_from_counts() -> None:
    latent = [0, 1, 2, 0]
    churn = [0, 10, 20, 30]
    pain = pain_from_counts(latent, churn, 0.25)
    assert pain.dtype == np.float32
    expected = np.array(latent) + 0.25 * np.log1p(np.array(churn) / 15.0)
    np.testing.assert_allclose(pain, expected.astype(np.float32), rtol=1e-6)
    # median 0 is replaced by 1
    np.testing.assert_allclose(
        pain_from_counts([0, 0, 0], [0, 0, 1], 0.25), [0, 0, 0.25 * np.log1p(1.0)], rtol=1e-6
    )
    np.testing.assert_array_equal(pain_from_counts(latent, churn, 0.0), np.array(latent, dtype=np.float32))
    empty = pain_from_counts([], [], 0.25)
    assert empty.dtype == np.float32 and empty.shape == (0,)


@pytest.mark.parametrize(
    ("latent", "churn", "gamma"),
    [
        ([1, 2], [1], 0.25),
        ([[1]], [[1]], 0.25),
        ([-1], [1], 0.25),
        ([1], [-1], 0.25),
        ([float("nan")], [1], 0.25),
        ([1], [1], -0.1),
        ([1], [1], float("inf")),
    ],
)
def test_pain_from_counts_validation(latent: object, churn: object, gamma: float) -> None:
    with pytest.raises(ValueError):
        pain_from_counts(latent, churn, gamma)  # type: ignore[arg-type]


def test_grades_from_pain_on_a_hand_made_vector() -> None:
    nonzero = [0.5, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 10, 11, 12, 13, 14, 15]
    grades = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4]
    values = [0.0, *nonzero[:10], 0.0, *nonzero[10:]]
    expected = [0, *grades[:10], 0, *grades[10:]]
    order = np.random.default_rng(42).permutation(len(values))
    got = grades_from_pain(np.array(values)[order])
    assert got.dtype == np.int8
    assert got.tolist() == np.array(expected)[order].tolist()
    assert GRADE_CUTS == (0.50, 0.80, 0.95)


def test_grades_from_pain_edge_cases() -> None:
    assert grades_from_pain([0.0, 0.0]).tolist() == [0, 0]
    assert grades_from_pain([0.0, 3.5]).tolist() == [0, 4]
    assert grades_from_pain([]).tolist() == []
    for bad in ([float("nan")], [-1.0], [[1.0]], [float("inf")]):
        with pytest.raises(ValueError):
            grades_from_pain(bad)


# ------------------------------------------------------------------------------------------- index frame
def test_index_frame() -> None:
    plus5 = timezone(timedelta(hours=5))
    snap = Snapshot("o__r", 3, datetime(2022, 1, 1, 5, 30, tzinfo=plus5), "x")
    frame = index_frame(snap, ["b.py", "a.py", "a.py"])
    assert tuple(frame.columns) == INDEX_COLUMNS
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["path"].tolist() == ["a.py", "b.py"]
    assert frame["repo"].tolist() == ["o__r", "o__r"]
    assert frame["snapshot_idx"].dtype == np.int16 and frame["snapshot_idx"].tolist() == [3, 3]
    assert str(frame["snapshot_time"].dtype) == "datetime64[us, UTC]"
    assert frame["snapshot_time"].iloc[0] == pd.Timestamp("2022-01-01 00:30", tz="UTC")
    assert frame["repo"].dtype == pd.Series(["x"], dtype="str").dtype == frame["path"].dtype
    empty = index_frame(snap, [])
    assert empty.empty and empty.dtypes.equals(frame.dtypes)
    with pytest.raises(ValueError, match="aware"):
        index_frame(Snapshot("o__r", 0, datetime(2022, 1, 1), "x"), ["a.py"])
    with pytest.raises(ValueError, match="int16"):
        index_frame(Snapshot("o__r", 40_000, T0, "x"), ["a.py"])


# ------------------------------------------------------------------------------------------- time mode
def test_time_mode_latent_defects() -> None:
    commits = [
        _c("i0", 5),
        _c("i1", 10),
        _c("i1b", 12),
        _c("x", 100),
        _c("f4", 50),
        _c("f8", 100),
        _c("i2", 150),
        _c("f1", 200),
        _c("f2", 210),
        _c("f6", 250),
        _c("f5", 300),
        _c("f7", 465),
        _c("f3", 500),
    ]
    rows: list[SzzRow] = [
        ("f1", 200, "a.py", "i0", 5),
        ("f1", 200, "a.py", "i1", 10),
        ("f1", 200, "a.py", "i1b", 12),
        ("f2", 210, "a.py", "i2", 150),
        ("f2", 210, "b.py", "i2", 150),
        ("f3", 500, "c.py", "i1", 10),
        ("f4", 50, "d.py", "i1", 10),
        ("f5", 300, "e.py", "i1", 10),
        ("f5", 300, "f.py", "i1", 10),
        ("f6", 250, "g.py", "x", 100),
        ("f7", 465, "h.py", "i1", 10),
        ("f8", 100, "i.py", "i1", 10),
    ]
    paths = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py", "i.py"]
    frame = make_labels(_hist(commits), _snap(100), _lineage(*paths), paths, _szz(rows))
    expected = {
        "a.py": 1,
        "b.py": 0,
        "c.py": 0,
        "d.py": 0,
        "e.py": 1,
        "f.py": 1,
        "g.py": 1,
        "h.py": 1,
        "i.py": 0,
    }
    assert _col(frame, "latent_defects") == expected
    assert _col(frame, "has_latent_defect") == {p: v > 0 for p, v in expected.items()}


def test_reachable_mode_uses_the_sets_not_the_dates() -> None:
    commits = [
        _c("a", 0),
        _c("f0", 80),
        _c("x", 150),
        _c("l", 50),
        _c("p", 60),
        _c("f", 200),
        _c("g", 220),
        _c("h", 90),
        _c("z", 200),
    ]
    changes: list[Change] = [
        ("l", "a.py", None, 5, 1),
        ("x", "b.py", None, 7, 0),
        ("z", "a.py", None, 100, 0),
    ]
    rows: list[SzzRow] = [
        ("l", 50, "a.py", "a", 0),  # late merge: a window fix authored before T
        ("f", 200, "b.py", "x", 150),  # x is prior although authored after T
        ("g", 220, "c.py", "p", 60),  # p was authored before T but merged after T
        ("h", 900, "d.py", "a", 0),  # fix_time is ignored
        ("x", 150, "e.py", "a", 0),  # x is prior, not a window fix
        ("z", 200, "f.py", "a", 0),  # z never reaches the branch
    ]
    reach_t = frozenset({"a", "f0", "x"})
    reach_end = reach_t | {"l", "p", "f", "g", "h"}
    paths = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py"]
    frame = make_labels(
        _hist(commits, changes),
        _snap(100),
        _lineage(*paths),
        paths,
        _szz(rows),
        reachable_T=reach_t,
        reachable_end=reach_end,
    )
    assert _col(frame, "latent_defects") == {"a.py": 1, "b.py": 1, "c.py": 0, "d.py": 1, "e.py": 0, "f.py": 0}
    assert _col(frame, "future_churn") == {"a.py": 6, "b.py": 0, "c.py": 0, "d.py": 0, "e.py": 0, "f.py": 0}


def _churn_history() -> MinedHistory:
    commits = [
        _c("p1", 50),
        _c("w1", 150),
        _c("w2", 160, bulk=True),
        _c("r1", 170),
        _c("r2", 180),
        _c("q1", 190),
        _c("e1", 500),
    ]
    changes: list[Change] = [
        ("p1", "a.py", None, 10, 0),
        ("w1", "a.py", None, 3, 2),
        ("w2", "a.py", None, 100, 0),
        ("r1", "r2.py", "r.py", 1, 1),
        ("r2", "r2.py", None, 4, 0),
        ("q1", "q2.py", "q.py", 0, 0),
        ("e1", "a.py", None, 7, 0),
    ]
    return _hist(commits, changes)


def test_future_churn() -> None:
    paths = ["a.py", "r.py", "q.py"]
    frame = make_labels(_churn_history(), _snap(100), _lineage(*paths), paths, _empty_szz())
    assert _col(frame, "future_churn") == {"a.py": 5, "q.py": 0, "r.py": 6}
    assert _col(frame, "latent_defects") == {"a.py": 0, "q.py": 0, "r.py": 0}


def test_window_renames_chains_and_stale_names() -> None:
    commits = [
        _c("i", 10),
        _c("k", 150, bulk=True),
        _c("m", 160),
        _c("n", 170),
        _c("r1", 180),
        _c("r2", 190),
        _c("m3", 200),
        _c("n3", 210),
        _c("s", 220),
        _c("n4", 230),
    ]
    changes: list[Change] = [
        ("k", "src/a.py", "a.py", 0, 0),
        ("k", "src/other.py", "other.py", 0, 0),
        ("m", "src/a.py", None, 2, 0),
        ("r1", "c2.py", "c.py", 0, 0),
        ("r2", "c3.py", "c2.py", 0, 0),
        ("m3", "c3.py", None, 3, 0),
        ("s", "old.py", None, 4, 0),
    ]
    rows: list[SzzRow] = [
        ("n", 170, "src/a.py", "i", 10),
        ("n3", 210, "c3.py", "i", 10),
        ("n4", 230, "old.py", "i", 10),
    ]
    paths = ["a.py", "c.py", "new.py"]
    lineage = _with(_lineage(*paths), "new.py", "old.py")
    frame = make_labels(_hist(commits, changes), _snap(100), lineage, paths, _szz(rows))
    assert _col(frame, "latent_defects") == {"a.py": 1, "c.py": 1, "new.py": 1}
    assert _col(frame, "future_churn") == {"a.py": 2, "c.py": 3, "new.py": 4}


def test_a_name_another_live_file_uses_counts_only_for_that_file() -> None:
    commits = [_c("i", 10), _c("w1", 150), _c("f1", 160)]
    changes: list[Change] = [("w1", "app.py", None, 5, 0)]
    rows: list[SzzRow] = [("f1", 160, "app.py", "i", 10)]
    paths = ["app.py", "sansio/app.py"]
    lineage = _with(_lineage(*paths), "sansio/app.py", "app.py")
    frame = make_labels(_hist(commits, changes), _snap(100), lineage, paths, _szz(rows))
    assert _col(frame, "latent_defects") == {"app.py": 1, "sansio/app.py": 0}
    assert _col(frame, "future_churn") == {"app.py": 5, "sansio/app.py": 0}
    # A name alive only as a lineage key (not ranked) is excluded too.
    only_split = make_labels(_hist(commits, changes), _snap(100), lineage, ["sansio/app.py"], _szz(rows))
    assert _col(only_split, "future_churn") == {"sansio/app.py": 0}


def test_copies_count_once_and_copies_of_prior_commits_never() -> None:
    commits = [
        _c("i", 5, subject="init"),
        _c("q", 50, subject="port"),
        _c("q2", 50, subject="port"),
        _c("x", 150, subject="fix crash"),
        _c("x2", 150, subject="fix crash"),
        _c("z", 200, subject="fix z"),
    ]
    changes: list[Change] = [
        ("q2", "a.py", None, 5, 0),
        ("x", "b.py", None, 3, 0),
        ("x2", "b.py", None, 3, 0),
    ]
    rows: list[SzzRow] = [
        ("q2", 50, "a.py", "i", 5),
        ("x", 150, "b.py", "i", 5),
        ("x2", 150, "b.py", "i", 5),
        ("z", 200, "c.py", "q2", 50),  # q2 is a window copy of the prior commit q
    ]
    paths = ["a.py", "b.py", "c.py"]
    frame = make_labels(
        _hist(commits, changes),
        _snap(100),
        _lineage(*paths),
        paths,
        _szz(rows),
        reachable_T={"i", "q"},
        reachable_end={"i", "q", "q2", "x", "x2", "z"},
    )
    assert _col(frame, "latent_defects") == {"a.py": 0, "b.py": 1, "c.py": 1}
    assert _col(frame, "future_churn") == {"a.py": 0, "b.py": 3, "c.py": 0}


def test_pain_and_grade_columns_follow_the_helpers() -> None:
    paths = ["a.py", "r.py", "q.py"]
    rows: list[SzzRow] = [("w1", 150, "a.py", "p1", 50)]
    settings = with_overrides(DEFAULT_SETTINGS, gamma=0.5)
    frame = make_labels(_churn_history(), _snap(100), _lineage(*paths), paths, _szz(rows), settings)
    latent = frame["latent_defects"].to_numpy()
    churn = frame["future_churn"].to_numpy()
    np.testing.assert_array_equal(frame["pain"].to_numpy(), pain_from_counts(latent, churn, 0.5))
    np.testing.assert_array_equal(frame["grade"].to_numpy(), grades_from_pain(frame["pain"].to_numpy()))
    assert _col(frame, "latent_defects") == {"a.py": 1, "q.py": 0, "r.py": 0}
    # The median is taken over the query: dropping r.py changes a.py's pain.
    pair = make_labels(_churn_history(), _snap(100), _lineage(*paths), ["a.py", "q.py"], _szz(rows), settings)
    assert _col(frame, "pain")["a.py"] == pytest.approx(1 + 0.5 * np.log1p(5 / 5), rel=1e-6)
    assert _col(pair, "pain")["a.py"] == pytest.approx(1 + 0.5 * np.log1p(5 / 2.5), rel=1e-6)


def test_label_frame_layout() -> None:
    paths = ["r.py", "a.py", "q.py", "a.py"]
    frame = make_labels(_churn_history(), _snap(100, index=7), _lineage(*paths), paths, _empty_szz())
    assert tuple(frame.columns) == LABEL_FRAME_COLUMNS == INDEX_COLUMNS + LABEL_COLUMNS
    assert isinstance(frame.index, pd.RangeIndex)
    assert frame["path"].tolist() == ["a.py", "q.py", "r.py"]
    dtypes = [str(t) for t in frame.dtypes]
    assert dtypes == [
        "str",
        "int16",
        "datetime64[us, UTC]",
        "str",
        "int32",
        "int32",
        "float32",
        "int8",
        "bool",
    ]
    assert frame["snapshot_idx"].tolist() == [7, 7, 7]
    untouched = make_labels(_churn_history(), _snap(1000), _lineage(*paths), ["zzz.py"], _empty_szz())
    assert untouched[list(LABEL_COLUMNS)].iloc[0].tolist() == [0, 0, 0.0, 0, False]
    empty = make_labels(_churn_history(), _snap(100), {}, [], _empty_szz())
    assert empty.empty and tuple(empty.columns) == LABEL_FRAME_COLUMNS
    assert empty.dtypes.equals(frame.dtypes)


def test_make_labels_validation() -> None:
    hist, paths = _churn_history(), ["a.py"]
    lineage = _lineage(*paths)
    naive = Snapshot("o__r", 0, datetime(2020, 5, 1), "0" * 40)
    with pytest.raises(ValueError, match="aware"):
        make_labels(hist, naive, lineage, paths, _empty_szz())
    with pytest.raises(ValueError, match="both"):
        make_labels(hist, _snap(100), lineage, paths, _empty_szz(), reachable_T={"p1"})
    with pytest.raises(ValueError, match="both"):
        make_labels(hist, _snap(100), lineage, paths, _empty_szz(), reachable_end={"p1"})
    with pytest.raises(ValueError, match="subset"):
        make_labels(hist, _snap(100), lineage, paths, _empty_szz(), reachable_T={"p1"}, reachable_end={"w1"})
    with pytest.raises(ValueError, match="inducing_time"):
        make_labels(hist, _snap(100), lineage, paths, _empty_szz().drop_columns(["inducing_time"]))


def test_szz_rows_with_nulls_and_other_units_are_handled() -> None:
    hist = _hist([_c("i", 10), _c("f", 150)])
    table = pa.Table.from_pydict(
        {
            "fix_sha": ["f", "f", None],
            "fix_time": pa.array([_at(150), None, _at(150)], type=pa.timestamp("ms", tz="UTC")),
            "path": ["a.py", "b.py", "c.py"],
            "inducing_sha": ["i", "i", "i"],
            "inducing_time": pa.array([_at(10)] * 3, type=pa.timestamp("ns", tz="UTC")),
        }
    )
    paths = ["a.py", "b.py", "c.py"]
    frame = make_labels(hist, _snap(100), _lineage(*paths), paths, table)
    assert _col(frame, "latent_defects") == {"a.py": 1, "b.py": 0, "c.py": 0}


# ------------------------------------------------------------------------------------------- features
def _feature_history() -> tuple[MinedHistory, pa.Table, pa.Table, dict[str, frozenset[str]]]:
    commits = [
        _c("i", 5),
        _c("f1", 10),
        _c("f2", 20),
        _c("g", 25),
        _c("n1", 30),
        _c("f4", 40, subject="fix b"),
        _c("f4c", 40, subject="fix b"),
        _c("fb", 45, bulk=True),
        _c("f3", 150),
    ]
    changes: list[Change] = [
        ("f1", "old.py", None, 1, 1),
        ("f2", "new.py", None, 1, 1),
        ("g", "b.py", None, 1, 1),
        ("n1", "new.py", None, 1, 1),
        ("f4", "b.py", None, 1, 1),
        ("f4c", "b.py", None, 1, 1),
        ("fb", "b.py", None, 1, 1),
        ("f3", "new.py", None, 1, 1),
    ]
    bugfix = _bugfix(["f1", "f2", "f3", "f4", "f4c", "fb"], others=["n1", "i", "g"])
    szz = _szz(
        [
            ("f1", 10, "old.py", "i", 5),
            ("f2", 20, "new.py", "i", 5),
            ("f3", 150, "new.py", "i", 5),
            ("n1", 30, "new.py", "i", 5),
            ("g", 25, "b.py", "i", 5),
            ("f4", 40, "b.py", "i", 5),
            ("f4c", 40, "b.py", "i", 5),
            ("fb", 45, "b.py", "i", 5),
        ]
    )
    lineage = _with(_lineage("new.py", "b.py"), "new.py", "old.py")
    return _hist(commits, changes), bugfix, szz, lineage


def _features(frame: pd.DataFrame) -> dict[str, list[int]]:
    return {str(p): [int(v) for v in row] for p, row in zip(frame.index, frame.to_numpy(), strict=True)}


def test_defect_history_features() -> None:
    hist, bugfix, szz, lineage = _feature_history()
    paths = ["new.py", "b.py", "b.py", "none.py"]
    frame = defect_history_features(hist, _snap(100), lineage, paths, bugfix, szz)
    assert tuple(frame.columns) == DEFECT_HISTORY_COLUMNS == ("prior_bugfix_commits", "prior_latent_bugs")
    assert frame.index.name == "path" and frame.index.dtype == pd.Index(["x"], dtype="str").dtype
    assert all(frame[c].dtype == np.int64 for c in DEFECT_HISTORY_COLUMNS)
    assert _features(frame) == {"b.py": [1, 1], "new.py": [2, 2], "none.py": [0, 0]}

    newest = defect_history_features(hist, _snap(100), lineage, paths, bugfix, szz, max_fixes=1)
    assert _features(newest) == {"b.py": [1, 1], "new.py": [2, 0], "none.py": [0, 0]}
    two = defect_history_features(hist, _snap(100), lineage, paths, bugfix, szz, max_fixes=3)
    assert _features(two) == {"b.py": [1, 1], "new.py": [2, 1], "none.py": [0, 0]}

    reach = {"i", "f1", "n1", "f4", "f4c", "f3"}
    reached = defect_history_features(hist, _snap(100), lineage, paths, bugfix, szz, reachable_T=reach)
    assert _features(reached) == {"b.py": [1, 1], "new.py": [2, 2], "none.py": [0, 0]}
    # reachable_end is accepted for symmetry with make_labels and never read.
    both = defect_history_features(
        hist,
        _snap(100),
        lineage,
        paths,
        bugfix,
        szz,
        reachable_T=reach,
        reachable_end=reach | {"f2", "g", "fb"},
    )
    assert_frame_equal(both, reached)
    early = defect_history_features(hist, _snap(100), lineage, paths, bugfix, szz, reachable_T={"i", "f1"})
    assert _features(early) == {"b.py": [0, 0], "new.py": [1, 1], "none.py": [0, 0]}

    empty = defect_history_features(hist, _snap(100), lineage, [], bugfix, szz)
    assert empty.empty and tuple(empty.columns) == DEFECT_HISTORY_COLUMNS
    assert empty.dtypes.equals(frame.dtypes) and empty.index.name == "path"
    no_fixes = defect_history_features(hist, _snap(100), lineage, paths, _bugfix([], others=["f1"]), szz)
    assert _features(no_fixes) == {"b.py": [0, 0], "new.py": [0, 0], "none.py": [0, 0]}


def test_defect_history_features_validation() -> None:
    hist, bugfix, szz, lineage = _feature_history()
    naive = Snapshot("o__r", 0, datetime(2020, 5, 1), "0" * 40)
    with pytest.raises(ValueError, match="aware"):
        defect_history_features(hist, naive, lineage, ["b.py"], bugfix, szz)
    with pytest.raises(ValueError, match="max_fixes"):
        defect_history_features(hist, _snap(100), lineage, ["b.py"], bugfix, szz, max_fixes=-1)
    with pytest.raises(ValueError, match="together"):
        defect_history_features(hist, _snap(100), lineage, ["b.py"], bugfix, szz, reachable_end={"i"})
    with pytest.raises(ValueError, match="subset"):
        defect_history_features(
            hist, _snap(100), lineage, ["b.py"], bugfix, szz, reachable_T={"i", "f1"}, reachable_end={"i"}
        )
    with pytest.raises(ValueError, match="is_bugfix"):
        defect_history_features(hist, _snap(100), lineage, ["b.py"], bugfix.select(["sha"]), szz)
    with pytest.raises(ValueError, match="fix_sha"):
        defect_history_features(hist, _snap(100), lineage, ["b.py"], bugfix, szz.drop_columns(["fix_sha"]))


# ------------------------------------------------------------------------------------------- git histories
def _lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


CORE = ("def calc(x):", "    y = x + 1", "    return y", "", "", "def other():", "    return 0")
UTIL = ("def ratio(a, b):", "    return a / b", "", "", "def unused():", "    return None")


def _core(**lines: str) -> str:
    rows = list(CORE)
    for key, value in lines.items():
        rows[int(key[1:]) - 1] = value
    return _lines(*rows)


def _util(line2: str = "    return a / b") -> str:
    rows = list(UTIL)
    rows[1] = line2
    return _lines(*rows)


def _git(
    repo: Path,
    config: Path,
    *args: str,
    authored: datetime | None = None,
    committed: datetime | None = None,
) -> str:
    """Run git in ``repo`` isolated from the user's config, optionally pinning the author or commit date."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_"))}
    env.update({"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    if authored is not None:
        env["GIT_AUTHOR_DATE"] = authored.isoformat()
    if committed is not None:
        env["GIT_COMMITTER_DATE"] = committed.isoformat()
    result = subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    return result.stdout.decode("utf-8", errors="replace").strip()


def _day(t0: datetime, n: float) -> datetime:
    return t0 + timedelta(days=n)


def _source_paths(repo: Path, sha: str) -> list[str]:
    return sorted(p for p in list_files(repo, sha) if default_is_source(p))


def _lineage_at(hist: MinedHistory, reach: frozenset[str]) -> dict[str, frozenset[str]]:
    commits = hist.commits.filter(pc.is_in(hist.commits.column("sha"), value_set=pa.array(sorted(reach))))
    until = max(commits.column("author_time").to_pylist())
    return path_lineage(hist.file_changes, commits, until)


def test_defect_history_features_never_read_after_t(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo = repo_builder.path
    repo_builder.commit({"pkg/core.py": _core(), "pkg/util.py": _util()}, "initial", _day(t0, 0))
    repo_builder.commit({"pkg/core.py": _core(l3="    return y * 2")}, "handle x", _day(t0, 5))
    repo_builder.commit({"pkg/core.py": _core(l3="    return abs(y) * 2")}, "fix: negative x", _day(t0, 10))
    repo_builder.commit({"pkg/util.py": _util("    return a // b")}, "speed up util", _day(t0, 15))
    t = _day(t0, 20)
    repo_builder.commit(
        {"pkg/util.py": _util("    return a / b if b else 0.0")}, "fix crash in util", _day(t0, 30)
    )
    repo_builder.rename("pkg/util.py", "pkg/helpers.py", "move util", _day(t0, 40))
    repo_builder.commit(
        {"pkg/helpers.py": _util("    return a / b if b else 1.0")}, "fix wrong ratio", _day(t0, 50)
    )
    late = repo_builder.commit(
        {"pkg/core.py": _core(l3="    return abs(y) * 3")}, "fix sign again", _day(t0, 60)
    )
    last = repo_builder.commit(
        {"pkg/core.py": _core(l3="    return abs(y) * 3", l7="    return 5")}, "tweak", _day(t0, 70)
    )

    hist = mine_history(repo)
    bugfix = classify_bugfix(hist.commits, hist.file_changes)
    szz = szz_table(repo, hist, bugfix)
    snap = snapshot_at("o__r", hist.commits, t, repo_dir=repo)
    reach = reachable_shas(repo, snap.sha)
    hist_t = mine_history(repo, snap.sha)
    bugfix_t = classify_bugfix(hist_t.commits, hist_t.file_changes)
    szz_t = szz_table(repo, hist_t, bugfix_t)
    lineage = _lineage_at(hist, reach)
    paths = _source_paths(repo, snap.sha)
    assert paths == ["pkg/core.py", "pkg/util.py"]

    for cap in (None, 300):
        full = defect_history_features(
            hist, snap, lineage, paths, bugfix, szz, reachable_T=reach, max_fixes=cap
        )
        cut = defect_history_features(
            hist_t, snap, lineage, paths, bugfix_t, szz_t, reachable_T=reach, max_fixes=cap
        )
        assert_frame_equal(full, cut)
        assert _features(full) == {"pkg/core.py": [1, 1], "pkg/util.py": [0, 0]}

    snap_time = snapshot_at("o__r", hist.commits, t)
    full_time = defect_history_features(hist, snap_time, lineage, paths, bugfix, szz)
    cut_time = defect_history_features(hist_t, snap_time, lineage, paths, bugfix_t, szz_t)
    assert_frame_equal(full_time, cut_time)

    extra_bugfix = pa.concat_tables([bugfix, _bugfix([last])])
    extra_szz = pa.concat_tables(
        [
            szz,
            pa.Table.from_pylist(
                [
                    {
                        "fix_sha": late,
                        "fix_time": _day(t0, 60),
                        "path": "pkg/util.py",
                        "inducing_sha": snap.sha,
                        "inducing_time": _day(t0, 15),
                        "n_lines": 1,
                    },
                    {
                        "fix_sha": last,
                        "fix_time": _day(t0, 70),
                        "path": "pkg/core.py",
                        "inducing_sha": snap.sha,
                        "inducing_time": _day(t0, 15),
                        "n_lines": 1,
                    },
                ],
                schema=SZZ_SCHEMA,
            ),
        ]
    )
    noisy = defect_history_features(hist, snap, lineage, paths, extra_bugfix, extra_szz, reachable_T=reach)
    assert_frame_equal(noisy, full)


def test_labels_end_to_end_with_a_late_merge(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> None:
    repo = repo_builder.path
    config = tmp_path / "isolated.gitconfig"
    config.touch()
    a = repo_builder.commit({"pkg/core.py": _core(), "pkg/util.py": _util()}, "initial", _day(t0, 0))
    b = repo_builder.commit({"pkg/core.py": _core(l3="    return y * 2")}, "handle x", _day(t0, 5))
    _git(repo, config, "checkout", "-q", "-b", "feature")
    p = repo_builder.commit({"pkg/util.py": _util("    return a // b")}, "speed up util", _day(t0, 15))
    _git(repo, config, "checkout", "-q", "main")
    c = repo_builder.commit(
        {"pkg/core.py": _core(l3="    return abs(y) * 2")}, "fix: negative x", _day(t0, 30)
    )
    _git(
        repo,
        config,
        "merge",
        "-q",
        "--no-ff",
        "-m",
        "Merge feature",
        "feature",
        authored=_day(t0, 40),
        committed=_day(t0, 40),
    )
    d = repo_builder.commit(
        {"pkg/util.py": _util("    return a / b if b else 0.0")}, "fix: divide by zero", _day(t0, 50)
    )
    e = repo_builder.commit({"pkg/core.py": _core(l3="    return abs(y) * 3")}, "fix crash", _day(t0, 400))

    hist = mine_history(repo)
    snap = snapshot_at("o__r", hist.commits, _day(t0, 20), repo_dir=repo)
    assert snap.sha == b
    reach_t, reach_end = reachable_window(repo, snap)
    assert reach_t == {a, b} and reach_end == {a, b, p, c, d}
    lineage = _lineage_at(hist, reach_t)
    paths = _source_paths(repo, snap.sha)
    bugfix = classify_bugfix(hist.commits, hist.file_changes)
    flags = zip(bugfix.column("sha").to_pylist(), bugfix.column("is_bugfix").to_pylist(), strict=True)
    fixes = {sha for sha, flag in flags if flag}
    assert fixes == {c, d, e}
    szz = szz_table(repo, hist, bugfix)
    pairs = set(zip(szz.column("fix_sha").to_pylist(), szz.column("inducing_sha").to_pylist(), strict=True))
    assert pairs == {(c, b), (d, p), (e, c)}

    churn = {
        (sha, path): added + deleted
        for sha, path, added, deleted in zip(
            *(hist.file_changes.column(n).to_pylist() for n in ("sha", "path", "added", "deleted")),
            strict=True,
        )
    }
    reachable = make_labels(hist, snap, lineage, paths, szz, reachable_T=reach_t, reachable_end=reach_end)
    assert _col(reachable, "latent_defects") == {"pkg/core.py": 1, "pkg/util.py": 0}
    assert _col(reachable, "future_churn") == {
        "pkg/core.py": churn[(c, "pkg/core.py")],
        "pkg/util.py": churn[(p, "pkg/util.py")] + churn[(d, "pkg/util.py")],
    }
    timed = make_labels(hist, snap, lineage, paths, szz)
    assert _col(timed, "latent_defects") == {"pkg/core.py": 1, "pkg/util.py": 1}
    assert _col(timed, "future_churn") == {
        "pkg/core.py": churn[(c, "pkg/core.py")],
        "pkg/util.py": churn[(d, "pkg/util.py")],
    }


def test_reachable_window(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo = repo_builder.path
    a = repo_builder.commit({"a.py": "a = 1\n"}, "a", _day(t0, 0))
    b = repo_builder.commit({"a.py": "a = 2\n"}, "b", _day(t0, 100))
    c = repo_builder.commit({"a.py": "a = 3\n"}, "c", _day(t0, 200))
    d = repo_builder.commit({"a.py": "a = 4\n"}, "d", _day(t0, 500))
    hist = mine_history(repo)
    snap = snapshot_at("o__r", hist.commits, _day(t0, 150), repo_dir=repo)
    assert reachable_window(repo, snap) == (frozenset({a, b}), frozenset({a, b, c, d}))
    short = with_overrides(DEFAULT_SETTINGS, horizon_days=100)
    assert reachable_window(repo, snap, short) == (frozenset({a, b}), frozenset({a, b, c}))
    with pytest.raises(ValueError, match="no state"):
        reachable_window(repo, Snapshot("o__r", 0, _day(t0, -400), a))
    with pytest.raises(ValueError, match="aware"):
        reachable_window(repo, Snapshot("o__r", 0, datetime(2022, 6, 1), a))


# ------------------------------------------------------------------------------------------- speed
def test_speed_on_a_large_synthetic_history() -> None:
    rng = np.random.default_rng(42)
    n_files, n_commits, n_rows = 2_000, 20_000, 10_000
    files = [f"pkg/m{i:04d}.py" for i in range(n_files)]
    days = np.sort(rng.uniform(0, 2_000, n_commits))
    shas = [f"{i:040x}" for i in range(n_commits)]
    commit_table = pa.Table.from_pydict(
        {
            "sha": shas,
            "author_name": ["Ana"] * n_commits,
            "author_email": [f"dev{i % 50}@example.com" for i in range(n_commits)],
            "author_time": pa.array((T0.timestamp() + days * 86_400).astype(np.int64), type=pa.int64()).cast(
                pa.timestamp("s", tz="UTC")
            ),
            "parents": [[] for _ in range(n_commits)],
            "subject": [f"change {i}" for i in range(n_commits)],
            "message": [f"change {i}" for i in range(n_commits)],
            "files_touched": [2] * n_commits,
            "is_bulk": (rng.random(n_commits) < 0.01).tolist(),
        },
        schema=COMMITS_SCHEMA,
    )
    touched = rng.integers(0, n_files, size=(n_commits, 2))
    change_sha = np.repeat(np.arange(n_commits), 2)
    change_path = [files[i] for i in touched.ravel()]
    old_path: list[str | None] = [None] * len(change_path)
    for k in rng.choice(len(change_path), 100, replace=False):
        old_path[k] = f"old/{change_path[k]}"
    change_table = pa.Table.from_pydict(
        {
            "sha": [shas[i] for i in change_sha],
            "path": change_path,
            "old_path": old_path,
            "added": rng.integers(0, 20, len(change_path)).tolist(),
            "deleted": rng.integers(0, 20, len(change_path)).tolist(),
            "is_binary": [False] * len(change_path),
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    hist = MinedHistory(commit_table, change_table)
    fix_idx = rng.integers(1_000, n_commits, n_rows)
    ind_idx = (fix_idx * rng.random(n_rows)).astype(np.int64)
    times = commit_table.column("author_time")
    szz = pa.Table.from_pydict(
        {
            "fix_sha": [shas[i] for i in fix_idx],
            "fix_time": pc.take(times, pa.array(fix_idx)),
            "path": [files[i] for i in rng.integers(0, n_files, n_rows)],
            "inducing_sha": [shas[i] for i in ind_idx],
            "inducing_time": pc.take(times, pa.array(ind_idx)),
            "n_lines": [1] * n_rows,
        },
        schema=SZZ_SCHEMA,
    )
    lineage = {f: frozenset({f, f"old/{f}"}) if i % 20 == 0 else frozenset({f}) for i, f in enumerate(files)}
    snap = Snapshot("o__r", 0, T0 + timedelta(days=1_000), "0" * 40)
    reach_t = frozenset(s for s, d in zip(shas, days, strict=True) if d <= 1_000)
    reach_end = frozenset(s for s, d in zip(shas, days, strict=True) if d <= 1_365)
    bugfix = _bugfix(shas[::7])

    started = time.perf_counter()
    labels = make_labels(hist, snap, lineage, files, szz, reachable_T=reach_t, reachable_end=reach_end)
    label_s = time.perf_counter() - started
    started = time.perf_counter()
    features = defect_history_features(
        hist, snap, lineage, files, bugfix, szz, reachable_T=reach_t, max_fixes=300
    )
    feature_s = time.perf_counter() - started
    assert len(labels) == len(features) == n_files
    assert labels["latent_defects"].sum() > 0 and labels["future_churn"].sum() > 0
    assert features["prior_bugfix_commits"].sum() > 0 and features["prior_latent_bugs"].sum() > 0
    assert label_s < 10 and feature_s < 10, (label_s, feature_s)
