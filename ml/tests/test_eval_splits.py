import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_ml.eval.splits import forward_folds, lopo_splits, temporal_split
from tdp_ml.eval.synthetic import SYNTHETIC_START, make_synthetic_dataset

H = 365


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset(n_repos=3, n_snapshots=6, n_files=5, seed=1)


# -------------------------------------------------------------------------------------- brute-force reference


def _ref_temporal(df: pd.DataFrame, h: int) -> tuple[set[object], set[object]]:
    train: set[object] = set()
    test: set[object] = set()
    times = pd.to_datetime(df["snapshot_time"], utc=True)
    for repo in df["repo"].unique():
        last = times[df["repo"] == repo].max()
        for label in df.index[df["repo"] == repo]:
            t = times[label]
            if t == last:
                test.add(label)
            elif t + timedelta(days=h) <= last:
                train.add(label)
    return train, test


def _ref_folds(df: pd.DataFrame, h: int, n_folds: int) -> list[tuple[set[object], set[object]]]:
    times = pd.to_datetime(df["snapshot_time"], utc=True)
    folds = []
    for j in range(n_folds):
        back = n_folds - 1 - j
        train: set[object] = set()
        valid: set[object] = set()
        for repo in df["repo"].unique():
            labels = df.index[df["repo"] == repo]
            distinct = sorted({times[label] for label in labels}, reverse=True)
            if len(distinct) <= back:
                continue
            v = distinct[back]
            for label in labels:
                t = times[label]
                if t == v:
                    valid.add(label)
                elif t < v and t + timedelta(days=h) <= v:
                    train.add(label)
        if train and valid:
            folds.append((train, valid))
    return folds


def _ref_lopo(df: pd.DataFrame) -> list[tuple[str, set[object], set[object]]]:
    times = pd.to_datetime(df["snapshot_time"], utc=True)
    out = []
    for repo in sorted(df["repo"].unique()):
        own = df["repo"] == repo
        last = times[own].max()
        train = set(df.index[~own])
        test = set(df.index[own & (times == last)])
        if train:
            out.append((str(repo), train, test))
    return out


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    rows = []
    for r in range(draw(st.integers(1, 4))):
        offsets = draw(st.lists(st.integers(0, 1500), min_size=1, max_size=6, unique=True))
        for offset in offsets:
            for f in range(draw(st.integers(1, 3))):
                rows.append((f"repo{r}", SYNTHETIC_START + pd.Timedelta(days=offset), f"f{f}.py"))
    order = draw(st.permutations(range(len(rows))))
    df = pd.DataFrame([rows[i] for i in order], columns=["repo", "snapshot_time", "path"])
    df.index = pd.Index([f"r{i}" for i in range(len(df))])
    return df


@settings(max_examples=200, deadline=None, derandomize=True)
@given(frames(), st.sampled_from([0, 1, 30, 365]), st.integers(1, 3))
def test_splits_match_brute_force(df: pd.DataFrame, h: int, n_folds: int) -> None:
    train, test = temporal_split(df, h)
    ref_train, ref_test = _ref_temporal(df, h)
    assert set(train) == ref_train and set(test) == ref_test
    assert list(train) == [x for x in df.index if x in ref_train]  # df order
    folds = forward_folds(df, h, n_folds)
    assert [(set(a), set(b)) for a, b in folds] == _ref_folds(df, h, n_folds)
    assert [(name, set(a), set(b)) for name, a, b in lopo_splits(df)] == _ref_lopo(df)


# -------------------------------------------------------------------------------------- temporal


def test_temporal_split_on_synthetic_data(data: pd.DataFrame) -> None:
    train, test = temporal_split(data, H)
    assert not set(train) & set(test)
    test_rows = data.loc[test]
    assert (test_rows["snapshot_idx"] == 5).all()
    assert sorted(test_rows["repo"].unique()) == sorted(data["repo"].unique())
    # snapshots are 182 days apart: 2 * 182 = 364 < 365, so the two snapshots before the test one are dropped
    assert sorted(data.loc[train, "snapshot_idx"].unique().tolist()) == [0, 1, 2]
    assert len(train) == 3 * 3 * 5


def test_temporal_split_zero_horizon_excludes_test_snapshot(data: pd.DataFrame) -> None:
    train, test = temporal_split(data, 0)
    assert sorted(data.loc[train, "snapshot_idx"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert not set(train) & set(test)


def test_temporal_split_edge_cases(data: pd.DataFrame) -> None:
    train, test = temporal_split(data.iloc[:0], H)
    assert len(train) == 0 and len(test) == 0
    one = data[data["snapshot_idx"] == 0]
    train, test = temporal_split(one, H)
    assert len(train) == 0 and len(test) == len(one)
    naive = data.assign(snapshot_time=data["snapshot_time"].dt.tz_localize(None))
    assert [list(x) for x in temporal_split(naive, H)] == [list(x) for x in temporal_split(data, H)]


def test_invalid_input(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="columns"):
        temporal_split(data.drop(columns=["repo"]), H)
    with pytest.raises(ValueError, match="nulls"):
        temporal_split(data.assign(snapshot_time=pd.NaT), H)
    with pytest.raises(ValueError, match="horizon_days"):
        temporal_split(data, -1)
    with pytest.raises(ValueError, match="horizon_days"):
        forward_folds(data, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_folds"):
        forward_folds(data, H, 0)
    dup = pd.concat([data.head(2), data.head(2)])
    for fn in (lambda d: temporal_split(d, H), lambda d: forward_folds(d, H), lopo_splits):
        with pytest.raises(ValueError, match="unique index"):
            fn(dup)


# -------------------------------------------------------------------------------------- forward folds


def test_forward_folds_inside_the_temporal_training_set() -> None:
    df = make_synthetic_dataset(n_repos=2, n_snapshots=10, n_files=3, seed=2)
    train, _ = temporal_split(df, H)
    inner = df.loc[train]
    assert sorted(inner["snapshot_idx"].unique().tolist()) == list(range(7))
    folds = forward_folds(inner, H)
    assert len(folds) == 2
    (train0, valid0), (train1, valid1) = folds
    assert (inner.loc[valid0, "snapshot_idx"] == 5).all() and (inner.loc[valid1, "snapshot_idx"] == 6).all()
    assert sorted(inner.loc[train0, "snapshot_idx"].unique().tolist()) == [0, 1, 2]
    assert sorted(inner.loc[train1, "snapshot_idx"].unique().tolist()) == [0, 1, 2, 3]
    assert not set(train1) & set(valid1)


def test_forward_folds_skip_empty_training_sets(caplog: pytest.LogCaptureFixture) -> None:
    df = make_synthetic_dataset(n_repos=2, n_snapshots=4, n_files=3, seed=2)
    with caplog.at_level(logging.WARNING, logger="tdp_ml.eval.splits"):
        folds = forward_folds(df, H, n_folds=3)
    # validation at snapshots 1 and 2 has nothing 365 days older; snapshot 3 trains on snapshot 0
    assert len(folds) == 1
    assert (df.loc[folds[0][1], "snapshot_idx"] == 3).all()
    assert (df.loc[folds[0][0], "snapshot_idx"] == 0).all()
    assert "skipped" in caplog.text
    assert forward_folds(df.iloc[:0], H) == []


# -------------------------------------------------------------------------------------- LOPO


def test_lopo_never_trains_on_the_held_out_repo(data: pd.DataFrame) -> None:
    splits = list(lopo_splits(data))
    assert [name for name, _, _ in splits] == sorted(data["repo"].unique().tolist())
    for name, train, test in splits:
        assert name not in set(data.loc[train, "repo"])
        assert len(train) == (data["repo"] != name).sum()
        test_rows = data.loc[test]
        assert (test_rows["repo"] == name).all() and (test_rows["snapshot_idx"] == 5).all()
        assert len(test_rows) == 5


def test_lopo_with_one_repo_yields_nothing(data: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    one = data[data["repo"] == data["repo"].iloc[0]]
    with caplog.at_level(logging.WARNING, logger="tdp_ml.eval.splits"):
        assert list(lopo_splits(one)) == []
    assert "skipped" in caplog.text


def test_lopo_validates_eagerly(data: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        lopo_splits(data.drop(columns=["snapshot_time"]))


def test_index_labels_are_returned(data: pd.DataFrame) -> None:
    labelled = data.set_index(pd.Index(np.arange(len(data)) * 10 + 3))
    train, test = temporal_split(labelled, H)
    assert set(test) <= set(labelled.index) and all(x % 10 == 3 for x in train)
