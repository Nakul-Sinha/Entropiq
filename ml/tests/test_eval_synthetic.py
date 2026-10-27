import time

import numpy as np
import pandas as pd
import pytest

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import BINARY_FEATURES, FEATURE_COLUMNS, FEATURE_INPUT_COLUMNS, PCT_COLUMNS
from tdp_ml.eval.metrics import evaluate
from tdp_ml.eval.synthetic import SYNTHETIC_COLUMNS, SYNTHETIC_START, make_synthetic_dataset


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_synthetic_dataset()


def test_columns_and_dtypes(data: pd.DataFrame) -> None:
    assert list(data.columns) == list(SYNTHETIC_COLUMNS)
    assert len(data) == 4 * 10 * 60 and isinstance(data.index, pd.RangeIndex)
    assert data["repo"].dtype == "str" and data["path"].dtype == "str"
    assert str(data["snapshot_idx"].dtype) == "int16"
    assert str(data["snapshot_time"].dtype) == "datetime64[s, UTC]"
    for col in FEATURE_INPUT_COLUMNS:
        assert data[col].dtype == np.float32, col
    expected = {
        "latent_defects": "int32",
        "future_churn": "int32",
        "pain": "float32",
        "grade": "int8",
        "has_latent_defect": "bool",
        "qid": "int64",
    }
    assert {c: str(data[c].dtype) for c in expected} == expected
    assert not data.isna().any().any()
    assert np.isfinite(data[list(FEATURE_INPUT_COLUMNS)].to_numpy()).all()


def test_deterministic_per_seed(data: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(make_synthetic_dataset(), data)
    other = make_synthetic_dataset(seed=DEFAULT_SETTINGS.seed + 1)
    assert not np.allclose(other["sloc"], data["sloc"])


def test_layout(data: pd.DataFrame) -> None:
    assert data["qid"].unique().tolist() == list(range(40))
    assert data.equals(data.sort_values(["qid", "path"]).reset_index(drop=True))
    assert sorted(data["repo"].unique()) == [f"synthetic__repo{r:02d}" for r in range(4)]
    assert data["path"].iloc[:3].tolist() == ["src/mod_000.py", "src/mod_001.py", "src/mod_002.py"]
    per_query = data.groupby("qid").agg(
        repo=("repo", "nunique"), snap=("snapshot_idx", "nunique"), files=("path", "nunique")
    )
    assert (per_query == pd.DataFrame({"repo": 1, "snap": 1, "files": 60}, index=per_query.index)).all().all()
    repo1 = data[data["repo"] == "synthetic__repo01"].drop_duplicates("snapshot_idx")
    expected = [SYNTHETIC_START + pd.Timedelta(days=23 + 182 * s) for s in range(10)]
    assert repo1["snapshot_time"].tolist() == expected


def test_time_grid_follows_step_days() -> None:
    df = make_synthetic_dataset(n_repos=1, n_snapshots=3, n_files=2, step_days=30)
    times = df.drop_duplicates("qid")["snapshot_time"]
    assert times.diff().dropna().tolist() == [pd.Timedelta(days=30)] * 2
    assert df.drop_duplicates("qid")["snapshot_idx"].tolist() == [0, 1, 2]


def test_features_are_plausible(data: pd.DataFrame) -> None:
    for col in BINARY_FEATURES:
        assert set(data[col].unique().tolist()) <= {0.0, 1.0}, col
    ratios = ("dup_ratio", "owner_share", "cochange_max_conf", "test_cochange_rate", "pkg_test_ratio")
    for col in (*ratios, "satd_density", "pagerank", "betweenness"):
        assert data[col].between(0, 1).all(), col
    counts = [c for c in FEATURE_COLUMNS if c not in ("cc_trend", "burstiness")]
    assert (data[counts] >= 0).all().all()
    assert (data["sloc"] >= 1).all() and (data["transitive_dependents"] >= data["fan_in"]).all()
    assert (data["prior_bugfix_commits"] >= data["prior_latent_bugs"]).all()


def test_percentile_columns(data: pd.DataFrame) -> None:
    for col in PCT_COLUMNS:
        raw = data[col[4:]]
        expected = raw.groupby(data["qid"]).rank(pct=True, method="average").astype("float32")
        np.testing.assert_array_equal(data[col].to_numpy(), expected.to_numpy(), err_msg=col)


def test_labels_follow_the_definitions(data: pd.DataFrame) -> None:
    churn = data["future_churn"].astype("float64")
    median = churn.groupby(data["qid"]).transform("median").clip(lower=1.0)
    pain = data["latent_defects"] + DEFAULT_SETTINGS.gamma * np.log1p(churn / median)
    np.testing.assert_allclose(data["pain"], pain, rtol=1e-6)
    assert (data["has_latent_defect"] == (data["latent_defects"] > 0)).all()
    nonzero = data["pain"].where(data["pain"] > 0)
    pct = nonzero.groupby(data["qid"]).rank(pct=True, method="average")
    grade = np.select([data["pain"] <= 0, pct <= 0.5, pct <= 0.8, pct <= 0.95], [0, 1, 2, 3], 4)
    assert (data["grade"].to_numpy() == grade).all()
    assert (data.groupby("qid")["pain"].max() > 0).all()
    assert set(data["grade"].unique().tolist()) == {0, 1, 2, 3, 4}


def test_every_query_has_pain_even_when_tiny() -> None:
    df = make_synthetic_dataset(n_repos=5, n_snapshots=20, n_files=2, seed=0, gamma=0.0)
    assert (df.groupby("qid")["pain"].max() > 0).all()


def test_hotspot_beats_random() -> None:
    df = make_synthetic_dataset(n_repos=6, n_snapshots=4, n_files=100, seed=9)
    rng = np.random.default_rng(1)
    df["hotspot"] = df["pct_churn_decayed"].astype("float64") * df["pct_cc_sum"]
    df["random"] = rng.uniform(size=len(df))
    hot = evaluate(df, "hotspot")["ndcg_10"].mean()
    rnd = evaluate(df, "random")["ndcg_10"].mean()
    assert hot > rnd + 0.1


def test_build_is_fast() -> None:
    start = time.perf_counter()
    make_synthetic_dataset(n_repos=10, n_snapshots=10, n_files=200)
    assert time.perf_counter() - start < 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_repos": 0},
        {"n_snapshots": 0},
        {"n_files": 1},
        {"step_days": 0},
        {"gamma": -0.1},
        {"gamma": float("nan")},
        {"n_files": 2.5},
    ],
)
def test_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_synthetic_dataset(**kwargs)  # type: ignore[arg-type]
