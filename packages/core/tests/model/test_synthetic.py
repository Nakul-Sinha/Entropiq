from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import (
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_INPUT_COLUMNS,
    INDEX_COLUMNS,
    LABEL_COLUMNS,
    PCT_COLUMNS,
)
from tdp_core.model.synthetic import SYNTHETIC_START, synthetic_dataset

COUNT_COLUMNS = (
    "sloc",
    "cc_sum",
    "cc_max",
    "cog_max",
    "nesting_max",
    "n_functions",
    "n_classes",
    "longest_function",
    "params_max",
    "commits_total",
    "commits_90d",
    "commits_180d",
    "churn_total",
    "n_authors",
    "minor_authors",
    "prior_bugfix_commits",
    "prior_latent_bugs",
    "fan_in",
    "fan_out",
    "transitive_dependents",
    "scc_size",
    "cochange_partners",
    "hidden_coupling",
    "test_importers",
    "satd_count",
)


def col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=np.float64)


def test_layout_and_dtypes(synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    assert list(df.columns) == [*INDEX_COLUMNS, *FEATURE_INPUT_COLUMNS, *LABEL_COLUMNS, "qid"]
    assert len(df) == 6000
    assert isinstance(df.index, pd.RangeIndex)
    assert df["qid"].dtype == np.int32
    assert df["qid"].is_monotonic_increasing
    assert df.groupby("qid").size().tolist() == [200] * 30
    assert not df.duplicated(["qid", "path"]).any()
    assert df.groupby("qid")["path"].apply(lambda s: s.is_monotonic_increasing).all()
    repo_index = df["repo"].str.removeprefix("synthetic__repo").astype(int)
    assert (df["qid"] == repo_index * 5 + df["snapshot_idx"]).all()
    assert df["repo"].nunique() == 6
    assert pd.api.types.is_string_dtype(df["repo"])
    assert pd.api.types.is_string_dtype(df["path"])
    assert df["snapshot_idx"].dtype == np.int16
    assert isinstance(df["snapshot_time"].dtype, pd.DatetimeTZDtype)
    assert str(df["snapshot_time"].dt.tz) == "UTC"
    step = timedelta(days=DEFAULT_SETTINGS.snapshot_step_days)
    expected = [SYNTHETIC_START + int(k) * step for k in df["snapshot_idx"]]
    assert df["snapshot_time"].tolist() == [pd.Timestamp(t) for t in expected]
    for c in FEATURE_INPUT_COLUMNS:
        assert df[c].dtype == np.float32, c
    assert [df[c].dtype for c in LABEL_COLUMNS] == [np.int32, np.int32, np.float32, np.int8, np.bool_]
    paths = df.loc[df["repo"] == "synthetic__repo00"].groupby("snapshot_idx")["path"].apply(tuple)
    assert paths.nunique() == 1


def test_values_are_plausible(synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    feats = df[list(FEATURE_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(feats).all()
    for c in FEATURE_COLUMNS:
        if c != "cc_trend":
            assert (df[c] >= 0).all(), c
    for c in BINARY_FEATURES:
        assert set(np.unique(df[c])) <= {0.0, 1.0}, c
    for c in COUNT_COLUMNS:
        v = col(df, c)
        assert np.array_equal(v, np.rint(v)), c

    ct, c180, c90 = col(df, "commits_total"), col(df, "commits_180d"), col(df, "commits_90d")
    assert np.all(c90 <= c180)
    assert np.all(c180 <= ct)
    assert np.all(col(df, "staleness_days") <= col(df, "age_days"))
    assert np.array_equal(col(df, "is_new") == 1, col(df, "age_days") < DEFAULT_SETTINGS.new_file_days)

    nf, cc_sum = col(df, "n_functions"), col(df, "cc_sum")
    cc_mean, cc_max = col(df, "cc_mean"), col(df, "cc_max")
    fn = nf > 0
    assert np.all(cc_sum >= 1)
    assert np.all(cc_mean[fn] <= cc_max[fn] + 1e-4)
    assert np.all(cc_max[fn] <= cc_sum[fn] - 1)
    assert np.all(cc_mean[~fn] == 0)
    assert np.all(cc_max[~fn] == 0)
    assert np.all(col(df, "longest_function") <= col(df, "sloc"))

    na = col(df, "n_authors")
    share = col(df, "owner_share")
    assert np.all((share >= 0) & (share <= 1))
    assert np.all(col(df, "author_entropy") <= np.log2(np.maximum(na, 1)) + 1e-6)
    assert np.all(col(df, "minor_authors") <= np.maximum(na - 1, 0))
    assert np.all(na <= ct)

    assert np.all(col(df, "prior_latent_bugs") <= col(df, "prior_bugfix_commits"))
    assert np.all(col(df, "prior_bugfix_commits") <= ct)

    fan_in, td = col(df, "fan_in"), col(df, "transitive_dependents")
    assert np.all(fan_in <= td)
    assert np.all(td <= 199)
    assert np.all(col(df, "fan_out") <= 199)
    np.testing.assert_allclose(df.groupby("qid")["pagerank"].sum(), 1.0, atol=1e-4)
    for c in ("betweenness", "cochange_max_conf", "test_cochange_rate", "dup_ratio"):
        v = col(df, c)
        assert np.all((v >= 0) & (v <= 1)), c
    assert np.all(col(df, "scc_size") >= 1)
    assert np.all(col(df, "hidden_coupling") <= col(df, "cochange_partners"))
    sloc, satd = col(df, "sloc"), col(df, "satd_count")
    np.testing.assert_allclose(col(df, "satd_density"), 1000 * satd / np.maximum(sloc, 1), rtol=1e-5)
    assert np.all(col(df, "satd_age_days")[satd == 0] == 0)


def test_pct_columns_are_within_query_ranks(synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    for c in PCT_COLUMNS:
        base = c[4:]
        expected = df.groupby("qid")[base].rank(pct=True, method="average").to_numpy(dtype=np.float64)
        np.testing.assert_allclose(col(df, c), expected, rtol=0, atol=1e-7, err_msg=c)
        assert col(df, c).min() > 0
        assert col(df, c).max() <= 1


def test_labels_follow_the_label_rules(synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    gamma = DEFAULT_SETTINGS.gamma
    for _, q in df.groupby("qid"):
        latent = q["latent_defects"].to_numpy(dtype=np.float64)
        churn = q["future_churn"].to_numpy(dtype=np.float64)
        pain = (latent + gamma * np.log1p(churn / max(float(np.median(churn)), 1.0))).astype(np.float32)
        np.testing.assert_array_equal(q["pain"].to_numpy(), pain)
        grade = np.zeros(len(q), dtype=np.int8)
        nz = pain != 0
        ranks = pd.Series(pain[nz]).rank(pct=True, method="average").to_numpy()
        grade[nz] = np.where(ranks <= 0.5, 1, np.where(ranks <= 0.8, 2, np.where(ranks <= 0.95, 3, 4)))
        np.testing.assert_array_equal(q["grade"].to_numpy(), grade)
    assert (df["has_latent_defect"] == (df["latent_defects"] > 0)).all()
    assert ((df["grade"] == 0) == (df["pain"] == 0)).all()
    assert (df["latent_defects"] >= 0).all()
    assert (df["future_churn"] >= 0).all()


def test_label_mix_is_useful(synthetic_frame: pd.DataFrame) -> None:
    df = synthetic_frame
    shares = df["grade"].value_counts(normalize=True)
    assert set(shares.index) == {0, 1, 2, 3, 4}
    assert 0.25 <= shares[0] <= 0.70
    assert 0.08 <= df["has_latent_defect"].mean() <= 0.35
    rho = df["churn_decayed"].rank().corr(df["pain"].rank())
    assert rho > 0.2


def test_deterministic_per_seed(synthetic_frame: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(synthetic_dataset(seed=42), synthetic_frame)
    other = synthetic_dataset(seed=43)
    assert not other[list(FEATURE_INPUT_COLUMNS)].equals(synthetic_frame[list(FEATURE_INPUT_COLUMNS)])


def test_custom_sizes() -> None:
    df = synthetic_dataset(4, 10, snapshots_per_repo=2, seed=1)
    assert len(df) == 40
    assert df["repo"].nunique() == 2
    assert sorted(df["snapshot_idx"].unique().tolist()) == [0, 1]
    assert df["transitive_dependents"].max() <= 9
    one = synthetic_dataset(1, 1)
    assert len(one) == 1
    assert one["pct_sloc"].iloc[0] == 1.0
    assert one["pagerank"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"n_queries": 0}, {"rows_per_query": 0}, {"snapshots_per_repo": 0}],
)
def test_rejects_sizes_below_one(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        synthetic_dataset(**kwargs)  # type: ignore[arg-type]
