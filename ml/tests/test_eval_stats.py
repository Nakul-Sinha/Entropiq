import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as scipy_stats

from tdp_ml.eval.stats import (
    SUMMARY_COLUMNS,
    TEST_COLUMNS,
    cliffs_delta,
    cliffs_magnitude,
    holm,
    summarize,
    wilcoxon_holm,
)


def _per_query(values: dict[str, list[float]], repos: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    frame.insert(0, "repo", repos)
    frame.insert(0, "qid", np.arange(len(frame), dtype=np.int64))
    return frame


def _run(qids: list[int], values: list[float], metric: str = "ndcg_10") -> pd.DataFrame:
    return pd.DataFrame({"qid": np.asarray(qids, dtype=np.int64), metric: values})


# ------------------------------------------------------------------------------------------------ summarize


@pytest.fixture(scope="module")
def per_query() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 40
    values = {"ndcg_10": rng.uniform(0.2, 0.9, n).tolist(), "popt": rng.uniform(0.3, 0.8, n).tolist()}
    values["ndcg_10"][3] = math.nan
    return _per_query(values, [f"repo{i % 8}" for i in range(n)])


def test_summarize_mean_and_interval(per_query: pd.DataFrame) -> None:
    out = summarize(per_query, n_boot=500)
    assert list(out.columns) == list(SUMMARY_COLUMNS)
    assert out["metric"].tolist() == ["ndcg_10", "popt"]
    for row in out.itertuples(index=False):
        assert row.mean == pytest.approx(per_query[row.metric].mean())  # NaN-aware query mean
        assert row.ci_low <= row.mean <= row.ci_high
        assert row.n_repos == 8
    assert out.loc[0, "n_queries"] == 39 and out.loc[1, "n_queries"] == 40
    assert str(out["n_queries"].dtype) == "int64" and str(out["n_repos"].dtype) == "int64"


def test_summarize_is_deterministic_per_seed(per_query: pd.DataFrame) -> None:
    a = summarize(per_query, n_boot=300, seed=1)
    pd.testing.assert_frame_equal(a, summarize(per_query, n_boot=300, seed=1))
    b = summarize(per_query, n_boot=300, seed=2)
    assert not np.allclose(a["ci_low"], b["ci_low"])
    pd.testing.assert_series_equal(a["mean"], b["mean"])


def test_summarize_resamples_repositories() -> None:
    frame = _per_query({"ndcg_10": [0.0, 0.0, 0.0, 1.0]}, ["a", "a", "a", "b"])
    out = summarize(frame, n_boot=2000)
    assert out.loc[0, "mean"] == pytest.approx(0.25)
    # drawing only repository a gives 0, only b gives 1: the interval spans both
    assert out.loc[0, "ci_low"] == 0.0 and out.loc[0, "ci_high"] == 1.0


def test_summarize_degenerate_intervals() -> None:
    single = _per_query({"ndcg_10": [0.1, 0.5, 0.9]}, ["a", "a", "a"])
    row = summarize(single).iloc[0]
    assert row["ci_low"] == pytest.approx(row["mean"]) and row["ci_high"] == pytest.approx(row["mean"])
    same = _per_query({"ndcg_10": [0.3] * 6}, ["a", "b", "c", "a", "b", "c"])
    row = summarize(same).iloc[0]
    assert row["ci_low"] == pytest.approx(0.3, abs=1e-12) and row["ci_high"] == pytest.approx(0.3, abs=1e-12)


def test_summarize_all_nan_metric() -> None:
    frame = _per_query({"ndcg_10": [math.nan, math.nan], "popt": [0.5, 0.7]}, ["a", "b"])
    row = summarize(frame).iloc[0]
    assert row["metric"] == "ndcg_10"
    assert math.isnan(row["mean"]) and math.isnan(row["ci_low"]) and math.isnan(row["ci_high"])
    assert row["n_queries"] == 0 and row["n_repos"] == 0


def test_summarize_groups_by_model_and_protocol(per_query: pd.DataFrame) -> None:
    frames = []
    for model, shift in (("hotspot", 0.0), ("ranker", 0.1)):
        for protocol in ("lopo", "temporal"):
            part = per_query.assign(ndcg_10=per_query["ndcg_10"] + shift)
            part.insert(0, "protocol", protocol)
            part.insert(0, "model", model)
            frames.append(part)
    out = summarize(pd.concat(frames, ignore_index=True), n_boot=200, metrics=["ndcg_10"])
    assert list(out.columns) == ["model", "protocol", *SUMMARY_COLUMNS]
    assert out[["model", "protocol"]].values.tolist() == [
        ["hotspot", "lopo"],
        ["hotspot", "temporal"],
        ["ranker", "lopo"],
        ["ranker", "temporal"],
    ]
    assert out.loc[2, "mean"] - out.loc[0, "mean"] == pytest.approx(0.1)
    # the same seed draws the same repositories for every group
    assert out.loc[2, "ci_low"] - out.loc[0, "ci_low"] == pytest.approx(0.1)


def test_summarize_errors_and_empty(per_query: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="repo"):
        summarize(per_query.drop(columns=["repo"]))
    with pytest.raises(ValueError, match="n_boot"):
        summarize(per_query, n_boot=0)
    with pytest.raises(ValueError, match="nulls"):
        summarize(per_query.assign(repo=None))
    with pytest.raises(ValueError, match="missing metric"):
        summarize(per_query, metrics=["ndcg_10", "nope"])
    with pytest.raises(ValueError, match="no metric"):
        summarize(per_query[["qid", "repo"]])
    empty = summarize(per_query.iloc[:0])
    assert empty.empty and list(empty.columns) == list(SUMMARY_COLUMNS)


# ------------------------------------------------------------------------------------------------ Holm


def test_holm_hand_computed() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([0.01, math.nan, 0.03])
    assert out[0] == pytest.approx(0.02) and math.isnan(out[1]) and out[2] == pytest.approx(0.03)
    assert holm([0.5, 0.6]).tolist() == [1.0, 1.0]
    assert holm([]).tolist() == []
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm([0.5, 1.5])
    with pytest.raises(ValueError, match="1-D"):
        holm([[0.5]])


@settings(max_examples=200, deadline=None, derandomize=True)
@given(st.lists(st.one_of(st.floats(0.0, 1.0), st.just(math.nan)), max_size=12))
def test_holm_bounds(p: list[float]) -> None:
    raw = np.asarray(p, dtype=float)
    adj = holm(raw)
    known = ~np.isnan(raw)
    assert np.array_equal(np.isnan(adj), ~known)
    assert (adj[known] >= raw[known] - 1e-15).all() and (adj[known] <= 1.0).all()


# -------------------------------------------------------------------------------------------- Cliff's delta


def test_cliffs_delta() -> None:
    delta, magnitude = cliffs_delta([1, 2, 3, 4], [2, 2, 5])
    assert delta == pytest.approx(-1 / 6) and magnitude == "small"
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == (1.0, "large")
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == (0.0, "negligible")
    assert cliffs_delta([2, 2, 5], [1, 2, 3, 4])[0] == pytest.approx(1 / 6)
    assert cliffs_delta([1.0, math.nan], [0.0]) == (1.0, "large")
    with pytest.raises(ValueError, match="non-empty"):
        cliffs_delta([], [1.0])
    with pytest.raises(ValueError, match="non-empty"):
        cliffs_delta([1.0], [math.nan])
    with pytest.raises(ValueError, match="1-D"):
        cliffs_delta([[1.0]], [1.0])


def test_cliffs_magnitude_thresholds() -> None:
    assert cliffs_magnitude(0.0) == "negligible"
    assert cliffs_magnitude(0.1469) == "negligible"
    assert cliffs_magnitude(0.147) == "small"
    assert cliffs_magnitude(-0.3299) == "small"
    assert cliffs_magnitude(0.33) == "medium"
    assert cliffs_magnitude(0.4739) == "medium"
    assert cliffs_magnitude(-0.474) == "large"
    assert cliffs_magnitude(1.0) == "large"
    assert cliffs_magnitude(math.nan) == ""


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    st.lists(st.integers(0, 5), min_size=1, max_size=15),
    st.lists(st.integers(0, 5), min_size=1, max_size=15),
)
def test_cliffs_delta_matches_pairwise_definition(a: list[int], b: list[int]) -> None:
    expected = sum((x > y) - (x < y) for x in a for y in b) / (len(a) * len(b))
    assert cliffs_delta(a, b)[0] == pytest.approx(expected)
    assert cliffs_delta(b, a)[0] == pytest.approx(-expected)


# ------------------------------------------------------------------------------------------------ Wilcoxon


@pytest.fixture(scope="module")
def runs() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    base = rng.uniform(0.3, 0.7, 20)
    qids = list(range(20))
    return {
        "model": _run(qids, (base + 0.05 + rng.normal(0, 0.01, 20)).tolist()),
        "close": _run(qids, (base + rng.normal(0, 0.05, 20)).tolist()),
        "worse": _run(qids, (base - 0.03 + rng.normal(0, 0.02, 20)).tolist()),
        "same": _run(qids, base.tolist()),
    }


def test_wilcoxon_detects_a_better_model(runs: dict[str, pd.DataFrame]) -> None:
    out = wilcoxon_holm(runs["model"], {"same": runs["same"]}, "ndcg_10")
    assert list(out.columns) == list(TEST_COLUMNS)
    row = out.iloc[0]
    assert row["n"] == 20 and row["n_a_higher"] == 20 and row["n_equal"] == 0 and row["n_a_lower"] == 0
    assert row["p_value"] < 0.05 and row["significant"]
    assert row["mean_diff"] == pytest.approx(row["mean_a"] - row["mean_b"])
    assert row["cliffs_delta"] > 0 and row["magnitude"] in {"small", "medium", "large"}


def test_wilcoxon_identical_results(runs: dict[str, pd.DataFrame]) -> None:
    row = wilcoxon_holm(runs["same"], {"same": runs["same"]}, "ndcg_10").iloc[0]
    assert row["p_value"] == 1.0 and row["statistic"] == 0.0 and not row["significant"]
    assert row["n_equal"] == 20 and row["cliffs_delta"] == 0.0 and row["magnitude"] == "negligible"


def test_wilcoxon_pairs_on_qid_and_drops_nan() -> None:
    a = _run([1, 2, 3, 4, 5, 6], [0.5, 0.6, math.nan, 0.8, 0.9, 0.4])
    b = _run([6, 5, 4, 3, 2, 10], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    row = wilcoxon_holm(a, {"b": b}, "ndcg_10").iloc[0]
    assert row["n"] == 4  # qids 2, 4, 5, 6; qid 3 has a NaN
    assert row["mean_a"] == pytest.approx(np.mean([0.6, 0.8, 0.9, 0.4]))
    assert row["mean_b"] == pytest.approx(np.mean([0.5, 0.3, 0.2, 0.1]))
    none = wilcoxon_holm(a, {"b": _run([100], [0.5])}, "ndcg_10").iloc[0]
    assert none["n"] == 0 and math.isnan(none["p_value"]) and not none["significant"]
    assert math.isnan(none["cliffs_delta"]) and none["magnitude"] == ""


def test_wilcoxon_matches_scipy_and_holm(runs: dict[str, pd.DataFrame]) -> None:
    others = {"close": runs["close"], "worse": runs["worse"], "same": runs["same"]}
    out = wilcoxon_holm(runs["model"], others, "ndcg_10")
    assert out["baseline"].tolist() == ["close", "worse", "same"]
    x = runs["model"]["ndcg_10"].to_numpy()
    for row in out.itertuples(index=False):
        y = others[row.baseline]["ndcg_10"].to_numpy()
        expected = scipy_stats.wilcoxon(x, y, zero_method="wilcox")
        assert row.p_value == pytest.approx(expected.pvalue)
        assert row.statistic == pytest.approx(expected.statistic)
    assert out["p_holm"].to_numpy() == pytest.approx(holm(out["p_value"]))
    assert out["significant"].dtype == bool


def test_wilcoxon_one_sided(runs: dict[str, pd.DataFrame]) -> None:
    greater = wilcoxon_holm(runs["model"], {"same": runs["same"]}, "ndcg_10", alternative="greater").iloc[0]
    less = wilcoxon_holm(runs["model"], {"same": runs["same"]}, "ndcg_10", alternative="less").iloc[0]
    assert greater["p_value"] < 0.001 < 0.99 < less["p_value"]


def test_wilcoxon_errors_and_empty(runs: dict[str, pd.DataFrame]) -> None:
    dup = pd.concat([runs["model"], runs["model"].head(1)])
    with pytest.raises(ValueError, match="duplicate qids"):
        wilcoxon_holm(dup, {"same": runs["same"]}, "ndcg_10")
    with pytest.raises(ValueError, match="duplicate qids"):
        wilcoxon_holm(runs["model"], {"dup": dup}, "ndcg_10")
    with pytest.raises(ValueError, match="popt"):
        wilcoxon_holm(runs["model"], {"same": runs["same"]}, "popt")
    with pytest.raises(ValueError, match="alternative"):
        wilcoxon_holm(runs["model"], {}, "ndcg_10", alternative="bigger")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        wilcoxon_holm(runs["model"], {}, "ndcg_10", alpha=1.5)
    empty = wilcoxon_holm(runs["model"], {}, "ndcg_10")
    assert empty.empty and list(empty.columns) == list(TEST_COLUMNS)
