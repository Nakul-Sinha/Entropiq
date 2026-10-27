import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score

from tdp_ml.eval.metrics import (
    HIGHER_IS_BETTER,
    METRIC_LABELS,
    METRICS,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    auc_pr,
    brier,
    defect_density,
    ece,
    effort_curve,
    ndcg_at_k,
    popt,
    precision_at_k,
    rank_order,
    recall_at_loc,
    reliability_table,
    spearman,
)

FAST = settings(max_examples=150, deadline=None, derandomize=True)


def test_metric_names_and_labels() -> None:
    assert METRICS == RANKING_METRICS + PROBABILITY_METRICS
    assert set(METRIC_LABELS) == set(METRICS) == set(HIGHER_IS_BETTER)
    assert METRIC_LABELS["ndcg_10"] == "NDCG@10"
    assert not HIGHER_IS_BETTER["brier"] and not HIGHER_IS_BETTER["ece"] and HIGHER_IS_BETTER["popt"]


# ---------------------------------------------------------------------------------------------- rank_order


def test_rank_order_is_stable_descending() -> None:
    assert rank_order([1.0, 3.0, 3.0, 2.0, 1.0]).tolist() == [1, 2, 3, 0, 4]
    assert rank_order([-np.inf, 0.0, np.inf]).tolist() == [2, 1, 0]
    assert rank_order([]).tolist() == []


def test_rank_order_rejects_nan_and_2d() -> None:
    with pytest.raises(ValueError, match="NaN"):
        rank_order([1.0, math.nan])
    with pytest.raises(ValueError, match="1-D"):
        rank_order([[1.0, 2.0]])


# ---------------------------------------------------------------------------------------------- NDCG


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_ndcg_perfect_ranking_is_one(k: int) -> None:
    grades = [4, 3, 3, 1, 0, 0, 2]
    assert ndcg_at_k(grades, grades, k) == pytest.approx(1.0)
    # equal grades get equal scores: any order among them is still perfect
    assert ndcg_at_k([2, 2, 0, 1], [5.0, 5.0, 1.0, 3.0], k) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_hand_computed() -> None:
    grades, scores = [3, 2, 1, 0], [0.0, 1.0, 2.0, 3.0]
    # ranked grades are 0, 1, 2, 3 with gains 0, 1, 3, 7; the ideal has gains 7, 3, 1, 0
    dcg4 = 1 / math.log2(3) + 3 / math.log2(4) + 7 / math.log2(5)
    idcg4 = 7 + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(grades, scores, 4) == pytest.approx(dcg4 / idcg4, abs=1e-15)
    assert ndcg_at_k(grades, scores, 4) == pytest.approx(0.547831481922746, abs=1e-12)
    assert ndcg_at_k(grades, scores, 10) == pytest.approx(0.547831481922746, abs=1e-12)
    assert ndcg_at_k(grades, scores, 3) == pytest.approx(0.22686868558673368, abs=1e-12)
    assert ndcg_at_k(grades, scores, 2) == pytest.approx(0.07094846566967604, abs=1e-12)
    assert ndcg_at_k(grades, scores, 1) == 0.0


def test_ndcg_ties_in_scores_keep_input_order() -> None:
    assert ndcg_at_k([0, 3], [1.0, 1.0], 2) == pytest.approx(0.6309297535714575, abs=1e-12)
    assert ndcg_at_k([3, 0], [1.0, 1.0], 2) == 1.0


def test_ndcg_undefined_cases_are_nan() -> None:
    assert math.isnan(ndcg_at_k([0, 0, 0], [3.0, 2.0, 1.0], 10))
    assert math.isnan(ndcg_at_k([], [], 10))


def test_ndcg_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k([1, 0], [1.0, 0.0], 0)
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k([1, 0], [1.0, 0.0], True)
    with pytest.raises(ValueError, match="NaN"):
        ndcg_at_k([1, 0], [1.0, math.nan], 2)
    with pytest.raises(ValueError, match=">= 0"):
        ndcg_at_k([1, -1], [1.0, 0.0], 2)
    with pytest.raises(ValueError, match="equal lengths"):
        ndcg_at_k([1, 0, 2], [1.0, 0.0], 2)
    with pytest.raises(ValueError, match="numeric"):
        ndcg_at_k(["a"], [1.0], 2)


grade_lists = st.lists(st.integers(0, 4), min_size=1, max_size=30)


@FAST
@given(grade_lists, st.integers(1, 40), st.integers(0, 2**32 - 1))
def test_ndcg_bounds_and_joint_permutation(grades: list[int], k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    g = np.asarray(grades, dtype=float)
    scores = rng.permutation(len(g)).astype(float)  # distinct scores
    value = ndcg_at_k(g, scores, k)
    assert math.isnan(value) == (g.sum() == 0)
    if not math.isnan(value):
        assert 0.0 <= value <= 1.0
    perm = rng.permutation(len(g))
    again = ndcg_at_k(g[perm], scores[perm], k)
    assert (math.isnan(value) and math.isnan(again)) or again == pytest.approx(value, abs=1e-12)


# ---------------------------------------------------------------------------------------------- precision


def test_precision_at_k() -> None:
    grades, scores = [4, 0, 3, 1, 2], [5.0, 4.0, 3.0, 2.0, 1.0]
    assert precision_at_k(grades, scores, 3) == pytest.approx(2 / 3)
    assert precision_at_k(grades, scores, 10) == pytest.approx(0.4)  # k > n uses n
    assert precision_at_k(grades, scores, 2, min_grade=0) == 1.0
    assert precision_at_k(grades, scores, 1) == 1.0
    assert math.isnan(precision_at_k([], [], 10))
    with pytest.raises(ValueError, match="k must be"):
        precision_at_k(grades, scores, -1)


# ---------------------------------------------------------------------------------------------- recall@LOC


def test_recall_at_loc_hand_computed() -> None:
    scores, sloc, defects = [4.0, 3.0, 2.0, 1.0], [10, 10, 30, 50], [1, 0, 2, 1]
    assert recall_at_loc(scores, sloc, defects, 0.2) == pytest.approx(0.25)  # boundary is inclusive
    assert recall_at_loc(scores, sloc, defects, 0.5) == pytest.approx(0.75)
    assert recall_at_loc(scores, sloc, defects, 1.0) == 1.0
    assert recall_at_loc(scores, sloc, defects, 0.05) == 0.0
    assert recall_at_loc(scores, sloc, defects) == pytest.approx(0.25)  # default fraction 0.2
    assert recall_at_loc(scores[::-1], sloc, defects, 0.2) == 0.0  # the 50-line file does not fit


def test_recall_at_loc_stops_at_first_file_that_does_not_fit() -> None:
    # ranked sloc 10, 30, 5: the 30-line file breaks the 20% budget, the 5-line file after it is not taken
    assert recall_at_loc([3.0, 2.0, 1.0, 0.0], [10, 30, 5, 55], [1, 1, 1, 1], 0.2) == pytest.approx(0.25)


def test_recall_at_loc_undefined_and_invalid() -> None:
    assert math.isnan(recall_at_loc([1.0, 2.0], [10, 20], [0, 0]))
    assert math.isnan(recall_at_loc([], [], []))
    assert recall_at_loc([1.0, 2.0], [0, 0], [1, 0]) == 1.0  # zero total effort: everything fits
    with pytest.raises(ValueError, match="frac"):
        recall_at_loc([1.0], [10], [1], 1.5)
    with pytest.raises(ValueError, match="sloc must be >= 0"):
        recall_at_loc([1.0, 2.0], [10, -1], [1, 0])
    with pytest.raises(ValueError, match="defects must be finite"):
        recall_at_loc([1.0, 2.0], [10, 1], [1, math.inf])


# ---------------------------------------------------------------------------------------------- effort curve


@FAST
@given(st.integers(1, 25), st.integers(0, 2**32 - 1))
def test_effort_curve_shape(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sloc = rng.integers(0, 50, n).astype(float)
    sloc[0] += 1
    defects = rng.integers(0, 3, n).astype(float)
    defects[-1] += 1
    x, y = effort_curve(rng.normal(size=n), sloc, defects)
    assert len(x) == len(y) == n + 1
    assert (x[0], y[0]) == (0.0, 0.0) and (x[-1], y[-1]) == (1.0, 1.0)
    assert (np.diff(x) >= 0).all() and (np.diff(y) >= 0).all()


def test_effort_curve_follows_rank_order() -> None:
    x, y = effort_curve([1.0, 3.0, 2.0], [10, 20, 70], [1, 0, 1])
    assert x.tolist() == pytest.approx([0.0, 0.2, 0.9, 1.0])
    assert y.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])


def test_effort_curve_needs_effort_and_defects() -> None:
    with pytest.raises(ValueError, match="sloc"):
        effort_curve([1.0, 2.0], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="defects"):
        effort_curve([1.0, 2.0], [1, 1], [0, 0])
    with pytest.raises(ValueError):
        effort_curve([], [], [])


# ---------------------------------------------------------------------------------------------- Popt


def test_popt_hand_computed() -> None:
    sloc, defects = [10, 20, 30, 40], [2, 1, 1, 0]
    assert popt([3.0, 4.0, 1.0, 2.0], sloc, defects) == pytest.approx(0.72, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_popt_optimal_is_one_and_worst_is_zero(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sloc = rng.integers(1, 200, 30).astype(float)
    defects = rng.poisson(1.0, 30).astype(float)
    defects[0] += 1
    density = defects / sloc
    assert popt(density, sloc, defects) == pytest.approx(1.0, abs=1e-12)
    assert popt(-density, sloc, defects) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= popt(rng.normal(size=30), sloc, defects) <= 1.0


def test_defect_density_handles_files_without_sloc() -> None:
    sloc, defects = [0, 0, 10, 4], [1, 0, 5, 0]
    density = defect_density(sloc, defects)
    assert density.tolist() == [np.inf, 0.0, 0.5, 0.0]  # 0 / 0 is 0, never NaN
    x, y = effort_curve(density, sloc, defects)  # the optimal curve of a lift chart
    assert y.tolist() == pytest.approx([0.0, 1 / 6, 1.0, 1.0, 1.0])
    assert x.tolist() == pytest.approx([0.0, 0.0, 10 / 14, 10 / 14, 1.0])
    assert popt(density, sloc, defects) == 1.0 and popt(-density, sloc, defects) == 0.0
    with pytest.raises(ValueError, match="equal lengths"):
        defect_density([1, 2], [1])
    with pytest.raises(ValueError, match="sloc must be >= 0"):
        defect_density([-1], [1])


def test_popt_edge_cases() -> None:
    # equal densities: every order is optimal
    assert popt([1.0, 2.0, 3.0], [10, 20, 30], [1, 2, 3]) == 1.0
    assert math.isnan(popt([1.0, 2.0], [10, 20], [0, 0]))
    assert math.isnan(popt([1.0, 2.0], [0, 0], [1, 0]))
    assert math.isnan(popt([], [], []))
    # a defective file without SLOC has infinite density and belongs first
    sloc, defects = [0, 10, 10], [1, 1, 0]
    assert popt([3.0, 2.0, 1.0], sloc, defects) == 1.0
    assert popt([1.0, 2.0, 3.0], sloc, defects) == 0.0
    assert 0.0 < popt([2.0, 3.0, 1.0], sloc, defects) < 1.0


@FAST
@given(st.integers(1, 20), st.integers(0, 2**32 - 1))
def test_popt_bounds(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sloc = rng.integers(0, 100, n).astype(float)
    defects = rng.integers(0, 4, n).astype(float)
    value = popt(rng.integers(0, 5, n).astype(float), sloc, defects)
    if sloc.sum() == 0 or defects.sum() == 0:
        assert math.isnan(value)
    else:
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------------------------- Spearman


def test_spearman_basic() -> None:
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(spearman([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))
    assert math.isnan(spearman([1.0], [2.0]))
    with pytest.raises(ValueError, match="pain must be finite"):
        spearman([1.0, 2.0], [1.0, math.nan])


@pytest.mark.parametrize("seed", range(10))
def test_spearman_matches_scipy_with_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 5, 40).astype(float)
    pain = rng.integers(0, 4, 40).astype(float)
    expected = scipy_stats.spearmanr(scores, pain).statistic
    assert spearman(scores, pain) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------------------------- probability


def test_brier() -> None:
    assert brier([0.2, 0.8], [0, 1]) == pytest.approx(0.04)
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert math.isnan(brier([], []))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier([1.2], [1])
    with pytest.raises(ValueError, match="only 0 and 1"):
        brier([0.5], [2])


def test_reliability_table() -> None:
    table = reliability_table([0.05, 1.0, 0.31, 0.39], [0, 1, 1, 0], bins=10)
    assert list(table.columns) == ["bin", "lower", "upper", "count", "mean_prob", "frac_positive"]
    assert len(table) == 10 and str(table["bin"].dtype) == "int64" and str(table["count"].dtype) == "int64"
    assert table["count"].tolist() == [1, 0, 0, 2, 0, 0, 0, 0, 0, 1]  # 1.0 falls in the last bin
    assert table.loc[3, "mean_prob"] == pytest.approx(0.35) and table.loc[3, "frac_positive"] == 0.5
    assert table.loc[1, ["mean_prob", "frac_positive"]].isna().all()
    assert table["lower"].iloc[0] == 0.0 and table["upper"].iloc[-1] == 1.0
    assert len(reliability_table([0.5], [1], bins=1)) == 1
    with pytest.raises(ValueError, match="bins"):
        reliability_table([0.5], [1], bins=0)


def test_ece_hand_computed_and_overconfident() -> None:
    assert ece([0.05, 0.15, 0.95], [0, 0, 1]) == pytest.approx(1 / 12)
    y = np.array([0, 1] * 500)
    assert ece(np.full(1000, 0.9), y) == pytest.approx(0.4)
    assert math.isnan(ece([], []))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ece_of_calibrated_data_is_small(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=20_000)
    y = rng.uniform(size=20_000) < p
    assert ece(p, y) < 0.03


def test_auc_pr_hand_computed() -> None:
    assert auc_pr([0.1, 0.9, 0.8, 0.3], [0, 1, 0, 1]) == pytest.approx(5 / 6)
    assert auc_pr([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)
    assert auc_pr([0.2, 0.7], [1, 1]) == 1.0
    assert math.isnan(auc_pr([0.2, 0.7], [0, 0]))
    assert math.isnan(auc_pr([], []))


def test_auc_pr_matches_sklearn_on_tied_data() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(200):
        n = int(rng.integers(1, 60))
        p = rng.integers(0, 6, n) / 5
        y = rng.integers(0, 2, n)
        if y.sum() == 0:
            continue
        assert auc_pr(p, y) == pytest.approx(average_precision_score(y, p), abs=1e-12)
        checked += 1
    assert checked > 150
