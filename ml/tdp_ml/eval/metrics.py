"""Ranking and probability metrics for one query, and per-query evaluation of a scored dataset.

A *query* is one (repository, snapshot) pair: the files ranked together. Every ranking metric here takes
the model scores of one query (higher means "more pain expected") and the labels of the same files.

Ordering rule: files are ranked by score descending; ties keep their input order (stable sort).
:func:`evaluate` sorts each query by ``path`` first, so ties are broken by path ascending and every result
is deterministic.

NaN policy: a metric that is undefined for a query (no relevant files, no defects, a constant input)
returns NaN instead of raising, so summaries can skip it with a NaN-aware mean. Invalid inputs (NaN
scores, negative counts, probabilities outside [0, 1], arrays of different lengths) raise ``ValueError``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

NDCG_KS: Final = (5, 10, 20)
PRECISION_K: Final = 10
MIN_GRADE: Final = 3
LOC_FRACTION: Final = 0.2
ECE_BINS: Final = 10

RANKING_METRICS: Final = ("ndcg_5", "ndcg_10", "ndcg_20", "precision_10", "recall_20loc", "popt", "spearman")
PROBABILITY_METRICS: Final = ("brier", "ece", "auc_pr")
METRICS: Final = RANKING_METRICS + PROBABILITY_METRICS

METRIC_LABELS: Final[Mapping[str, str]] = {
    "ndcg_5": "NDCG@5",
    "ndcg_10": "NDCG@10",
    "ndcg_20": "NDCG@20",
    "precision_10": "P@10",
    "recall_20loc": "Recall@20%LOC",
    "popt": "Popt",
    "spearman": "Spearman",
    "brier": "Brier",
    "ece": "ECE",
    "auc_pr": "AUC-PR",
}
HIGHER_IS_BETTER: Final[Mapping[str, bool]] = {m: m not in ("brier", "ece") for m in METRICS}

QUERY_COLUMNS: Final = ("qid", "repo", "snapshot_idx", "snapshot_time", "n_files", "n_defective", "n_defects")
RUN_COLUMNS: Final = ("model", "protocol", "split")
GROUP_COLUMNS: Final = ("model", "protocol")
EVAL_INPUT_COLUMNS: Final = (
    "qid",
    "repo",
    "snapshot_idx",
    "snapshot_time",
    "path",
    "grade",
    "pain",
    "latent_defects",
    "sloc",
)

_SLOC_TOLERANCE: Final = 1e-9
_POPT_TOLERANCE: Final = 1e-12


# ---------------------------------------------------------------------------------------------------------
# input checks


def _vector(values: npt.ArrayLike, name: str) -> FloatArray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _scores(values: npt.ArrayLike, name: str = "scores") -> FloatArray:
    arr = _vector(values, name)
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    return arr


def _finite(values: npt.ArrayLike, name: str) -> FloatArray:
    arr = _vector(values, name)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _non_negative(values: npt.ArrayLike, name: str) -> FloatArray:
    arr = _finite(values, name)
    if (arr < 0).any():
        raise ValueError(f"{name} must be >= 0")
    return arr


def _probabilities(values: npt.ArrayLike, name: str = "prob") -> FloatArray:
    arr = _finite(values, name)
    if ((arr < 0) | (arr > 1)).any():
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr


def _binary(values: npt.ArrayLike, name: str = "y") -> FloatArray:
    arr = _vector(values, name)
    if not np.isin(arr, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return arr


def _same_length(*arrays: tuple[str, FloatArray]) -> int:
    lengths = {name: len(arr) for name, arr in arrays}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"arrays must have equal lengths, got {lengths}")
    return len(arrays[0][1])


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


# ---------------------------------------------------------------------------------------------------------
# ranking metrics


def rank_order(scores: npt.ArrayLike) -> IntArray:
    """Positions of the items from highest to lowest score.

    Ties keep their input order (stable sort); :func:`evaluate` sorts every query by path first, so ties
    are broken by path ascending. ``+inf`` ranks first and ``-inf`` last.

    Raises ``ValueError`` if ``scores`` is not 1-D or contains NaN.
    """
    s = _scores(scores)
    return np.argsort(-s, kind="stable")


def ndcg_at_k(grades: npt.ArrayLike, scores: npt.ArrayLike, k: int) -> float:
    """Normalised discounted cumulative gain of the top ``k`` ranked items, in [0, 1].

    Gain is ``2**grade - 1`` and the discount of the item at 1-based rank ``i`` is ``log2(i + 1)``. The
    ideal ranking sorts the grades descending. When ``k`` exceeds the number of items ``n``, the top ``n``
    are used.

    Returns NaN for an empty query or when every grade is 0 (the ideal gain is 0).
    Raises ``ValueError`` for ``k < 1``, NaN scores, negative or non-finite grades, or unequal lengths.
    """
    k = _positive_int(k, "k")
    g = _non_negative(grades, "grades")
    s = _scores(scores)
    n = _same_length(("grades", g), ("scores", s))
    if n == 0:
        return math.nan
    m = min(k, n)
    discounts = 1.0 / np.log2(np.arange(2, m + 2, dtype=np.float64))
    gains = np.exp2(g) - 1.0
    ideal = float(np.sort(gains)[::-1][:m] @ discounts)
    if ideal <= 0.0:
        return math.nan
    dcg = float(gains[rank_order(s)[:m]] @ discounts)
    return min(1.0, dcg / ideal)


def precision_at_k(
    grades: npt.ArrayLike, scores: npt.ArrayLike, k: int, min_grade: float = MIN_GRADE
) -> float:
    """Share of the top ``k`` ranked items whose grade is at least ``min_grade``, in [0, 1].

    The denominator is ``min(k, n)``, so a query with fewer than ``k`` files is not penalised.

    Returns NaN for an empty query.
    Raises ``ValueError`` for ``k < 1``, NaN scores, negative or non-finite grades, or unequal lengths.
    """
    k = _positive_int(k, "k")
    g = _non_negative(grades, "grades")
    s = _scores(scores)
    n = _same_length(("grades", g), ("scores", s))
    if n == 0:
        return math.nan
    m = min(k, n)
    top = g[rank_order(s)[:m]]
    return float(np.count_nonzero(top >= min_grade)) / m


def _check_fraction(frac: float) -> float:
    if not isinstance(frac, int | float | np.floating) or not 0.0 <= float(frac) <= 1.0:
        raise ValueError(f"frac must lie in [0, 1], got {frac!r}")
    return float(frac)


def recall_at_loc(
    scores: npt.ArrayLike, sloc: npt.ArrayLike, defects: npt.ArrayLike, frac: float = LOC_FRACTION
) -> float:
    """Share of all defects found by inspecting ranked files within ``frac`` of the total SLOC, in [0, 1].

    Files are taken in ranked order while the cumulative SLOC stays at or below ``frac * sum(sloc)``
    (inclusive, with a relative tolerance of 1e-9); the first file that does not fit ends the prefix. A
    first file larger than the budget gives 0.0. When the total SLOC is 0 every file fits.

    Units: ``sloc`` is source lines of code per file, ``defects`` a count per file (latent defects).
    Returns NaN for an empty query or when ``sum(defects) == 0``.
    Raises ``ValueError`` for ``frac`` outside [0, 1], NaN scores, negative or non-finite SLOC or defect
    counts, or unequal lengths.
    """
    frac = _check_fraction(frac)
    s = _scores(scores)
    effort = _non_negative(sloc, "sloc")
    d = _non_negative(defects, "defects")
    n = _same_length(("scores", s), ("sloc", effort), ("defects", d))
    total_defects = float(d.sum())
    if n == 0 or total_defects == 0.0:
        return math.nan
    total_sloc = float(effort.sum())
    order = rank_order(s)
    budget = frac * total_sloc + _SLOC_TOLERANCE * total_sloc
    # SLOC is non-negative, so the cumulative sum is non-decreasing and the files that fit form a prefix.
    taken = int(np.count_nonzero(np.cumsum(effort[order]) <= budget))
    return float(d[order][:taken].sum()) / total_defects


def effort_curve(
    scores: npt.ArrayLike, sloc: npt.ArrayLike, defects: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Cumulative effort and defect fractions when files are inspected in ranked order.

    Returns ``(x, y)`` of length ``n + 1`` starting at ``(0, 0)`` and ending at exactly ``(1, 1)``: ``x``
    is the cumulative SLOC fraction and ``y`` the cumulative defect fraction after each ranked file. Both
    are non-decreasing. Pass :func:`defect_density` as ``scores`` to get the optimal curve (lift charts).

    Raises ``ValueError`` when ``sum(sloc) == 0`` or ``sum(defects) == 0`` (including an empty query), for
    NaN scores, negative or non-finite SLOC or defect counts, or unequal lengths.
    """
    s = _scores(scores)
    effort = _non_negative(sloc, "sloc")
    d = _non_negative(defects, "defects")
    _same_length(("scores", s), ("sloc", effort), ("defects", d))
    order = rank_order(s)
    cum_effort = np.concatenate(([0.0], np.cumsum(effort[order])))
    cum_defects = np.concatenate(([0.0], np.cumsum(d[order])))
    if cum_effort[-1] <= 0.0:
        raise ValueError("effort curve needs sum(sloc) > 0")
    if cum_defects[-1] <= 0.0:
        raise ValueError("effort curve needs sum(defects) > 0")
    return cum_effort / cum_effort[-1], cum_defects / cum_defects[-1]


def defect_density(sloc: npt.ArrayLike, defects: npt.ArrayLike) -> FloatArray:
    """Defects per SLOC of every file: the scores of the optimal effort-aware order.

    ``defects / sloc``; a file with 0 SLOC has density ``+inf`` when it has defects and 0 otherwise, so
    the result never contains NaN. Passed as ``scores`` to :func:`effort_curve` it gives the optimal curve
    and its negation the worst one (the curves :func:`popt` compares against).

    Units: ``sloc`` is source lines of code per file, ``defects`` a count per file (latent defects).
    Raises ``ValueError`` for negative or non-finite SLOC or defect counts, or unequal lengths.
    """
    effort = _non_negative(sloc, "sloc")
    d = _non_negative(defects, "defects")
    _same_length(("sloc", effort), ("defects", d))
    density = np.divide(d, effort, out=np.zeros_like(d), where=effort > 0)
    density[(effort == 0) & (d > 0)] = np.inf
    return density


def _curve_area(scores: FloatArray, sloc: FloatArray, defects: FloatArray) -> float:
    x, y = effort_curve(scores, sloc, defects)
    return float(np.trapezoid(y, x))


def popt(scores: npt.ArrayLike, sloc: npt.ArrayLike, defects: npt.ArrayLike) -> float:
    """Effort-aware ranking quality: the model's effort curve area relative to the optimal and worst.

    ``popt = 1 - (A_opt - A_model) / (A_opt - A_worst)``, where ``A`` is the trapezoid area under the
    :func:`effort_curve`. The optimal order ranks files by defect density (defects per SLOC) descending,
    the worst order by density ascending; a defective file with 0 SLOC has infinite density (it comes first
    in the optimal order). When ``A_opt - A_worst <= 1e-12`` (all densities equal) every order is optimal
    and the result is 1.0. The result is clipped to [0, 1].

    Returns NaN for an empty query, ``sum(sloc) == 0`` or ``sum(defects) == 0``.
    Raises ``ValueError`` for NaN scores, negative or non-finite SLOC or defect counts, or unequal lengths.
    """
    s = _scores(scores)
    effort = _non_negative(sloc, "sloc")
    d = _non_negative(defects, "defects")
    n = _same_length(("scores", s), ("sloc", effort), ("defects", d))
    if n == 0 or float(effort.sum()) == 0.0 or float(d.sum()) == 0.0:
        return math.nan
    density = defect_density(effort, d)
    a_model = _curve_area(s, effort, d)
    a_opt = _curve_area(density, effort, d)
    a_worst = _curve_area(-density, effort, d)
    span = a_opt - a_worst
    if span <= _POPT_TOLERANCE:
        return 1.0
    return float(np.clip(1.0 - (a_opt - a_model) / span, 0.0, 1.0))


def spearman(scores: npt.ArrayLike, pain: npt.ArrayLike) -> float:
    """Spearman rank correlation between scores and pain, in [-1, 1].

    Pearson correlation of the average ranks (ties share their mean rank), the same as
    ``scipy.stats.spearmanr``.

    Returns NaN when there are fewer than 2 items or either input is constant.
    Raises ``ValueError`` for NaN scores, non-finite pain, or unequal lengths.
    """
    s = _scores(scores)
    p = _finite(pain, "pain")
    n = _same_length(("scores", s), ("pain", p))
    if n < 2:
        return math.nan
    rs = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
    rp = pd.Series(p).rank(method="average").to_numpy(dtype=np.float64)
    rs = rs - rs.mean()
    rp = rp - rp.mean()
    denom = math.sqrt(float(rs @ rs) * float(rp @ rp))
    if denom == 0.0:
        return math.nan
    return float(np.clip(float(rs @ rp) / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------------------------------------
# probability metrics


def brier(prob: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Brier score: mean squared difference between predicted probability and outcome, in [0, 1].

    Lower is better. Returns NaN for empty input.
    Raises ``ValueError`` for probabilities outside [0, 1] or not finite, outcomes other than 0 and 1, or
    unequal lengths.
    """
    p = _probabilities(prob)
    t = _binary(y)
    n = _same_length(("prob", p), ("y", t))
    if n == 0:
        return math.nan
    return float(np.mean((p - t) ** 2))


def reliability_table(prob: npt.ArrayLike, y: npt.ArrayLike, bins: int = ECE_BINS) -> pd.DataFrame:
    """Calibration table with ``bins`` equal-width probability bins over [0, 1].

    Bins are left-closed: the bin of probability ``p`` is ``min(floor(p * bins), bins - 1)``, so 1.0 falls
    in the last bin. Columns: ``bin`` (int64), ``lower``, ``upper`` (bin edges), ``count`` (int64),
    ``mean_prob`` (mean predicted probability) and ``frac_positive`` (observed share of positives). Every
    bin is present; empty bins have count 0 and NaN means.

    Raises ``ValueError`` for ``bins < 1``, invalid probabilities or outcomes, or unequal lengths.
    """
    bins = _positive_int(bins, "bins")
    p = _probabilities(prob)
    t = _binary(y)
    _same_length(("prob", p), ("y", t))
    idx = np.minimum(np.floor(p * bins).astype(np.intp), bins - 1)
    count = np.bincount(idx, minlength=bins)
    sum_prob = np.bincount(idx, weights=p, minlength=bins)
    sum_pos = np.bincount(idx, weights=t, minlength=bins)
    nan = np.full(bins, np.nan)
    mean_prob = np.divide(sum_prob, count, out=nan.copy(), where=count > 0)
    frac_positive = np.divide(sum_pos, count, out=nan.copy(), where=count > 0)
    edges = np.arange(bins + 1, dtype=np.float64) / bins
    return pd.DataFrame(
        {
            "bin": np.arange(bins, dtype=np.int64),
            "lower": edges[:-1],
            "upper": edges[1:],
            "count": count.astype(np.int64),
            "mean_prob": mean_prob,
            "frac_positive": frac_positive,
        }
    )


def ece(prob: npt.ArrayLike, y: npt.ArrayLike, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width bins, in [0, 1].

    ``sum over non-empty bins of count / n * |frac_positive - mean_prob|`` with the bins of
    :func:`reliability_table`. Lower is better. Returns NaN for empty input.
    Raises ``ValueError`` for ``bins < 1``, invalid probabilities or outcomes, or unequal lengths.
    """
    table = reliability_table(prob, y, bins)
    counts = table["count"].to_numpy(dtype=np.float64)
    n = float(counts.sum())
    if n == 0.0:
        return math.nan
    filled = counts > 0
    gaps = np.abs(table["frac_positive"].to_numpy()[filled] - table["mean_prob"].to_numpy()[filled])
    return float((counts[filled] / n) @ gaps)


def auc_pr(prob: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Area under the precision-recall curve as average precision (no interpolation), in [0, 1].

    ``AP = sum_i (R_i - R_{i-1}) * P_i`` over the distinct probability thresholds from high to low, the
    same definition as ``sklearn.metrics.average_precision_score``. Tied probabilities form one threshold.

    Returns NaN for empty input or when there are no positives.
    Raises ``ValueError`` for invalid probabilities or outcomes, or unequal lengths.
    """
    p = _probabilities(prob)
    t = _binary(y)
    n = _same_length(("prob", p), ("y", t))
    n_pos = float(t.sum())
    if n == 0 or n_pos == 0.0:
        return math.nan
    order = np.argsort(-p, kind="stable")
    p_sorted = p[order]
    last_of_tie = np.concatenate((np.flatnonzero(np.diff(p_sorted)), [n - 1]))
    true_pos = np.cumsum(t[order])[last_of_tie]
    precision = true_pos / (last_of_tie + 1)
    recall = true_pos / n_pos
    return float(np.diff(np.concatenate(([0.0], recall))) @ precision)


# ---------------------------------------------------------------------------------------------------------
# per-query evaluation


def _empty_result(df: pd.DataFrame, prob: bool) -> pd.DataFrame:
    metrics = RANKING_METRICS + (PROBABILITY_METRICS if prob else ())
    columns: dict[str, pd.Series] = {
        "qid": pd.Series([], dtype="int64"),
        "repo": pd.Series([], dtype="str"),
        "snapshot_idx": pd.Series([], dtype="int64"),
        "snapshot_time": pd.to_datetime(df["snapshot_time"].iloc[:0], utc=True).reset_index(drop=True),
        "n_files": pd.Series([], dtype="int64"),
        "n_defective": pd.Series([], dtype="int64"),
        "n_defects": pd.Series([], dtype="int64"),
    }
    columns.update({m: pd.Series([], dtype="float64") for m in metrics})
    return pd.DataFrame(columns)


def _check_eval_input(df: pd.DataFrame) -> None:
    if not pd.api.types.is_integer_dtype(df["qid"].dtype):
        raise ValueError(f"evaluate: qid must have an integer dtype, got {df['qid'].dtype}")
    for col in ("qid", "repo", "snapshot_idx", "snapshot_time", "path"):
        if df[col].isna().any():
            raise ValueError(f"evaluate: column {col!r} contains nulls")
    if df.duplicated(["qid", "path"]).any():
        dup = df.loc[df.duplicated(["qid", "path"]), ["qid", "path"]].iloc[0]
        raise ValueError(f"evaluate: duplicate (qid, path) rows, first ({dup['qid']}, {dup['path']!r})")
    per_qid = df.groupby("qid", sort=True)[["repo", "snapshot_idx", "snapshot_time"]].nunique()
    mixed = per_qid[(per_qid > 1).any(axis=1)]
    if len(mixed):
        raise ValueError(
            f"evaluate: qid {mixed.index[0]} mixes several repositories or snapshots; one qid is one query"
        )


def evaluate(df: pd.DataFrame, score_col: str, prob_col: str | None = None) -> pd.DataFrame:
    """Compute every metric for every query (``qid``) of a scored dataset.

    ``df`` needs :data:`EVAL_INPUT_COLUMNS` plus ``score_col`` (higher means more expected pain); with
    ``prob_col`` (a probability in [0, 1]) it also needs ``has_latent_defect``. Rows are sorted by
    ``(qid, path)`` first, so ties in the score are broken by path ascending and the output does not depend
    on the row order of ``df``.

    Per query: ``ndcg_5``, ``ndcg_10``, ``ndcg_20`` on ``grade``; ``precision_10`` (grade >= 3);
    ``recall_20loc`` and ``popt`` with ``sloc`` as effort and ``latent_defects`` as defect counts;
    ``spearman`` against ``pain``; with ``prob_col`` also ``brier``, ``ece`` (10 bins) and ``auc_pr``
    against ``has_latent_defect``. Undefined metrics are NaN (see the module docstring).

    Returns one row per qid sorted by qid on a RangeIndex, with columns :data:`QUERY_COLUMNS`, then
    :data:`RANKING_METRICS`, then :data:`PROBABILITY_METRICS` when ``prob_col`` is given. ``qid``,
    ``snapshot_idx`` and the ``n_`` counts are int64, ``repo`` is str, ``snapshot_time`` is UTC and the
    metrics are float64. An empty ``df`` gives an empty frame with these columns.

    Raises ``ValueError`` for missing columns (listed), a non-integer or null ``qid``, nulls in the index
    columns, duplicate ``(qid, path)`` rows, a qid spanning several repositories or snapshots, or invalid
    metric inputs (the message then starts with ``qid <q>: ``).
    """
    required = [*EVAL_INPUT_COLUMNS, score_col]
    if prob_col is not None:
        required += [prob_col, "has_latent_defect"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"evaluate: missing columns {missing}")
    if len(df) == 0:
        return _empty_result(df, prob_col is not None)
    _check_eval_input(df)

    columns = list(dict.fromkeys(required))  # only what is needed: the dataset has about 100 columns
    data = df[columns].sort_values(["qid", "path"], kind="stable").reset_index(drop=True)
    qids = data["qid"].to_numpy(dtype=np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(qids)) + 1))
    ends = np.concatenate((starts[1:], [len(data)]))

    grade = data["grade"].to_numpy(dtype=np.float64)
    score = data[score_col].to_numpy(dtype=np.float64)
    pain = data["pain"].to_numpy(dtype=np.float64)
    defects = data["latent_defects"].to_numpy(dtype=np.float64)
    sloc = data["sloc"].to_numpy(dtype=np.float64)
    prob = data[prob_col].to_numpy(dtype=np.float64) if prob_col is not None else None
    outcome = data["has_latent_defect"].to_numpy(dtype=np.float64) if prob_col is not None else None

    metrics = RANKING_METRICS + (PROBABILITY_METRICS if prob_col is not None else ())
    values: dict[str, list[float]] = {m: [] for m in metrics}
    n_files: list[int] = []
    n_defective: list[int] = []
    n_defects: list[int] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        part = slice(start, end)
        g, s, d, effort = grade[part], score[part], defects[part], sloc[part]
        try:
            for k in NDCG_KS:
                values[f"ndcg_{k}"].append(ndcg_at_k(g, s, k))
            values["precision_10"].append(precision_at_k(g, s, PRECISION_K, MIN_GRADE))
            values["recall_20loc"].append(recall_at_loc(s, effort, d, LOC_FRACTION))
            values["popt"].append(popt(s, effort, d))
            values["spearman"].append(spearman(s, pain[part]))
            if prob is not None and outcome is not None:
                values["brier"].append(brier(prob[part], outcome[part]))
                values["ece"].append(ece(prob[part], outcome[part], ECE_BINS))
                values["auc_pr"].append(auc_pr(prob[part], outcome[part]))
        except ValueError as exc:
            raise ValueError(f"qid {int(qids[start])}: {exc}") from exc
        n_files.append(end - start)
        n_defective.append(int(np.count_nonzero(d > 0)))
        n_defects.append(round(float(d.sum())))

    first = data.iloc[starts].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "qid": first["qid"].astype("int64"),
            "repo": first["repo"].astype("str"),
            "snapshot_idx": first["snapshot_idx"].astype("int64"),
            "snapshot_time": pd.to_datetime(first["snapshot_time"], utc=True),
            "n_files": np.asarray(n_files, dtype=np.int64),
            "n_defective": np.asarray(n_defective, dtype=np.int64),
            "n_defects": np.asarray(n_defects, dtype=np.int64),
        }
    )
    for m in metrics:
        out[m] = np.asarray(values[m], dtype=np.float64)
    logger.debug("evaluated %d queries on %r", len(out), score_col)
    return out
