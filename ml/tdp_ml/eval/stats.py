"""Summaries with bootstrap intervals and paired significance tests over per-query metrics.

Inputs are per-query frames in the format of :func:`tdp_ml.eval.metrics.evaluate` (optionally with the
run columns ``model``, ``protocol`` and ``split``). Queries of one repository are correlated, so the
bootstrap resamples whole repositories, and paired tests pair two models on the same ``qid``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_ml.eval.metrics import GROUP_COLUMNS, METRICS, FloatArray

logger = logging.getLogger(__name__)

Alternative = Literal["two-sided", "greater", "less"]

CLIFFS_THRESHOLDS: Final = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))
SUMMARY_COLUMNS: Final = ("metric", "mean", "ci_low", "ci_high", "n_queries", "n_repos")
TEST_COLUMNS: Final = (
    "metric",
    "baseline",
    "n",
    "mean_a",
    "mean_b",
    "mean_diff",
    "median_diff",
    "n_a_higher",
    "n_equal",
    "n_a_lower",
    "statistic",
    "p_value",
    "p_holm",
    "significant",
    "cliffs_delta",
    "magnitude",
)
_ALTERNATIVES: Final = ("two-sided", "greater", "less")


def _group_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in GROUP_COLUMNS if c in frame.columns]


def _groups(frame: pd.DataFrame, cols: list[str]) -> list[tuple[tuple[Hashable, ...], pd.DataFrame]]:
    if not cols:
        return [((), frame)]
    out: list[tuple[tuple[Hashable, ...], pd.DataFrame]] = []
    for key, part in frame.groupby(cols, sort=True, dropna=False):
        out.append((key if isinstance(key, tuple) else (key,), part))
    return out


def _bootstrap_group(
    part: pd.DataFrame, metrics: list[str], n_boot: int, seed: int
) -> list[dict[str, object]]:
    codes, repos = pd.factorize(part["repo"], sort=True)
    n_repos = len(repos)
    values = part[metrics].to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    onehot = np.zeros((len(part), n_repos))
    onehot[np.arange(len(part)), codes] = 1.0
    sums = onehot.T @ np.where(present, values, 0.0)  # (repos, metrics)
    counts = onehot.T @ present.astype(np.float64)  # (repos, metrics)
    weights = np.random.default_rng(seed).multinomial(n_repos, np.full(n_repos, 1.0 / n_repos), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (weights @ sums) / (weights @ counts)  # (n_boot, metrics)
    rows: list[dict[str, object]] = []
    for j, metric in enumerate(metrics):
        total = float(counts[:, j].sum())
        mean = float(sums[:, j].sum()) / total if total > 0 else math.nan
        replicates = boot[:, j][~np.isnan(boot[:, j])]
        if replicates.size:
            low, high = (float(v) for v in np.percentile(replicates, [2.5, 97.5]))
        else:
            low = high = math.nan
        rows.append(
            {
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_queries": int(total),
                "n_repos": int(np.count_nonzero(counts[:, j] > 0)),
            }
        )
    return rows


def summarize(
    per_query: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
    *,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean of every metric with a 95% bootstrap interval that resamples repositories.

    The point estimate is the mean over queries, ignoring NaN. The interval is the 2.5th and 97.5th
    percentile of ``n_boot`` replicates; each replicate draws repositories with replacement (multinomial
    weights, the same draws for every group and metric given ``seed``) and takes the query-level mean of the
    drawn repositories. Replicates with no values are ignored.

    Results are grouped by the run columns ``model`` and ``protocol`` when present. ``metrics`` defaults to
    the columns of :data:`METRICS` present in ``per_query``, in that order.

    Returns the group columns, then ``metric``, ``mean``, ``ci_low``, ``ci_high``, ``n_queries`` (non-NaN
    values) and ``n_repos`` (repositories with at least one value), one row per group and metric. A metric
    with no values has NaN mean and interval and zero counts. An empty input gives an empty frame.

    Raises ``ValueError`` without a ``repo`` column or with null repositories, for ``n_boot < 1``, when no
    metric column is present, or when a requested metric is missing.
    """
    if "repo" not in per_query.columns:
        raise ValueError("summarize needs a 'repo' column")
    if per_query["repo"].isna().any():
        raise ValueError("summarize: column 'repo' contains nulls")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int | np.integer) or n_boot < 1:
        raise ValueError(f"n_boot must be an int >= 1, got {n_boot!r}")
    if metrics is None:
        chosen = [m for m in METRICS if m in per_query.columns]
        if not chosen:
            raise ValueError("summarize found no metric columns")
    else:
        chosen = list(metrics)
        missing = [m for m in chosen if m not in per_query.columns]
        if missing or not chosen:
            raise ValueError(f"summarize: missing metric columns {missing}")
    group_cols = _group_columns(per_query)
    columns = group_cols + list(SUMMARY_COLUMNS)
    if len(per_query) == 0:
        empty = {c: pd.Series([], dtype=per_query[c].dtype) for c in group_cols}
        empty |= {
            "metric": pd.Series([], dtype="str"),
            "mean": pd.Series([], dtype="float64"),
            "ci_low": pd.Series([], dtype="float64"),
            "ci_high": pd.Series([], dtype="float64"),
            "n_queries": pd.Series([], dtype="int64"),
            "n_repos": pd.Series([], dtype="int64"),
        }
        return pd.DataFrame(empty, columns=columns)
    rows: list[dict[str, object]] = []
    for key, part in _groups(per_query, group_cols):
        for row in _bootstrap_group(part, chosen, int(n_boot), seed):
            rows.append(dict(zip(group_cols, key, strict=True)) | row)
    out = pd.DataFrame(rows, columns=columns)
    for col in ("n_queries", "n_repos"):
        out[col] = out[col].astype("int64")
    for col in ("mean", "ci_low", "ci_high"):
        out[col] = out[col].astype("float64")
    out["metric"] = out["metric"].astype("str")
    return out


def holm(p_values: npt.ArrayLike) -> FloatArray:
    """Holm step-down adjustment of p-values, in the input order.

    With the ``m`` non-NaN p-values sorted ascending, ``adj_i = min(1, max_{j <= i} (m - j) * p_j)`` for
    0-based ``i``. NaN stays NaN and does not count towards ``m``.

    Raises ``ValueError`` for a non 1-D input or values outside [0, 1].
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be 1-D, got shape {p.shape}")
    known = np.flatnonzero(~np.isnan(p))
    values = p[known]
    if ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must lie in [0, 1]")
    out = np.full(p.shape, np.nan)
    order = np.argsort(values, kind="stable")
    m = len(values)
    scaled = (m - np.arange(m)) * values[order]
    out[known[order]] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return out


def cliffs_magnitude(delta: float) -> str:
    """Magnitude label of a Cliff's delta: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.

    Returns ``""`` for NaN.
    """
    if math.isnan(delta):
        return ""
    size = abs(delta)
    for threshold, label in CLIFFS_THRESHOLDS:
        if size < threshold:
            return label
    return "large"


def cliffs_delta(a: npt.ArrayLike, b: npt.ArrayLike) -> tuple[float, str]:
    """Cliff's delta ``P(a > b) - P(a < b)`` over all pairs, in [-1, 1], with its magnitude label.

    Positive when values of ``a`` tend to be larger. NaN values are dropped from each sample first.
    Raises ``ValueError`` when either sample is empty after dropping NaN or is not 1-D.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("cliffs_delta needs 1-D samples")
    x = x[~np.isnan(x)]
    y = np.sort(y[~np.isnan(y)])
    if x.size == 0 or y.size == 0:
        raise ValueError("cliffs_delta needs two non-empty samples")
    below = np.searchsorted(y, x, side="left")  # pairs with b < a
    above = y.size - np.searchsorted(y, x, side="right")  # pairs with b > a
    delta = float(below.sum() - above.sum()) / (x.size * y.size)
    return delta, cliffs_magnitude(delta)


def _paired_values(frame: pd.DataFrame, name: str, metric: str) -> pd.DataFrame:
    for col in ("qid", metric):
        if col not in frame.columns:
            raise ValueError(f"wilcoxon_holm: {name} has no column {col!r}")
    if frame["qid"].duplicated().any():
        raise ValueError(f"wilcoxon_holm: {name} has duplicate qids; pass one run per frame")
    return frame[["qid", metric]]


def _test_row(
    x: FloatArray, y: FloatArray, metric: str, baseline: str, alternative: Alternative
) -> dict[str, object]:
    n = len(x)
    diff = x - y
    row: dict[str, object] = {
        "metric": metric,
        "baseline": baseline,
        "n": n,
        "mean_a": float(x.mean()) if n else math.nan,
        "mean_b": float(y.mean()) if n else math.nan,
        "mean_diff": float(diff.mean()) if n else math.nan,
        "median_diff": float(np.median(diff)) if n else math.nan,
        "n_a_higher": int(np.count_nonzero(diff > 0)),
        "n_equal": int(np.count_nonzero(diff == 0)),
        "n_a_lower": int(np.count_nonzero(diff < 0)),
    }
    if n == 0:
        row |= {"statistic": math.nan, "p_value": math.nan, "cliffs_delta": math.nan, "magnitude": ""}
        return row
    if not np.any(diff != 0):
        row |= {"statistic": 0.0, "p_value": 1.0}
    else:
        from scipy import stats  # imported lazily: scipy.stats takes over a second to import

        result = stats.wilcoxon(x, y, zero_method="wilcox", alternative=alternative)
        row |= {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
    delta, magnitude = cliffs_delta(x, y)
    row |= {"cliffs_delta": delta, "magnitude": magnitude}
    return row


def wilcoxon_holm(
    a: pd.DataFrame,
    others: Mapping[str, pd.DataFrame],
    metric: str,
    *,
    alternative: Alternative = "two-sided",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests of model ``a`` against each baseline, with Holm correction.

    Each frame holds one run in the per-query format (``qid`` plus ``metric``). For every baseline, in the
    order of ``others``, the two frames are paired on ``qid`` (inner join, pairs with a NaN dropped) and
    ``scipy.stats.wilcoxon(x, y, zero_method="wilcox", alternative=alternative)`` is run on the model values
    ``x`` and baseline values ``y``; ``alternative="greater"`` tests whether the model is better on a metric
    where higher is better. When every difference is zero the test is skipped with statistic 0 and p 1.
    Without pairs the statistic, p-value and Cliff's delta are NaN. ``p_holm`` applies :func:`holm` across
    the baselines and ``significant`` is ``p_holm < alpha`` (False when NaN). ``cliffs_delta`` and
    ``magnitude`` compare the paired values.

    Returns the columns :data:`TEST_COLUMNS`, one row per baseline (empty when ``others`` is empty).
    Raises ``ValueError`` when a frame lacks ``qid`` or ``metric``, has duplicate qids, for an unknown
    ``alternative``, or for ``alpha`` outside (0, 1).
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    model = _paired_values(a, "the model frame", metric)
    rows: list[dict[str, object]] = []
    for name, other in others.items():
        baseline = _paired_values(other, f"baseline {name!r}", metric)
        pairs = (
            model.merge(baseline, on="qid", how="inner", suffixes=("_a", "_b"))
            .dropna(subset=[f"{metric}_a", f"{metric}_b"])
            .sort_values("qid", kind="stable")
        )
        x = pairs[f"{metric}_a"].to_numpy(dtype=np.float64)
        y = pairs[f"{metric}_b"].to_numpy(dtype=np.float64)
        rows.append(_test_row(x, y, metric, str(name), alternative))
    out = pd.DataFrame(rows, columns=list(TEST_COLUMNS))
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"].to_numpy(dtype=np.float64))
    out["significant"] = (out["p_holm"] < alpha).astype(bool)  # NaN compares as False
    for col in ("n", "n_a_higher", "n_equal", "n_a_lower"):
        out[col] = out[col].astype("int64")
    for col in ("metric", "baseline", "magnitude"):
        out[col] = out[col].astype("str")
    return out
