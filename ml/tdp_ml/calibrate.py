"""Calibration of ranker scores into risk probabilities with ranges, evaluated on the temporal test set.

Inputs are score files in the layout of :data:`tdp_ml.train.SCORE_COLUMNS` (``<model>_oof.parquet``: the
LOPO cross-fit scores, every row scored by a model that never saw its repository). The procedure:

1. **Evaluation fit.** An isotonic curve is fitted on the embargoed rows of :func:`temporal_split` (rows
   whose label window closes before their repository's last snapshot) and evaluated on each repository's
   last snapshot: pooled Brier, ECE (10 equal-width bins) and AUC-PR with 95% intervals that resample
   repositories, the same metrics per query, and a reliability table with Wilson intervals.
2. **Shipped fit.** The curve that ships (``calibrator.json``) is refitted on every row, which matches the
   score scale of the production booster (trained on every row with the same parameters).

Ranges (``y_low``, ``y_high``) are the 5th and 95th percentiles of bootstrap refits evaluated on the main
curve's grid, as in :class:`tdp_core.model.calibration.IsotonicCalibrator`. By default the bootstrap
resamples whole repositories (:func:`fit_calibrator`), because rows of one repository are correlated (the
same file repeats across snapshots and base rates differ by repository); ``bootstrap="row"`` reproduces
:meth:`IsotonicCalibrator.fit` exactly.

matplotlib and MLflow are imported inside the functions that use them, and xgboost is never imported.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.model.calibration import HIGH_PERCENTILE, LOW_PERCENTILE, IsotonicCalibrator
from tdp_ml.eval import (
    ECE_BINS,
    EVAL_INPUT_COLUMNS,
    PROBABILITY_METRICS,
    auc_pr,
    brier,
    ece,
    evaluate,
    reliability_table,
    summarize,
    temporal_split,
)
from tdp_ml.pipeline.layout import read_json, write_frame, write_json

logger = logging.getLogger(__name__)

type FloatArray = npt.NDArray[np.float64]
type IntpArray = npt.NDArray[np.intp]

DEFAULT_MODEL: Final = "ranker"
MODEL_NAME: Final = re.compile(r"^[a-z][a-z0-9_]*$")
BOOTSTRAP_UNITS: Final = ("repo", "row")
DEFAULT_BOOTSTRAP: Final = "repo"
DEFAULT_N_BOOT: Final = 200
DEFAULT_CI_BOOT: Final = 1000
ECE_TARGET: Final = 0.05
Z_95: Final = 1.959963984540054
RECORD_FORMAT: Final = 1
FIGURE_DPI: Final = 200
PRIMARY_SET: Final = "oof"
TEMPORAL_SET: Final = "temporal_model"
SET_LABELS: Final[Mapping[str, str]] = {
    PRIMARY_SET: "LOPO out-of-fold scores",
    TEMPORAL_SET: "final temporal model scores",
}
SCORE_INPUT_COLUMNS: Final = (*EVAL_INPUT_COLUMNS, "has_latent_defect", "score")
RELIABILITY_COLUMNS: Final = (
    "set",
    "bin",
    "lower",
    "upper",
    "count",
    "mean_prob",
    "frac_positive",
    "ci_low",
    "ci_high",
)
SET_METRICS: Final = (
    "mean_risk",
    "mean_band_width",
    "brier",
    "brier_ci_low",
    "brier_ci_high",
    "brier_base_rate",
    "brier_skill",
    "ece",
    "ece_ci_low",
    "ece_ci_high",
    "auc_pr",
    "auc_pr_ci_low",
    "auc_pr_ci_high",
    "query_brier",
    "query_brier_ci_low",
    "query_brier_ci_high",
    "query_ece",
    "query_ece_ci_low",
    "query_ece_ci_high",
    "query_auc_pr",
    "query_auc_pr_ci_low",
    "query_auc_pr_ci_high",
)
FINAL_STATS: Final = ("thresholds", "risk_min", "risk_max", "mean_band_width", "zero_width_rows")
"""Keys of :attr:`CalibrationOutcome.final_stats` (the shipped curve applied to every row)."""

_NOT_NULL: Final = ("qid", "repo", "snapshot_idx", "snapshot_time", "path", "has_latent_defect", "score")
_ZERO_WIDTH: Final = 1e-12


# ---------------------------------------------------------------------------------------------------------
# inputs


def check_scores(frame: pd.DataFrame, *, name: str = "scores") -> None:
    """Validate a score frame (the layout of ``<model>_oof.parquet``); returns nothing.

    Raises ``ValueError`` with the prefix ``"<name>: "`` when a column of :data:`SCORE_INPUT_COLUMNS` is
    missing (all missing ones are listed), the frame is empty, ``qid`` is not an integer dtype, one of
    ``qid``, ``repo``, ``snapshot_idx``, ``snapshot_time``, ``path``, ``has_latent_defect`` or ``score``
    holds nulls, a score is not finite, ``has_latent_defect`` holds values other than 0 and 1 (booleans
    are fine), or a ``(qid, path)`` pair repeats.
    """
    missing = [c for c in SCORE_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if len(frame) == 0:
        raise ValueError(f"{name}: no rows")
    if not pd.api.types.is_integer_dtype(frame["qid"].dtype):
        raise ValueError(f"{name}: qid must have an integer dtype, got {frame['qid'].dtype}")
    for col in _NOT_NULL:
        if frame[col].isna().any():
            raise ValueError(f"{name}: column {col!r} contains nulls")
    score = frame["score"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(score)):
        raise ValueError(f"{name}: scores must be finite (no NaN or infinity)")
    label = frame["has_latent_defect"].to_numpy(dtype=np.float64)
    if not np.all((label == 0.0) | (label == 1.0)):
        raise ValueError(f"{name}: has_latent_defect must hold only 0 and 1")
    if frame.duplicated(["qid", "path"]).any():
        dup = frame.loc[frame.duplicated(["qid", "path"]), ["qid", "path"]].iloc[0]
        raise ValueError(f"{name}: duplicate (qid, path) rows, first ({dup['qid']}, {dup['path']!r})")


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["qid", "path"], kind="stable").reset_index(drop=True)


def load_scores(path: Path) -> pd.DataFrame:
    """Read a score parquet file, validate it with :func:`check_scores` and sort it by ``(qid, path)``.

    Returns the frame on a RangeIndex. Raises ``FileNotFoundError`` when ``path`` does not exist and
    ``ValueError`` (prefixed with the path) when the content is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"score file not found: {path}")
    frame = pd.read_parquet(path)
    check_scores(frame, name=str(path))
    return _sorted(frame)


# ---------------------------------------------------------------------------------------------------------
# fitting


def _check_count(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


def _check_bootstrap(bootstrap: str) -> str:
    if bootstrap not in BOOTSTRAP_UNITS:
        raise ValueError(f"bootstrap must be one of {list(BOOTSTRAP_UNITS)}, got {bootstrap!r}")
    return bootstrap


def _group_members(groups: npt.ArrayLike, n: int) -> list[IntpArray]:
    """Row positions of every group, groups sorted by their string value."""
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {values.shape}")
    if values.size != n:
        raise ValueError(f"groups has {values.size} values but scores has {n}")
    if bool(pd.isna(values).any()):
        raise ValueError("groups must not contain nulls")
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    return [np.flatnonzero(codes == k) for k in range(len(uniques))]


def _resample(members: list[IntpArray], rng: np.random.Generator) -> IntpArray:
    """One cluster bootstrap draw: ``K`` groups with replacement, their rows concatenated."""
    k = len(members)
    pick = rng.integers(0, k, size=k)
    return np.concatenate([members[int(j)] for j in pick])


def fit_calibrator(
    scores: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    groups: npt.ArrayLike | None = None,
    bootstrap: str = DEFAULT_BOOTSTRAP,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SETTINGS.seed,
) -> IsotonicCalibrator:
    """Fit an isotonic calibrator of outcomes ``y`` (0 or 1) on raw ranker ``scores`` with a range.

    The curve is always :meth:`IsotonicCalibrator.fit` without bootstrap. ``bootstrap`` picks the unit the
    range resamples:

    - ``"row"``: exactly ``IsotonicCalibrator.fit(scores, y, n_boot=n_boot, seed=seed)`` (rows drawn with
      replacement); ``groups`` is ignored.
    - ``"repo"`` (default): every one of ``n_boot`` refits draws ``K`` groups (``K`` = number of distinct
      ``groups`` values, usually repositories) with replacement and uses all their rows. Each refit is
      evaluated on the main curve's ``x`` grid (clipped at its ends); ``y_low`` and ``y_high`` are the 5th
      and 95th percentiles, made non-decreasing and clamped to bracket ``y`` the same way as P3.0. With one
      group, or ``n_boot=0``, the range equals the curve.

    Deterministic for a given ``seed``. Runtime is about ``n_boot`` isotonic fits of ``len(scores)`` rows
    (1 to 3 s for 200 refits on 27,000 rows). Raises ``ValueError`` for an unknown ``bootstrap``,
    ``n_boot`` that is not an int >= 0, the input errors of :meth:`IsotonicCalibrator.fit`, and, with
    ``"repo"``, missing, multi-dimensional, mismatched or null ``groups``.
    """
    _check_bootstrap(bootstrap)
    n_boot = _check_count(n_boot, "n_boot", 0)
    if bootstrap == "row":
        return IsotonicCalibrator.fit(scores, y, n_boot=n_boot, seed=seed)
    if groups is None:
        raise ValueError("bootstrap='repo' needs groups (one repository name per row)")
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    main = IsotonicCalibrator.fit(s, t, n_boot=0)
    members = _group_members(groups, s.size)
    if n_boot == 0:
        return main
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, main.x.size), dtype=np.float64)
    for b in range(n_boot):
        idx = _resample(members, rng)
        fb = IsotonicCalibrator.fit(s[idx], t[idx], n_boot=0)
        boots[b] = np.interp(main.x, fb.x, fb.y)
    lo, hi = np.percentile(boots, [LOW_PERCENTILE, HIGH_PERCENTILE], axis=0)
    lo = np.clip(np.minimum(np.maximum.accumulate(lo), main.y), 0.0, 1.0)
    hi = np.clip(np.maximum(np.maximum.accumulate(hi), main.y), 0.0, 1.0)
    logger.debug("repository bootstrap: %d groups, %d refits, %d thresholds", len(members), n_boot, lo.size)
    return IsotonicCalibrator(main.x, main.y, lo, hi)


# ---------------------------------------------------------------------------------------------------------
# metrics


def wilson_interval(
    successes: npt.ArrayLike, trials: npt.ArrayLike, *, z: float = Z_95
) -> tuple[FloatArray, FloatArray]:
    """Wilson score interval of a binomial proportion, element-wise, clipped to [0, 1].

    ``successes`` and ``trials`` broadcast against each other; ``z`` is the normal quantile (1.96 for a
    95% interval). Entries with ``trials == 0`` give NaN. Example: 0 of 10 gives (0, 0.2775) and 5 of 10
    gives (0.2366, 0.7634). Raises ``ValueError`` for negative or non-finite trials, successes outside
    ``[0, trials]``, or ``z`` that is not a finite number > 0.
    """
    if not isinstance(z, int | float | np.floating) or not math.isfinite(z) or z <= 0:
        raise ValueError(f"z must be a finite number > 0, got {z!r}")
    k, n = np.broadcast_arrays(np.asarray(successes, dtype=np.float64), np.asarray(trials, dtype=np.float64))
    if not (np.all(np.isfinite(n)) and np.all(n >= 0)):
        raise ValueError("trials must be finite and >= 0")
    if not (np.all(np.isfinite(k)) and np.all(k >= 0) and np.all(k <= n)):
        raise ValueError("successes must lie in [0, trials]")
    z2 = float(z) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        half = float(z) * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    low = np.where(n > 0, np.clip(center - half, 0.0, 1.0), np.nan)
    high = np.where(n > 0, np.clip(center + half, 0.0, 1.0), np.nan)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def reliability(prob: npt.ArrayLike, y: npt.ArrayLike, *, bins: int = ECE_BINS) -> pd.DataFrame:
    """The reliability table of :func:`tdp_ml.eval.reliability_table` with 95% Wilson intervals.

    Columns: ``bin``, ``lower``, ``upper``, ``count``, ``mean_prob``, ``frac_positive``, ``ci_low`` and
    ``ci_high`` (the interval of ``frac_positive``, from ``round(frac_positive * count)`` positives out of
    ``count``). Empty bins have NaN means and intervals. Raises ``ValueError`` as
    :func:`reliability_table` does.
    """
    table = reliability_table(prob, y, bins)
    count = table["count"].to_numpy(dtype=np.float64)
    frac = table["frac_positive"].to_numpy(dtype=np.float64)
    positives = np.where(count > 0, np.rint(np.nan_to_num(frac) * count), 0.0)
    low, high = wilson_interval(positives, count)
    table["ci_low"] = low
    table["ci_high"] = high
    return table


def _nan_interval(values: FloatArray) -> tuple[float, float]:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def pooled_metrics(
    prob: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    groups: npt.ArrayLike,
    base_rate: float,
    bins: int = ECE_BINS,
    n_boot: int = DEFAULT_CI_BOOT,
    seed: int = DEFAULT_SETTINGS.seed,
) -> dict[str, float]:
    """Pooled Brier, ECE and AUC-PR of probabilities ``prob`` against outcomes ``y`` with 95% intervals.

    Point values use every row: ``brier``, ``ece`` (``bins`` equal-width bins) and ``auc_pr`` from
    :mod:`tdp_ml.eval`. ``brier_base_rate`` is the Brier score of always predicting ``base_rate`` (a
    probability, usually the positive rate of the fitting rows) and ``brier_skill = 1 - brier /
    brier_base_rate`` (NaN when that is 0; higher is better, 0 means no better than the base rate).

    ``<m>_ci_low`` and ``<m>_ci_high`` for ``m`` in ``brier``, ``ece``, ``auc_pr`` are the 2.5th and 97.5th
    percentiles over ``n_boot`` draws that resample ``groups`` (repositories) with replacement and pool
    their rows (the draw of :func:`fit_calibrator`); draws where a metric is undefined are ignored. ECE is
    non-negative and biased upward, so its interval can lie above the point value.

    Every value is a Python float (NaN when undefined, for example AUC-PR without positives). Raises
    ``ValueError`` for ``n_boot < 1``, a ``base_rate`` outside [0, 1], invalid ``groups`` (see
    :func:`fit_calibrator`) and the input errors of the metrics.
    """
    n_boot = _check_count(n_boot, "n_boot", 1)
    if not isinstance(base_rate, int | float | np.floating) or not 0.0 <= float(base_rate) <= 1.0:
        raise ValueError(f"base_rate must be a probability in [0, 1], got {base_rate!r}")
    p = np.asarray(prob, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    out: dict[str, float] = {
        "brier": brier(p, t),
        "ece": ece(p, t, bins),
        "auc_pr": auc_pr(p, t),
    }
    reference = brier(np.full(t.shape, float(base_rate)), t)
    out["brier_base_rate"] = reference
    out["brier_skill"] = 1.0 - out["brier"] / reference if reference > 0 else math.nan
    members = _group_members(groups, p.size)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(PROBABILITY_METRICS)), dtype=np.float64)
    for b in range(n_boot):
        idx = _resample(members, rng)
        pb, tb = p[idx], t[idx]
        boots[b] = (brier(pb, tb), ece(pb, tb, bins), auc_pr(pb, tb))
    for j, metric in enumerate(PROBABILITY_METRICS):
        out[f"{metric}_ci_low"], out[f"{metric}_ci_high"] = _nan_interval(boots[:, j])
    return {k: float(v) for k, v in out.items()}


@dataclass(frozen=True, slots=True)
class RowSet:
    """Size of a set of score rows: ``rows`` (files), ``queries`` (distinct qid), ``repos`` and
    ``positive_rate`` (share of rows with a latent defect, 0..1)."""

    rows: int
    queries: int
    repos: int
    positive_rate: float

    @classmethod
    def of(cls, frame: pd.DataFrame) -> RowSet:
        """Describe ``frame`` (needs ``qid``, ``repo`` and ``has_latent_defect``); NaN rate when empty."""
        label = frame["has_latent_defect"].to_numpy(dtype=np.float64)
        return cls(
            rows=len(frame),
            queries=int(frame["qid"].nunique()),
            repos=int(frame["repo"].nunique()),
            positive_rate=float(label.mean()) if label.size else math.nan,
        )

    def to_json(self) -> dict[str, int | float]:
        """``{"rows", "queries", "repos", "positive_rate"}`` as Python numbers."""
        return {
            "rows": self.rows,
            "queries": self.queries,
            "repos": self.repos,
            "positive_rate": self.positive_rate,
        }


@dataclass(frozen=True, eq=False, slots=True)
class EvaluationSet:
    """Calibration quality of one set of scores.

    ``metrics`` has exactly the keys of :data:`SET_METRICS` (Python floats, NaN when undefined);
    ``reliability`` has the columns :data:`RELIABILITY_COLUMNS` with ``set`` equal to ``name``.
    """

    name: str
    rows: RowSet
    metrics: dict[str, float]
    reliability: pd.DataFrame


def evaluate_calibrator(
    calibrator: IsotonicCalibrator,
    frame: pd.DataFrame,
    *,
    name: str,
    base_rate: float,
    bins: int = ECE_BINS,
    n_boot: int = DEFAULT_CI_BOOT,
    seed: int = DEFAULT_SETTINGS.seed,
) -> EvaluationSet:
    """Apply ``calibrator`` to ``frame["score"]`` and measure the risk against ``has_latent_defect``.

    ``frame`` is a valid score frame (see :func:`check_scores`). Metrics: ``mean_risk`` and
    ``mean_band_width`` (mean of ``high - low``); the pooled metrics of :func:`pooled_metrics` with
    repositories as groups; ``query_brier``, ``query_ece`` and ``query_auc_pr`` (the mean over queries of
    the per-query metric, from :func:`tdp_ml.eval.evaluate`) with their ``_ci_low`` / ``_ci_high``
    repository bootstrap intervals from :func:`tdp_ml.eval.summarize`. ``reliability`` comes from
    :func:`reliability`. Raises ``ValueError`` for invalid input.
    """
    risk, low, high = calibrator.predict(frame["score"].to_numpy(dtype=np.float64))
    label = frame["has_latent_defect"].to_numpy(dtype=np.float64)
    metrics: dict[str, float] = {
        "mean_risk": float(risk.mean()),
        "mean_band_width": float((high - low).mean()),
    }
    metrics |= pooled_metrics(
        risk, label, groups=frame["repo"], base_rate=base_rate, bins=bins, n_boot=n_boot, seed=seed
    )
    per_query = evaluate(frame.assign(risk=risk), "score", prob_col="risk")
    summary = summarize(per_query, n_boot=n_boot, seed=seed, metrics=PROBABILITY_METRICS)
    columns = {c: summary[c].to_numpy(dtype=np.float64) for c in ("mean", "ci_low", "ci_high")}
    for i, metric in enumerate(summary["metric"].astype("str").tolist()):
        metrics[f"query_{metric}"] = float(columns["mean"][i])
        metrics[f"query_{metric}_ci_low"] = float(columns["ci_low"][i])
        metrics[f"query_{metric}_ci_high"] = float(columns["ci_high"][i])
    table = reliability(risk, label, bins=bins)
    table.insert(0, "set", name)
    table["set"] = table["set"].astype("str")
    return EvaluationSet(
        name=name,
        rows=RowSet.of(frame),
        metrics={k: metrics[k] for k in SET_METRICS},
        reliability=table[list(RELIABILITY_COLUMNS)].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------------------------------------
# procedure


@dataclass(frozen=True, eq=False, slots=True)
class CalibrationOutcome:
    """Result of :func:`calibrate`.

    ``calibrator`` is the curve that ships (fitted on every row); ``eval_calibrator`` was fitted on the
    embargoed rows only and produced every number in ``evaluations`` (the primary set
    :data:`PRIMARY_SET` first, then :data:`TEMPORAL_SET` when temporal scores were given). ``fit_rows``
    describes the embargoed fitting rows, ``final_rows`` all rows; ``final_stats`` has the keys
    :data:`FINAL_STATS` (``calibrator`` applied to every row: number of thresholds, lowest and highest
    risk, mean band width and the count of rows whose band has zero width). ``seconds`` is the wall time.
    """

    model: str
    calibrator: IsotonicCalibrator
    eval_calibrator: IsotonicCalibrator
    fit_rows: RowSet
    final_rows: RowSet
    evaluations: tuple[EvaluationSet, ...]
    final_stats: dict[str, float]
    settings: dict[str, Any]
    seconds: float

    @property
    def primary(self) -> EvaluationSet:
        """The evaluation of the primary set (out-of-fold scores of the temporal test snapshots)."""
        return self.evaluations[0]

    @property
    def meets_target(self) -> bool:
        """True when the primary pooled ECE is at most :data:`ECE_TARGET` (0.05)."""
        value = self.primary.metrics["ece"]
        return math.isfinite(value) and value <= ECE_TARGET


def _check_model_column(frame: pd.DataFrame, model: str, name: str) -> None:
    if "model" not in frame.columns:
        return
    values = sorted({str(v) for v in frame["model"].dropna().unique()})
    if values != [model]:
        raise ValueError(f"{name}: the model column holds {values}, expected only {model!r}")


def calibrate(
    scores: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    temporal_scores: pd.DataFrame | None = None,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    bootstrap: str = DEFAULT_BOOTSTRAP,
    n_boot: int = DEFAULT_N_BOOT,
    ci_boot: int = DEFAULT_CI_BOOT,
    bins: int = ECE_BINS,
    seed: int = DEFAULT_SETTINGS.seed,
) -> CalibrationOutcome:
    """Fit, evaluate and refit the risk calibration of one model's out-of-fold ``scores``.

    1. Both frames are validated (:func:`check_scores`); a ``model`` column must hold only ``model``.
    2. :func:`temporal_split` with ``horizon_days`` gives the test rows (each repository's last snapshot)
       and the embargoed fitting rows (label window closed before that snapshot). The evaluation curve is
       :func:`fit_calibrator` on the fitting rows (``bootstrap``, ``n_boot``, ``seed``).
    3. It is evaluated on the test rows (set :data:`PRIMARY_SET`) and, when given, on every row of
       ``temporal_scores`` (set :data:`TEMPORAL_SET`, usually the final temporal model's test scores),
       with ``bins`` ECE bins, ``ci_boot`` repository bootstrap draws and the fitting rows' positive rate
       as the Brier reference.
    4. The shipped curve is :func:`fit_calibrator` on every row of ``scores``.

    Raises ``ValueError`` for invalid frames or options, or when no row is old enough to fit on (every
    repository needs a snapshot at least ``horizon_days`` before its last one).
    """
    started = time.perf_counter()
    if not MODEL_NAME.match(model):
        raise ValueError(f"model name must match {MODEL_NAME.pattern}, got {model!r}")
    _check_bootstrap(bootstrap)
    n_boot = _check_count(n_boot, "n_boot", 0)
    ci_boot = _check_count(ci_boot, "ci_boot", 1)
    bins = _check_count(bins, "bins", 1)
    check_scores(scores, name="scores")
    _check_model_column(scores, model, "scores")
    if temporal_scores is not None:
        check_scores(temporal_scores, name="temporal_scores")
        _check_model_column(temporal_scores, model, "temporal_scores")
    frame = _sorted(scores)
    train_idx, test_idx = temporal_split(frame, horizon_days)
    if len(train_idx) == 0:
        raise ValueError(
            f"no rows to fit on: every repository needs a snapshot at least {horizon_days} days before "
            "its last snapshot"
        )
    train = frame.loc[train_idx]
    test = frame.loc[test_idx].reset_index(drop=True)

    def fit(rows: pd.DataFrame) -> IsotonicCalibrator:
        return fit_calibrator(
            rows["score"].to_numpy(dtype=np.float64),
            rows["has_latent_defect"].to_numpy(dtype=np.float64),
            groups=rows["repo"].astype("str").to_numpy(),
            bootstrap=bootstrap,
            n_boot=n_boot,
            seed=seed,
        )

    eval_cal = fit(train)
    base_rate = float(train["has_latent_defect"].to_numpy(dtype=np.float64).mean())
    common: dict[str, Any] = {"base_rate": base_rate, "bins": bins, "n_boot": ci_boot, "seed": seed}
    evaluations = [evaluate_calibrator(eval_cal, test, name=PRIMARY_SET, **common)]
    if temporal_scores is not None:
        evaluations.append(
            evaluate_calibrator(eval_cal, _sorted(temporal_scores), name=TEMPORAL_SET, **common)
        )
    shipped = fit(frame)
    risk, low, high = shipped.predict(frame["score"].to_numpy(dtype=np.float64))
    width = high - low
    final_stats = {
        "thresholds": float(shipped.x.size),
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
        "mean_band_width": float(width.mean()),
        "zero_width_rows": float(np.count_nonzero(width <= _ZERO_WIDTH)),
    }
    settings: dict[str, Any] = {
        "horizon_days": int(horizon_days),
        "bootstrap": bootstrap,
        "n_boot": n_boot,
        "ci_boot": ci_boot,
        "bins": bins,
        "seed": int(seed),
        "low_percentile": float(LOW_PERCENTILE),
        "high_percentile": float(HIGH_PERCENTILE),
        "target_ece": ECE_TARGET,
    }
    seconds = time.perf_counter() - started
    logger.info(
        "calibrated %s: fit %d rows, test %d rows, ECE %.4f, %.1f s",
        model,
        len(train),
        len(test),
        evaluations[0].metrics["ece"],
        seconds,
    )
    return CalibrationOutcome(
        model=model,
        calibrator=shipped,
        eval_calibrator=eval_cal,
        fit_rows=RowSet.of(train),
        final_rows=RowSet.of(frame),
        evaluations=tuple(evaluations),
        final_stats=final_stats,
        settings=settings,
        seconds=seconds,
    )


# ---------------------------------------------------------------------------------------------------------
# outputs


def _check_model_name(model: str) -> str:
    if not isinstance(model, str) or not MODEL_NAME.match(model):
        raise ValueError(f"model name must match {MODEL_NAME.pattern}, got {model!r}")
    return model


def output_paths(results_dir: Path, model: str) -> dict[str, Path]:
    """Files written by :func:`write_outputs` for ``model`` under ``results_dir``.

    Keys: ``calibrator`` (``<model>_calibrator.json``, the bundle's ``calibrator.json`` content),
    ``record`` (``<model>_calibration.json``), ``reliability`` (``<model>_reliability.parquet``),
    ``summary`` (``<model>_calibration_summary.md``) and ``figure`` (``figures/reliability.png`` for the
    default model ``ranker``, else ``figures/<model>_reliability.png``). Raises ``ValueError`` when
    ``model`` does not match :data:`MODEL_NAME`.
    """
    _check_model_name(model)
    root = Path(results_dir)
    figure = "reliability.png" if model == DEFAULT_MODEL else f"{model}_reliability.png"
    return {
        "calibrator": root / f"{model}_calibrator.json",
        "record": root / f"{model}_calibration.json",
        "reliability": root / f"{model}_reliability.parquet",
        "summary": root / f"{model}_calibration_summary.md",
        "figure": root / "figures" / figure,
    }


def _jsonable(value: Any) -> Any:
    """Strict JSON values: NaN and infinity become None, numpy and Path values plain Python."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _nested(mapping: Mapping[str, Any], *keys: str) -> str:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or value.get(key) is None:
            return ""
        value = value[key]
    return str(value)


def calibration_record(outcome: CalibrationOutcome, *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON record of a calibration (``<model>_calibration.json``), strict JSON (no NaN).

    Keys: ``format`` (:data:`RECORD_FORMAT`), ``created_at`` (``provenance["created_at"]`` or now, UTC),
    ``model``, ``provenance`` (as given: score paths and sha256 values, run id, code version),
    ``settings``, ``target_ece``, ``meets_target``, ``fit`` (the embargoed fitting rows), ``evaluation``
    (``{set name: {rows, queries, repos, positive_rate, **metrics}}``), ``calibrator`` (the shipped curve
    on every row: ``rows``, ``queries``, ``repos``, ``positive_rate`` and :data:`FINAL_STATS`) and
    ``seconds``. Undefined values are null.
    """
    final = outcome.final_rows.to_json() | {
        k: (int(v) if k in ("thresholds", "zero_width_rows") else v) for k, v in outcome.final_stats.items()
    }
    record: dict[str, Any] = {
        "format": RECORD_FORMAT,
        "created_at": str(provenance.get("created_at") or _now()),
        "model": outcome.model,
        "provenance": dict(provenance),
        "settings": dict(outcome.settings),
        "target_ece": ECE_TARGET,
        "meets_target": outcome.meets_target,
        "fit": outcome.fit_rows.to_json(),
        "evaluation": {e.name: e.rows.to_json() | e.metrics for e in outcome.evaluations},
        "calibrator": final,
        "seconds": outcome.seconds,
    }
    out: dict[str, Any] = _jsonable(record)
    return out


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def _ci(metrics: Mapping[str, float], key: str) -> str:
    point = _num(metrics[key])
    low, high = metrics[f"{key}_ci_low"], metrics[f"{key}_ci_high"]
    if not (math.isfinite(low) and math.isfinite(high)):
        return point
    return f"{point} [{low:.3f}, {high:.3f}]"


def summary_markdown(outcome: CalibrationOutcome) -> str:
    """A Markdown report of the calibration: what risk means, the procedure, metrics and limitations."""
    s = outcome.settings
    fit, final, stats = outcome.fit_rows, outcome.final_rows, outcome.final_stats
    primary = outcome.primary
    lines = [
        f"# Risk calibration: {outcome.model}",
        "",
        "Risk is the calibrated probability that a file has at least one latent defect within "
        f"{s['horizon_days']} days of the snapshot (a bug fixed in that window whose lines already existed "
        "in the file). It is an isotonic regression of the label on the raw ranker score. The range is the "
        f"{s['low_percentile']:.0f}th to {s['high_percentile']:.0f}th percentile of {s['n_boot']} bootstrap "
        f"refits that resample {'repositories' if s['bootstrap'] == 'repo' else 'rows'}.",
        "",
        f"Evaluation: the curve was fitted on {fit.rows:,} out-of-fold rows ({fit.queries} queries, "
        f"{fit.repos} repositories) whose {s['horizon_days']}-day label windows close by their "
        "repository's last snapshot, and tested on that last snapshot of every repository "
        f"({primary.rows.rows:,} files, {primary.rows.queries} queries). The curve was never fitted on a "
        "test row or on a row whose label window overlaps the test window.",
        "",
        "| set | files | positive rate | mean risk | Brier [95% CI] | Brier skill | ECE [95% CI] | "
        "AUC-PR [95% CI] | per-query ECE [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in outcome.evaluations:
        m = e.metrics
        lines.append(
            f"| {SET_LABELS.get(e.name, e.name)} | {e.rows.rows:,} | {_num(e.rows.positive_rate)} | "
            f"{_num(m['mean_risk'])} | {_ci(m, 'brier')} | {_num(m['brier_skill'])} | {_ci(m, 'ece')} | "
            f"{_ci(m, 'auc_pr')} | {_ci(m, 'query_ece')} |"
        )
    verdict = "met" if outcome.meets_target else "missed"
    lines += [
        "",
        f"ECE target {ECE_TARGET:.2f} on the temporal test set: {verdict} "
        f"(pooled ECE {_num(primary.metrics['ece'], 4)}, {s['bins']} equal-width bins).",
        "",
        f"Shipped calibrator: refitted on all {final.rows:,} rows ({final.queries} queries, {final.repos} "
        f"repositories, positive rate {_num(final.positive_rate)}) with {int(stats['thresholds'])} "
        f"thresholds; risk on those rows runs from {_num(stats['risk_min'])} to {_num(stats['risk_max'])}, "
        f"mean band width {_num(stats['mean_band_width'])}, {int(stats['zero_width_rows'])} rows with a "
        "zero-width band.",
        "",
        "Notes:",
        "",
        f"- Intervals resample repositories ({s['ci_boot']} draws), because files of one repository are "
        "correlated. ECE is non-negative and biased upward in small samples, so its interval can sit above "
        "the point estimate.",
        "- Brier skill compares with always predicting the positive rate of the fitting rows "
        f"({_num(fit.positive_rate)}); 0 means no better than that constant.",
        "- Per-query ECE is much higher than pooled ECE: risk is one pooled curve for every repository, and "
        "ranker margins carry no repository-level base rate.",
        "- The range describes the uncertainty of the pooled curve, not of one file's outcome. At the "
        "extremes the isotonic curve is flat at 0 or 1 and the range can collapse.",
        "",
    ]
    return "\n".join(lines)


def _atomic_temp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")


def _write_text(path: Path, text: str) -> Path:
    tmp = _atomic_temp(path)
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def reliability_figure(outcome: CalibrationOutcome, path: Path, *, dpi: int = FIGURE_DPI) -> Path:
    """Draw the reliability diagram of ``outcome`` as a PNG at ``path`` (``dpi`` dots per inch).

    Top panel: observed share of files with a latent defect against mean predicted risk per bin, with
    95% Wilson intervals for the primary set and a dashed line for the temporal model set; the diagonal
    is perfect calibration. Bottom panel: files per bin of the primary set (log scale). Uses
    ``matplotlib.figure.Figure`` directly (no pyplot, the global backend is untouched); the file is
    written atomically and parent folders are created. Returns ``path``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    handles: list[Any] = []
    labels: list[str] = []
    for e in outcome.evaluations:
        table = e.reliability[e.reliability["count"] > 0]
        x = table["mean_prob"].to_numpy(dtype=np.float64)
        y = table["frac_positive"].to_numpy(dtype=np.float64)
        label = (
            f"{SET_LABELS.get(e.name, e.name)} (ECE {_num(e.metrics['ece'])}, "
            f"Brier {_num(e.metrics['brier'])})"
        )
        if e is outcome.primary:
            err = np.vstack(
                [
                    np.clip(y - table["ci_low"].to_numpy(dtype=np.float64), 0.0, None),
                    np.clip(table["ci_high"].to_numpy(dtype=np.float64) - y, 0.0, None),
                ]
            )
            handles.append(top.errorbar(x, y, yerr=err, marker="o", linestyle="-", capsize=3, color="C0"))
        else:
            handles += top.plot(x, y, marker="o", linestyle="--", markerfacecolor="none", color="C1")
        labels.append(label)
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_ylabel("Observed share with a latent defect")
    top.set_title(f"Reliability on the temporal test set: {outcome.model}")
    handles += top.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1, zorder=1)
    labels.append("perfect calibration")
    top.legend(handles, labels, loc="upper left", fontsize="small")
    table = outcome.primary.reliability
    widths = (table["upper"] - table["lower"]).to_numpy(dtype=np.float64)
    counts = table["count"].to_numpy(dtype=np.float64)
    bottom.bar(
        table["lower"].to_numpy(dtype=np.float64), counts, width=widths * 0.9, align="edge", color="C0"
    )
    if counts.max() > 0:
        bottom.set_yscale("log")
    bottom.set_xlabel("Mean predicted risk")
    bottom.set_ylabel("Files per bin")
    path = Path(path)
    tmp = _atomic_temp(path)
    try:
        fig.savefig(tmp, dpi=dpi, format="png")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_outputs(
    outcome: CalibrationOutcome, results_dir: Path, *, provenance: Mapping[str, Any]
) -> dict[str, Path]:
    """Write every output of :func:`output_paths` atomically and return that mapping.

    ``calibrator`` holds :meth:`IsotonicCalibrator.to_json` of the shipped curve (the bundle's
    ``calibrator.json``), ``record`` :func:`calibration_record`, ``reliability`` the reliability tables of
    every set (:data:`RELIABILITY_COLUMNS`), ``summary`` :func:`summary_markdown` and ``figure``
    :func:`reliability_figure`. Folders are created as needed.
    """
    paths = output_paths(results_dir, outcome.model)
    write_json(paths["calibrator"], outcome.calibrator.to_json())
    write_json(paths["record"], calibration_record(outcome, provenance=provenance))
    tables = pd.concat([e.reliability for e in outcome.evaluations], ignore_index=True)
    write_frame(paths["reliability"], tables[list(RELIABILITY_COLUMNS)])
    _write_text(paths["summary"], summary_markdown(outcome))
    reliability_figure(outcome, paths["figure"])
    return paths


# ---------------------------------------------------------------------------------------------------------
# MLflow


def run_payload(
    outcome: CalibrationOutcome, *, provenance: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, str]]:
    """Metrics and tags that :func:`log_to_run` adds to the model's training run.

    Metrics: ``calibration_<k>`` for every :data:`SET_METRICS` key of the primary set (so the ECE gate is
    ``calibration_ece``), ``calibration_temporal_model_<k>`` for the temporal model set when present,
    plus ``calibration_fit_rows``, ``calibration_test_rows``, ``calibration_thresholds``,
    ``calibration_final_mean_band_width``, ``calibration_final_zero_width_rows`` and
    ``calibration_seconds``; values that are not finite are dropped. Tags (strings):
    ``calibration_created_at``, ``calibration_bootstrap``, ``calibration_n_boot``, ``calibration_seed``,
    ``calibration_horizon_days``, ``calibration_scores_sha256``, ``calibration_code_git_sha`` and
    ``calibration_meets_target`` (``true`` or ``false``).
    """
    metrics: dict[str, float] = {}
    for e in outcome.evaluations:
        prefix = "calibration_" if e is outcome.primary else f"calibration_{e.name}_"
        metrics |= {f"{prefix}{k}": v for k, v in e.metrics.items()}
    metrics |= {
        "calibration_fit_rows": float(outcome.fit_rows.rows),
        "calibration_test_rows": float(outcome.primary.rows.rows),
        "calibration_thresholds": outcome.final_stats["thresholds"],
        "calibration_final_mean_band_width": outcome.final_stats["mean_band_width"],
        "calibration_final_zero_width_rows": outcome.final_stats["zero_width_rows"],
        "calibration_seconds": outcome.seconds,
    }
    s = outcome.settings
    tags = {
        "calibration_created_at": str(provenance.get("created_at") or _now()),
        "calibration_bootstrap": str(s["bootstrap"]),
        "calibration_n_boot": str(s["n_boot"]),
        "calibration_seed": str(s["seed"]),
        "calibration_horizon_days": str(s["horizon_days"]),
        "calibration_scores_sha256": _nested(provenance, "scores", "sha256"),
        "calibration_code_git_sha": str(provenance.get("code_git_sha") or ""),
        "calibration_meets_target": "true" if outcome.meets_target else "false",
    }
    finite = {k: float(v) for k, v in metrics.items() if math.isfinite(float(v))}
    return finite, tags


def default_run_id(results_dir: Path, model: str) -> str:
    """The MLflow run id recorded in ``<results_dir>/<model>_lopo_run.json`` (written by ``tdp-ml train``).

    The record must be a JSON object with ``protocol`` ``"lopo"``, ``model`` equal to ``model`` and a
    non-empty ``run_id``. Every failure (missing or unreadable file, wrong content, bad model name) raises
    ``ValueError`` saying to pass ``--run-id`` or ``--no-tracking``.
    """
    hint = "pass --run-id or --no-tracking"
    try:
        _check_model_name(model)
    except ValueError as exc:
        raise ValueError(f"{exc}; {hint}") from exc
    path = Path(results_dir) / f"{model}_lopo_run.json"
    try:
        data = read_json(path)
    except RuntimeError as exc:  # PipelineError
        raise ValueError(f"{exc}; {hint}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object; {hint}")
    if data.get("protocol") != "lopo" or data.get("model") != model:
        raise ValueError(
            f"{path} describes protocol {data.get('protocol')!r} and model {data.get('model')!r}, "
            f"expected 'lopo' and {model!r}; {hint}"
        )
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"{path} has no run_id; {hint}")
    return run_id.strip()


def find_run(run_id: str, *, uri: str | None = None) -> dict[str, str]:
    """Look up an MLflow run; returns ``run_id``, ``experiment_id``, ``run_name``, ``status`` and
    ``tracking_uri``.

    The store is ``uri`` or ``MLFLOW_TRACKING_URI`` (through :func:`tdp_ml.tracking.configure_mlflow`).
    Raises ``ValueError("MLflow run <id> not found in <uri>")`` when the store does not know the run.
    """
    from tdp_ml.tracking import configure_mlflow

    used = configure_mlflow(uri)
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    try:
        run = MlflowClient(tracking_uri=used).get_run(run_id)
    except MlflowException as exc:
        raise ValueError(f"MLflow run {run_id} not found in {used}") from exc
    return {
        "run_id": str(run.info.run_id),
        "experiment_id": str(run.info.experiment_id),
        "run_name": str(run.info.run_name or ""),
        "status": str(run.info.status),
        "tracking_uri": used,
    }


def _artifact_key(key: str) -> PurePosixPath:
    if not isinstance(key, str) or not key or "\\" in key or key.endswith("/"):
        raise ValueError(f"artifact key must be a relative POSIX path, got {key!r}")
    pure = PurePosixPath(key)
    if pure.is_absolute() or ".." in pure.parts or pure.name in ("", ".") or ":" in pure.parts[0]:
        raise ValueError(f"artifact key must be a relative POSIX path without '..', got {key!r}")
    return pure


def log_to_run(
    run_id: str,
    *,
    artifacts: Mapping[str, Path],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    uri: str | None = None,
) -> None:
    """Add artifacts, metrics and tags to an existing MLflow run without reopening it.

    ``artifacts`` maps a relative POSIX artifact path (for example ``model/calibrator.json``) to a local
    file; the file is uploaded under that exact name. Metrics (non-finite values skipped) and tags go in
    one ``log_batch`` call at step 0. The run's status, end time and parameters are never changed, and
    a second call replaces the files and tags and adds newer metric values (the latest one wins), so
    reruns are safe. Raises ``ValueError`` for an unknown run (see :func:`find_run`) or an artifact key
    that is absolute, contains ``..`` or has no file name, and ``FileNotFoundError`` for a missing source.
    """
    info = find_run(run_id, uri=uri)
    keys = {key: _artifact_key(key) for key in artifacts}
    for key, source in artifacts.items():
        if not Path(source).is_file():
            raise FileNotFoundError(f"artifact source not found: {source} (for {key})")
    from mlflow.entities import Metric, RunTag
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=info["tracking_uri"])
    with tempfile.TemporaryDirectory(prefix="tdp-calibration-") as tmp:
        for n, (key, source) in enumerate(artifacts.items()):
            pure = keys[key]
            folder = Path(tmp) / str(n)
            folder.mkdir()
            staged = folder / pure.name
            shutil.copyfile(source, staged)
            parent = pure.parent.as_posix()
            client.log_artifact(run_id, str(staged), artifact_path=None if parent == "." else parent)
    now_ms = int(time.time() * 1000)
    batch_metrics = [
        Metric(str(k), float(v), now_ms, 0) for k, v in sorted(metrics.items()) if math.isfinite(float(v))
    ]
    batch_tags = [RunTag(str(k), str(v)) for k, v in sorted(tags.items())]
    if batch_metrics or batch_tags:
        client.log_batch(run_id, metrics=batch_metrics, tags=batch_tags)
    logger.info("logged %d artifacts and %d metrics to run %s", len(artifacts), len(batch_metrics), run_id)
