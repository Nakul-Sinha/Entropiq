"""Baseline rankers that the learned model must beat, run under the evaluation protocols.

Every baseline turns the test rows of a split into one score per file (higher means more expected pain)
and goes through :func:`tdp_ml.eval.run_protocol`, so its per-query metrics line up with the ranker's on
``qid``. The baselines:

- ``random``: a random shuffle; the per-query metrics are the mean over ``n_shuffles`` (100) seeded
  shuffles, so the row is the expected value of a random order rather than one lucky draw.
- ``loc``: ``sloc``. ``complexity``: ``cc_sum``.
- ``hotspot``: ``pct_churn_decayed * pct_cc_sum`` (the classic churn times complexity hotspot).
- ``prior_bugs``: ``prior_latent_bugs``, ties broken by ``prior_bugfix_commits``.
- ``rf``: ``RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
  class_weight="balanced_subsample")`` on ``has_latent_defect``; the score is the positive-class
  probability.
- ``xgb_point``: ``XGBRegressor(n_estimators=500, learning_rate=0.05)`` (other parameters default) on
  ``pain``, a pointwise regressor.

The two learned baselines use :data:`FEATURE_INPUT_COLUMNS` as inputs and fit on the training rows only;
the harness hides the label columns of the test rows. scikit-learn and XGBoost are imported lazily, so
the heuristic baselines do not pay their import time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import FEATURE_INPUT_COLUMNS
from tdp_ml.eval import (
    EVAL_INPUT_COLUMNS,
    METRICS,
    QUERY_COLUMNS,
    RUN_COLUMNS,
    FitPredict,
    run_protocol,
)

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

BASELINES: Final = ("random", "loc", "complexity", "hotspot", "prior_bugs", "rf", "xgb_point")
"""Baseline names in report order."""

N_SHUFFLES: Final = 100
"""Seeded shuffles averaged by the ``random`` baseline."""

DEFAULT_JOBS: Final = 4
"""Default thread count of the learned baselines (the laptop's resource limit)."""

RF_PARAMS: Final = {
    "n_estimators": 500,
    "min_samples_leaf": 5,
    "class_weight": "balanced_subsample",
}
XGB_POINT_PARAMS: Final = {"n_estimators": 500, "learning_rate": 0.05, "tree_method": "hist"}

_SPLIT_COLUMNS: Final = ("qid", "repo", "snapshot_idx", "snapshot_time", "path")


def _column(test: pd.DataFrame, name: str) -> FloatArray:
    if name not in test.columns:
        raise ValueError(f"baseline needs the column {name!r}")
    values = test[name].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"column {name!r} contains NaN")
    return values


def loc_scores(test: pd.DataFrame) -> FloatArray:
    """``loc`` baseline: the file's ``sloc`` (source lines). Raises ``ValueError`` if missing or NaN."""
    return _column(test, "sloc")


def complexity_scores(test: pd.DataFrame) -> FloatArray:
    """``complexity`` baseline: ``cc_sum`` (summed cyclomatic complexity). Raises like :func:`loc_scores`."""
    return _column(test, "cc_sum")


def hotspot_scores(test: pd.DataFrame) -> FloatArray:
    """``hotspot`` baseline: ``pct_churn_decayed * pct_cc_sum`` (both percentiles in [0, 1]).

    Raises ``ValueError`` when either column is missing or contains NaN.
    """
    return _column(test, "pct_churn_decayed") * _column(test, "pct_cc_sum")


def prior_bugs_scores(test: pd.DataFrame) -> FloatArray:
    """``prior_bugs`` baseline: ``prior_latent_bugs``, ties broken by ``prior_bugfix_commits``.

    Returns the dense rank (0, 1, 2, ...) of each row's ``(prior_latent_bugs, prior_bugfix_commits)``
    pair in lexicographic order, so a file with more prior latent bugs always scores higher and equal
    counts are separated by prior bug-fix commits. Rows equal on both keep equal scores (the evaluator
    then breaks ties by path). Raises ``ValueError`` when a column is missing or contains NaN.
    """
    primary = _column(test, "prior_latent_bugs")
    secondary = _column(test, "prior_bugfix_commits")
    scores = np.zeros(len(primary), dtype=np.float64)
    if len(primary) == 0:
        return scores
    order = np.lexsort((secondary, primary))
    p, s = primary[order], secondary[order]
    changed = np.concatenate(([False], (p[1:] != p[:-1]) | (s[1:] != s[:-1])))
    scores[order] = np.cumsum(changed, dtype=np.int64).astype(np.float64)
    return scores


def _features(frame: pd.DataFrame) -> npt.NDArray[np.float32]:
    missing = [c for c in FEATURE_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"learned baselines need every feature column; missing {missing[:5]}")
    return frame[list(FEATURE_INPUT_COLUMNS)].to_numpy(dtype=np.float32)


def _label(train: pd.DataFrame, name: str) -> FloatArray:
    if name not in train.columns:
        raise ValueError(f"training rows need the label column {name!r}")
    values = train[name].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError(f"label column {name!r} contains NaN")
    return values


def rf_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_jobs: int = DEFAULT_JOBS,
    seed: int = DEFAULT_SETTINGS.seed,
) -> FloatArray:
    """``rf`` baseline: random forest on ``has_latent_defect``; returns the positive-class probability.

    Fits ``RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
    class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)`` on the training rows'
    :data:`FEATURE_INPUT_COLUMNS` and scores the test rows (probabilities in [0, 1]). When every training
    row has the same label, the forest cannot rank, so every test row gets that label (0.0 or 1.0).
    ``n_jobs=-1`` uses every core. Raises ``ValueError`` for missing feature or label columns, an empty
    training set, or NaN labels.
    """
    from sklearn.ensemble import RandomForestClassifier

    if len(train) == 0:
        raise ValueError("rf needs at least one training row")
    y = _label(train, "has_latent_defect").astype(np.int64)
    x_test = _features(test)
    if np.unique(y).size < 2:
        logger.warning("rf: every training row has label %d; scores are constant", int(y[0]))
        return np.full(len(test), float(y[0]), dtype=np.float64)
    model = RandomForestClassifier(**RF_PARAMS, n_jobs=n_jobs, random_state=seed)
    model.fit(_features(train), y)
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    return np.asarray(model.predict_proba(x_test)[:, positive], dtype=np.float64)


def xgb_point_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_jobs: int = DEFAULT_JOBS,
    seed: int = DEFAULT_SETTINGS.seed,
) -> FloatArray:
    """``xgb_point`` baseline: pointwise XGBoost regression of ``pain``; returns the predicted pain.

    Fits ``XGBRegressor(n_estimators=500, learning_rate=0.05, tree_method="hist", n_jobs=n_jobs,
    random_state=seed)`` (squared error, other parameters default) on the training rows'
    :data:`FEATURE_INPUT_COLUMNS` and predicts the test rows. ``n_jobs=-1`` uses every core. Raises
    ``ValueError`` for missing feature or label columns, an empty training set, or NaN labels.
    """
    import xgboost as xgb

    if len(train) == 0:
        raise ValueError("xgb_point needs at least one training row")
    y = _label(train, "pain")
    model = xgb.XGBRegressor(**XGB_POINT_PARAMS, n_jobs=n_jobs, random_state=seed)
    model.fit(_features(train), y)
    return np.asarray(model.predict(_features(test)), dtype=np.float64)


def _heuristic(score: Callable[[pd.DataFrame], FloatArray]) -> FitPredict:
    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        return score(test)

    return fit_predict


def baseline_fit_predict(
    name: str, *, n_jobs: int = DEFAULT_JOBS, seed: int = DEFAULT_SETTINGS.seed
) -> FitPredict:
    """The ``fit_predict(train, test)`` callable of a deterministic baseline, for :func:`run_protocol`.

    ``name`` is one of :data:`BASELINES` except ``random`` (use :func:`run_random`, which averages many
    shuffles). ``n_jobs`` and ``seed`` apply to ``rf`` and ``xgb_point`` only.
    Raises ``ValueError`` for an unknown name or ``random``.
    """
    if name == "loc":
        return _heuristic(loc_scores)
    if name == "complexity":
        return _heuristic(complexity_scores)
    if name == "hotspot":
        return _heuristic(hotspot_scores)
    if name == "prior_bugs":
        return _heuristic(prior_bugs_scores)
    if name == "rf":
        return lambda train, test: rf_fit_predict(train, test, n_jobs=n_jobs, seed=seed)
    if name == "xgb_point":
        return lambda train, test: xgb_point_fit_predict(train, test, n_jobs=n_jobs, seed=seed)
    if name == "random":
        raise ValueError("the random baseline averages many shuffles; use run_random")
    raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")


def run_random(
    df: pd.DataFrame,
    protocol: str,
    *,
    n_shuffles: int = N_SHUFFLES,
    seed: int = DEFAULT_SETTINGS.seed,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
) -> pd.DataFrame:
    """Per-query metrics of the ``random`` baseline: the mean over ``n_shuffles`` seeded shuffles.

    Shuffle ``i`` scores the test rows of each split with a permutation drawn from
    ``numpy.random.default_rng([seed, i])``. Each metric of a query is the mean over the shuffles (NaN
    when the metric is undefined for that query, for example NDCG when every grade is 0). The output has
    the format of :attr:`tdp_ml.eval.ProtocolRun.per_query` with ``model`` set to ``"random"``.
    Raises ``ValueError`` for ``n_shuffles < 1`` and for the problems :func:`run_protocol` reports.
    """
    if isinstance(n_shuffles, bool) or not isinstance(n_shuffles, int) or n_shuffles < 1:
        raise ValueError(f"n_shuffles must be an int >= 1, got {n_shuffles!r}")
    narrow = df[[c for c in dict.fromkeys((*_SPLIT_COLUMNS, *EVAL_INPUT_COLUMNS)) if c in df.columns]]
    frames: list[pd.DataFrame] = []
    for i in range(n_shuffles):
        rng = np.random.default_rng([seed, i])

        def shuffle(train: pd.DataFrame, test: pd.DataFrame, rng: np.random.Generator = rng) -> FloatArray:
            return rng.permutation(len(test)).astype(np.float64)

        run = run_protocol(narrow, protocol, shuffle, horizon_days=horizon_days, model="random")
        frames.append(run.per_query)
    first = frames[0]
    keys = [*RUN_COLUMNS, *QUERY_COLUMNS]
    for frame in frames[1:]:
        if not frame[keys].equals(first[keys]):
            raise RuntimeError("random shuffles produced different query sets")
    metrics = [m for m in METRICS if m in first.columns]
    stacked = pd.concat([f[metrics] for f in frames], keys=range(len(frames)))
    mean = stacked.groupby(level=1).mean()
    result = first.copy()
    result[metrics] = mean[metrics].to_numpy(dtype=np.float64)
    return result


def run_baseline(
    df: pd.DataFrame,
    name: str,
    protocol: str,
    *,
    n_jobs: int = DEFAULT_JOBS,
    seed: int = DEFAULT_SETTINGS.seed,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    n_shuffles: int = N_SHUFFLES,
) -> pd.DataFrame:
    """Per-query metrics of one baseline under one protocol (``"temporal"`` or ``"lopo"``).

    Returns the :attr:`tdp_ml.eval.ProtocolRun.per_query` frame with ``model`` set to ``name``.
    ``n_jobs`` and ``seed`` go to the learned baselines, ``n_shuffles`` and ``seed`` to ``random``,
    ``horizon_days`` is the temporal embargo. Raises ``ValueError`` for an unknown baseline or protocol
    and for missing columns.
    """
    if name == "random":
        return run_random(df, protocol, n_shuffles=n_shuffles, seed=seed, horizon_days=horizon_days)
    fit_predict = baseline_fit_predict(name, n_jobs=n_jobs, seed=seed)
    return run_protocol(df, protocol, fit_predict, horizon_days=horizon_days, model=name).per_query


def check_baselines(names: Sequence[str]) -> list[str]:
    """Validate and de-duplicate baseline names, keeping the :data:`BASELINES` report order.

    Raises ``ValueError`` for an empty list or an unknown name.
    """
    unknown = sorted(set(names) - set(BASELINES))
    if unknown:
        raise ValueError(f"unknown baselines {unknown}; expected some of {BASELINES}")
    chosen: list[str] = [b for b in BASELINES if b in set(names)]
    if not chosen:
        raise ValueError("no baseline selected")
    return chosen


def order_per_query(per_query: pd.DataFrame) -> pd.DataFrame:
    """Sort a per-query frame by protocol, baseline report order (unknown names last), then qid."""
    rank = {name: i for i, name in enumerate(BASELINES)}
    key = per_query["model"].map(lambda m: rank.get(str(m), len(BASELINES)))
    ordered = per_query.assign(_order=key).sort_values(["protocol", "_order", "model", "qid"], kind="stable")
    return ordered.drop(columns="_order").reset_index(drop=True)


def run_baselines(
    df: pd.DataFrame,
    protocol: str,
    names: Sequence[str] = BASELINES,
    *,
    n_jobs: int = DEFAULT_JOBS,
    seed: int = DEFAULT_SETTINGS.seed,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    n_shuffles: int = N_SHUFFLES,
) -> pd.DataFrame:
    """Run several baselines under one protocol and concatenate their per-query frames.

    ``names`` defaults to every baseline; the output is ordered by :func:`order_per_query`, and each
    ``(model, protocol)`` group has one row per test query. The other arguments are those of
    :func:`run_baseline`. Raises ``ValueError`` like :func:`check_baselines` and :func:`run_baseline`.
    """
    frames: list[pd.DataFrame] = []
    for name in check_baselines(names):
        start = time.perf_counter()
        frames.append(
            run_baseline(
                df,
                name,
                protocol,
                n_jobs=n_jobs,
                seed=seed,
                horizon_days=horizon_days,
                n_shuffles=n_shuffles,
            )
        )
        logger.info("baseline %s (%s) took %.1f s", name, protocol, time.perf_counter() - start)
    return order_per_query(pd.concat(frames, ignore_index=True))
