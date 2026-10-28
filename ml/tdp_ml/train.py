"""LambdaMART ranker training and tuning on the dataset of ``tdp-ml dataset``.

The ranker is an ``xgboost.XGBRanker`` with ``objective="rank:ndcg"`` over the 89 model input columns
(``FEATURE_INPUT_COLUMNS``), grouped by ``qid`` and trained on ``grade`` (0 to 4). The procedure:

1. :func:`tune` runs Optuna TPE (seed from the config) over the search space of ``ml/configs/ranker.yaml``.
   A trial fits one model per forward fold inside the temporal training set
   (:func:`tdp_ml.eval.forward_folds`), with early stopping on the fold's validation queries, and scores
   the mean NDCG@10 over the folds. Early stopping uses the same validation fold that is scored, which is
   standard and slightly optimistic. The final number of rounds is ``rounds_factor`` times the mean
   ``best_iteration + 1`` of the best trial's folds, rounded half up.
2. :func:`run_temporal` fits the final temporal model on the embargoed temporal training rows and scores
   each repository's last snapshot. :func:`run_lopo` fits one model per held-out repository with the same
   parameters (tuned once on temporal folds), scores every row of the held-out repository (the
   out-of-fold scores for calibration) and then fits the production candidate on every row.
3. :func:`compare` sets the per-query results against the baselines with bootstrap intervals and paired
   Wilcoxon tests; :func:`train` ties it together and writes the result files.

Queries whose grades are all 0 carry no ranking signal and are dropped from every training and tuning
validation set (:func:`training_rows`); test queries are never filtered. Scores are raw margins
(unitless, higher means more expected pain) computed with :class:`tdp_core.model.ranker.Ranker`, so
stored scores equal the scores of a bundle built from the same booster.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import FEATURE_COLUMNS, FEATURE_INPUT_COLUMNS
from tdp_core.mining.git import run_git
from tdp_core.model.ranker import Ranker
from tdp_ml.eval import (
    PROTOCOLS,
    RANKING_METRICS,
    TEST_COLUMNS,
    comparison_markdown,
    evaluate,
    forward_folds,
    load_per_query,
    per_query_path,
    run_protocol,
    save_per_query,
    summarize,
    summary_markdown,
    temporal_split,
    wilcoxon_holm,
)
from tdp_ml.pipeline.config import DEFAULT_CONFIG_DIR, ConfigError
from tdp_ml.pipeline.dataset import validate_dataset
from tdp_ml.pipeline.layout import PipelineError, read_json, write_frame, write_json

logger = logging.getLogger(__name__)

DEFAULT_RANKER_CONFIG: Final[Path] = DEFAULT_CONFIG_DIR / "ranker.yaml"
DEVICES: Final = ("auto", "cpu", "cuda")
DEFAULT_NTHREAD: Final = 4
TUNING_ONLY_PARAMS: Final = ("early_stopping_rounds", "eval_metric")
RUNTIME_PARAMS: Final = ("device", "n_jobs", "nthread", "monotone_constraints")
RESERVED_PARAMS: Final = (*RUNTIME_PARAMS, "random_state", "seed")
PARAMS_FORMAT: Final = 1
COMPARE_METRICS: Final = ("ndcg_10", "precision_10", "recall_20loc", "popt")
SCORE_COLUMNS: Final = (
    "model",
    "protocol",
    "split",
    "qid",
    "repo",
    "snapshot_idx",
    "snapshot_time",
    "path",
    "sloc",
    "grade",
    "pain",
    "latent_defects",
    "has_latent_defect",
    "is_test",
    "score",
)

_F: Final = list(FEATURE_INPUT_COLUMNS)
_CONFIG_KEYS: Final = (
    "experiment",
    "model_name",
    "seed",
    "n_trials",
    "n_folds",
    "rounds_factor",
    "device",
    "nthread",
    "fixed",
    "search",
    "monotone_features",
)
_SEARCH_KEYS: Final = frozenset({"type", "low", "high", "log"})
_MODEL_NAME: Final = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------- configuration ----------


@dataclass(frozen=True, slots=True)
class SearchParam:
    """One hyperparameter of the search space: an int or float range, optionally on a log scale."""

    name: str
    kind: Literal["int", "float"]
    low: float
    high: float
    log: bool = False

    def suggest(self, trial: Any) -> int | float:
        """Draw a value from an Optuna ``trial`` (``suggest_int`` or ``suggest_float``, bounds inclusive)."""
        if self.kind == "int":
            return int(trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log))
        return float(trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log))

    def to_json(self) -> dict[str, Any]:
        """``{"name", "type", "low", "high", "log"}``; int bounds are written as ints."""
        low: int | float = int(self.low) if self.kind == "int" else float(self.low)
        high: int | float = int(self.high) if self.kind == "int" else float(self.high)
        return {"name": self.name, "type": self.kind, "low": low, "high": high, "log": self.log}


@dataclass(frozen=True, slots=True)
class RankerConfig:
    """Training and tuning settings read from ``ml/configs/ranker.yaml`` (see :func:`load_ranker_config`).

    ``fixed`` holds the XGBRanker parameters every fit shares (``early_stopping_rounds`` and
    ``eval_metric`` apply only while tuning); ``search`` the tuned ranges in file order; ``seed`` feeds
    the TPE sampler and ``random_state``; ``rounds_factor`` scales the mean early-stopping rounds into the
    final ``n_estimators``; ``device`` is ``auto``, ``cpu`` or ``cuda``; ``nthread`` the XGBoost threads;
    ``monotone_features`` the base features constrained to +1 by ``--monotone``.
    """

    fixed: Mapping[str, Any]
    search: tuple[SearchParam, ...]
    experiment: str = "entropiq-ranker"
    model_name: str = "ranker"
    seed: int = 42
    n_trials: int = 50
    n_folds: int = 2
    rounds_factor: float = 1.1
    device: str = "auto"
    nthread: int = DEFAULT_NTHREAD
    monotone_features: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready copy in the layout of the YAML file, so :func:`load_ranker_config` reads it back."""
        return {
            "experiment": self.experiment,
            "model_name": self.model_name,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "n_folds": self.n_folds,
            "rounds_factor": self.rounds_factor,
            "device": self.device,
            "nthread": self.nthread,
            "fixed": _plain(self.fixed),
            "search": {p.name: {k: v for k, v in p.to_json().items() if k != "name"} for p in self.search},
            "monotone_features": list(self.monotone_features),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | bool) or _is_number(value)


def _check_fixed(path: Path, raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"{path}: fixed must be a non-empty mapping")
    fixed: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfigError(f"{path}: fixed key {key!r} must be a string")
        if key in RESERVED_PARAMS:
            raise ConfigError(f"{path}: fixed.{key} is set by the code and must not appear in the config")
        if isinstance(value, list):
            if not all(_is_scalar(v) for v in value):
                raise ConfigError(f"{path}: fixed.{key} must hold scalars only")
            fixed[key] = list(value)
        elif _is_scalar(value):
            fixed[key] = value
        else:
            raise ConfigError(f"{path}: fixed.{key} must be a scalar or a list of scalars")
    if not isinstance(fixed.get("objective"), str):
        raise ConfigError(f"{path}: fixed.objective is required (a string)")
    for key in ("n_estimators", "early_stopping_rounds"):
        value = fixed.get(key)
        if not _is_int(value) or cast(int, value) < 1:
            raise ConfigError(f"{path}: fixed.{key} is required (an int >= 1)")
    return fixed


def _check_search(path: Path, raw: object, fixed: Mapping[str, Any]) -> tuple[SearchParam, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigError(f"{path}: search must be a non-empty mapping")
    params: list[SearchParam] = []
    for name, spec in raw.items():
        where = f"search.{name}"
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{path}: search key {name!r} must be a non-empty string")
        if name in fixed or name in RESERVED_PARAMS:
            raise ConfigError(f"{path}: {where} is fixed or set by the code and cannot be searched")
        if not isinstance(spec, Mapping):
            raise ConfigError(f"{path}: {where} must be a mapping with type, low and high")
        unknown = sorted(str(k) for k in set(spec) - _SEARCH_KEYS)
        if unknown:
            raise ConfigError(f"{path}: unknown key(s) in {where}: {', '.join(unknown)}")
        kind = spec.get("type")
        if kind not in ("int", "float"):
            raise ConfigError(f"{path}: {where}.type must be int or float")
        low, high, log = spec.get("low"), spec.get("high"), spec.get("log", False)
        if not _is_number(low) or not _is_number(high):
            raise ConfigError(f"{path}: {where}.low and .high must be finite numbers")
        if not isinstance(log, bool):
            raise ConfigError(f"{path}: {where}.log must be true or false")
        lo, hi = float(cast(float, low)), float(cast(float, high))
        if lo >= hi:
            raise ConfigError(f"{path}: {where} needs low < high, got {lo} and {hi}")
        if log and lo <= 0:
            raise ConfigError(f"{path}: {where} is on a log scale and needs low > 0")
        if kind == "int" and (not lo.is_integer() or not hi.is_integer()):
            raise ConfigError(f"{path}: {where} is an int range and needs integral bounds")
        params.append(SearchParam(name, cast(Literal["int", "float"], kind), lo, hi, log))
    return tuple(params)


def _check_monotone(path: Path, raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: monotone_features must be a list of feature names")
    seen: list[str] = []
    for name in raw:
        if not isinstance(name, str) or name not in FEATURE_COLUMNS:
            raise ConfigError(f"{path}: monotone_features holds an unknown feature {name!r}")
        if name in seen:
            raise ConfigError(f"{path}: monotone_features lists {name!r} twice")
        seen.append(name)
    return tuple(seen)


def load_ranker_config(path: Path = DEFAULT_RANKER_CONFIG) -> RankerConfig:
    """Read and validate a ranker YAML config (default ``ml/configs/ranker.yaml``).

    Top-level keys: ``experiment`` (non-empty str), ``model_name`` (``^[a-z][a-z0-9_]*$``), ``seed`` (int),
    ``n_trials`` and ``n_folds`` (ints >= 1), ``rounds_factor`` (> 0), ``device`` (auto, cpu or cuda),
    ``nthread`` (int >= 1), ``fixed`` and ``search`` (required) and ``monotone_features`` (base feature
    names). Missing optional keys take the :class:`RankerConfig` defaults.

    Raises :class:`ConfigError` naming the file and the offending key for an unreadable file or invalid
    YAML, an unknown key, a wrong type or range, a ``fixed`` mapping without ``objective``,
    ``n_estimators`` or ``early_stopping_rounds`` or with a parameter the code sets itself
    (:data:`RESERVED_PARAMS`), an empty search, a search entry with a bad type, ``low >= high``, a log
    scale with ``low <= 0``, non-integral int bounds or a name that is fixed or reserved, and an unknown
    or repeated monotone feature.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the file ({exc})") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    unknown = sorted(str(k) for k in set(data) - set(_CONFIG_KEYS))
    if unknown:
        raise ConfigError(f"{path}: unknown key(s): {', '.join(unknown)}")
    base = RankerConfig(fixed={}, search=())
    experiment = data.get("experiment", base.experiment)
    if not isinstance(experiment, str) or not experiment.strip():
        raise ConfigError(f"{path}: experiment must be a non-empty string")
    model_name = data.get("model_name", base.model_name)
    if not isinstance(model_name, str) or not _MODEL_NAME.fullmatch(model_name):
        raise ConfigError(f"{path}: model_name must match ^[a-z][a-z0-9_]*$, got {model_name!r}")
    seed = data.get("seed", base.seed)
    if not _is_int(seed):
        raise ConfigError(f"{path}: seed must be an int, got {seed!r}")
    counts: dict[str, int] = {}
    for key in ("n_trials", "n_folds", "nthread"):
        value = data.get(key, getattr(base, key))
        if not _is_int(value) or value < 1:
            raise ConfigError(f"{path}: {key} must be an int >= 1, got {value!r}")
        counts[key] = int(value)
    factor = data.get("rounds_factor", base.rounds_factor)
    if not _is_number(factor) or factor <= 0:
        raise ConfigError(f"{path}: rounds_factor must be a number > 0, got {factor!r}")
    device = data.get("device", base.device)
    if device not in DEVICES:
        raise ConfigError(f"{path}: device must be one of {DEVICES}, got {device!r}")
    if "fixed" not in data or "search" not in data:
        raise ConfigError(f"{path}: fixed and search are required")
    fixed = _check_fixed(path, data["fixed"])
    search = _check_search(path, data["search"], fixed)
    monotone = _check_monotone(path, data.get("monotone_features", []))
    return RankerConfig(
        fixed=fixed,
        search=search,
        experiment=experiment.strip(),
        model_name=model_name,
        seed=int(seed),
        n_trials=counts["n_trials"],
        n_folds=counts["n_folds"],
        rounds_factor=float(factor),
        device=str(device),
        nthread=counts["nthread"],
        monotone_features=monotone,
    )


def monotone_columns(features: Sequence[str]) -> tuple[str, ...]:
    """Model input columns constrained by ``features``: each base feature and its ``pct_`` copy.

    The result follows ``FEATURE_INPUT_COLUMNS`` order (8 columns for the default four features, whose
    ``pct_`` copies all exist). Raises ``ValueError`` for a name that is not a base feature.
    """
    unknown = [f for f in features if f not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"unknown base features {unknown}")
    wanted = set(features) | {f"pct_{f}" for f in features}
    return tuple(c for c in FEATURE_INPUT_COLUMNS if c in wanted)


def monotone_constraints(columns: Sequence[str]) -> dict[str, int]:
    """``{column: 1}`` for every column: a non-decreasing score in each (XGBoost takes feature names)."""
    return dict.fromkeys(columns, 1)


# ---------- data ----------


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes, read in 1 MiB chunks. Raises ``OSError`` when it cannot be read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate ``dataset.parquet``, sorted by ``(qid, path)`` on a RangeIndex.

    Raises :class:`PipelineError` when the file is missing (the message says to run ``tdp-ml dataset``)
    or fails :func:`tdp_ml.pipeline.dataset.validate_dataset` (wrong columns or dtypes, non-finite
    features, missing labels, duplicate ``(qid, path)``).
    """
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"{path} does not exist; build it with `tdp-ml dataset`")
    frame = pd.read_parquet(path)
    validate_dataset(frame)
    return frame.sort_values(["qid", "path"], kind="stable").reset_index(drop=True)


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the queries with at least one non-zero grade, sorted by ``(qid, path)``, index labels kept.

    A query whose grades are all 0 has no ranking signal (its NDCG is undefined). Features are not
    checked, so NaN (missing, as used by ablation masking) passes through. An empty frame gives an empty
    frame.
    """
    if frame.empty:
        return frame.copy()
    keep = frame.groupby("qid", sort=False)["grade"].transform("max") > 0
    return frame.loc[keep.to_numpy(dtype=bool)].sort_values(["qid", "path"], kind="stable")


def test_mask(frame: pd.DataFrame) -> pd.Series:
    """True for the rows of each repository's last snapshot: the test rows of both protocols."""
    times = pd.to_datetime(frame["snapshot_time"], utc=True)
    last = times.groupby(frame["repo"].astype("str"), sort=False).transform("max")
    return (times == last).astype(bool).rename("is_test")


test_mask.__test__ = False  # type: ignore[attr-defined]  # not a pytest test despite its name


def code_git_sha() -> str:
    """The code version: ``TDP_CODE_SHA`` when set (the box copy has no ``.git``), else ``git rev-parse
    HEAD`` of this checkout, else ``"unknown"``."""
    env = os.environ.get("TDP_CODE_SHA", "").strip()
    if env:
        return env
    try:
        root = Path(__file__).resolve().parents[2]
        return run_git(["rev-parse", "HEAD"], cwd=root).decode("ascii").strip() or "unknown"
    except Exception:  # any failure means the version is unknown, never an error
        return "unknown"


# ---------- device ----------


def _cuda_probe() -> bool:
    """True when XGBoost was built with CUDA and one round on the GPU succeeds."""
    try:
        if not xgb.build_info().get("USE_CUDA"):
            return False
        data = xgb.DMatrix(np.array([[0.0], [1.0]], dtype=np.float32), label=np.array([0.0, 1.0]))
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, data, num_boost_round=1)
    except Exception as exc:  # XGBoostError, missing driver libraries, ...
        logger.debug("CUDA probe failed: %s", exc)
        return False
    return True


def resolve_device(device: str) -> str:
    """The XGBoost device to train on: ``"cpu"`` or ``"cuda"``.

    ``"cpu"`` is returned as is; ``"auto"`` gives ``"cuda"`` when the CUDA probe succeeds and ``"cpu"``
    otherwise; ``"cuda"`` gives ``"cuda"`` or raises ``RuntimeError`` when CUDA is not available. Raises
    ``ValueError`` for any other value.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "cpu":
        return "cpu"
    available = _cuda_probe()
    if device == "cuda":
        if not available:
            raise RuntimeError("CUDA is not available for XGBoost")
        return "cuda"
    chosen = "cuda" if available else "cpu"
    logger.info("device auto: training on %s", chosen)
    return chosen


# ---------- fitting ----------


@dataclass(frozen=True, eq=False, slots=True)
class FitResult:
    """One fitted ranker: the booster (set to CPU scoring), the early-stopping rounds (``best_iteration +
    1``, None without validation), the training rows and queries used, and the fit time in seconds."""

    booster: xgb.Booster
    best_rounds: int | None
    rows: int
    queries: int
    seconds: float


def fit_ranker(
    train: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    valid: pd.DataFrame | None = None,
    device: str = "cpu",
    nthread: int = DEFAULT_NTHREAD,
) -> FitResult:
    """Fit an ``XGBRanker`` on the rows of ``train`` with a non-zero-grade query.

    ``params`` are XGBRanker keyword arguments; ``device`` and ``n_jobs`` (= ``nthread``) are added here.
    Without ``valid`` the tuning-only keys (:data:`TUNING_ONLY_PARAMS`) are dropped, so exactly
    ``n_estimators`` rounds are fitted. With ``valid`` its non-zero-grade queries are the early-stopping
    set and ``params`` must hold ``early_stopping_rounds``. Feature names are recorded as
    ``FEATURE_INPUT_COLUMNS``. After fitting the booster is switched to CPU prediction with ``nthread``
    threads, so scoring never warns about mismatched devices.

    Raises ``ValueError`` when no training (or validation) query has a non-zero grade, or when
    ``valid`` is given without ``early_stopping_rounds``.
    """
    started = time.perf_counter()
    tr = training_rows(train)
    if tr.empty:
        raise ValueError("no training query with a non-zero grade")
    kwargs = dict(params) | {"device": device, "n_jobs": nthread}
    fit_kwargs: dict[str, Any] = {}
    if valid is None:
        for key in TUNING_ONLY_PARAMS:
            kwargs.pop(key, None)
    else:
        vr = training_rows(valid)
        if vr.empty:
            raise ValueError("no validation query with a non-zero grade")
        if "early_stopping_rounds" not in kwargs:
            raise ValueError("early stopping needs early_stopping_rounds in params")
        fit_kwargs = {"eval_set": [(vr[_F], vr["grade"])], "eval_qid": [vr["qid"].to_numpy()]}
    model = xgb.XGBRanker(**kwargs)
    model.fit(tr[_F], tr["grade"], qid=tr["qid"].to_numpy(), verbose=False, **fit_kwargs)
    booster = model.get_booster()
    booster.set_param({"device": "cpu", "nthread": nthread})
    best = int(model.best_iteration) + 1 if valid is not None else None
    return FitResult(
        booster=booster,
        best_rounds=best,
        rows=len(tr),
        queries=int(tr["qid"].nunique()),
        seconds=time.perf_counter() - started,
    )


def score_rows(booster: xgb.Booster, frame: pd.DataFrame, *, rounds: int | None = None) -> pd.Series:
    """Raw ranker margins of ``frame`` (float64 Series named ``score`` on ``frame.index``).

    With ``rounds=None`` every round is used through :class:`tdp_core.model.ranker.Ranker` (the runtime
    path, so stored scores equal bundle scores). With ``rounds`` only the first ``rounds`` trees are used
    (early-stopped fold models). NaN features mean missing. An empty frame gives an empty Series.
    """
    if len(frame) == 0:
        return pd.Series([], index=frame.index, dtype="float64", name="score")
    features = frame[_F]
    if rounds is None:
        values = Ranker(booster).score(features)
    else:
        values = booster.inplace_predict(
            features.to_numpy(np.float32),
            predict_type="margin",
            iteration_range=(0, int(rounds)),
            validate_features=False,
        )
    return pd.Series(np.asarray(values, dtype=np.float64), index=frame.index, name="score")


# ---------- parameters ----------


def rounds_to_estimators(rounds: Sequence[int], factor: float) -> int:
    """Final ``n_estimators``: ``max(1, floor(factor * mean(rounds) + 0.5))`` (rounded half up).

    ``rounds`` are the early-stopping rounds (``best_iteration + 1``) of the tuning folds. Raises
    ``ValueError`` for an empty input, a round below 1 or a factor that is not above 0.
    """
    values = [int(r) for r in rounds]
    if not values:
        raise ValueError("rounds must not be empty")
    if any(r < 1 for r in values):
        raise ValueError(f"every round must be >= 1, got {values}")
    if not factor > 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    return max(1, math.floor(factor * (sum(values) / len(values)) + 0.5))


def tuning_params(config: RankerConfig, trial_params: Mapping[str, Any]) -> dict[str, Any]:
    """Parameters of one tuning fit: the fixed ones, the trial's draw and ``random_state`` = seed."""
    return {**config.fixed, **trial_params, "random_state": config.seed}


def final_params(config: RankerConfig, best_params: Mapping[str, Any], n_estimators: int) -> dict[str, Any]:
    """Parameters of the final fits: fixed without the tuning-only keys, the best draw, ``n_estimators``
    and ``random_state`` = seed."""
    fixed = {k: v for k, v in config.fixed.items() if k not in TUNING_ONLY_PARAMS}
    return {**fixed, **best_params, "n_estimators": int(n_estimators), "random_state": config.seed}


def with_monotone(params: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    """A copy of ``params`` with ``monotone_constraints`` = +1 on ``columns``."""
    return {**params, "monotone_constraints": monotone_constraints(columns)}


# ---------- tuning ----------


@dataclass(frozen=True, eq=False, slots=True)
class TuningResult:
    """Outcome of :func:`tune`.

    ``best_value`` is the best trial's mean NDCG@10 over the folds; ``fold_rounds`` its early-stopping
    rounds per fold; ``n_estimators`` the final round count; ``trials`` one row per trial (``number``,
    ``state``, ``value``, ``seconds``, the search parameters, ``fold<k>_ndcg_10`` and ``fold<k>_rounds``);
    ``folds`` the row and query counts of each fold; ``seconds`` the wall time.
    """

    best_params: dict[str, Any]
    best_value: float
    fold_rounds: tuple[int, ...]
    n_estimators: int
    trials: pd.DataFrame
    folds: tuple[dict[str, int], ...]
    seconds: float


def _tuning_folds(
    frame: pd.DataFrame, horizon_days: int, n_folds: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    train_idx, _ = temporal_split(frame, horizon_days)
    train = frame.loc[train_idx]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for k, (tr_idx, va_idx) in enumerate(forward_folds(train, horizon_days, n_folds=n_folds)):
        tr, va = training_rows(train.loc[tr_idx]), training_rows(train.loc[va_idx])
        if tr.empty or va.empty:
            raise ValueError(f"forward fold {k} has no training or validation query with a non-zero grade")
        folds.append((tr, va))
    if not folds:
        raise ValueError("the temporal training set has no usable forward fold")
    return folds


def _trials_frame(study: Any, search: Sequence[SearchParam], n_folds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        duration = trial.duration
        row: dict[str, Any] = {
            "number": int(trial.number),
            "state": str(trial.state.name),
            "value": float(trial.value) if trial.value is not None else math.nan,
            "seconds": duration.total_seconds() if duration is not None else math.nan,
        }
        for p in search:
            row[p.name] = trial.params.get(p.name, math.nan)
        values = trial.user_attrs.get("fold_ndcg_10", [])
        rounds = trial.user_attrs.get("fold_rounds", [])
        for k in range(n_folds):
            row[f"fold{k}_ndcg_10"] = float(values[k]) if k < len(values) else math.nan
            row[f"fold{k}_rounds"] = int(rounds[k]) if k < len(rounds) else -1
        rows.append(row)
    return pd.DataFrame(rows)


def tune(
    frame: pd.DataFrame,
    config: RankerConfig,
    *,
    n_trials: int | None = None,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    device: str = "cpu",
    nthread: int | None = None,
    monotone: bool = False,
) -> TuningResult:
    """Tune the search space with Optuna TPE on forward folds inside the temporal training set.

    The temporal training rows of ``frame`` (embargo ``horizon_days``) are cut into ``config.n_folds``
    forward folds; each fold's training and validation sets keep only non-zero-grade queries. A trial
    draws parameters, fits one early-stopped model per fold (``device``, ``nthread`` threads, default
    ``config.nthread``), scores the validation rows with the first ``best_iteration + 1`` trees and
    returns the mean over folds of the per-query mean NDCG@10. The sampler is ``TPESampler(seed=
    config.seed)`` and trials run one at a time, so a CPU run is deterministic. With ``monotone`` every
    fit carries the monotone constraints of ``config.monotone_features``.

    Early stopping uses the same validation fold that is scored, which is standard and slightly
    optimistic for ``best_value``. Raises ``ValueError`` for ``n_trials < 1`` (default
    ``config.n_trials``), no usable fold, or a fold without a non-zero-grade query.
    """
    import optuna

    started = time.perf_counter()
    n = config.n_trials if n_trials is None else int(n_trials)
    if n < 1:
        raise ValueError(f"n_trials must be >= 1, got {n}")
    threads = config.nthread if nthread is None else int(nthread)
    folds = _tuning_folds(frame, horizon_days, config.n_folds)
    columns = monotone_columns(config.monotone_features) if monotone else ()

    def objective(trial: Any) -> float:
        trial_started = time.perf_counter()
        params = tuning_params(config, {p.name: p.suggest(trial) for p in config.search})
        if monotone:
            params = with_monotone(params, columns)
        values: list[float] = []
        rounds: list[int] = []
        for tr, va in folds:
            fit = fit_ranker(tr, params, valid=va, device=device, nthread=threads)
            best = cast(int, fit.best_rounds)
            pred = score_rows(fit.booster, va, rounds=best)
            per_query = evaluate(va.assign(score=pred.to_numpy()), "score")
            values.append(float(per_query["ndcg_10"].mean()))
            rounds.append(best)
        trial.set_user_attr("fold_ndcg_10", values)
        trial.set_user_attr("fold_rounds", rounds)
        mean = float(np.mean(values))
        logger.info(
            "trial %d/%d ndcg@10 %.4f (folds %s) rounds %s in %.1f s",
            trial.number + 1,
            n,
            mean,
            ", ".join(f"{v:.4f}" for v in values),
            ", ".join(str(r) for r in rounds),
            time.perf_counter() - trial_started,
        )
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=n, n_jobs=1)
    best = study.best_trial
    fold_rounds = tuple(int(r) for r in best.user_attrs["fold_rounds"])
    folds_info = tuple(
        {
            "train_rows": len(tr),
            "train_queries": int(tr["qid"].nunique()),
            "valid_rows": len(va),
            "valid_queries": int(va["qid"].nunique()),
        }
        for tr, va in folds
    )
    return TuningResult(
        best_params=dict(best.params),
        best_value=float(cast(float, best.value)),
        fold_rounds=fold_rounds,
        n_estimators=rounds_to_estimators(fold_rounds, config.rounds_factor),
        trials=_trials_frame(study, config.search, len(folds)),
        folds=folds_info,
        seconds=time.perf_counter() - started,
    )


# ---------- params file ----------


@dataclass(frozen=True, slots=True)
class TunedParams:
    """Tuned parameters as stored in ``results/<prefix>_params.json``.

    ``params`` are the final XGBRanker parameters (fixed plus best, ``n_estimators`` and
    ``random_state``; never runtime keys or monotone constraints, which ``--monotone`` adds at fit time);
    ``best_params`` the best draw; ``best_value`` its tuning NDCG@10; ``fold_rounds`` its early-stopping
    rounds; ``info`` the provenance (trials, folds, search space, dataset sha256, device, run id, ...).
    """

    params: dict[str, Any]
    best_params: dict[str, Any]
    best_value: float | None
    fold_rounds: tuple[int, ...]
    info: dict[str, Any]

    @property
    def n_estimators(self) -> int:
        """Rounds of the final models."""
        return int(self.params["n_estimators"])

    def to_json(self) -> dict[str, Any]:
        """``{"format": 1, "params", "best_params", "best_value", "fold_rounds", "info"}``.

        Raises ``ValueError`` when a value is NaN, infinite or not JSON-serialisable.
        """
        data = {
            "format": PARAMS_FORMAT,
            "params": _plain(self.params),
            "best_params": _plain(self.best_params),
            "best_value": self.best_value,
            "fold_rounds": [int(r) for r in self.fold_rounds],
            "info": _plain(self.info),
        }
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tuned params are not finite JSON: {exc}") from exc
        return data

    @classmethod
    def from_json(cls, data: Any) -> TunedParams:
        """Parse :meth:`to_json` output. Raises :class:`ConfigError` unless ``format`` is 1, ``params`` is
        a mapping with an int ``n_estimators >= 1`` and no tuning-only or runtime keys, and the other
        fields have the right types."""
        if not isinstance(data, Mapping):
            raise ConfigError("tuned params must be a JSON object")
        fmt = data.get("format")
        if not _is_int(fmt) or fmt != PARAMS_FORMAT:
            raise ConfigError(f"tuned params format must be {PARAMS_FORMAT}, got {fmt!r}")
        params = data.get("params")
        if not isinstance(params, Mapping):
            raise ConfigError("tuned params need a params mapping")
        n_estimators = params.get("n_estimators")
        if not _is_int(n_estimators) or cast(int, n_estimators) < 1:
            raise ConfigError("tuned params need an int n_estimators >= 1")
        bad = sorted(k for k in params if k in TUNING_ONLY_PARAMS or k in RUNTIME_PARAMS)
        if bad:
            raise ConfigError(f"tuned params must not hold {bad}; they are set at fit time")
        best_params = data.get("best_params", {})
        info = data.get("info", {})
        if not isinstance(best_params, Mapping) or not isinstance(info, Mapping):
            raise ConfigError("tuned params need best_params and info mappings")
        best_value = data.get("best_value")
        if best_value is not None and not _is_number(best_value):
            raise ConfigError(f"best_value must be a number or null, got {best_value!r}")
        rounds = data.get("fold_rounds", [])
        if not isinstance(rounds, list) or not all(_is_int(r) for r in rounds):
            raise ConfigError("fold_rounds must be a list of ints")
        return cls(
            params=dict(params),
            best_params=dict(best_params),
            best_value=None if best_value is None else float(best_value),
            fold_rounds=tuple(int(r) for r in rounds),
            info=dict(info),
        )


def tuned_params(config: RankerConfig, tuning: TuningResult, *, info: Mapping[str, Any]) -> TunedParams:
    """Final parameters of a tuning run (:func:`final_params`) with its provenance.

    ``info`` gets ``n_trials``, ``n_folds``, ``seed``, ``rounds_factor``, ``folds``, ``search`` and the
    tuning seconds, then the given ``info`` (for example ``created_at``, ``dataset_sha256``,
    ``horizon_days``, ``device``, ``monotone``, ``run_id``).
    """
    meta: dict[str, Any] = {
        "n_trials": len(tuning.trials),
        "n_folds": len(tuning.folds),
        "seed": config.seed,
        "rounds_factor": config.rounds_factor,
        "folds": [dict(f) for f in tuning.folds],
        "search": [p.to_json() for p in config.search],
        "tuning_seconds": round(tuning.seconds, 3),
    }
    meta.update(info)
    return TunedParams(
        params=final_params(config, tuning.best_params, tuning.n_estimators),
        best_params=dict(tuning.best_params),
        best_value=tuning.best_value,
        fold_rounds=tuning.fold_rounds,
        info=meta,
    )


def save_params(path: Path, tuned: TunedParams) -> Path:
    """Write ``tuned`` as JSON atomically and return ``path`` (``ValueError`` for non-finite values)."""
    return write_json(Path(path), tuned.to_json())


def load_params(path: Path) -> TunedParams:
    """Read a params file of :func:`save_params`. Raises :class:`ConfigError` when it is missing, not
    JSON or invalid (see :meth:`TunedParams.from_json`)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path} does not exist; tune with `tdp-ml train --protocol temporal` first")
    try:
        data = read_json(path)
    except PipelineError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        return TunedParams.from_json(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


# ---------- protocols ----------


@dataclass(frozen=True, eq=False, slots=True)
class ProtocolResult:
    """A protocol run of the ranker.

    ``per_query`` is the P3.1 per-query frame; ``scores`` holds :data:`SCORE_COLUMNS` (temporal: the test
    rows; LOPO: every row, scored by the model that never saw its repository); ``booster`` is the final
    temporal model or, for LOPO, the model fitted on every row; ``training_rows`` and
    ``training_queries`` describe that booster's training set; ``seconds`` is the wall time.
    """

    protocol: str
    model: str
    per_query: pd.DataFrame
    scores: pd.DataFrame
    booster: xgb.Booster
    training_rows: int
    training_queries: int
    seconds: float


def _score_frame(
    rows: pd.DataFrame, scores: pd.Series, *, model: str, protocol: str, split: pd.Series, is_test: pd.Series
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "model": model,
            "protocol": protocol,
            "split": split.astype("str").to_numpy(),
            "qid": rows["qid"].to_numpy(),
            "repo": rows["repo"].astype("str").to_numpy(),
            "snapshot_idx": rows["snapshot_idx"].to_numpy(),
            "snapshot_time": rows["snapshot_time"].to_numpy(),
            "path": rows["path"].astype("str").to_numpy(),
            "sloc": rows["sloc"].to_numpy(),
            "grade": rows["grade"].to_numpy(),
            "pain": rows["pain"].to_numpy(),
            "latent_defects": rows["latent_defects"].to_numpy(),
            "has_latent_defect": rows["has_latent_defect"].to_numpy(),
            "is_test": is_test.to_numpy(dtype=bool),
            "score": scores.reindex(rows.index).to_numpy(dtype=np.float64),
        },
        columns=list(SCORE_COLUMNS),
    )
    out["snapshot_time"] = pd.to_datetime(out["snapshot_time"], utc=True)
    return out.sort_values(["qid", "path"], kind="stable").reset_index(drop=True)


def run_temporal(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    device: str = "cpu",
    nthread: int = DEFAULT_NTHREAD,
    model: str = "ranker",
) -> ProtocolResult:
    """Fit the final temporal model on the embargoed temporal training rows and score the test queries.

    ``params`` are final parameters (no early stopping: exactly ``n_estimators`` rounds). The test set is
    each repository's last snapshot; training rows are older snapshots whose label window
    (``horizon_days``) closed by then. Raises ``ValueError`` when the split has no training rows or no
    training query has a non-zero grade.
    """
    started = time.perf_counter()
    fits: list[FitResult] = []

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        fit = fit_ranker(train, params, device=device, nthread=nthread)
        fits.append(fit)
        return score_rows(fit.booster, test)

    run = run_protocol(frame, "temporal", fit_predict, horizon_days=horizon_days, model=model)
    fit = fits[-1]
    rows = frame.loc[run.scores.index]
    scores = _score_frame(
        rows,
        run.scores,
        model=model,
        protocol="temporal",
        split=pd.Series("temporal", index=rows.index),
        is_test=pd.Series(True, index=rows.index),
    )
    return ProtocolResult(
        protocol="temporal",
        model=model,
        per_query=run.per_query,
        scores=scores,
        booster=fit.booster,
        training_rows=fit.rows,
        training_queries=fit.queries,
        seconds=time.perf_counter() - started,
    )


def run_lopo(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    device: str = "cpu",
    nthread: int = DEFAULT_NTHREAD,
    model: str = "ranker",
) -> ProtocolResult:
    """Leave one project out: one model per held-out repository, then a model on every row.

    For each repository a model is fitted on every other repository and scores all rows of the held-out
    repository (every snapshot): these are the out-of-fold scores, and the held-out last snapshot is
    evaluated from the same values. ``scores`` holds every scored row with ``split`` = the held-out
    repository and ``is_test`` marking each repository's last snapshot. Finally a model with the same
    ``params`` is fitted on every row of ``frame`` (the production candidate); ``booster``,
    ``training_rows`` and ``training_queries`` describe it. ``horizon_days`` is unused by the split and
    kept for a uniform signature.

    Raises ``RuntimeError`` if a split's training rows contain the held-out repository (a leak guard),
    and ``ValueError`` as :func:`run_temporal` does.
    """
    started = time.perf_counter()
    repos = frame["repo"].astype("str")
    stash: list[pd.Series] = []

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        held = sorted(test["repo"].astype("str").unique().tolist())
        if len(held) != 1:
            raise RuntimeError(f"a LOPO test set must hold one repository, got {held}")
        if (train["repo"].astype("str") == held[0]).any():
            raise RuntimeError(f"LOPO leak: the training rows contain the held-out repository {held[0]}")
        fit = fit_ranker(train, params, device=device, nthread=nthread)
        oof = score_rows(fit.booster, frame.loc[(repos == held[0]).to_numpy()])
        stash.append(oof)
        return oof.loc[test.index]

    run = run_protocol(frame, "lopo", fit_predict, horizon_days=horizon_days, model=model)
    oof = pd.concat(stash)
    rows = frame.loc[oof.index]
    scores = _score_frame(
        rows,
        oof,
        model=model,
        protocol="lopo",
        split=rows["repo"].astype("str"),
        is_test=test_mask(frame).loc[oof.index],
    )
    full = fit_ranker(frame, params, device=device, nthread=nthread)
    return ProtocolResult(
        protocol="lopo",
        model=model,
        per_query=run.per_query,
        scores=scores,
        booster=full.booster,
        training_rows=full.rows,
        training_queries=full.queries,
        seconds=time.perf_counter() - started,
    )


# ---------- comparison ----------


def _hotspot(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    return test["pct_churn_decayed"].astype("float64") * test["pct_cc_sum"].astype("float64")


def hotspot_per_query(frame: pd.DataFrame, protocol: str, *, horizon_days: int) -> pd.DataFrame:
    """Per-query metrics of the hotspot baseline (``pct_churn_decayed * pct_cc_sum``) under ``protocol``."""
    return run_protocol(frame, protocol, _hotspot, horizon_days=horizon_days, model="hotspot").per_query


def _one_run(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if "protocol" in frame.columns:
        frame = frame.loc[(frame["protocol"].astype("str") == protocol).to_numpy()]
    return frame.reset_index(drop=True)


def _read_runs(path: Path, protocol: str) -> dict[str, pd.DataFrame] | None:
    try:
        frame = _one_run(load_per_query(path), protocol)
    except (OSError, ValueError) as exc:
        logger.warning("ignoring %s: %s", path, exc)
        return None
    if "model" not in frame.columns:
        logger.warning("ignoring %s: it has no model column", path)
        return None
    names = list(dict.fromkeys(frame["model"].astype("str").tolist()))
    model = frame["model"].astype("str")
    return {n: frame.loc[(model == n).to_numpy()].reset_index(drop=True) for n in names}


def _same_dataset(note: Path, dataset_sha256: str | None) -> bool:
    if dataset_sha256 is None or not note.is_file():
        return True
    try:
        data = json.loads(note.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    recorded = data.get("dataset_sha256") if isinstance(data, dict) else None
    return recorded is None or recorded == dataset_sha256


def comparison_runs(
    frame: pd.DataFrame,
    protocol: str,
    per_query: pd.DataFrame,
    *,
    results_dir: Path,
    horizon_days: int,
    extra: Mapping[str, Path] | None = None,
    dataset_sha256: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-query frames of the models to compare against, keyed by model name.

    The baselines file ``<results_dir>/baselines_<protocol>.parquet`` is used when it exists and every
    model in it covers exactly the qids of ``per_query``, in file order. With ``dataset_sha256`` the
    file's provenance note (``baselines_<protocol>.json``, when present) must name the same dataset.
    Otherwise the whole file is ignored with a warning. The hotspot baseline is computed here when the
    file does not provide it. Each ``extra`` entry (name to per-query file, for example the unconstrained
    ranker in a monotone run) is added when the file exists and its qids match.
    """
    qids = set(per_query["qid"].tolist())
    others: dict[str, pd.DataFrame] = {}
    path = per_query_path("baselines", protocol, root=results_dir)
    if path.is_file() and not _same_dataset(path.with_suffix(".json"), dataset_sha256):
        logger.warning("ignoring %s: it was computed on a different dataset", path)
    elif path.is_file():
        runs = _read_runs(path, protocol)
        if runs is not None:
            bad = sorted(n for n, f in runs.items() if set(f["qid"].tolist()) != qids)
            if bad or not runs:
                logger.warning("ignoring %s: the qids of %s differ from the ranker's", path, bad)
            else:
                others.update(runs)
    if "hotspot" not in others:
        others["hotspot"] = hotspot_per_query(frame, protocol, horizon_days=horizon_days)
    for name, extra_path in (extra or {}).items():
        if name in others or not Path(extra_path).is_file():
            continue
        found = _read_runs(Path(extra_path), protocol)
        if found is None or len(found) != 1:
            continue
        run = next(iter(found.values()))
        if set(run["qid"].tolist()) != qids:
            logger.warning("ignoring %s: its qids differ from the ranker's", extra_path)
            continue
        others[name] = run
    return others


def compare(
    per_query: pd.DataFrame, others: Mapping[str, pd.DataFrame], *, n_boot: int = 1000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap summary and paired tests of the model against ``others``.

    ``summary`` is :func:`tdp_ml.eval.summarize` over the model and every other run (ranking metrics,
    ``n_boot`` repository resamples, ``seed``). ``tests`` stacks :func:`tdp_ml.eval.wilcoxon_holm` (two
    sided, Holm across the baselines within a metric) for each of :data:`COMPARE_METRICS`; it is empty
    with ``TEST_COLUMNS`` when ``others`` is empty.
    """
    combined = pd.concat([per_query, *others.values()], ignore_index=True)
    summary = summarize(combined, n_boot, seed, metrics=list(RANKING_METRICS))
    if not others:
        return summary, pd.DataFrame(columns=list(TEST_COLUMNS))
    tests = pd.concat([wilcoxon_holm(per_query, others, m) for m in COMPARE_METRICS], ignore_index=True)
    return summary, tests


def p_greater(per_query: pd.DataFrame, baseline: pd.DataFrame, metric: str = "ndcg_10") -> float:
    """One-sided paired Wilcoxon p-value that the model beats ``baseline`` on ``metric`` (NaN without
    pairs). This is the promotion gate against the hotspot baseline."""
    test = wilcoxon_holm(per_query, {"baseline": baseline}, metric, alternative="greater")
    return float(test["p_value"].iloc[0])


def summary_report(
    protocol: str, model: str, summary: pd.DataFrame, tests: pd.DataFrame, *, notes: Sequence[str] = ()
) -> str:
    """Markdown report of a run: title, notes, the summary table and the paired comparison table."""
    lines = [f"# Ranker results ({protocol})", ""]
    lines += [f"- {note}" for note in notes]
    if notes:
        lines.append("")
    text = "\n".join(lines) + "\n" + summary_markdown(summary) + "\n"
    text += (
        f"Paired two-sided Wilcoxon signed-rank tests of {model} against each baseline on the same test "
        "queries, Holm-corrected across the baselines within each metric; means are over queries and "
        "intervals are 95% repository bootstrap intervals.\n\n"
    )
    return text + comparison_markdown(tests)


# ---------- orchestration ----------


@dataclass(frozen=True, eq=False, slots=True)
class TrainOutcome:
    """Everything one :func:`train` call produced.

    ``model`` is the run's model name (``ranker`` or ``ranker_monotone``); ``params`` the parameters used
    (tuned here when ``tuning`` is set); ``others`` the comparison runs; ``files`` every result file
    written plus the params file used (key ``params``); ``seconds`` the ``tuning``, ``fit`` and ``total``
    wall times.
    """

    protocol: str
    model: str
    params: TunedParams
    tuning: TuningResult | None
    result: ProtocolResult
    summary: pd.DataFrame
    tests: pd.DataFrame
    others: dict[str, pd.DataFrame]
    files: dict[str, Path]
    seconds: dict[str, float] = field(default_factory=dict)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _report_notes(
    outcome_params: TunedParams,
    tuned_here: bool,
    device: str,
    result: ProtocolResult,
    provenance: Mapping[str, str],
) -> list[str]:
    where = "tuned in this run" if tuned_here else "loaded from the params file (tuned on temporal folds)"
    notes = [
        f"Parameters {where}; n_estimators {outcome_params.n_estimators}, device {device}.",
        f"Test queries: {result.per_query['qid'].nunique()}; final model trained on "
        f"{result.training_rows} rows in {result.training_queries} queries.",
    ]
    sha = provenance.get("dataset_sha256")
    if sha:
        notes.append(f"Dataset sha256 {sha}.")
    return notes


def train(
    frame: pd.DataFrame,
    config: RankerConfig,
    *,
    protocol: str,
    results_dir: Path,
    params: TunedParams | None = None,
    params_path: Path | None = None,
    n_trials: int | None = None,
    monotone: bool = False,
    device: str = "cpu",
    nthread: int | None = None,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    run_id: str | None = None,
    provenance: Mapping[str, str] | None = None,
) -> TrainOutcome:
    """Tune (when ``params`` is None), run ``protocol`` and write the result files to ``results_dir``.

    The model name and file prefix are ``config.model_name`` plus ``_monotone`` with ``monotone``. When
    tuning, ``<prefix>_params.json`` and ``<prefix>_trials.parquet`` are written; ``params_path`` names a
    loaded params file for the record. Then ``<prefix>_<protocol>.parquet`` (per query),
    ``<prefix>_scores_temporal.parquet`` or ``<prefix>_oof.parquet`` (scores) and
    ``<prefix>_<protocol>_summary.md`` (comparison with the baselines, see :func:`comparison_runs`).
    ``device`` must already be resolved (``cpu`` or ``cuda``); ``nthread`` defaults to
    ``config.nthread``; ``run_id`` and ``provenance`` go into the params info.

    Raises ``ValueError`` for an unknown protocol and for the errors of :func:`tune` and the protocol
    runners.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    total_started = time.perf_counter()
    threads = config.nthread if nthread is None else int(nthread)
    prefix = config.model_name + ("_monotone" if monotone else "")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    tuning: TuningResult | None = None
    seconds = {"tuning": 0.0}
    columns = monotone_columns(config.monotone_features) if monotone else ()
    if params is None:
        tuning = tune(
            frame,
            config,
            n_trials=n_trials,
            horizon_days=horizon_days,
            device=device,
            nthread=threads,
            monotone=monotone,
        )
        info: dict[str, Any] = {
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "horizon_days": horizon_days,
            "device": device,
            "monotone": monotone,
            "run_id": run_id,
        }
        info.update(provenance or {})
        params = tuned_params(config, tuning, info=info)
        files["params"] = save_params(results_dir / f"{prefix}_params.json", params)
        files["trials"] = write_frame(results_dir / f"{prefix}_trials.parquet", tuning.trials)
        seconds["tuning"] = tuning.seconds
    elif params_path is not None:
        files["params"] = Path(params_path)
    if bool(params.info.get("monotone", False)) != monotone:
        logger.info(
            "params were tuned with monotone=%s and are used with monotone=%s",
            params.info.get("monotone", False),
            monotone,
        )
    fit_params = with_monotone(params.params, columns) if monotone else dict(params.params)
    runner: Callable[..., ProtocolResult] = run_temporal if protocol == "temporal" else run_lopo
    result = runner(
        frame, fit_params, horizon_days=horizon_days, device=device, nthread=threads, model=prefix
    )
    files["per_query"] = save_per_query(result.per_query, per_query_path(prefix, protocol, root=results_dir))
    score_name = f"{prefix}_scores_temporal.parquet" if protocol == "temporal" else f"{prefix}_oof.parquet"
    files["scores"] = write_frame(results_dir / score_name, result.scores)
    extra = (
        {config.model_name: per_query_path(config.model_name, protocol, root=results_dir)}
        if monotone
        else None
    )
    others = comparison_runs(
        frame,
        protocol,
        result.per_query,
        results_dir=results_dir,
        horizon_days=horizon_days,
        extra=extra,
        dataset_sha256=(provenance or {}).get("dataset_sha256"),
    )
    summary, tests = compare(result.per_query, others, seed=config.seed)
    notes = _report_notes(params, tuning is not None, device, result, provenance or {})
    files["summary"] = _write_text(
        results_dir / f"{prefix}_{protocol}_summary.md",
        summary_report(protocol, prefix, summary, tests, notes=notes),
    )
    seconds["fit"] = result.seconds
    seconds["total"] = time.perf_counter() - total_started
    return TrainOutcome(
        protocol=protocol,
        model=prefix,
        params=params,
        tuning=tuning,
        result=result,
        summary=summary,
        tests=tests,
        others=others,
        files=files,
        seconds=seconds,
    )
