"""Deterministic synthetic dataset in the layout of ``data/dataset.parquet``, for tests and development.

Every file has a hidden, persistent *risk* (plus noise per snapshot) and a *size*. The feature columns are
noisy functions of both (complexity, churn, prior bugs and dependents rise with risk), and the labels
follow the real definitions: latent defects and future churn are drawn from risk and size, pain and grade
use the formulas of the labelling step. A good ranker therefore beats simple baselines, which beat random.

The data is not realistic in detail; it only has the right columns, dtypes and qualitative signal.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import (
    FEATURE_COLUMNS,
    FEATURE_INPUT_COLUMNS,
    INDEX_COLUMNS,
    LABEL_COLUMNS,
    PCT_COLUMNS,
)

logger = logging.getLogger(__name__)

SYNTHETIC_START: Final = pd.Timestamp("2016-01-04", tz="UTC")
SYNTHETIC_COLUMNS: Final = INDEX_COLUMNS + FEATURE_INPUT_COLUMNS + LABEL_COLUMNS + ("qid",)
_REPO_OFFSET_DAYS: Final = 23

FloatArray = npt.NDArray[np.float64]


def _check_int(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


def _sigmoid(x: FloatArray) -> FloatArray:
    out: FloatArray = 1.0 / (1.0 + np.exp(-x))
    return out


def _grades(pain: npt.NDArray[np.float32], qid: npt.NDArray[np.int64]) -> npt.NDArray[np.int8]:
    """Grade rule of the labelling step: 0 for zero pain, else quantile bins of the non-zero pains."""
    nonzero = pd.Series(np.where(pain > 0, pain.astype(np.float64), np.nan))
    pct = nonzero.groupby(qid).rank(pct=True, method="average").to_numpy(dtype=np.float64)
    grade = np.select(
        [~(pain > 0), pct <= 0.50, pct <= 0.80, pct <= 0.95],
        [0, 1, 2, 3],
        default=4,
    )
    return grade.astype(np.int8)


def make_synthetic_dataset(
    n_repos: int = 4,
    n_snapshots: int = 10,
    n_files: int = 60,
    *,
    seed: int = DEFAULT_SETTINGS.seed,
    step_days: int = DEFAULT_SETTINGS.snapshot_step_days,
    gamma: float = DEFAULT_SETTINGS.gamma,
) -> pd.DataFrame:
    """Build a synthetic dataset with the columns and dtypes of ``data/dataset.parquet``.

    Repositories are ``synthetic__repo00``, ``synthetic__repo01``, ...; repository ``r`` has snapshots
    ``s = 0 .. n_snapshots - 1`` at ``SYNTHETIC_START + 23 * r + s * step_days`` days (UTC, second
    resolution), and the same ``n_files`` paths (``src/mod_000.py``, ...) at every snapshot. ``qid`` is
    ``r * n_snapshots + s``. Rows are sorted by qid, then path, on a RangeIndex.

    Columns are :data:`SYNTHETIC_COLUMNS`: index columns (repo and path str, snapshot_idx int16,
    snapshot_time UTC), the 89 feature inputs as float32 (``pct_`` columns are within-query percentile
    ranks of the float32 raw column), the labels (latent_defects and future_churn int32, pain float32,
    grade int8, has_latent_defect bool) and qid int64. ``pain = latent + gamma * log1p(churn /
    max(median churn of the query, 1))``; grades follow the labelling rule; every query has a non-zero pain.

    The result is a deterministic function of the arguments. Raises ``ValueError`` for ``n_repos < 1``,
    ``n_snapshots < 1``, ``n_files < 2``, ``step_days < 1`` or a negative or non-finite ``gamma``.
    """
    n_repos = _check_int(n_repos, "n_repos", 1)
    n_snapshots = _check_int(n_snapshots, "n_snapshots", 1)
    n_files = _check_int(n_files, "n_files", 2)
    step_days = _check_int(step_days, "step_days", 1)
    if not isinstance(gamma, int | float | np.floating) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError(f"gamma must be a finite number >= 0, got {gamma!r}")

    rng = np.random.default_rng(seed)
    shape = (n_repos, n_snapshots, n_files)
    n_queries = n_repos * n_snapshots
    n_rows = n_queries * n_files

    def normal(scale: float = 1.0) -> FloatArray:
        return rng.normal(0.0, scale, shape)

    base = rng.standard_normal((n_repos, 1, n_files))
    size = np.broadcast_to(rng.standard_normal((n_repos, 1, n_files)), shape)
    risk = 0.8 * base + 0.6 * normal()
    snap = np.broadcast_to(np.arange(n_snapshots, dtype=np.float64)[None, :, None], shape)
    history_days = 365.0 + snap * step_days

    f: dict[str, npt.NDArray[Any]] = {}
    # static structure
    sloc = np.maximum(1.0, np.round(np.exp(4.3 + 0.8 * size + 0.2 * risk + normal(0.2))))
    f["sloc"] = sloc
    f["cc_sum"] = np.round(sloc * np.exp(-2.3 + 0.3 * risk + normal(0.3))) + 1.0
    n_functions = np.maximum(1.0, np.round(sloc * np.exp(-2.8 + normal(0.3))))
    f["n_functions"] = n_functions
    f["cc_mean"] = f["cc_sum"] / n_functions
    spread = np.exp(0.7 + 0.2 * risk + normal(0.3))
    f["cc_max"] = np.maximum(np.ceil(f["cc_mean"]), np.round(f["cc_mean"] * spread))
    f["cog_max"] = np.round(f["cc_max"] * np.exp(0.2 + normal(0.3)))
    f["nesting_max"] = np.clip(np.round(1.0 + 0.6 * np.log(f["cc_max"]) + normal(0.5)), 0.0, None)
    f["n_classes"] = rng.poisson(sloc / 150.0).astype(np.float64)
    f["longest_function"] = np.minimum(sloc, np.round(sloc / n_functions * np.exp(1.0 + normal(0.3))))
    f["params_max"] = rng.poisson(np.exp(1.0 + 0.1 * risk)).astype(np.float64)
    f["dup_ratio"] = np.clip(rng.beta(1.0, 12.0, shape) + 0.02 * risk, 0.0, 1.0)
    # evolution
    f["commits_total"] = 1.0 + rng.poisson(np.exp(2.0 + 0.4 * risk + 0.3 * size) * (1.0 + 0.3 * snap))
    f["commits_90d"] = rng.poisson(np.exp(0.3 + 0.7 * risk)).astype(np.float64)
    f["commits_180d"] = f["commits_90d"] + rng.poisson(np.exp(0.3 + 0.6 * risk))
    f["commits_decayed"] = np.exp(0.5 + 0.7 * risk + normal(0.4))
    f["churn_total"] = np.round(np.exp(5.0 + 0.5 * risk + 0.5 * size + normal(0.5)))
    f["churn_decayed"] = np.exp(2.5 + 0.8 * risk + 0.3 * size + normal(0.5))
    f["rel_churn_decayed"] = f["churn_decayed"] / sloc
    f["age_days"] = history_days * rng.uniform(0.3, 1.0, shape)
    f["staleness_days"] = np.exp(4.0 - 0.5 * risk + normal(0.8))
    f["cc_trend"] = rng.normal(0.2 * risk, 1.0)
    f["burstiness"] = np.clip(rng.normal(0.2 * risk, 0.3), -1.0, 1.0)
    f["is_new"] = (rng.uniform(size=shape) < 0.08).astype(np.float64)
    # ownership
    n_authors = 1.0 + rng.poisson(np.exp(0.8 + 0.4 * risk + 0.2 * size))
    f["n_authors"] = n_authors
    f["owner_share"] = 1.0 / n_authors + (1.0 - 1.0 / n_authors) * rng.beta(2.0, 2.0, shape)
    f["minor_authors"] = rng.binomial((n_authors - 1.0).astype(np.int64), 0.5).astype(np.float64)
    f["author_entropy"] = np.log2(n_authors) * rng.uniform(0.3, 1.0, shape)
    f["owner_active"] = (rng.uniform(size=shape) < _sigmoid(1.0 - 0.5 * risk)).astype(np.float64)
    # defect history
    f["prior_latent_bugs"] = rng.poisson(np.exp(-0.5 + 0.7 * risk)).astype(np.float64)
    f["prior_bugfix_commits"] = f["prior_latent_bugs"] + rng.poisson(np.exp(-0.2 + 0.5 * risk))
    # impact
    f["fan_in"] = rng.poisson(np.exp(0.5 + 0.5 * risk + 0.2 * size)).astype(np.float64)
    f["fan_out"] = rng.poisson(np.exp(1.0 + 0.3 * size)).astype(np.float64)
    f["transitive_dependents"] = f["fan_in"] + rng.poisson(np.exp(1.0 + 0.6 * risk))
    f["pagerank"] = np.clip(np.exp(-5.0 + 0.5 * np.log1p(f["transitive_dependents"]) + normal(0.3)), 0.0, 1.0)
    f["betweenness"] = np.clip(np.exp(-5.0 + 0.8 * risk + normal(1.0)), 0.0, 1.0)
    f["scc_size"] = 1.0 + (rng.uniform(size=shape) < 0.1) * rng.poisson(2.0, shape)
    f["cochange_partners"] = rng.poisson(np.exp(0.5 + 0.5 * risk)).astype(np.float64)
    f["cochange_max_conf"] = np.where(f["cochange_partners"] > 0, rng.uniform(0.5, 1.0, shape), 0.0)
    f["hidden_coupling"] = rng.binomial(f["cochange_partners"].astype(np.int64), 0.4).astype(np.float64)
    # tests
    f["has_test_file"] = (rng.uniform(size=shape) < _sigmoid(0.3 - 0.6 * risk)).astype(np.float64)
    f["test_importers"] = f["has_test_file"] * rng.poisson(1.5, shape)
    f["test_cochange_rate"] = rng.beta(2.0, 5.0, shape)
    f["pkg_test_ratio"] = np.broadcast_to(rng.beta(2.0, 3.0, (n_repos, 1, 1)), shape).copy()
    # self-admitted debt
    satd = np.minimum(sloc, rng.poisson(np.exp(-1.0 + 0.4 * risk + 0.3 * size)))
    f["satd_count"] = satd
    f["satd_density"] = satd / sloc
    f["satd_age_days"] = np.where(satd > 0, np.minimum(history_days, np.exp(5.0 + normal(0.7))), 0.0)

    # labels
    latent = rng.poisson(np.exp(-1.6 + 0.9 * risk + 0.2 * size)).reshape(n_queries, n_files)
    touched = rng.uniform(size=shape) < _sigmoid(0.8 + 0.8 * risk)
    churn_amount = np.round(np.exp(3.0 + 0.7 * risk + 0.3 * size + normal(0.8)))
    churn = (touched * churn_amount).reshape(n_queries, n_files)
    churn_pain = gamma * np.log1p(churn / np.maximum(np.median(churn, axis=1, keepdims=True), 1.0))
    pain_2d = (latent + churn_pain).astype(np.float32)
    quiet = np.flatnonzero(pain_2d.max(axis=1) <= 0)
    if quiet.size:  # a query without any pain cannot be ranked: give its riskiest file one defect
        latent[quiet, risk.reshape(n_queries, n_files)[quiet].argmax(axis=1)] = 1
        pain_2d = (latent + churn_pain).astype(np.float32)
    pain = pain_2d.ravel()

    qid = np.repeat(np.arange(n_queries, dtype=np.int64), n_files)
    repo_idx = np.repeat(np.arange(n_repos), n_snapshots * n_files)
    snap_idx = np.tile(np.repeat(np.arange(n_snapshots), n_files), n_repos)
    width = max(3, len(str(n_files - 1)))
    paths = np.array([f"src/mod_{i:0{width}d}.py" for i in range(n_files)], dtype=object)
    days = _REPO_OFFSET_DAYS * repo_idx + step_days * snap_idx
    times = (SYNTHETIC_START + pd.to_timedelta(days, unit="D")).as_unit("s")

    columns: dict[str, object] = {
        "repo": pd.array([f"synthetic__repo{r:02d}" for r in repo_idx.tolist()], dtype="str"),
        "snapshot_idx": snap_idx.astype(np.int16),
        "snapshot_time": times,
        "path": pd.array(np.tile(paths, n_queries), dtype="str"),
    }
    raw = {name: np.asarray(f[name], dtype=np.float64).reshape(n_rows).astype(np.float32) for name in f}
    raw = {name: raw[name] for name in FEATURE_COLUMNS}
    columns.update(raw)
    pct_source = pd.DataFrame({name[4:]: raw[name[4:]] for name in PCT_COLUMNS})
    pct = pct_source.groupby(qid).rank(pct=True, method="average")
    columns.update({f"pct_{name}": pct[name].to_numpy(dtype=np.float32) for name in pct_source.columns})
    columns.update(
        {
            "latent_defects": latent.ravel().astype(np.int32),
            "future_churn": churn.ravel().astype(np.int32),
            "pain": pain,
            "grade": _grades(pain, qid),
            "has_latent_defect": latent.ravel() > 0,
            "qid": qid,
        }
    )
    frame = pd.DataFrame(columns)
    frame = frame[list(SYNTHETIC_COLUMNS)]
    logger.debug("synthetic dataset: %d repos x %d snapshots x %d files", n_repos, n_snapshots, n_files)
    return frame
