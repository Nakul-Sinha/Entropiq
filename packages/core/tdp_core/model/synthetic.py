"""Synthetic training data in the ``data/dataset.parquet`` layout.

:func:`synthetic_dataset` draws random but plausible feature rows for a few fake repositories and labels them
with the pain and grade rules of the labelling stage. It exists so the dev model bundle and the tests of
later stages (training, what-if, platform) can run long before the real dataset is mined. Nothing here is
evidence about real code.

Generation outline (all draws from one ``numpy`` generator seeded with ``seed``): three hidden traits per file
(size, activity, centrality) drive the static, evolution, ownership, defect history, impact, test and SATD
columns so the usual correlations appear (big files are complex, active files have many authors and bug
fixes, central files have many dependents). The expected number of latent defects grows with recent churn,
prior latent bugs, transitive dependents and, more weakly, complexity, duplication, SATD and import cycles,
and shrinks with tests. Pain and grades then follow the real label rules exactly.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import (
    FEATURE_COLUMNS,
    FEATURE_INPUT_COLUMNS,
    INDEX_COLUMNS,
    LABEL_COLUMNS,
    PCT_COLUMNS,
)

logger = logging.getLogger(__name__)

SYNTHETIC_START: Final = datetime(2019, 1, 7, tzinfo=UTC)
"""Snapshot time of snapshot 0 in every synthetic repository (a Monday, UTC)."""

SYNTHETIC_REPO_PREFIX: Final = "synthetic__repo"

type _F = npt.NDArray[np.float64]


def synthetic_dataset(
    n_queries: int = 30,
    rows_per_query: int = 200,
    *,
    snapshots_per_repo: int = 5,
    seed: int = 42,
    settings: Settings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """Return a synthetic dataset frame of ``n_queries * rows_per_query`` rows.

    Columns are ``INDEX_COLUMNS + FEATURE_INPUT_COLUMNS + LABEL_COLUMNS + ("qid",)``; rows are sorted by
    ``qid`` then ``path`` with a ``RangeIndex``. Query ``q`` is snapshot ``q % snapshots_per_repo`` of
    repository ``synthetic__repo{q // snapshots_per_repo:02d}``; its snapshot time is ``SYNTHETIC_START`` plus
    ``snapshot_idx * settings.snapshot_step_days`` days (UTC) and ``qid == q``. Paths
    (``pkgNN/module_III.py``) repeat across the snapshots of a repository.

    Dtypes: repo and path ``str``, ``snapshot_idx`` int16, ``snapshot_time`` UTC timestamp, features float32
    (binary features 0 or 1), ``latent_defects`` and ``future_churn`` int32, ``pain`` float32, ``grade`` int8,
    ``has_latent_defect`` bool, ``qid`` int32. Units follow the real features: lines, commits, days, counts
    and ratios. ``pct_`` columns are the within-query ``rank(pct=True, method="average")`` of the stored
    float32 values, and ``pain`` / ``grade`` follow the labelling rules with ``settings.gamma``.

    The output depends only on the arguments (same seed, same frame). Raises ``ValueError`` if any size
    argument is below 1.
    """
    if n_queries < 1 or rows_per_query < 1 or snapshots_per_repo < 1:
        raise ValueError("n_queries, rows_per_query and snapshots_per_repo must all be >= 1")
    rng = np.random.default_rng(seed)
    rows = rows_per_query
    n = n_queries * rows
    cap = float(rows - 1)
    qid = np.repeat(np.arange(n_queries, dtype=np.int64), rows)

    # hidden traits
    size = rng.normal(0.0, 1.0, n)
    act = 0.4 * size + rng.normal(0.0, 1.0, n)
    cen = rng.normal(0.0, 1.0, n)

    f: dict[str, _F] = {}

    # S: static structure
    sloc = np.clip(np.rint(np.exp(4.4 + 0.9 * size)), 1.0, 6000.0)
    nf = rng.poisson(sloc / 22.0).astype(np.float64)
    has_fn = nf > 0
    nf_safe = np.maximum(nf, 1.0)
    raw = 1.0 + rng.gamma(2.0, 0.6 + 0.25 * np.maximum(size, 0.0))
    cc_sum = 1.0 + np.rint(nf * raw)
    cc_mean = np.where(has_fn, (cc_sum - 1.0) / nf_safe, 0.0)
    cc_max_raw = np.rint(cc_mean * (1.0 + rng.exponential(1.0, n)))
    cc_max = np.where(has_fn, np.clip(cc_max_raw, np.ceil(cc_mean), np.maximum(cc_sum - 1.0, 0.0)), 0.0)
    f["sloc"] = sloc
    f["cc_sum"] = cc_sum
    f["cc_mean"] = cc_mean
    f["cc_max"] = cc_max
    f["cog_max"] = np.rint(cc_max * rng.uniform(0.5, 2.0, n))
    f["nesting_max"] = np.minimum(np.rint(np.log2(1.0 + cc_max)), 8.0)
    f["n_functions"] = nf
    f["n_classes"] = rng.poisson(nf / 6.0).astype(np.float64)
    longest = np.minimum(np.rint(sloc / nf_safe * (1.0 + rng.exponential(1.0, n))), sloc)
    f["longest_function"] = np.where(has_fn, longest, 0.0)
    f["params_max"] = np.where(has_fn, 1.0 + rng.poisson(2.0, n), 0.0)
    f["dup_ratio"] = np.where(rng.uniform(size=n) < 0.6, 0.0, rng.beta(0.8, 6.0, n))

    # E: evolution
    ct = rng.poisson(np.exp(1.8 + 0.8 * act)).astype(np.float64)
    c180 = rng.binomial(ct.astype(np.int64), _sigmoid(act - 1.5)).astype(np.float64)
    c90 = rng.binomial(c180.astype(np.int64), 0.55).astype(np.float64)
    commits_decayed = 0.75 * c90 + 0.35 * (c180 - c90) + 0.03 * (ct - c180)
    per_commit = rng.lognormal(np.log(5.0 + 0.15 * sloc), 0.6)
    churn_decayed = commits_decayed * per_commit
    age = np.where(ct > 0, rng.uniform(10.0, 3650.0, n), 0.0)
    f["commits_total"] = ct
    f["commits_90d"] = c90
    f["commits_180d"] = c180
    f["commits_decayed"] = commits_decayed
    f["churn_total"] = np.rint(ct * per_commit)
    f["churn_decayed"] = churn_decayed
    f["rel_churn_decayed"] = churn_decayed / np.maximum(sloc, 1.0)
    f["age_days"] = age
    f["staleness_days"] = age * rng.beta(1.0, 1.0 + ct / 4.0)
    f["cc_trend"] = np.where(c180 > 0, rng.normal(0.5, 2.0, n), 0.0)
    f["burstiness"] = np.where(ct >= 3, rng.gamma(2.0, 0.5, n), 0.0)
    f["is_new"] = (age < settings.new_file_days).astype(np.float64)

    # O: ownership
    na = np.where(ct > 0, np.minimum(1.0 + rng.poisson(1.2 * np.log1p(ct)), ct), 0.0)
    na_safe = np.maximum(na, 1.0)
    owner_share = np.where(na > 0, 1.0 / na_safe + (1.0 - 1.0 / na_safe) * rng.beta(2.0, 2.0, n), 0.0)
    f["n_authors"] = na
    f["owner_share"] = owner_share
    f["minor_authors"] = rng.binomial(np.maximum(na - 1.0, 0.0).astype(np.int64), 0.4).astype(np.float64)
    f["author_entropy"] = np.minimum(1.3 * np.log2(na_safe) * (1.0 - owner_share), np.log2(na_safe))
    f["owner_active"] = ((ct > 0) & (rng.uniform(size=n) < 0.7)).astype(np.float64)

    # D: defect history
    prior_fix = rng.binomial(ct.astype(np.int64), 0.12).astype(np.float64)
    prior_latent = rng.binomial(prior_fix.astype(np.int64), 0.6).astype(np.float64)
    f["prior_bugfix_commits"] = prior_fix
    f["prior_latent_bugs"] = prior_latent

    # I: impact
    fan_in = np.minimum(rng.poisson(np.exp(0.3 + 0.9 * cen)).astype(np.float64), cap)
    fan_out = np.minimum(rng.poisson(np.exp(0.9 + 0.3 * size)).astype(np.float64), cap)
    dependents = np.minimum(fan_in + rng.poisson(1.5 * fan_in * np.exp(0.3 * cen)), cap)
    weight = (1.0 + dependents) * rng.uniform(0.8, 1.2, n)
    partners = rng.poisson(0.25 * np.sqrt(ct)).astype(np.float64)
    weak_conf = np.where(rng.uniform(size=n) < 0.3, rng.uniform(0.0, 0.5, n), 0.0)
    f["fan_in"] = fan_in
    f["fan_out"] = fan_out
    f["transitive_dependents"] = dependents
    f["pagerank"] = weight / np.bincount(qid, weights=weight)[qid]
    f["betweenness"] = np.where((fan_in > 0) & (fan_out > 0), rng.beta(0.4, 30.0, n), 0.0)
    f["scc_size"] = np.where(rng.uniform(size=n) < 0.06, rng.integers(2, 7, n), 1).astype(np.float64)
    f["cochange_partners"] = partners
    f["cochange_max_conf"] = np.where(partners > 0, rng.uniform(0.5, 1.0, n), weak_conf)
    f["hidden_coupling"] = rng.binomial(partners.astype(np.int64), 0.5).astype(np.float64)

    # T: tests
    has_test = rng.uniform(size=n) < 0.5
    test_importers = rng.poisson(1.5, n) * has_test + rng.poisson(0.3, n)
    f["has_test_file"] = has_test.astype(np.float64)
    f["test_importers"] = test_importers.astype(np.float64)
    f["test_cochange_rate"] = np.where(test_importers > 0, rng.beta(2.0, 3.0, n), 0.0)
    f["pkg_test_ratio"] = rng.uniform(0.0, 1.5, n_queries)[qid] * rng.uniform(0.8, 1.2, n)

    # Q: self-admitted debt
    satd = rng.poisson(sloc / 400.0).astype(np.float64)
    f["satd_count"] = satd
    f["satd_density"] = 1000.0 * satd / np.maximum(sloc, 1.0)
    f["satd_age_days"] = np.where(satd > 0, age * rng.uniform(size=n), 0.0)

    # labels
    lin = (
        -3.6
        + 0.35 * np.log1p(churn_decayed)
        + 0.60 * np.log1p(prior_latent)
        + 0.35 * np.log1p(dependents)
        + 0.25 * np.log1p(cc_max)
        + 0.80 * f["dup_ratio"]
        + 0.20 * np.log1p(satd)
        + 0.30 * (f["scc_size"] > 1)
        - 0.50 * has_test
        - 0.30 * np.log1p(test_importers)
        + rng.normal(0.0, 0.5, n)
    )
    latent = rng.poisson(np.exp(lin)).astype(np.int64)
    changed = rng.uniform(size=n) < _sigmoid(-0.2 + act)
    edits = np.rint(rng.lognormal(np.log(3.0 + 0.08 * sloc) + 0.4 * act, 0.8))
    fix_churn = np.rint(rng.lognormal(np.log(6.0), 0.5, n))
    future_churn = (changed * edits + latent * fix_churn).astype(np.int64)

    repo_idx = qid // snapshots_per_repo
    snap_idx = qid % snapshots_per_repo
    local = np.tile(np.arange(rows), n_queries)
    step = timedelta(days=settings.snapshot_step_days)
    times = [SYNTHETIC_START + int(k) * step for k in range(snapshots_per_repo)]
    data: dict[str, object] = {
        "repo": pd.array([f"{SYNTHETIC_REPO_PREFIX}{r:02d}" for r in repo_idx], dtype="str"),
        "snapshot_idx": snap_idx.astype(np.int16),
        "snapshot_time": pd.DatetimeIndex([times[k] for k in snap_idx]),
        "path": pd.array([f"pkg{i // 20:02d}/module_{i:03d}.py" for i in local], dtype="str"),
    }
    for col in FEATURE_COLUMNS:
        data[col] = f[col].astype(np.float32)
    df = pd.DataFrame(data)
    df["qid"] = qid.astype(np.int32)
    df["latent_defects"] = latent.astype(np.int32)
    df["future_churn"] = future_churn.astype(np.int32)

    groups = df.groupby("qid", sort=False)
    pct = {c: groups[c[4:]].rank(pct=True, method="average").astype(np.float32) for c in PCT_COLUMNS}
    df = pd.concat([df, pd.DataFrame(pct, index=df.index)], axis=1)

    df["pain"] = _pain(df["latent_defects"], df["future_churn"], df["qid"], settings.gamma)
    df["grade"] = _grades(df["pain"], df["qid"])
    df["has_latent_defect"] = df["latent_defects"] > 0

    order = [*INDEX_COLUMNS, *FEATURE_INPUT_COLUMNS, *LABEL_COLUMNS, "qid"]
    out = df[order].sort_values(["qid", "path"], kind="stable").reset_index(drop=True)
    logger.debug("synthetic dataset: %d queries x %d rows, seed %d", n_queries, rows, seed)
    return out


def _pain(latent: pd.Series, future_churn: pd.Series, qid: pd.Series, gamma: float) -> pd.Series:
    """Pain per row: ``latent + gamma * log1p(future_churn / max(median future_churn in the query, 1))``.

    ``latent`` (latent defects) and ``future_churn`` (lines added plus deleted) are counts; the median is
    taken within each ``qid``. Returns float32 with the input index.
    """
    churn = future_churn.astype(np.float64)
    med = churn.groupby(qid).transform("median").to_numpy(dtype=np.float64)
    pain = latent.to_numpy(dtype=np.float64) + gamma * np.log1p(churn.to_numpy() / np.maximum(med, 1.0))
    return pd.Series(pain.astype(np.float32), index=latent.index, name="pain")


def _grades(pain: pd.Series, qid: pd.Series) -> pd.Series:
    """Grade 0 to 4 per row from pain, within each ``qid``.

    Grade 0 if pain is 0; otherwise the ``rank(pct=True, method="average")`` among the non-zero pains of
    the query maps ``<= 0.50`` to 1, ``<= 0.80`` to 2, ``<= 0.95`` to 3 and anything higher to 4.
    Returns int8 with the input index.
    """
    nonzero = pain != 0
    rank = pain.where(nonzero).groupby(qid).rank(pct=True, method="average")
    grade = np.select([rank <= 0.50, rank <= 0.80, rank <= 0.95], [1, 2, 3], default=4)
    grade = np.where(nonzero.to_numpy(), grade, 0)
    return pd.Series(grade.astype(np.int8), index=pain.index, name="grade")


def _sigmoid(z: _F) -> _F:
    out: _F = 1.0 / (1.0 + np.exp(-z))
    return out
