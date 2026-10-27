"""Leakage-safe train/test splits over a dataset of (repository, snapshot, file) rows.

Labels of a snapshot at time T look at (T, T + H], so a training row is only safe for testing at time V
when its label window has closed by then: ``T + H <= V``. Every split here applies this embargo per
repository, which is the rule of the temporal protocol; LOPO (leave one project out) holds out a whole
repository instead, so no embargo is needed there.

Every function returns index *labels* of ``df`` (``df.index[mask]``) in ``df`` order. ``snapshot_time`` is
read with ``pd.to_datetime(utc=True)``, so naive times are taken as UTC.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _snapshot_times(df: pd.DataFrame) -> pd.Series:
    missing = sorted({"repo", "snapshot_time"} - set(df.columns))
    if missing:
        raise ValueError(f"splits need the columns {missing}")
    if not df.index.is_unique:
        raise ValueError("splits need a unique index")
    for col in ("repo", "snapshot_time"):
        if df[col].isna().any():
            raise ValueError(f"column {col!r} contains nulls")
    return pd.to_datetime(df["snapshot_time"], utc=True)


def _horizon(horizon_days: int) -> pd.Timedelta:
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int | np.integer) or horizon_days < 0:
        raise ValueError(f"horizon_days must be an int >= 0, got {horizon_days!r}")
    return pd.Timedelta(days=int(horizon_days))


def _labels(df: pd.DataFrame, mask: pd.Series) -> pd.Index:
    return df.index[mask.to_numpy(dtype=bool)]


def temporal_split(df: pd.DataFrame, horizon_days: int) -> tuple[pd.Index, pd.Index]:
    """Split into ``(train, test)``: test is each repository's last snapshot, train its embargoed past.

    For a repository whose last snapshot time is ``L``, a row at time ``T`` is a test row when ``T == L``
    and a training row when ``T < L`` and ``T + horizon_days <= L`` (its label window ends before the test
    snapshot). The strict ``T < L`` matters only for ``horizon_days == 0``. A repository with a single
    snapshot contributes test rows only; the training set may be empty.

    ``horizon_days`` is the label horizon H in days. Returns two index-label arrays in ``df`` order.
    Raises ``ValueError`` when ``repo`` or ``snapshot_time`` is missing or has nulls, the index is not
    unique, or ``horizon_days`` is not an int >= 0.
    """
    horizon = _horizon(horizon_days)
    times = _snapshot_times(df)
    last = times.groupby(df["repo"], sort=False).transform("max")
    test = times == last
    train = (times < last) & (times + horizon <= last)
    return _labels(df, train), _labels(df, test)


def forward_folds(df: pd.DataFrame, horizon_days: int, n_folds: int = 2) -> list[tuple[pd.Index, pd.Index]]:
    """Forward-chaining ``(train, valid)`` folds for tuning inside a training set, oldest fold first.

    Within each repository the distinct snapshot times are ranked from the newest (rank 1). Fold ``j``
    (``j = 0 .. n_folds - 1``) validates on the snapshot with rank ``n_folds - j`` of every repository that
    has one, so the last fold validates on the newest snapshot. Its training rows are the same
    repository's rows with ``T < V`` and ``T + horizon_days <= V`` (the embargo of :func:`temporal_split`),
    where ``V`` is that repository's validation time; repositories too short for a fold take no part in it.

    A fold whose training or validation set is empty is skipped with a warning, so the result can hold
    fewer than ``n_folds`` folds (possibly none). Pass the training rows of :func:`temporal_split`.

    Raises ``ValueError`` for ``n_folds < 1`` and for the input problems listed in :func:`temporal_split`.
    """
    horizon = _horizon(horizon_days)
    if isinstance(n_folds, bool) or not isinstance(n_folds, int | np.integer) or n_folds < 1:
        raise ValueError(f"n_folds must be an int >= 1, got {n_folds!r}")
    times = _snapshot_times(df)
    repo = df["repo"]
    rank = times.groupby(repo, sort=False).rank(method="dense", ascending=False)
    folds: list[tuple[pd.Index, pd.Index]] = []
    for j in range(int(n_folds)):
        back = int(n_folds) - 1 - j
        valid_time = times.where(rank == back + 1).groupby(repo, sort=False).transform("max")
        valid = times == valid_time
        train = valid_time.notna() & (times < valid_time) & (times + horizon <= valid_time)
        if not train.any() or not valid.any():
            logger.warning(
                "forward fold %d of %d skipped: %d training rows, %d validation rows",
                j,
                n_folds,
                int(train.sum()),
                int(valid.sum()),
            )
            continue
        folds.append((_labels(df, train), _labels(df, valid)))
    return folds


def lopo_splits(df: pd.DataFrame) -> Iterator[tuple[str, pd.Index, pd.Index]]:
    """Leave-one-project-out splits ``(held_out_repo, train, test)``, repositories in sorted order.

    For each repository, test is its last snapshot and train is every row of every other repository. A
    split with an empty training set (a dataset with one repository) is skipped with a warning. The input
    is validated eagerly, when the function is called.

    Raises ``ValueError`` when ``repo`` or ``snapshot_time`` is missing or has nulls, or the index is not
    unique.
    """
    times = _snapshot_times(df)
    repo = df["repo"].astype("str")
    last = times.groupby(repo, sort=False).transform("max")
    splits: list[tuple[str, pd.Index, pd.Index]] = []
    for name in sorted(repo.unique().tolist()):
        held_out = repo == name
        train = ~held_out
        test = held_out & (times == last)
        if not train.any():
            logger.warning("LOPO split for %s skipped: no other repository to train on", name)
            continue
        splits.append((str(name), _labels(df, train), _labels(df, test)))
    return iter(splits)
