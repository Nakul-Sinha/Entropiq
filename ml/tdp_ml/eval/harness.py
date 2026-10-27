"""Run a scoring model under an evaluation protocol and collect per-query metrics.

Every model (baselines, the ranker, ablations) goes through :func:`run_protocol`, so all of them see the
same splits and produce per-query frames in the same format; paired tests can then match them on ``qid``.

Protocols:

- ``temporal``: one split; test is each repository's last snapshot, train is the embargoed past of every
  repository (:func:`tdp_ml.eval.splits.temporal_split`).
- ``lopo``: one split per repository; test is its last snapshot, train is every other repository
  (:func:`tdp_ml.eval.splits.lopo_splits`).

Both protocols test the same queries (each repository's last snapshot), so their qids line up.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import LABEL_COLUMNS
from tdp_ml.eval.metrics import evaluate
from tdp_ml.eval.splits import lopo_splits, temporal_split

logger = logging.getLogger(__name__)

ProtocolName = Literal["temporal", "lopo"]
PROTOCOLS: Final = ("temporal", "lopo")

FitPredict = Callable[[pd.DataFrame, pd.DataFrame], npt.ArrayLike]
"""``fit_predict(train, test) -> scores``: fit on the training rows, return one score per test row."""


@dataclass(frozen=True, eq=False, slots=True)
class Split:
    """One train/test split: ``name`` is ``"temporal"`` or the held-out repository slug."""

    name: str
    train: pd.Index
    test: pd.Index


@dataclass(frozen=True, eq=False, slots=True)
class ProtocolRun:
    """Result of :func:`run_protocol`.

    ``per_query`` has the columns ``model``, ``protocol``, ``split``, then the :func:`evaluate` columns, one
    row per test query sorted by qid. ``scores`` holds the model score (float64, named ``score``) of every
    test row, indexed by the row labels of the input frame in input order.
    """

    protocol: str
    model: str
    per_query: pd.DataFrame
    scores: pd.Series


def protocol_splits(
    df: pd.DataFrame, protocol: str, *, horizon_days: int = DEFAULT_SETTINGS.horizon_days
) -> list[Split]:
    """The splits of ``protocol`` (``"temporal"`` or ``"lopo"``) over ``df``.

    ``horizon_days`` is the label horizon used for the temporal embargo. Splits may have an empty training
    set (for example a temporal split of single-snapshot repositories); :func:`run_protocol` drops them.
    Raises ``ValueError`` for an unknown protocol and for the input problems of the split functions.
    """
    if protocol == "temporal":
        train, test = temporal_split(df, horizon_days)
        return [Split("temporal", train, test)]
    if protocol == "lopo":
        return [Split(repo, train, test) for repo, train, test in lopo_splits(df)]
    raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def _as_scores(raw: npt.ArrayLike, test: pd.DataFrame, split: str) -> npt.NDArray[np.float64]:
    if isinstance(raw, pd.Series):
        if not raw.index.is_unique or not raw.index.sort_values().equals(test.index.sort_values()):
            raise ValueError(f"split {split}: a score Series must be indexed by the test rows")
        raw = raw.reindex(test.index)
    scores = np.asarray(raw, dtype=np.float64)
    if scores.shape != (len(test),):
        raise ValueError(f"split {split}: expected {len(test)} scores, got shape {scores.shape}")
    if np.isnan(scores).any():
        raise ValueError(f"split {split}: scores contain NaN")
    return scores


def run_protocol(
    df: pd.DataFrame,
    protocol: str,
    fit_predict: FitPredict,
    *,
    horizon_days: int = DEFAULT_SETTINGS.horizon_days,
    model: str = "model",
) -> ProtocolRun:
    """Fit and score a model on every split of ``protocol`` and evaluate each test query.

    For each split with non-empty train and test sets, ``fit_predict(train, test)`` gets the training rows
    (all columns) and the test rows **without the label columns** (``LABEL_COLUMNS`` are dropped, so a
    model cannot peek at the answers), and must return one finite-or-infinite, non-NaN score per test row:
    an array in test-row order, or a Series indexed by the test row labels. Higher scores mean more
    expected pain. The scores are evaluated with :func:`evaluate` against the true labels.

    ``df`` needs the dataset columns used by :func:`evaluate` and the splits. ``horizon_days`` is the
    temporal embargo; ``model`` names the run in the output.

    Raises ``ValueError`` for an unknown protocol, when no split has both training and test rows, when the
    scores have the wrong length or contain NaN, and for the input problems of :func:`evaluate`.
    """
    usable = [
        s for s in protocol_splits(df, protocol, horizon_days=horizon_days) if len(s.train) and len(s.test)
    ]
    if not usable:
        raise ValueError(f"protocol {protocol!r} has no split with both training and test rows")
    frames: list[pd.DataFrame] = []
    parts: list[pd.Series] = []
    for split in usable:
        train = df.loc[split.train]
        test = df.loc[split.test]
        hidden = test.drop(columns=[c for c in LABEL_COLUMNS if c in test.columns])
        logger.info(
            "%s %s split %s: %d training rows, %d test rows",
            model,
            protocol,
            split.name,
            len(train),
            len(test),
        )
        scores = _as_scores(fit_predict(train, hidden), test, split.name)
        per_query = evaluate(test.assign(score=scores), "score")
        per_query.insert(0, "split", split.name)
        frames.append(per_query)
        parts.append(pd.Series(scores, index=split.test))
    result = pd.concat(frames, ignore_index=True).sort_values("qid", kind="stable").reset_index(drop=True)
    result.insert(0, "protocol", protocol)
    result.insert(0, "model", model)
    for col in ("model", "protocol", "split"):
        result[col] = result[col].astype("str")
    scored = pd.concat(parts)
    order = df.index[df.index.isin(scored.index)]
    all_scores = scored.reindex(order).astype("float64").rename("score")
    return ProtocolRun(protocol=protocol, model=model, per_query=result, scores=all_scores)
