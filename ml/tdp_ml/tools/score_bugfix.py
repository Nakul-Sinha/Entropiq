"""Score the bug-fix detector against hand labels from a ``sample_bugfix`` sample.

Every labelled commit stands for ``N_s / labelled_s`` commits of its stratum s (stratum population over
labelled rows in it), so precision and recall are estimates for the whole eligible population, not just
the sample. Labelling a prefix of the shuffled rows (rows 1..K in order) is a valid random subsample of every
stratum, so a partly labelled sample can be scored as long as each stratum has at least one label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tdp_ml.tools.sample_bugfix import KEY_COLUMNS, STRATA


@dataclass(frozen=True)
class StratumScore:
    stratum: str
    population: int  # eligible commits in the stratum (N_s)
    labelled: int
    bugs: int  # labelled rows marked as bug fixes
    rate: float  # bugs / labelled
    ci_low: float  # Wilson 95% interval of rate
    ci_high: float


@dataclass(frozen=True)
class BugfixScore:
    labelled: int
    precision: float  # NaN when nothing is predicted
    recall: float  # NaN when no labelled bug fix exists
    f1: float
    strata: tuple[StratumScore, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready dict; NaN becomes None."""

        def clean(value: Any) -> Any:
            if isinstance(value, float) and math.isnan(value):
                return None
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list | tuple):
                return [clean(v) for v in value]
            return value

        result: dict[str, Any] = clean(asdict(self))
        return result


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; ``(nan, nan)`` when ``n == 0``."""
    if n == 0:
        return math.nan, math.nan
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in 0..{n}, got {successes}")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def read_labels(sample_csv: Path) -> pd.DataFrame:
    """Hand labels from a sample CSV: columns ``sha`` and ``label`` (int 0/1), unlabelled rows dropped.

    Labels are stripped; anything other than ``0``, ``1`` or empty raises ``ValueError`` naming the rows.
    """
    frame = pd.read_csv(sample_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [c for c in ("sha", "label") if c not in frame.columns]
    if missing:
        raise ValueError(f"{sample_csv} lacks columns: {', '.join(missing)}")
    labels = frame["label"].str.strip()
    rows = frame["row"] if "row" in frame.columns else pd.Series(range(1, len(frame) + 1)).astype(str)
    bad = [str(r) for r, lb in zip(rows, labels, strict=True) if lb not in ("", "0", "1")]
    if bad:
        raise ValueError(f"labels must be 0, 1 or empty; bad rows: {', '.join(bad)}")
    keep = labels != ""
    out = pd.DataFrame({"sha": frame.loc[keep, "sha"].str.strip(), "label": labels[keep].astype(int)})
    if out["sha"].duplicated().any():
        raise ValueError(f"{sample_csv} labels the same commit twice")
    return out.reset_index(drop=True)


def read_key(key_csv: Path) -> pd.DataFrame:
    """The key file of a sample (``KEY_COLUMNS``; sizes and ``predicted`` as ints)."""
    frame = pd.read_csv(key_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = [c for c in KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{key_csv} lacks columns: {', '.join(missing)}")
    frame = frame[list(KEY_COLUMNS)].copy()
    for column in ("stratum_size", "sample_size", "predicted"):
        frame[column] = frame[column].astype(int)
    return frame


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def score_bugfix(
    labels: pd.DataFrame,
    key: pd.DataFrame,
    predicted: Mapping[str, bool] | None = None,
) -> BugfixScore:
    """Design-weighted precision, recall and F1 of the detector, plus per-stratum label rates.

    ``labels`` has ``sha`` and ``label`` (0/1, from ``read_labels``), ``key`` the sample key (``read_key``).
    A labelled row in stratum s weighs ``N_s / labelled_s``. The prediction is ``key.predicted`` or, to
    re-score a newer detector on the same labels, ``predicted[sha]``. With TP, PP and AP the weighted sums
    of prediction and label, prediction and label: precision = TP/PP, recall = TP/AP, F1 = 2TP/(PP+AP); NaN
    when a denominator is 0. Strata keep the sampling design's grouping even when re-scoring.

    Raises ``ValueError`` when a label's sha is not in the key, a stratum in the key has no label, the key
    gives two sizes for one stratum, or ``predicted`` lacks a labelled sha.
    """
    unknown = sorted(set(labels["sha"]) - set(key["sha"]))
    if unknown:
        raise ValueError(f"labelled commits missing from the key: {', '.join(unknown[:5])}")
    merged = labels.merge(key, on="sha", how="left", validate="one_to_one")
    if predicted is not None:
        missing = [sha for sha in merged["sha"] if sha not in predicted]
        if missing:
            raise ValueError(f"predicted lacks labelled commits: {', '.join(missing[:5])}")
        merged["p"] = [int(bool(predicted[sha])) for sha in merged["sha"]]
    else:
        merged["p"] = merged["predicted"].astype(int)

    names = [s for s in STRATA if s in set(key["stratum"])]
    names += sorted(set(key["stratum"]) - set(STRATA))
    tp = pp = ap = 0.0
    strata: list[StratumScore] = []
    for stratum in names:
        sizes = set(key.loc[key["stratum"] == stratum, "stratum_size"])
        if len(sizes) != 1:
            raise ValueError(f"key gives several sizes for stratum {stratum!r}: {sorted(sizes)}")
        population = int(sizes.pop())
        rows = merged[merged["stratum"] == stratum]
        labelled = len(rows)
        if population > 0 and labelled == 0:
            raise ValueError(f"stratum {stratum!r} has no labelled commit; label at least one")
        bugs = int(rows["label"].sum())
        if labelled:
            weight = population / labelled
            tp += weight * float((rows["p"] * rows["label"]).sum())
            pp += weight * float(rows["p"].sum())
            ap += weight * bugs
        low, high = wilson_interval(bugs, labelled)
        strata.append(StratumScore(stratum, population, labelled, bugs, _ratio(bugs, labelled), low, high))
    return BugfixScore(
        labelled=len(merged),
        precision=_ratio(tp, pp),
        recall=_ratio(tp, ap),
        f1=_ratio(2 * tp, pp + ap),
        strata=tuple(strata),
    )
