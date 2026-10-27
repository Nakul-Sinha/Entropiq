"""Per-query result files and Markdown tables.

The per-query file format (Parquet, one row per test query and run) has the optional run columns
:data:`RUN_COLUMNS` (``model``, ``protocol``, ``split``), then :data:`QUERY_COLUMNS`, then the
:data:`RANKING_METRICS` columns, then the :data:`PROBABILITY_METRICS` columns when probabilities were
evaluated. :func:`tdp_ml.eval.harness.run_protocol` produces it; several runs can be concatenated in one
file as long as each ``(model, protocol)`` group has unique qids.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from tdp_core.paths import results_dir
from tdp_ml.eval.metrics import GROUP_COLUMNS, METRIC_LABELS, METRICS

logger = logging.getLogger(__name__)

_NAME: Final = re.compile(r"[A-Za-z0-9_.-]+")


def per_query_path(prefix: str, protocol: str, *, root: Path | None = None) -> Path:
    """Path of a per-query result file: ``<root>/<prefix>_<protocol>.parquet``.

    ``root`` defaults to :func:`tdp_core.paths.results_dir` (``TDP_RESULTS_DIR``). For example
    ``per_query_path("baselines", "lopo")`` is ``results/baselines_lopo.parquet``.
    Raises ``ValueError`` when ``prefix`` or ``protocol`` is not made of letters, digits, ``_``, ``.`` and
    ``-``.
    """
    for name, value in (("prefix", prefix), ("protocol", protocol)):
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise ValueError(f"{name} must match [A-Za-z0-9_.-]+, got {value!r}")
    return (root if root is not None else results_dir()) / f"{prefix}_{protocol}.parquet"


def _check_per_query(per_query: pd.DataFrame) -> None:
    missing = [c for c in ("qid", "repo") if c not in per_query.columns]
    if missing:
        raise ValueError(f"per-query frame is missing the columns {missing}")
    if not any(m in per_query.columns for m in METRICS):
        raise ValueError("per-query frame has no metric column")
    keys = [c for c in GROUP_COLUMNS if c in per_query.columns] + ["qid"]
    if per_query.duplicated(keys).any():
        raise ValueError(f"per-query frame has duplicate qids within a run (keys {keys})")


def save_per_query(per_query: pd.DataFrame, path: Path) -> Path:
    """Write a per-query frame to Parquet atomically and return ``path``.

    The parent folder is created. The frame is written to ``<name>.tmp`` next to ``path`` and then moved
    into place, so readers never see a partial file; the temporary file is removed on failure.
    Raises ``ValueError`` without ``qid`` or ``repo``, without any metric column, or when a qid repeats
    within a ``(model, protocol)`` run.
    """
    _check_per_query(per_query)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        per_query.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("wrote %d per-query rows to %s", len(per_query), path)
    return path


def load_per_query(path: Path) -> pd.DataFrame:
    """Read a per-query file written by :func:`save_per_query`, with the same checks."""
    frame = pd.read_parquet(path)
    _check_per_query(frame)
    return frame


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def _number(value: float, digits: int) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _metric_order(present: Sequence[str]) -> list[str]:
    known = [m for m in METRICS if m in present]
    return known + [m for m in dict.fromkeys(present) if m not in known]


def summary_markdown(summary: pd.DataFrame, *, metrics: Sequence[str] | None = None, digits: int = 3) -> str:
    """Markdown table of a :func:`tdp_ml.eval.stats.summarize` result.

    One row per group (``model`` and ``protocol`` when present), one column per metric headed by its
    display label (for example ``NDCG@10``). Cells read ``mean [ci_low, ci_high]`` with ``digits``
    decimals; a NaN mean or a missing metric prints ``n/a`` and a NaN interval prints the mean alone.
    ``metrics`` defaults to the metrics of ``summary`` in canonical order. ``|`` in names is escaped.
    Raises ``ValueError`` when a summary column is missing or there is no metric to show.
    """
    missing = [c for c in ("metric", "mean", "ci_low", "ci_high") if c not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing the columns {missing}")
    chosen = list(metrics) if metrics is not None else _metric_order(summary["metric"].astype(str).tolist())
    if not chosen:
        raise ValueError("summary has no metric to show")
    group_cols = [c for c in GROUP_COLUMNS if c in summary.columns]
    header = [_escape(c) for c in group_cols] + [_escape(METRIC_LABELS.get(m, m)) for m in chosen]
    rows: list[list[str]] = []
    groups = summary.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), summary)]
    for key, part in groups:
        keys = key if isinstance(key, tuple) else (key,)
        by_metric = {str(r["metric"]): r for r in part.to_dict("records")}
        cells = [_escape(k) for k in keys]
        for m in chosen:
            r = by_metric.get(m)
            mean = math.nan if r is None else float(r["mean"])
            low = math.nan if r is None else float(r["ci_low"])
            high = math.nan if r is None else float(r["ci_high"])
            if math.isnan(mean):
                cells.append("n/a")
            elif math.isnan(low) or math.isnan(high):
                cells.append(_number(mean, digits))
            else:
                cells.append(f"{_number(mean, digits)} [{_number(low, digits)}, {_number(high, digits)}]")
        rows.append(cells)
    return _table(header, rows)


def _label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value) == "":
        return "n/a"
    return _escape(value)


def _p_value(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3g}"


def comparison_markdown(tests: pd.DataFrame, *, digits: int = 3) -> str:
    """Markdown table of a :func:`tdp_ml.eval.stats.wilcoxon_holm` result, one row per baseline.

    Columns: metric label, baseline, n, mean (model), mean (baseline), mean diff, p, p (Holm), Cliff's
    delta and magnitude. Means and deltas use ``digits`` decimals, p-values three significant digits, NaN
    prints ``n/a``. An empty input gives the header only. Raises ``ValueError`` when a column is missing.
    """
    needed = ("metric", "baseline", "n", "mean_a", "mean_b", "mean_diff", "p_value", "p_holm")
    missing = [c for c in (*needed, "cliffs_delta", "magnitude") if c not in tests.columns]
    if missing:
        raise ValueError(f"tests are missing the columns {missing}")
    header = [
        "Metric",
        "Baseline",
        "n",
        "Mean (model)",
        "Mean (baseline)",
        "Mean diff",
        "p",
        "p (Holm)",
        "Cliff's delta",
        "Magnitude",
    ]
    rows = [
        [
            _escape(METRIC_LABELS.get(str(r["metric"]), str(r["metric"]))),
            _escape(r["baseline"]),
            str(int(r["n"])),
            _number(float(r["mean_a"]), digits),
            _number(float(r["mean_b"]), digits),
            _number(float(r["mean_diff"]), digits),
            _p_value(float(r["p_value"])),
            _p_value(float(r["p_holm"])),
            _number(float(r["cliffs_delta"]), digits),
            _label(r["magnitude"]),
        ]
        for r in tests.to_dict("records")
    ]
    return _table(header, rows)
