"""Helpers shared by the report writers: file order, the summary block, number formatting, tool identity.

The summary has exactly the shape of ``Analysis.summary`` in the HTTP API (plan 3.7), so the worker can store
``summarize(result.files, result.extras)`` as is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Final, TypedDict

import tdp_core
from tdp_core.contracts import BANDS, DEBT_TYPES, Band, FileExtra, FileScore

TOOL_NAME: Final = "tdp"
BAND_ORDER: Final[tuple[Band, ...]] = tuple(name for name, _ in BANDS)
HISTOGRAM_BUCKETS: Final = 10


class ReportSummary(TypedDict):
    """Counts for one analysis, the ``Analysis.summary`` object of the HTTP API.

    ``critical_hours`` is the estimated fix time of the Critical files in hours, rounded to 2 decimals.
    ``debt_types`` maps every name in ``DEBT_TYPES`` (in that order, zeros included) to the number of files
    tagged with it; a name repeated within one file counts once. ``priority_histogram`` holds 10 file counts
    for priorities 0-9, 10-19, ..., 90-100.
    """

    files: int
    critical: int
    high: int
    medium: int
    low: int
    critical_hours: float
    debt_types: dict[str, int]
    priority_histogram: list[int]


def tool_version() -> str:
    """The installed ``tdp_core`` version, read at call time (tests may monkeypatch it)."""
    return tdp_core.__version__


def ranked_files(files: Iterable[FileScore]) -> list[FileScore]:
    """Files in report order: priority descending, then path ascending (the ``AnalysisResult`` order).

    Returns a new list and never changes the input; equal paths keep their input order.
    """
    return sorted(files, key=lambda f: (-f.priority, f.path))


def summarize(files: Iterable[FileScore], extras: Mapping[str, FileExtra]) -> ReportSummary:
    """Summary counts for ``files``.

    Band counts use ``FileScore.band``; a band outside ``BANDS`` is counted in ``files`` only.
    ``critical_hours`` adds ``extras[path].cost.hours`` over the Critical files in input order, rounded to 2
    decimals; a Critical file without an entry in ``extras``, or with a non-finite cost, adds 0. Each debt
    type counts files, so a name repeated within one file counts once; names outside ``DEBT_TYPES`` are
    appended after the known ones in sorted order. A priority falls into bucket ``priority // 10``, clamped
    to 0..9, so 100 shares the last bucket with 90-99 and out of range priorities land in the first or last
    bucket. Empty input gives all zeros.
    """
    bands = dict.fromkeys(BAND_ORDER, 0)
    debt_types = dict.fromkeys(DEBT_TYPES, 0)
    unknown_types: dict[str, int] = {}
    histogram = [0] * HISTOGRAM_BUCKETS
    count = 0
    critical_hours = 0.0
    for score in files:
        count += 1
        if score.band in bands:
            bands[score.band] += 1
        if score.band == "Critical":
            extra = extras.get(score.path)
            if extra is not None and math.isfinite(extra.cost.hours):
                critical_hours += extra.cost.hours
        for name in dict.fromkeys(score.debt_types):
            if name in debt_types:
                debt_types[name] += 1
            else:
                unknown_types[name] = unknown_types.get(name, 0) + 1
        bucket = min(max(int(score.priority), 0) // 10, HISTOGRAM_BUCKETS - 1)
        histogram[bucket] += 1
    for name in sorted(unknown_types):
        debt_types[name] = unknown_types[name]
    return ReportSummary(
        files=count,
        critical=bands["Critical"],
        high=bands["High"],
        medium=bands["Medium"],
        low=bands["Low"],
        critical_hours=round(critical_hours, 2),
        debt_types=debt_types,
        priority_histogram=histogram,
    )


def format_percent(value: float) -> str:
    """A 0..1 probability as a whole percentage: 0.64 -> ``"64%"``, 1.0 -> ``"100%"``.

    Rounds half to even on the binary value (Python's ``%`` format). NaN and infinities give ``"n/a"``.
    """
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.0%}"
