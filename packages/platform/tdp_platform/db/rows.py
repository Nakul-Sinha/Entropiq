"""Rows of ``file_results``: the stored JSON shapes and their conversion from an ``AnalysisResult``.

This module is the single source of truth for what the worker writes and the API reads:

- ``features``: ``{column: float}`` for every column of ``FEATURE_INPUT_COLUMNS`` (JSONB does not keep key
  order; reorder by ``FEATURE_INPUT_COLUMNS`` when rebuilding a frame).
- ``shap``: ``{"by_family": {family: float}, "by_feature": {feature: float}}`` (``by_feature`` may be empty).
- ``reasons``: ``{"reasons": [Reason...], "mitigations": [Reason...]}`` as produced by ``to_jsonable``.
- ``extra``: ``to_jsonable(FileExtra)``; rebuild it with ``from_jsonable(FileExtra, extra)``.

Scalar columns are copied from ``FileScore``; ``cost_low``/``cost_high`` (hours) come from
``FileExtra.cost`` and ``top_reason`` is the text of the first reason ("" when there is none).
"""

from __future__ import annotations

import math
import operator
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from tdp_core.contracts import BANDS, DEBT_TYPES, FEATURE_INPUT_COLUMNS, AnalysisResult
from tdp_core.serialize import to_jsonable
from tdp_platform.db.session import json_dumps

COPY_COLUMNS: Final[tuple[str, ...]] = (
    "analysis_id",
    "path",
    "blob_sha",
    "priority",
    "band",
    "risk",
    "risk_low",
    "risk_high",
    "score",
    "cost_low",
    "cost_high",
    "debt_types",
    "top_reason",
    "features",
    "shap",
    "reasons",
    "extra",
)

_BLOB_SHA = re.compile(r"^[0-9a-f]{40}$")
_BAD_TEXT = re.compile("[\x00\ud800-\udfff]")  # Postgres text cannot hold NUL or lone surrogates
_BAND_NAMES: Final = frozenset(name for name, _ in BANDS)
_FLOAT_FIELDS: Final = ("risk", "risk_low", "risk_high", "score", "cost_low", "cost_high")


@dataclass(frozen=True, slots=True)
class FileResultRow:
    """One ``file_results`` row before insertion (the id and analysis id are assigned when stored).

    Validated on creation, raising ``ValueError`` that names the field: ``priority`` an integer 0..100
    (numpy integers accepted, bools rejected), the six float fields finite (``cost_*`` in hours),
    ``debt_types`` a subset of ``DEBT_TYPES`` (stored as a tuple), ``band`` one of the band names,
    ``blob_sha`` 40 lower case hex characters, ``path`` non-empty, and text free of NUL characters and lone
    surrogates. JSON mappings are checked when serialised (NaN raises ``ValueError`` there).
    """

    path: str
    blob_sha: str
    priority: int
    band: str
    risk: float
    risk_low: float
    risk_high: float
    score: float
    cost_low: float
    cost_high: float
    debt_types: tuple[str, ...]
    top_reason: str
    features: Mapping[str, float]
    shap: Mapping[str, Any]
    reasons: Mapping[str, Any]
    extra: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path or _BAD_TEXT.search(self.path):
            raise ValueError(f"path: invalid path {self.path!r}")
        if not isinstance(self.blob_sha, str) or not _BLOB_SHA.fullmatch(self.blob_sha):
            raise ValueError(f"blob_sha: expected 40 lower case hex characters, got {self.blob_sha!r}")
        try:
            priority = operator.index(self.priority)
        except TypeError:
            raise ValueError(f"priority: expected an integer, got {self.priority!r}") from None
        if isinstance(self.priority, bool) or not 0 <= priority <= 100:
            raise ValueError(f"priority: expected an integer 0..100, got {self.priority!r}")
        object.__setattr__(self, "priority", priority)
        if self.band not in _BAND_NAMES:
            raise ValueError(f"band: expected one of {sorted(_BAND_NAMES)}, got {self.band!r}")
        for name in _FLOAT_FIELDS:
            raw = getattr(self, name)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{name}: expected a number, got {raw!r}") from None
            if isinstance(raw, bool) or not math.isfinite(value):
                raise ValueError(f"{name}: expected a finite number, got {raw!r}")
            object.__setattr__(self, name, value)
        if isinstance(self.debt_types, str):
            raise ValueError("debt_types: expected a sequence of debt types, got a string")
        debt_types = tuple(self.debt_types)
        unknown = [d for d in debt_types if d not in DEBT_TYPES]
        if unknown:
            raise ValueError(f"debt_types: unknown debt types {unknown!r}")
        object.__setattr__(self, "debt_types", debt_types)
        if not isinstance(self.top_reason, str) or _BAD_TEXT.search(self.top_reason):
            raise ValueError("top_reason: expected text without NUL characters")


def rows_from_result(
    result: AnalysisResult, *, shap_by_feature: Mapping[str, Mapping[str, float]] | None = None
) -> list[FileResultRow]:
    """One row per entry of ``result.files``, in that order (priority desc, then path).

    Stored ids follow this order, so ties in a ranking keep path order. ``shap_by_feature`` optionally maps
    a path to per-feature SHAP values. Raises ``ValueError`` when a file has no ``extras`` entry, no row in
    ``result.features`` (indexed by path), or the frame lacks a ``FEATURE_INPUT_COLUMNS`` column; extra
    columns are ignored.
    """
    frame = result.features
    missing_cols = [c for c in FEATURE_INPUT_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"features frame lacks columns {missing_cols[:5]!r} ({len(missing_cols)} in total)")
    if not frame.index.is_unique:
        raise ValueError("features frame has duplicate paths in its index")
    paths = [fs.path for fs in result.files]
    positions = frame.index.get_indexer(pd.Index(paths))
    values = frame.loc[:, list(FEATURE_INPUT_COLUMNS)].to_numpy(dtype="float64")
    by_feature = shap_by_feature or {}
    rows: list[FileResultRow] = []
    for fs, pos in zip(result.files, positions, strict=True):
        extra = result.extras.get(fs.path)
        if extra is None:
            raise ValueError(f"no extras for {fs.path!r}")
        if pos < 0:
            raise ValueError(f"no feature row for {fs.path!r}")
        features = dict(zip(FEATURE_INPUT_COLUMNS, values[pos].tolist(), strict=True))
        rows.append(
            FileResultRow(
                path=fs.path,
                blob_sha=extra.blob_sha,
                priority=fs.priority,
                band=fs.band,
                risk=fs.risk,
                risk_low=fs.risk_low,
                risk_high=fs.risk_high,
                score=fs.score,
                cost_low=extra.cost.low,
                cost_high=extra.cost.high,
                debt_types=tuple(fs.debt_types),
                top_reason=fs.reasons[0].text if fs.reasons else "",
                features=features,
                shap={
                    "by_family": {str(k): float(v) for k, v in fs.shap_by_family.items()},
                    "by_feature": {str(k): float(v) for k, v in by_feature.get(fs.path, {}).items()},
                },
                reasons={"reasons": to_jsonable(fs.reasons), "mitigations": to_jsonable(fs.mitigations)},
                extra=to_jsonable(extra),
            )
        )
    return rows


def copy_records(analysis_id: uuid.UUID, rows: Sequence[FileResultRow]) -> list[tuple[Any, ...]]:
    """Records for asyncpg ``copy_records_to_table`` in ``COPY_COLUMNS`` order.

    JSONB columns are JSON text (the codec SQLAlchemy installs on asyncpg connections takes text),
    ``debt_types`` a list of strings, ``analysis_id`` a UUID. Raises ``ValueError`` for duplicate paths or
    JSON that cannot be stored (NaN or infinite numbers) before anything reaches the database.
    """
    if not isinstance(analysis_id, uuid.UUID):
        raise ValueError(f"analysis_id: expected a UUID, got {analysis_id!r}")
    seen: set[str] = set()
    records: list[tuple[Any, ...]] = []
    for row in rows:
        if row.path in seen:
            raise ValueError(f"duplicate path {row.path!r}")
        seen.add(row.path)
        records.append(
            (
                analysis_id,
                row.path,
                row.blob_sha,
                row.priority,
                row.band,
                row.risk,
                row.risk_low,
                row.risk_high,
                row.score,
                row.cost_low,
                row.cost_high,
                list(row.debt_types),
                row.top_reason,
                json_dumps(dict(row.features)),
                json_dumps(dict(row.shap)),
                json_dumps(dict(row.reasons)),
                json_dumps(dict(row.extra)),
            )
        )
    return records
