"""JSON report: the whole ``AnalysisResult`` plus a format header, the summary and per-file features.

The document is the ``to_jsonable`` form of the result. When every number in the result is finite,
``from_jsonable(AnalysisResult, json.loads(text))`` rebuilds every field except ``features`` (an empty
DataFrame; the values live in each file entry). A NaN or infinity is written as ``null``, which
``from_jsonable`` rejects for a float field. Keys, in order: ``format`` (``"tdp-report"``),
``format_version``, ``tool`` (``name``, ``version``), ``sha``, ``analyzed_at`` (ISO 8601 UTC ending in ``Z``),
``model_version``, ``extractor_version``, ``summary`` (see ``tdp_core.report.common.summarize``), ``files``
(each ``FileScore`` plus a ``features`` object), ``extras`` (by path, sorted), ``graph``, ``plan`` and
``stats``.

``format_version`` changes only when an existing key changes meaning or disappears; new keys keep it. Every
NaN or infinity becomes ``null``, and the text is strict, pure ASCII JSON.
"""

from __future__ import annotations

import json
import math
from typing import Any, Final

import numpy as np
import pandas as pd

from tdp_core.contracts import AnalysisResult
from tdp_core.report.common import TOOL_NAME, summarize, tool_version
from tdp_core.serialize import to_jsonable

JSON_FORMAT: Final = "tdp-report"
JSON_FORMAT_VERSION: Final = 1


def to_json_dict(result: AnalysisResult) -> dict[str, Any]:
    """The JSON report as fresh Python values (dicts, lists, str, int, float, bool, None).

    ``files`` keeps the order of ``result.files``. Each entry is the serialised ``FileScore`` plus
    ``features``: that path's row of ``result.features`` (index = path) as ``{column: value}`` in column
    order, or ``{}`` when the frame has no row for it. Feature values become ``int``, ``bool``, ``float`` or
    ``None`` (missing values, NaN and infinities); float32 values keep their shortest decimal form, so
    ``0.1`` stays ``0.1``. Raises ``ValueError`` when the feature index or the column names (as strings) are
    not unique or ``analyzed_at`` is naive, and ``TypeError`` for a feature value that is not a number,
    boolean or missing value, or for ``stats`` values that ``to_jsonable`` cannot serialise.
    """
    features = _feature_rows(result.features)
    files: list[dict[str, Any]] = []
    for score in result.files:
        entry: dict[str, Any] = to_jsonable(score)
        entry["features"] = features.get(score.path, {})
        files.append(entry)
    document: dict[str, Any] = {
        "format": JSON_FORMAT,
        "format_version": JSON_FORMAT_VERSION,
        "tool": {"name": TOOL_NAME, "version": tool_version()},
        "sha": result.sha,
        "analyzed_at": to_jsonable(result.analyzed_at),
        "model_version": result.model_version,
        "extractor_version": result.extractor_version,
        "summary": dict(summarize(result.files, result.extras)),
        "files": files,
        "extras": {path: to_jsonable(result.extras[path]) for path in sorted(result.extras)},
        "graph": to_jsonable(result.graph),
        "plan": to_jsonable(result.plan),
        "stats": to_jsonable(result.stats),
    }
    cleaned: dict[str, Any] = _finite(document)
    return cleaned


def to_json(result: AnalysisResult, *, indent: int | None = 2) -> str:
    """The JSON report as text: ``to_json_dict`` dumped as strict ASCII JSON, ending with one newline.

    ``indent=None`` writes a single line. Non-ASCII characters are written as ``\\u`` escapes, so the text is
    safe on any console; ``allow_nan=False`` holds because non-finite numbers are already ``null``.
    """
    return json.dumps(to_json_dict(result), indent=indent, ensure_ascii=True, allow_nan=False) + "\n"


def _feature_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if not frame.index.is_unique:
        raise ValueError("the features index must hold each path once")
    columns = [str(column) for column in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("feature column names must be unique")
    values = [[_feature_value(v) for v in frame.iloc[:, j].to_numpy()] for j in range(len(columns))]
    return {
        str(path): {column: values[j][i] for j, column in enumerate(columns)}
        for i, path in enumerate(frame.index)
    }


def _feature_value(value: object) -> bool | int | float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (np.float16, np.float32)):
        number = float(str(value))
    elif isinstance(value, (float, np.floating)):
        number = float(value)
    else:
        raise TypeError(f"feature values must be numbers, booleans or missing, got {type(value).__name__}")
    return number if math.isfinite(number) else None


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value
