from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tdp_core.contracts import FEATURE_INPUT_COLUMNS, AnalysisResult, FileExtra
from tdp_core.serialize import from_jsonable, to_jsonable
from tdp_platform.db.rows import COPY_COLUMNS, FileResultRow, copy_records, rows_from_result
from tdp_platform.db.session import json_dumps

MakeRow = Callable[..., FileResultRow]


def _row_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": "pkg/a.py",
        "blob_sha": "0" * 40,
        "priority": 50,
        "band": "Medium",
        "risk": 0.5,
        "risk_low": 0.4,
        "risk_high": 0.6,
        "score": 1.25,
        "cost_low": 1.0,
        "cost_high": 2.0,
        "debt_types": ("test",),
        "top_reason": "Few tests",
        "features": {"sloc": 10.0},
        "shap": {"by_family": {}, "by_feature": {}},
        "reasons": {"reasons": [], "mitigations": []},
        "extra": {},
    }
    kwargs.update(overrides)
    return kwargs


def test_row_coerces_numpy_values() -> None:
    row = FileResultRow(
        **_row_kwargs(
            priority=np.int64(87), band="High", risk=np.float32(0.5), debt_types=["test", "complexity"]
        )
    )
    assert row.priority == 87 and type(row.priority) is int
    assert row.risk == 0.5 and type(row.risk) is float
    assert row.debt_types == ("test", "complexity")


@pytest.mark.parametrize(
    "overrides",
    [
        {"blob_sha": "abc"},
        {"blob_sha": "A" * 40},
        {"blob_sha": "g" * 40},
        {"priority": -1},
        {"priority": 101},
        {"priority": True},
        {"priority": 3.5},
        {"priority": "50"},
        {"band": "Severe"},
        {"risk": float("nan")},
        {"cost_high": float("inf")},
        {"score": "high"},
        {"risk_low": True},
        {"debt_types": ("bogus",)},
        {"debt_types": "test"},
        {"path": ""},
        {"path": "a\x00b"},
        {"top_reason": "bad\x00"},
    ],
)
def test_row_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    field = next(iter(overrides))
    with pytest.raises(ValueError, match=field):
        FileResultRow(**_row_kwargs(**overrides))


def test_rows_from_result_maps_every_field(analysis_result: AnalysisResult) -> None:
    rows = rows_from_result(analysis_result)
    assert [r.path for r in rows] == [f.path for f in analysis_result.files]
    for row, fs in zip(rows, analysis_result.files, strict=True):
        extra = analysis_result.extras[fs.path]
        assert row.blob_sha == extra.blob_sha
        assert (row.priority, row.band) == (fs.priority, fs.band)
        assert (row.risk, row.risk_low, row.risk_high, row.score) == (
            fs.risk,
            fs.risk_low,
            fs.risk_high,
            fs.score,
        )
        assert (row.cost_low, row.cost_high) == (extra.cost.low, extra.cost.high)
        assert row.debt_types == fs.debt_types
        assert row.top_reason == (fs.reasons[0].text if fs.reasons else "")
        assert list(row.features) == list(FEATURE_INPUT_COLUMNS)
        frame = analysis_result.features.loc[[fs.path], list(FEATURE_INPUT_COLUMNS)]
        assert list(row.features.values()) == frame.to_numpy(dtype=np.float64)[0].tolist()
        assert all(type(v) is float for v in row.features.values())
        assert row.shap == {"by_family": dict(fs.shap_by_family), "by_feature": {}}
        assert row.reasons == {"reasons": to_jsonable(fs.reasons), "mitigations": to_jsonable(fs.mitigations)}
        assert from_jsonable(FileExtra, row.extra) == extra
        json.loads(json_dumps(dict(row.extra)))  # storable
    assert rows[-1].top_reason == ""  # the last file has no reasons


def test_rows_from_result_with_feature_shap(analysis_result: AnalysisResult) -> None:
    rows = rows_from_result(
        analysis_result,
        shap_by_feature={"pkg/core.py": {"cc_max": np.float32(0.25)}},  # type: ignore[dict-item]
    )
    by_path = {r.path: r for r in rows}
    assert by_path["pkg/core.py"].shap["by_feature"] == {"cc_max": 0.25}
    assert by_path["pkg/util.py"].shap["by_feature"] == {}


def test_rows_from_result_needs_extras_and_features(analysis_result: AnalysisResult) -> None:
    extras = dict(analysis_result.extras)
    del extras["app.py"]
    with pytest.raises(ValueError, match="extras"):
        rows_from_result(dataclasses.replace(analysis_result, extras=extras))
    with pytest.raises(ValueError, match="columns"):
        rows_from_result(
            dataclasses.replace(analysis_result, features=analysis_result.features.drop(columns="sloc"))
        )
    with pytest.raises(ValueError, match="feature row"):
        rows_from_result(
            dataclasses.replace(analysis_result, features=analysis_result.features.drop(index="app.py"))
        )
    frame = analysis_result.features.copy()
    frame["unused"] = 1.0
    assert len(rows_from_result(dataclasses.replace(analysis_result, features=frame))) == 4


def test_copy_records(analysis_result: AnalysisResult) -> None:
    rows = rows_from_result(analysis_result)
    aid = uuid.uuid4()
    records = copy_records(aid, rows)
    assert len(records) == len(rows)
    for record, row in zip(records, rows, strict=True):
        assert len(record) == len(COPY_COLUMNS)
        values = dict(zip(COPY_COLUMNS, record, strict=True))
        assert values["analysis_id"] == aid
        assert values["path"] == row.path
        assert values["priority"] == row.priority
        assert values["debt_types"] == list(row.debt_types)
        for name in ("features", "shap", "reasons", "extra"):
            assert isinstance(values[name], str)
            assert values[name] == json_dumps(dict(getattr(row, name)))
        for name in ("risk", "risk_low", "risk_high", "score", "cost_low", "cost_high"):
            assert type(values[name]) is float
    assert "id" not in COPY_COLUMNS
    assert copy_records(aid, []) == []


def test_copy_records_rejects_bad_input(make_row: MakeRow) -> None:
    aid = uuid.uuid4()
    with pytest.raises(ValueError, match="duplicate"):
        copy_records(aid, [make_row("a.py"), make_row("b.py"), make_row("a.py")])
    nan_row = dataclasses.replace(make_row("a.py"), features={"sloc": float("nan")})
    with pytest.raises(ValueError):
        copy_records(aid, [nan_row])
    with pytest.raises(ValueError):
        copy_records("not-a-uuid", [make_row("a.py")])  # type: ignore[arg-type]
