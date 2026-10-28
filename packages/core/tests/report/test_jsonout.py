from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tdp_core.contracts import FEATURE_INPUT_COLUMNS, AnalysisResult, FileScore
from tdp_core.report.common import summarize
from tdp_core.report.jsonout import JSON_FORMAT, JSON_FORMAT_VERSION, to_json, to_json_dict
from tdp_core.serialize import from_jsonable, to_jsonable

MakeScore = Callable[..., FileScore]
MakeResult = Callable[..., AnalysisResult]
Golden = Callable[[str, str], None]

HEADER = ["format", "format_version", "tool", "sha", "analyzed_at", "model_version", "extractor_version"]


def reject_constant(name: str) -> Any:
    raise AssertionError(f"non-standard JSON constant {name}")


def strict_loads(text: str) -> Any:
    return json.loads(text, parse_constant=reject_constant)


def test_golden(sample_result: AnalysisResult, pinned_version: str, golden: Golden) -> None:
    text = to_json(sample_result)
    golden("report.json", text)
    assert text.endswith("}\n")
    assert text.isascii()
    strict_loads(text)


def test_header(sample_result: AnalysisResult, pinned_version: str) -> None:
    doc = to_json_dict(sample_result)
    assert list(doc) == [*HEADER, "summary", "files", "extras", "graph", "plan", "stats"]
    assert doc["format"] == JSON_FORMAT == "tdp-report"
    assert doc["format_version"] == JSON_FORMAT_VERSION == 1
    assert doc["tool"] == {"name": "tdp", "version": pinned_version}
    assert doc["sha"] == sample_result.sha
    assert doc["analyzed_at"] == "2026-01-15T09:30:05Z"
    assert doc["model_version"] == "dev-20260115T093005Z"
    assert doc["extractor_version"] == sample_result.extractor_version


def test_round_trip(sample_result: AnalysisResult) -> None:
    back = from_jsonable(AnalysisResult, json.loads(to_json(sample_result)))
    assert back.sha == sample_result.sha
    assert back.analyzed_at == sample_result.analyzed_at
    assert back.model_version == sample_result.model_version
    assert back.extractor_version == sample_result.extractor_version
    assert back.files == sample_result.files
    assert back.extras == sample_result.extras
    assert back.graph == sample_result.graph
    assert back.plan == sample_result.plan
    assert back.stats == sample_result.stats
    assert isinstance(back.features, pd.DataFrame)
    assert back.features.empty


def test_summary_and_extras(sample_result: AnalysisResult) -> None:
    doc = to_json_dict(sample_result)
    assert doc["summary"] == summarize(sample_result.files, sample_result.extras)
    assert list(doc["extras"]) == sorted(sample_result.extras)
    assert list(sample_result.extras) != sorted(sample_result.extras)
    assert doc["extras"]["src/payments/processor.py"] == to_jsonable(
        sample_result.extras["src/payments/processor.py"]
    )
    assert doc["graph"] == to_jsonable(sample_result.graph)
    assert doc["plan"] == to_jsonable(sample_result.plan)
    assert doc["stats"] == sample_result.stats


def test_files_keep_the_input_order(sample_result: AnalysisResult) -> None:
    shuffled = dataclasses.replace(sample_result, files=tuple(reversed(sample_result.files)))
    doc = to_json_dict(shuffled)
    assert [f["path"] for f in doc["files"]] == [f.path for f in reversed(sample_result.files)]


def test_file_entries(sample_result: AnalysisResult) -> None:
    doc = to_json_dict(sample_result)
    first = doc["files"][0]
    expected = to_jsonable(sample_result.files[0])
    assert list(first) == [*expected, "features"]
    assert {k: v for k, v in first.items() if k != "features"} == expected
    assert first["features"] == {"sloc": 640.0, "cc_max": 31.0, "dup_ratio": 0.05, "pct_sloc": 1.0}
    by_path = {f["path"]: f["features"] for f in doc["files"]}
    assert by_path["setup.py"] == {"sloc": 20.0, "cc_max": 1.0, "dup_ratio": 0.0, "pct_sloc": 0.2}
    assert by_path["src/legacy/old utils.py"]["dup_ratio"] == 0.32


def test_full_feature_frame(sample_result: AnalysisResult) -> None:
    paths = [f.path for f in sample_result.files]
    frame = pd.DataFrame(
        np.full((len(paths), len(FEATURE_INPUT_COLUMNS)), 0.1, dtype=np.float32),
        index=pd.Index(paths, name="path"),
        columns=list(FEATURE_INPUT_COLUMNS),
    )
    result = dataclasses.replace(sample_result, features=frame)
    doc = to_json_dict(result)
    for entry in doc["files"]:
        assert list(entry["features"]) == list(FEATURE_INPUT_COLUMNS)
        assert len(entry["features"]) == 89
        assert set(entry["features"].values()) == {0.1}
    text = to_json(result)
    assert '"pct_sloc": 0.1' in text
    assert "0.100000" not in text


def test_missing_rows_and_empty_frames(sample_result: AnalysisResult) -> None:
    partial = sample_result.features.drop(index=["setup.py"])
    doc = to_json_dict(dataclasses.replace(sample_result, features=partial))
    by_path = {f["path"]: f["features"] for f in doc["files"]}
    assert by_path["setup.py"] == {}
    assert by_path["src/pkg/__init__.py"]["sloc"] == 35.0
    empty = to_json_dict(dataclasses.replace(sample_result, features=pd.DataFrame()))
    assert all(f["features"] == {} for f in empty["files"])
    no_columns = pd.DataFrame(index=pd.Index(["setup.py"], name="path"))
    doc = to_json_dict(dataclasses.replace(sample_result, features=no_columns))
    assert all(f["features"] == {} for f in doc["files"])


def test_feature_value_types(make_score: MakeScore, make_result: MakeResult) -> None:
    index = pd.Index(["a.py", "b.py"], name="path")
    frame = pd.DataFrame(
        {
            "count": pd.Series([3, 4], index=index, dtype=np.int64),
            "flag": pd.Series([True, False], index=index, dtype=bool),
            "maybe": pd.Series([None, 2], index=index, dtype=object),
            "nullable": pd.Series(pd.array([pd.NA, True], dtype="boolean"), index=index),
            "nullable_int": pd.Series(pd.array([pd.NA, 7], dtype="Int64"), index=index),
            "half": pd.Series([0.1, 2.5], index=index, dtype=np.float16),
            "double": pd.Series([0.1, 1e300], index=index, dtype=np.float64),
            "python": pd.Series([1.5, 7], index=index, dtype=object),
            7: pd.Series([1.0, 2.0], index=index),
        }
    )
    result = make_result([make_score("a.py", 60), make_score("b.py", 50)], features=frame)
    features = [f["features"] for f in to_json_dict(result)["files"]]
    assert list(features[0]) == [
        "count",
        "flag",
        "maybe",
        "nullable",
        "nullable_int",
        "half",
        "double",
        "python",
        "7",
    ]
    a, b = features
    assert a["count"] == 3 and type(a["count"]) is int
    assert a["flag"] is True and b["flag"] is False
    assert a["maybe"] is None and b["maybe"] == 2 and type(b["maybe"]) is int
    assert a["nullable"] is None and b["nullable"] is True
    assert a["nullable_int"] is None and b["nullable_int"] == 7
    assert a["half"] == 0.1 and b["half"] == 2.5
    assert a["double"] == 0.1 and b["double"] == 1e300
    assert a["python"] == 1.5 and b["python"] == 7
    assert a["7"] == 1.0


def test_non_numeric_feature_is_rejected(make_score: MakeScore, make_result: MakeResult) -> None:
    frame = pd.DataFrame({"name": ["x"]}, index=pd.Index(["a.py"], name="path"))
    with pytest.raises(TypeError, match="feature values must be numbers"):
        to_json_dict(make_result([make_score("a.py", 60)], features=frame))


def test_duplicate_index_is_rejected(make_score: MakeScore, make_result: MakeResult) -> None:
    frame = pd.DataFrame({"sloc": [1.0, 2.0]}, index=pd.Index(["a.py", "a.py"], name="path"))
    with pytest.raises(ValueError, match="each path once"):
        to_json_dict(make_result([make_score("a.py", 60)], features=frame))


def test_duplicate_columns_are_rejected(make_score: MakeScore, make_result: MakeResult) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["a.py"], name="path"), columns=[1, "1"])
    with pytest.raises(ValueError, match="column names must be unique"):
        to_json_dict(make_result([make_score("a.py", 60)], features=frame))


def test_non_finite_values_become_null(make_score: MakeScore, make_result: MakeResult) -> None:
    index = pd.Index(["a.py"], name="path")
    frame = pd.DataFrame(
        {
            "nan32": pd.Series([np.nan], index=index, dtype=np.float32),
            "inf32": pd.Series([np.inf], index=index, dtype=np.float32),
            "ninf64": pd.Series([-np.inf], index=index, dtype=np.float64),
        }
    )
    score = make_score("a.py", 60, risk=math.nan, risk_low=-math.inf, risk_high=math.inf)
    result = make_result(
        [score],
        features=frame,
        stats={"n_files": 1, "timings_s": {"mine": math.nan, "parse": 1.0}, "ratios": [math.inf, 0.5]},
    )
    text = to_json(result)
    doc = strict_loads(text)
    entry = doc["files"][0]
    assert entry["features"] == {"nan32": None, "inf32": None, "ninf64": None}
    assert entry["risk"] is None and entry["risk_low"] is None and entry["risk_high"] is None
    assert doc["stats"] == {"n_files": 1, "timings_s": {"mine": None, "parse": 1.0}, "ratios": [None, 0.5]}
    assert "NaN" not in text and "Infinity" not in text


def test_non_finite_values_do_not_round_trip(make_score: MakeScore, make_result: MakeResult) -> None:
    text = to_json(make_result([make_score("a.py", 60, risk=math.nan)]))
    with pytest.raises(TypeError):
        from_jsonable(AnalysisResult, json.loads(text))


def test_stats_must_be_serialisable(make_result: MakeResult) -> None:
    with pytest.raises(TypeError, match="cannot serialise"):
        to_json_dict(make_result(stats={"bad": object()}))


def test_indent(sample_result: AnalysisResult) -> None:
    compact = to_json(sample_result, indent=None)
    assert compact.count("\n") == 1
    assert compact.endswith("}\n")
    assert strict_loads(compact) == strict_loads(to_json(sample_result))
    assert to_json(sample_result).count("\n") > 100


def test_non_ascii_paths(make_score: MakeScore, make_result: MakeResult) -> None:
    path = "src/caf\xe9/\u4e2d\u6587.py"
    result = make_result([make_score(path, 60)])
    text = to_json(result)
    assert text.isascii()
    assert "\\u00e9" in text
    back = from_jsonable(AnalysisResult, json.loads(text))
    assert back.files[0].path == path


def test_empty_result(make_result: MakeResult) -> None:
    doc = strict_loads(to_json(make_result()))
    assert doc["files"] == []
    assert doc["extras"] == {}
    assert doc["summary"]["files"] == 0
    assert doc["plan"]["steps"] == []


def test_naive_datetime_is_rejected(sample_result: AnalysisResult) -> None:
    naive = dataclasses.replace(sample_result, analyzed_at=datetime(2026, 1, 15, 9, 30))
    with pytest.raises(ValueError, match="naive"):
        to_json(naive)


def test_fresh_objects(sample_result: AnalysisResult) -> None:
    first = to_json_dict(sample_result)
    first["files"][0]["features"]["sloc"] = -1
    first["summary"]["debt_types"]["test"] = 99
    first["stats"]["timings_s"]["mine"] = 0
    second = to_json_dict(sample_result)
    assert second["files"][0]["features"]["sloc"] == 640.0
    assert second["summary"]["debt_types"]["test"] == 1
    assert second["stats"]["timings_s"]["mine"] == 1.5
    assert sample_result.stats["timings_s"]["mine"] == 1.5
