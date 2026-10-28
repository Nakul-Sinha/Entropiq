from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

import jsonschema  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.contracts import (
    AnalysisResult,
    CostEstimate,
    FileExtra,
    FileScore,
    FunctionInfo,
    Reason,
)
from tdp_core.report.sarif import (
    LEVELS,
    MAX_MESSAGE_REASONS,
    RULE_IDS,
    SARIF_BANDS,
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    SRCROOT,
    artifact_uri,
    region_start_line,
    to_sarif,
)

MakeScore = Callable[..., FileScore]
MakeResult = Callable[..., AnalysisResult]
Golden = Callable[[str, str], None]

SCHEMA_PATH = Path(__file__).parent / "fixtures" / "sarif-2.1.0.json"
NASTY_PATHS = (
    "src/legacy/old utils.py",
    "a:b.py",
    "c:/windows/x.py",
    "q?x=1#frag.py",
    "100%.py",
    "caf\xe9/\u4e2d.py",
    "tab\tnew\nline.py",
    "//host/share.py",
    "/abs.py",
    "back\\slash.py",
    "a\udc80b.py",
)


@pytest.fixture(scope="module")
def validator() -> Any:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def schema_errors(validator: Any, log: dict[str, Any]) -> list[str]:
    return [error.message for error in validator.iter_errors(log)]


def reason(text: str) -> Reason:
    return Reason("S", "sloc", 1.0, 50.0, 0.1, text)


def function(name: str, start: int, cyclomatic: int, cognitive: int) -> FunctionInfo:
    return FunctionInfo(
        name=name,
        start_line=start,
        end_line=start + 5,
        cyclomatic=cyclomatic,
        cognitive=cognitive,
        max_nesting=1,
        params=0,
    )


def extra(*functions: FunctionInfo) -> FileExtra:
    return FileExtra(
        blob_sha="c" * 40,
        functions=functions,
        dependents=(),
        dependencies=(),
        test_importers=(),
        commit_activity=(),
        cost=CostEstimate(hours=1.0, low=0.8, high=1.2, breakdown={}),
    )


def results_of(log: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], log["runs"][0]["results"])


def test_golden(sample_result: AnalysisResult, pinned_version: str, golden: Golden) -> None:
    golden("report.sarif", json.dumps(to_sarif(sample_result), indent=2) + "\n")


def test_schema_fixture() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["id"] == SARIF_SCHEMA_URI
    assert schema["$schema"] == "http://json-schema.org/draft-04/schema#"
    assert schema["properties"]["version"]["enum"] == [SARIF_VERSION]


def test_sample_is_schema_valid(sample_result: AnalysisResult, validator: Any) -> None:
    assert schema_errors(validator, to_sarif(sample_result)) == []
    assert schema_errors(validator, to_sarif(sample_result, min_band="Critical")) == []


def test_edge_cases_are_schema_valid(make_score: MakeScore, make_result: MakeResult, validator: Any) -> None:
    assert schema_errors(validator, to_sarif(make_result())) == []
    nasty = [make_score(path, 95 - i, reasons=(reason(path),)) for i, path in enumerate(NASTY_PATHS)]
    assert schema_errors(validator, to_sarif(make_result(nasty))) == []
    nan = make_score("nan.py", 99, risk=math.nan, risk_low=math.nan, risk_high=math.inf)
    log = to_sarif(make_result([nan]))
    assert schema_errors(validator, log) == []
    json.dumps(log, allow_nan=False)


def test_schema_rejects_a_broken_log(sample_result: AnalysisResult, validator: Any) -> None:
    log = copy.deepcopy(to_sarif(sample_result))
    first = results_of(log)[0]
    first["level"] = "fatal"
    first["locations"][0]["physicalLocation"]["region"]["startLine"] = 0
    log["runs"][0]["tool"]["driver"]["rules"][0]["unexpected"] = True
    assert len(schema_errors(validator, log)) == 3


def test_document_shape(sample_result: AnalysisResult, pinned_version: str) -> None:
    log = to_sarif(sample_result)
    assert list(log) == ["$schema", "version", "runs"]
    assert log["$schema"] == SARIF_SCHEMA_URI
    assert log["version"] == "2.1.0"
    assert len(log["runs"]) == 1
    run = log["runs"][0]
    driver = run["tool"]["driver"]
    assert driver["name"] == "tdp"
    assert driver["version"] == pinned_version
    assert run["properties"] == {
        "sha": sample_result.sha,
        "analyzed_at": "2026-01-15T09:30:05Z",
        "model_version": sample_result.model_version,
        "extractor_version": sample_result.extractor_version,
    }
    assert "partialFingerprints" not in json.dumps(log)
    assert "fingerprints" not in json.dumps(log)


def test_rules(sample_result: AnalysisResult) -> None:
    for min_band in SARIF_BANDS:
        rules = to_sarif(sample_result, min_band=min_band)["runs"][0]["tool"]["driver"]["rules"]
        assert [rule["id"] for rule in rules] == ["tdp/critical", "tdp/high"]
        assert [rule["name"] for rule in rules] == ["CriticalMaintenancePriority", "HighMaintenancePriority"]
        assert [rule["defaultConfiguration"]["level"] for rule in rules] == ["warning", "note"]
        assert rules[0]["shortDescription"]["text"] == "Critical maintenance priority"
        assert rules[1]["shortDescription"]["text"] == "High maintenance priority"
        assert "Priority 90 to 100 out of 100" in rules[0]["fullDescription"]["text"]
        assert "Priority 75 to 89 out of 100" in rules[1]["fullDescription"]["text"]
        for rule in rules:
            assert rule["help"]["text"].startswith("Priority is a percentile within the repository")
            assert rule["properties"]["tags"] == ["maintainability", "technical-debt"]
    assert dict(RULE_IDS) == {"Critical": "tdp/critical", "High": "tdp/high"}
    assert dict(LEVELS) == {"Critical": "warning", "High": "note"}


def test_results_for_the_sample(sample_result: AnalysisResult) -> None:
    log = to_sarif(sample_result)
    rules = log["runs"][0]["tool"]["driver"]["rules"]
    results = results_of(log)
    assert [r["ruleId"] for r in results] == ["tdp/critical", "tdp/high", "tdp/high"]
    assert [r["ruleIndex"] for r in results] == [0, 1, 1]
    assert [r["level"] for r in results] == ["warning", "note", "note"]
    assert [r["rank"] for r in results] == [95.0, 81.0, 81.0]
    locations = [r["locations"][0]["physicalLocation"] for r in results]
    assert [loc["artifactLocation"]["uri"] for loc in locations] == [
        "src/payments/processor.py",
        "src/billing/invoice.py",
        "src/legacy/old%20utils.py",
    ]
    assert {loc["artifactLocation"]["uriBaseId"] for loc in locations} == {SRCROOT}
    assert [loc["region"]["startLine"] for loc in locations] == [120, 12, 1]
    for r in results:
        rule = rules[r["ruleIndex"]]
        assert rule["id"] == r["ruleId"]
        assert rule["defaultConfiguration"]["level"] == r["level"]
    assert results[0]["message"]["text"] == (
        "Critical maintenance priority: 95 of 100. Estimated chance that a bug in its current code is fixed "
        "within 12 months: 64% (55% to 72%). Top reasons: Very high recent churn: 831 recency-weighted lines "
        "added or deleted (top 1% in repo); Many past bugs: 4 past bugs traced back to its code (top 1% in "
        "repo); No test file named after it."
    )
    assert results[2]["message"]["text"].endswith(
        " Top reason: Heavy duplication: 32% of its code is duplicated (top 3% in repo)."
    )
    assert results[0]["properties"] == {
        "priority": 95,
        "band": "Critical",
        "risk": 0.64,
        "risk_low": 0.55,
        "risk_high": 0.72,
        "debt_types": ["defect_hotspot", "test"],
    }


def test_min_band(sample_result: AnalysisResult) -> None:
    critical = results_of(to_sarif(sample_result, min_band="Critical"))
    assert [r["ruleId"] for r in critical] == ["tdp/critical"]
    default = results_of(to_sarif(sample_result))
    assert default == results_of(to_sarif(sample_result, min_band="High"))
    assert {r["properties"]["band"] for r in default} == {"Critical", "High"}


@pytest.mark.parametrize("band", ["Medium", "Low", "high", "critical", "x", ""])
def test_min_band_is_validated(sample_result: AnalysisResult, band: str) -> None:
    with pytest.raises(ValueError, match="min_band must be one of Critical, High"):
        to_sarif(sample_result, min_band=cast(Any, band))


def test_results_are_sorted(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [make_score("b.py", 91), make_score("a.py", 91), make_score("c.py", 99), make_score("m.py", 60)]
    results = results_of(to_sarif(make_result(files)))
    assert [r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for r in results] == [
        "c.py",
        "a.py",
        "b.py",
    ]
    assert [r["rank"] for r in results] == [99.0, 91.0, 91.0]
    assert all(isinstance(r["rank"], float) for r in results)


def test_rank_is_clamped(make_score: MakeScore, make_result: MakeResult) -> None:
    results = results_of(to_sarif(make_result([make_score("a.py", 130, band="Critical")])))
    assert results[0]["rank"] == 100.0
    assert results[0]["properties"]["priority"] == 130


def test_message_reasons(make_score: MakeScore, make_result: MakeResult) -> None:
    many = make_score(
        "many.py",
        99,
        reasons=tuple(reason(f"Reason number {i}.") for i in range(5)),
    )
    blank = make_score("blank.py", 98, reasons=(reason("   "), reason("Only\n  real\treason..")))
    none = make_score("none.py", 97, risk=math.nan, risk_low=0.1, risk_high=math.inf)
    results = results_of(to_sarif(make_result([many, blank, none])))
    texts = [r["message"]["text"] for r in results]
    assert texts[0].endswith(" Top reasons: Reason number 0; Reason number 1; Reason number 2.")
    assert texts[0].count("Reason number") == MAX_MESSAGE_REASONS == 3
    assert texts[1].endswith(" Top reason: Only real reason.")
    assert texts[2] == (
        "Critical maintenance priority: 97 of 100. Estimated chance that a bug in its current code is fixed "
        "within 12 months: n/a (10% to n/a)."
    )
    assert results[2]["properties"]["risk"] is None
    assert results[2]["properties"]["risk_low"] == 0.1
    assert results[2]["properties"]["risk_high"] is None
    two = make_score("two.py", 99, reasons=(reason("first"), reason("second")))
    text = results_of(to_sarif(make_result([two])))[0]["message"]["text"]
    assert text.endswith(" Top reasons: first; second.")


def test_region_start_line() -> None:
    assert region_start_line(None) == 1
    assert region_start_line(extra()) == 1
    assert region_start_line(extra(function("a", 10, 3, 1), function("b", 20, 5, 0))) == 20
    assert region_start_line(extra(function("a", 10, 5, 1), function("b", 20, 5, 7))) == 20
    assert region_start_line(extra(function("a", 30, 5, 7), function("b", 20, 5, 7))) == 20
    assert region_start_line(extra(function("b", 20, 5, 7), function("a", 20, 5, 7))) == 20
    assert region_start_line(extra(function("a", 0, 9, 9))) == 1
    assert region_start_line(extra(function("a", -4, 9, 9), function("b", 50, 1, 0))) == 1


def test_region_uses_the_extras(make_score: MakeScore, make_result: MakeResult) -> None:
    result = make_result(
        [make_score("a.py", 95), make_score("b.py", 92)],
        extras={"a.py": extra(function("f", 42, 3, 3))},
    )
    regions = [r["locations"][0]["physicalLocation"]["region"] for r in results_of(to_sarif(result))]
    assert regions == [{"startLine": 42}, {"startLine": 1}]


@pytest.mark.parametrize(
    ("path", "uri"),
    [
        ("src/a.py", "src/a.py"),
        ("src/legacy/old utils.py", "src/legacy/old%20utils.py"),
        ("a:b.py", "a%3Ab.py"),
        ("q?x=1#f", "q%3Fx%3D1%23f"),
        ("100%.py", "100%25.py"),
        ("caf\xe9.py", "caf%C3%A9.py"),
        ("a~b-c_d.e", "a~b-c_d.e"),
        ("/abs.py", "%2Fabs.py"),
        ("//host/x.py", "%2F/host/x.py"),
        ("a\udc80b", "a%3Fb"),
        ("", ""),
    ],
)
def test_artifact_uri(path: str, uri: str) -> None:
    assert artifact_uri(path) == uri


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=40))
def test_artifact_uri_round_trips(path: str) -> None:
    uri = artifact_uri(path)
    assert re.fullmatch(r"[A-Za-z0-9._~/%-]*", uri)
    assert not uri.startswith("/")
    assert unquote(uri) == path


def test_fresh_objects(sample_result: AnalysisResult) -> None:
    first = to_sarif(sample_result)
    first["runs"][0]["tool"]["driver"]["rules"][0]["properties"]["tags"].append("mutated")
    first["runs"][0]["results"][0]["properties"]["debt_types"].clear()
    first["runs"][0]["properties"]["sha"] = "x"
    second = to_sarif(sample_result)
    assert second["runs"][0]["tool"]["driver"]["rules"][0]["properties"]["tags"] == [
        "maintainability",
        "technical-debt",
    ]
    assert second["runs"][0]["results"][0]["properties"]["debt_types"] == ["defect_hotspot", "test"]
    assert second["runs"][0]["properties"]["sha"] == sample_result.sha


def test_naive_datetime_is_rejected(sample_result: AnalysisResult) -> None:
    naive = dataclasses.replace(sample_result, analyzed_at=datetime(2026, 1, 15, 9, 30))
    with pytest.raises(ValueError, match="naive"):
        to_sarif(naive)
