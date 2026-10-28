import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from tdp_core.advisor.schema import (
    Advice,
    AdviceError,
    AdviceStep,
    extract_json_object,
    parse_advice,
    response_schema,
)

MakeAdvice = Callable[..., dict[str, Any]]


def test_dump_matches_the_http_api_type(make_advice: MakeAdvice) -> None:
    advice = Advice.model_validate(make_advice())
    dumped = advice.model_dump(mode="json")
    assert set(dumped) == {"summary", "steps", "tests_to_add", "risks", "confidence", "dropped_steps"}
    assert dumped["dropped_steps"] == 0
    step = dumped["steps"][0]
    assert set(step) == {"order", "action", "targets", "effort_hours"}
    assert step["effort_hours"] == [1.0, 2.0]
    assert all(isinstance(x, float) for x in step["effort_hours"])
    assert json.loads(advice.model_dump_json()) == dumped


def test_models_are_frozen(make_advice: MakeAdvice) -> None:
    advice = Advice.model_validate(make_advice())
    with pytest.raises(ValidationError):
        advice.summary = "changed"
    with pytest.raises(ValidationError):
        advice.steps[0].order = 9


def test_response_schema_leaves_out_dropped_steps() -> None:
    schema = response_schema()
    assert "dropped_steps" not in schema["properties"]
    assert "dropped_steps" not in schema.get("required", [])
    assert set(schema["required"]) == {"summary", "steps", "tests_to_add", "risks", "confidence"}
    assert "AdviceStep" in schema["$defs"]
    assert set(schema["$defs"]["AdviceStep"]["required"]) == {"order", "action", "targets", "effort_hours"}
    assert schema["properties"]["confidence"]["enum"] == ["low", "medium", "high"]
    # The full model still has the field: the schema is a fresh copy every time.
    schema["properties"].clear()
    assert "summary" in response_schema()["properties"]
    assert "dropped_steps" in Advice.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go: {"a": {"b": "x}"}} and that is all.', {"a": {"b": "x}"}}),
        ('pre {not json} then {"a": 2}', {"a": 2}),
        ('{"a": 1} {"b": 2}', {"a": 1}),
        ('[1, 2] {"a": 3}', {"a": 3}),
        ('{"x": NaN}', None),
        ('{"x": Infinity}', None),
        ('{"x": -Infinity}', None),
        ('{"a": [1, 2,]}', None),
        ("no braces at all", None),
        ("", None),
        pytest.param("{" * 5000, None, id="many-open-braces"),
    ],
)
def test_extract_json_object(text: str, expected: dict[str, Any] | None) -> None:
    assert extract_json_object(text) == expected


def test_extract_json_object_survives_deep_nesting() -> None:
    # Too deep for the decoder from the outer braces (RecursionError), so a shallower inner object wins.
    depth = 3000
    found = extract_json_object('{"a": ' * depth + "1" + "}" * depth)
    assert isinstance(found, dict)
    assert "a" in found


def test_parse_advice_valid(valid_reply: str) -> None:
    advice = parse_advice(valid_reply)
    assert advice.confidence == "medium"
    assert [s.order for s in advice.steps] == [2, 1, 3]
    assert advice.steps[1].effort_hours == (2.0, 12.0)
    assert advice.dropped_steps == 0


def test_parse_advice_ignores_prose_and_fences(valid_reply: str) -> None:
    advice = parse_advice(f"Sure! Here is the plan:\n```json\n{valid_reply}\n```\nGood luck.")
    assert len(advice.steps) == 3


def test_parse_advice_normalises_confidence(make_advice: MakeAdvice) -> None:
    assert parse_advice(json.dumps(make_advice(confidence="High "))).confidence == "high"
    with pytest.raises(AdviceError, match="confidence"):
        parse_advice(json.dumps(make_advice(confidence=3)))


def test_parse_advice_discards_model_dropped_steps(make_advice: MakeAdvice) -> None:
    assert parse_advice(json.dumps(make_advice(dropped_steps=9))).dropped_steps == 0


def test_parse_advice_ignores_unknown_keys(make_advice: MakeAdvice) -> None:
    advice = parse_advice(json.dumps(make_advice(notes="extra")))
    assert "notes" not in advice.model_dump()


def test_parse_advice_missing_field_names_it(make_advice: MakeAdvice) -> None:
    data = make_advice()
    del data["confidence"]
    with pytest.raises(AdviceError, match="confidence") as info:
        parse_advice(json.dumps(data))
    assert "does not match the schema" in str(info.value)
    assert "https://" not in str(info.value)


def test_parse_advice_error_does_not_echo_values(make_advice: MakeAdvice) -> None:
    with pytest.raises(AdviceError) as info:
        parse_advice(json.dumps(make_advice(confidence="SECRET-VALUE-123")))
    message = str(info.value)
    assert "confidence" in message
    assert "SECRET-VALUE-123" not in message.upper()


def test_parse_advice_caps_reported_errors(make_advice: MakeAdvice) -> None:
    steps = [{"order": "x"} for _ in range(20)]
    with pytest.raises(AdviceError, match="more errors") as info:
        parse_advice(json.dumps(make_advice(steps=steps)))
    assert str(info.value).count(";") == 10


@pytest.mark.parametrize("bad", ["NaN", "Infinity"])
def test_parse_advice_rejects_non_finite_effort(valid_reply: str, bad: str) -> None:
    text = valid_reply.replace("[1, 2]", f"[1, {bad}]")
    assert bad in text
    with pytest.raises(AdviceError):
        parse_advice(text)


def test_non_finite_effort_rejected_by_the_model_too() -> None:
    with pytest.raises(ValidationError):
        AdviceStep(order=1, action="a", targets=["x"], effort_hours=(1.0, float("inf")))


def test_parse_advice_without_json() -> None:
    with pytest.raises(AdviceError, match="no JSON object"):
        parse_advice("I cannot help with that.")


def test_parse_advice_strips_whitespace(make_advice: MakeAdvice) -> None:
    data = make_advice(summary="  padded summary \n", tests_to_add=["  one  "])
    data["steps"][0]["targets"] = ["  _scan "]
    advice = parse_advice(json.dumps(data))
    assert advice.summary == "padded summary"
    assert advice.tests_to_add == ["one"]
    assert advice.steps[0].targets == ["_scan"]


@pytest.mark.parametrize("field", ["summary", "action"])
def test_empty_text_is_rejected(make_advice: MakeAdvice, field: str) -> None:
    data = make_advice()
    if field == "summary":
        data["summary"] = "   "
    else:
        data["steps"][0]["action"] = ""
    with pytest.raises(AdviceError, match=field):
        parse_advice(json.dumps(data))
