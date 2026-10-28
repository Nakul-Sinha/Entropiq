import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from tdp_core.advisor.advisor import (
    ADVICE_MAX_TOKENS,
    MIN_STEP_HOURS,
    PROMPT_VERSION,
    Advisor,
    ground_advice,
    normalize_target,
    valid_targets,
)
from tdp_core.advisor.client import FakeLLMClient, LLMError
from tdp_core.advisor.prompt import build_system_prompt
from tdp_core.advisor.schema import Advice, AdviceError
from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import CostEstimate, FileExtra, FileScore, FileSyntax, FunctionInfo

MakeAdvice = Callable[..., dict[str, Any]]


class Sample:
    """The sample file's inputs, bundled so tests can call ``advise`` in one line."""

    def __init__(
        self, path: str, source: str, syntax: FileSyntax, score: FileScore, extra: FileExtra
    ) -> None:
        self.path, self.source, self.syntax, self.score, self.extra = path, source, syntax, score, extra

    def advise(self, client: FakeLLMClient, **overrides: Any) -> Advice:
        kwargs: dict[str, Any] = {
            "path": self.path,
            "source": self.source,
            "syntax": self.syntax,
            "score": self.score,
            "extra": self.extra,
        }
        kwargs.update(overrides)
        return Advisor(client).advise(**kwargs)


@pytest.fixture
def sample(
    sample_path: str,
    sample_source: str,
    sample_syntax: FileSyntax,
    sample_score: FileScore,
    sample_extra: FileExtra,
) -> Sample:
    return Sample(sample_path, sample_source, sample_syntax, sample_score, sample_extra)


def reply_with_steps(make_advice: MakeAdvice, *steps: tuple[list[str], list[float]]) -> str:
    return json.dumps(
        make_advice(
            steps=[
                {"order": i, "action": f"step {i}", "targets": targets, "effort_hours": effort}
                for i, (targets, effort) in enumerate(steps, 1)
            ]
        )
    )


def test_valid_reply_is_parsed_and_grounded(sample: Sample, valid_reply: str) -> None:
    fake = FakeLLMClient([valid_reply])
    advice = sample.advise(fake)
    assert len(fake.calls) == 1
    assert fake.calls[0].max_tokens == ADVICE_MAX_TOKENS
    assert fake.calls[0].system == build_system_prompt()
    assert "<code_excerpt>" in fake.calls[0].user
    assert [s.order for s in advice.steps] == [1, 2, 3]
    assert [s.targets for s in advice.steps] == [["Parser.parse"], ["_scan"], ["pkg/parser.py"]]
    assert [s.effort_hours for s in advice.steps] == [(2.0, 7.2), (1.0, 2.0), (0.25, 0.25)]
    assert advice.dropped_steps == 0
    assert advice.confidence == "medium"
    assert advice.tests_to_add == ["parse skips comment lines", "parse keeps bang lines"]


def test_invalid_reply_is_retried_once(
    sample: Sample, valid_reply: str, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLMClient(["not json", valid_reply])
    with caplog.at_level(logging.WARNING, logger="tdp_core.advisor.advisor"):
        advice = sample.advise(fake)
    assert len(advice.steps) == 3
    assert len(fake.calls) == 2
    first, second = fake.calls
    assert second.system == first.system
    assert second.user.startswith(first.user)
    assert "Your previous reply could not be used: the reply contains no JSON object" in second.user
    assert "not json" not in second.user.removeprefix(first.user)
    assert "retrying once" in caplog.text


def test_schema_error_retry_names_the_field(
    sample: Sample, valid_reply: str, make_advice: MakeAdvice
) -> None:
    bad = make_advice()
    del bad["risks"]
    fake = FakeLLMClient([json.dumps(bad), valid_reply])
    sample.advise(fake)
    retry = fake.calls[1].user
    assert "does not match the schema: risks: Field required" in retry


def test_second_failure_raises(sample: Sample) -> None:
    fake = FakeLLMClient(["not json", '{"summary": ""}'])
    with pytest.raises(AdviceError, match=r"no valid advice for pkg/parser\.py after one retry") as info:
        sample.advise(fake)
    assert not isinstance(info.value, LLMError)
    assert isinstance(info.value.__cause__, AdviceError)
    assert len(fake.calls) == 2


def test_llm_error_propagates_without_retry(sample: Sample, valid_reply: str) -> None:
    error = LLMError("rate limited", retryable=True)
    fake = FakeLLMClient([error, valid_reply])
    with pytest.raises(LLMError) as info:
        sample.advise(fake)
    assert info.value is error
    assert len(fake.calls) == 1


def test_llm_error_on_the_retry_propagates(sample: Sample) -> None:
    error = LLMError("down", retryable=True)
    fake = FakeLLMClient(["not json", error])
    with pytest.raises(LLMError) as info:
        sample.advise(fake)
    assert info.value is error


def test_invented_function_is_dropped(sample: Sample, make_advice: MakeAdvice) -> None:
    reply = reply_with_steps(make_advice, (["Parser.frobnicate"], [1, 2]), (["load"], [1, 2]))
    advice = sample.advise(FakeLLMClient([reply]))
    assert advice.dropped_steps == 1
    assert [(s.order, s.targets) for s in advice.steps] == [(1, ["load"])]


def test_one_bad_target_drops_the_step(sample: Sample, make_advice: MakeAdvice) -> None:
    reply = reply_with_steps(make_advice, (["load", "Parser.parse", "Reader"], [1, 2]))
    advice = sample.advise(FakeLLMClient([reply]))
    assert advice.steps == []
    assert advice.dropped_steps == 1


@pytest.mark.parametrize(
    ("target", "normalized"),
    [
        ("Parser.parse", "Parser.parse"),
        ("parse", "parse"),
        ("Parser", "Parser"),
        ("pkg/parser.py", "pkg/parser.py"),
        ("parser.py", "parser.py"),
        ("parse()", "parse"),
        ("parse(data)", "parse"),
        ("self.parse", "parse"),
        ("cls.parse()", "parse"),
        ("`_scan`", "_scan"),
        (" '_scan.keep' ", "_scan.keep"),
        ("Parser._scan.keep", "Parser._scan.keep"),
    ],
)
def test_accepted_target_forms(sample: Sample, make_advice: MakeAdvice, target: str, normalized: str) -> None:
    advice = sample.advise(FakeLLMClient([reply_with_steps(make_advice, ([target], [1, 2]))]))
    assert advice.dropped_steps == 0
    assert advice.steps[0].targets == [normalized]


def test_steps_without_targets_are_dropped(sample: Sample, make_advice: MakeAdvice) -> None:
    reply = reply_with_steps(
        make_advice, ([], [1, 2]), (["", "  ", "``"], [1, 2]), (["load", "load()"], [1, 2])
    )
    advice = sample.advise(FakeLLMClient([reply]))
    assert advice.dropped_steps == 2
    assert advice.steps[0].targets == ["load"]


def test_reversed_effort_is_sorted(sample: Sample, make_advice: MakeAdvice) -> None:
    advice = sample.advise(FakeLLMClient([reply_with_steps(make_advice, (["load"], [5, 1]))]))
    assert advice.steps[0].effort_hours == (1.0, 5.0)


def test_effort_is_clipped_into_the_cost_range(sample: Sample, make_advice: MakeAdvice) -> None:
    tiny = replace(sample.extra, cost=CostEstimate(0.1, 0.08, 0.1, {}))
    reply = reply_with_steps(make_advice, (["load"], [0, 30]))
    advice = sample.advise(FakeLLMClient([reply]), extra=tiny)
    assert advice.steps[0].effort_hours == (MIN_STEP_HOURS, MIN_STEP_HOURS)


def test_steps_are_ordered_by_order_then_reply_position(sample: Sample, make_advice: MakeAdvice) -> None:
    steps = [
        {"order": 5, "action": "c", "targets": ["load"], "effort_hours": [1, 2]},
        {"order": 1, "action": "a", "targets": ["load"], "effort_hours": [1, 2]},
        {"order": 1, "action": "b", "targets": ["load"], "effort_hours": [1, 2]},
    ]
    advice = sample.advise(FakeLLMClient([json.dumps(make_advice(steps=steps))]))
    assert [(s.order, s.action) for s in advice.steps] == [(1, "a"), (2, "b"), (3, "c")]


def test_empty_list_items_are_removed(sample: Sample, make_advice: MakeAdvice) -> None:
    reply = json.dumps(make_advice(tests_to_add=["", "a test", "  "], risks=[" ", "a risk"]))
    advice = sample.advise(FakeLLMClient([reply]))
    assert advice.tests_to_add == ["a test"]
    assert advice.risks == ["a risk"]


def test_path_argument_wins_over_syntax_path(sample: Sample, make_advice: MakeAdvice) -> None:
    stale = replace(sample.syntax, path="old/name.py")
    reply = reply_with_steps(make_advice, (["pkg/parser.py"], [1, 2]), (["old/name.py"], [1, 2]))
    fake = FakeLLMClient([reply])
    advice = sample.advise(fake, syntax=stale)
    assert [s.targets for s in advice.steps] == [["pkg/parser.py"]]
    assert advice.dropped_steps == 1
    assert "File: pkg/parser.py" in fake.calls[0].user
    assert "old/name.py" not in fake.calls[0].user


def test_targets_come_from_syntax_functions(sample: Sample, make_advice: MakeAdvice) -> None:
    extra = replace(sample.extra, functions=())
    advice = sample.advise(FakeLLMClient([reply_with_steps(make_advice, (["load"], [1, 2]))]), extra=extra)
    assert advice.dropped_steps == 0


def test_constructor_validates_its_arguments() -> None:
    fake = FakeLLMClient([])
    with pytest.raises(ValueError, match="advisor_max_excerpt_lines"):
        Advisor(fake, with_overrides(DEFAULT_SETTINGS, advisor_max_excerpt_lines=-1))
    with pytest.raises(ValueError, match="max_tokens"):
        Advisor(fake, max_tokens=0)


def test_zero_excerpt_lines_sends_no_code(sample: Sample, valid_reply: str) -> None:
    fake = FakeLLMClient([valid_reply])
    settings = with_overrides(DEFAULT_SETTINGS, advisor_max_excerpt_lines=0)
    Advisor(fake, settings, max_tokens=500).advise(
        path=sample.path, source=sample.source, syntax=sample.syntax, score=sample.score, extra=sample.extra
    )
    assert "<code_excerpt>\n(no code available)\n</code_excerpt>" in fake.calls[0].user
    assert fake.calls[0].max_tokens == 500


def test_advisor_uses_the_configured_excerpt_cap(sample_score: FileScore, sample_extra: FileExtra) -> None:
    functions = (FunctionInfo("a", 1, 200, 9, 0, 0, 0), FunctionInfo("b", 201, 400, 8, 0, 0, 0))
    syntax = FileSyntax("m.py", "python", 400, 1, functions, 0, (), (), (), (), False)
    source = "".join(f"line {i}\n" for i in range(1, 401))
    empty = '{"summary": "s", "steps": [], "tests_to_add": [], "risks": [], "confidence": "low"}'
    fake = FakeLLMClient([empty])
    settings = with_overrides(DEFAULT_SETTINGS, advisor_max_excerpt_lines=40)
    advisor = Advisor(fake, settings)
    advisor.advise(path="m.py", source=source, syntax=syntax, score=sample_score, extra=sample_extra)
    prompt = fake.calls[0].user
    excerpt = prompt[prompt.index("<code_excerpt>") : prompt.index("</code_excerpt>")]
    assert len([line for line in excerpt.splitlines() if line.startswith("line ")]) == 40
    assert "# lines 1-40 (a, cyclomatic 9; truncated, the function ends at line 200)" in excerpt


def test_ground_advice_is_idempotent(sample: Sample, make_advice: MakeAdvice) -> None:
    reply = reply_with_steps(make_advice, (["nope"], [1, 2]), (["parse()"], [9, 0]))
    once = sample.advise(FakeLLMClient([reply]))
    twice = ground_advice(once, path=sample.path, syntax=sample.syntax, cost=sample.extra.cost)
    assert twice == once
    assert twice.dropped_steps == 1


def test_valid_targets(sample_path: str, sample_syntax: FileSyntax) -> None:
    assert valid_targets(sample_path, sample_syntax) == {
        "load",
        "Parser",
        "Parser.__init__",
        "__init__",
        "Parser.parse",
        "parse",
        "Parser._scan",
        "_scan",
        "Parser._scan.keep",
        "_scan.keep",
        "keep",
        "helper",
        "pkg/parser.py",
        "parser.py",
    }


def test_normalize_target() -> None:
    assert normalize_target('  "`self.parse(data)`" ') == "parse"
    assert normalize_target("f(g())") == "f(g())"
    assert normalize_target("") == ""


def test_cache_key() -> None:
    key = Advisor.cache_key("blob", "model")
    assert key == hashlib.sha256(f"blob|model|{PROMPT_VERSION}".encode()).hexdigest()
    assert key == hashlib.sha256(b"blob|model|1").hexdigest()
    assert len(key) == 64
    assert key == key.lower()
    assert all(c in "0123456789abcdef" for c in key)
    assert Advisor.cache_key("blob", "other") != key
    assert Advisor.cache_key("blob2", "model") != key


def test_logs_do_not_contain_code(sample: Sample, valid_reply: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="tdp_core.advisor"):
        sample.advise(FakeLLMClient(["not json", valid_reply]))
    assert "3 steps kept, 0 dropped" in caplog.text
    assert "splitlines" not in caplog.text
    assert "Ignore previous instructions" not in caplog.text


def test_advisor_modules_do_not_import_anthropic() -> None:
    code = "import sys, tdp_core.advisor.advisor, tdp_core.advisor.client; print('anthropic' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "False"
