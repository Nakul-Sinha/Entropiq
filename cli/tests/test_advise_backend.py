"""How /advise picks where fresh advice comes from: the Anthropic API or the Claude CLI login."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from tdp_cli.commands import advise
from tdp_cli.errors import UsageError
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_core.advisor.client import FakeLLMClient, LLMError
from tdp_core.config import DEFAULT_SETTINGS

MODEL = DEFAULT_SETTINGS.advisor_model
CLI = ("C:/tools/claude.exe",)
REPLY = json.dumps(
    {
        "summary": "Merge the two helpers.",
        "steps": [
            {
                "order": 1,
                "action": "Merge mu_0 and mu_1",
                "targets": ["mu_0", "mu_1"],
                "effort_hours": [0.5, 1.0],
            }
        ],
        "tests_to_add": [],
        "risks": [],
        "confidence": "high",
    }
)


def _refuse_api(model: str, key: str) -> Any:
    raise AssertionError("the API client should not be used here")


def _refuse_cli(model: str, command: Sequence[str]) -> Any:
    raise AssertionError("the Claude CLI client should not be used here")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(advise.ENV_API_KEY, raising=False)
    monkeypatch.delenv(advise.ENV_BACKEND, raising=False)
    monkeypatch.setattr(advise, "client_factory", _refuse_api)
    monkeypatch.setattr(advise, "cli_client_factory", _refuse_cli)
    monkeypatch.setattr(advise, "claude_cli", lambda environ=None: None)


def _with_cli(
    monkeypatch: pytest.MonkeyPatch, *replies: str | Exception
) -> list[tuple[str, tuple[str, ...]]]:
    """Make the Claude CLI available, answering with ``replies``; returns the (model, command) per build."""
    seen: list[tuple[str, tuple[str, ...]]] = []
    client = FakeLLMClient(list(replies))

    def factory(model: str, command: Sequence[str]) -> FakeLLMClient:
        seen.append((model, tuple(command)))
        return client

    monkeypatch.setattr(advise, "claude_cli", lambda environ=None: list(CLI))
    monkeypatch.setattr(advise, "cli_client_factory", factory)
    return seen


def _with_api(monkeypatch: pytest.MonkeyPatch, *replies: str | Exception) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    client = FakeLLMClient(list(replies))

    def factory(model: str, key: str) -> FakeLLMClient:
        seen.append((model, key))
        return client

    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")
    monkeypatch.setattr(advise, "client_factory", factory)
    return seen


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined]


def test_without_a_key_the_claude_cli_login_is_used(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _with_cli(monkeypatch, REPLY)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert seen == [(MODEL, CLI)]
    assert "(through the Claude CLI)" in text
    assert "AI suggestions for pkg/mu.py" in text
    assert "1. Merge mu_0 and mu_1" in text
    assert "AI fix advice is off" not in text


def test_a_set_key_wins_in_auto_mode(loaded_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(advise, "claude_cli", lambda environ=None: list(CLI))
    seen = _with_api(monkeypatch, REPLY)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert seen == [(MODEL, "test-key")]
    assert "through the Claude CLI" not in text
    assert "AI suggestions for pkg/mu.py" in text


def test_the_variable_can_force_the_claude_cli(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")
    monkeypatch.setenv(advise.ENV_BACKEND, "claude-cli")
    seen = _with_cli(monkeypatch, REPLY)
    dispatch(loaded_session, "/advise pkg/mu.py")
    assert seen == [(MODEL, CLI)]
    assert "(through the Claude CLI)" in _out(loaded_session)


def test_forcing_the_api_without_a_key_turns_advice_off(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(advise.ENV_BACKEND, "api")
    _with_cli(monkeypatch)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "AI fix advice is off" in text
    assert "Claude CLI" in text
    assert advise.ENV_API_KEY in text


def test_with_neither_the_panel_explains_both_ways(loaded_session: Session) -> None:
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "AI fix advice is off" in text
    assert "log in once (run claude)" in text
    assert advise.ENV_API_KEY in text
    assert f"{advise.ENV_BACKEND}=claude-cli" in text


def test_an_unknown_backend_name_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="ENTROPIQ_ADVISOR must be one of auto, api, claude-cli"):
        advise.choose_backend({advise.ENV_BACKEND: "openai"})


@pytest.mark.parametrize(
    ("environ", "found", "expected"),
    [
        ({}, None, None),
        ({}, list(CLI), advise.Backend("claude-cli", command=CLI)),
        ({advise.ENV_API_KEY: " k "}, list(CLI), advise.Backend("api", key="k")),
        (
            {advise.ENV_API_KEY: "k", advise.ENV_BACKEND: "CLAUDE-CLI"},
            list(CLI),
            advise.Backend("claude-cli", command=CLI),
        ),
        ({advise.ENV_API_KEY: "k", advise.ENV_BACKEND: "claude-cli"}, None, None),
        ({advise.ENV_BACKEND: "api"}, list(CLI), None),
        ({advise.ENV_BACKEND: " "}, list(CLI), advise.Backend("claude-cli", command=CLI)),
    ],
)
def test_choose_backend(
    monkeypatch: pytest.MonkeyPatch,
    environ: dict[str, str],
    found: list[str] | None,
    expected: advise.Backend | None,
) -> None:
    monkeypatch.setattr(advise, "claude_cli", lambda env=None: found)
    assert advise.choose_backend(environ) == expected


def test_backend_labels() -> None:
    assert advise.Backend("api", key="k").label == "the Anthropic API"
    assert advise.Backend("claude-cli", command=CLI).label == "the Claude CLI"


def test_a_logged_out_claude_cli_gets_the_login_hint(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_cli(monkeypatch, LLMError("the Claude CLI reported an error: Not logged in · Please run /login"))
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "The advice request failed" in text
    assert advise.LOGIN_HINT in text
    assert advise.RETRY_HINT not in text


def test_a_busy_claude_cli_gets_the_retry_hint(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_cli(monkeypatch, LLMError("the Claude CLI reported an error: overloaded", retryable=True))
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert advise.RETRY_HINT in text
    assert advise.LOGIN_HINT not in text
