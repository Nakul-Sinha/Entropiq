"""AnthropicClient against a stubbed SDK method: a real SDK client with a dummy key, no network."""

import logging
import sys
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from tdp_core.advisor.client import ANTHROPIC_FALLBACK_BETA, AnthropicClient, LLMError

MODEL = "claude-sonnet-5-5"


def message(
    *texts: str,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "x"}]
    content += [{"type": "text", "text": t} for t in texts]
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    )


class FakeCreate:
    """Stands in for ``sdk.beta.messages.create``: records keyword arguments, returns or raises."""

    def __init__(self, result: BetaMessage | Exception) -> None:
        self.result = result
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> BetaMessage:
        self.kwargs.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def sent(self) -> dict[str, Any]:
        """The last call's arguments without the omitted ones."""
        return {k: v for k, v in self.kwargs[-1].items() if not isinstance(v, anthropic.Omit)}


def stub(
    client: AnthropicClient, monkeypatch: pytest.MonkeyPatch, result: BetaMessage | Exception
) -> FakeCreate:
    fake = FakeCreate(result)
    monkeypatch.setattr(client._sdk.beta.messages, "create", fake)
    return fake


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="no Anthropic API key"):
        AnthropicClient(MODEL)


def test_empty_explicit_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    with pytest.raises(LLMError, match="no Anthropic API key"):
        AnthropicClient(MODEL, api_key="  ")


def test_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", " sk-from-env ")
    assert AnthropicClient(MODEL)._sdk.api_key == "sk-from-env"


def test_explicit_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    client = AnthropicClient(MODEL, api_key="sk-test")
    assert client._sdk.api_key == "sk-test"
    assert client.model == MODEL


def test_empty_model_raises() -> None:
    with pytest.raises(ValueError, match="model"):
        AnthropicClient(" ", api_key="sk-test")


def test_missing_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMError, match=r"tdp-core\[advisor\]"):
        AnthropicClient(MODEL, api_key="sk-test")


def test_timeout_and_retries_reach_the_sdk() -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    assert client._sdk.timeout == 300.0
    assert client._sdk.max_retries == 2
    other = AnthropicClient(MODEL, api_key="sk-test", timeout_s=30.0, max_retries=0)
    assert (other._sdk.timeout, other._sdk.max_retries) == (30.0, 0)


def test_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    fake = stub(client, monkeypatch, message("ok"))
    assert client.complete("SYSTEM", "USER", max_tokens=1234) == "ok"
    assert fake.sent() == {
        "model": MODEL,
        "max_tokens": 1234,
        "system": "SYSTEM",
        "messages": [{"role": "user", "content": "USER"}],
        "betas": [ANTHROPIC_FALLBACK_BETA],
        "fallbacks": "default",
    }


def test_effort_is_sent_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test", effort="medium")
    fake = stub(client, monkeypatch, message("ok"))
    client.complete("s", "u", max_tokens=10)
    assert fake.sent()["output_config"] == {"effort": "medium"}


def test_fallbacks_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test", fallbacks=False)
    fake = stub(client, monkeypatch, message("ok"))
    client.complete("s", "u", max_tokens=10)
    assert set(fake.sent()) == {"model", "max_tokens", "system", "messages"}


def test_text_blocks_are_concatenated(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, message("a", "b"))
    assert client.complete("s", "u", max_tokens=10) == "ab"


def test_no_text_gives_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, message())
    assert client.complete("s", "u", max_tokens=10) == ""


def test_refusal_raises_with_category(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    stub(client, monkeypatch, message(stop_reason="refusal", stop_details=details))
    with pytest.raises(LLMError, match=r"declined the request \(category cyber\)") as info:
        client.complete("s", "u", max_tokens=10)
    assert info.value.retryable is False


def test_refusal_without_details(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, message(stop_reason="refusal"))
    with pytest.raises(LLMError, match=r"declined the request$"):
        client.complete("s", "u", max_tokens=10)


@pytest.mark.parametrize("reason", ["max_tokens", "model_context_window_exceeded"])
def test_cut_off_reply_is_returned_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, reason: str
) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, message('{"summary": "cut', stop_reason=reason))
    with caplog.at_level(logging.WARNING, logger="tdp_core.advisor.client"):
        assert client.complete("s", "u", max_tokens=10) == '{"summary": "cut'
    assert reason in caplog.text
    assert "20 output tokens" in caplog.text


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    return cls("m", response=httpx2.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("error", "retryable", "fragment"),
    [
        (_status(anthropic.BadRequestError, 400), False, "HTTP 400"),
        (_status(anthropic.AuthenticationError, 401), False, "HTTP 401"),
        (_status(anthropic.RateLimitError, 429), True, "HTTP 429"),
        (_status(anthropic.InternalServerError, 500), True, "HTTP 500"),
        (_status(anthropic.APIStatusError, 529), True, "HTTP 529"),
        (_status(anthropic.APIStatusError, 408), True, "HTTP 408"),
        (anthropic.APIConnectionError(request=_REQUEST), True, "could not reach"),
        (anthropic.APITimeoutError(request=_REQUEST), True, "could not reach"),
        (anthropic.AnthropicError("odd"), False, "Anthropic client error: odd"),
    ],
)
def test_errors_are_mapped(
    monkeypatch: pytest.MonkeyPatch, error: Exception, retryable: bool, fragment: str
) -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, error)
    with pytest.raises(LLMError, match=fragment) as info:
        client.complete("s", "u", max_tokens=10)
    assert info.value.retryable is retryable
    assert info.value.__cause__ is error


def test_status_error_names_the_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx2.Response(500, request=_REQUEST, headers={"request-id": "req_123"})
    client = AnthropicClient(MODEL, api_key="sk-test")
    stub(client, monkeypatch, anthropic.InternalServerError("m", response=response, body=None))
    with pytest.raises(LLMError, match=r"HTTP 500 \(request req_123\)"):
        client.complete("s", "u", max_tokens=10)


def test_close() -> None:
    client = AnthropicClient(MODEL, api_key="sk-test")
    client.close()
    assert client._sdk.is_closed()
