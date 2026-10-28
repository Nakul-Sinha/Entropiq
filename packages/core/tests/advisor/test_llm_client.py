import pytest

from tdp_core.advisor.client import FakeLLMClient, LLMCall, LLMClient, LLMError
from tdp_core.advisor.schema import AdviceError


def test_fake_returns_replies_in_order_and_records_calls() -> None:
    fake = FakeLLMClient(["first", "second"])
    assert fake.remaining == 2
    assert fake.complete("sys", "user 1", max_tokens=10) == "first"
    assert fake.complete("sys", "user 2", max_tokens=20) == "second"
    assert fake.calls == [LLMCall("sys", "user 1", 10), LLMCall("sys", "user 2", 20)]
    assert fake.remaining == 0


def test_fake_raises_a_queued_exception_after_recording() -> None:
    error = LLMError("boom", retryable=True)
    fake = FakeLLMClient([error, "after"])
    with pytest.raises(LLMError) as info:
        fake.complete("s", "u", max_tokens=1)
    assert info.value is error
    assert len(fake.calls) == 1
    assert fake.complete("s", "u", max_tokens=1) == "after"


def test_exhausted_fake_raises_runtime_error_naming_the_call() -> None:
    fake = FakeLLMClient(["only"])
    fake.complete("s", "u", max_tokens=1)
    with pytest.raises(RuntimeError, match="call 2") as info:
        fake.complete("s", "u", max_tokens=1)
    assert not isinstance(info.value, AdviceError)
    assert len(fake.calls) == 2


def test_fake_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError, match="single string"):
        FakeLLMClient("reply")


def test_fake_copies_its_input() -> None:
    replies = ["a"]
    fake = FakeLLMClient(replies)
    replies.append("b")
    assert fake.remaining == 1


def test_fake_satisfies_the_protocol() -> None:
    client: LLMClient = FakeLLMClient([])
    assert isinstance(client, FakeLLMClient)


def test_llm_error_is_an_advice_error() -> None:
    error = LLMError("x", retryable=True)
    assert isinstance(error, AdviceError)
    assert isinstance(error, RuntimeError)
    assert error.retryable is True
    assert LLMError("y").retryable is False
    assert str(error) == "x"
