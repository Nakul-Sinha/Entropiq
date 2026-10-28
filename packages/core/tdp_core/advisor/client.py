"""Language model clients for the advisor: the ``LLMClient`` protocol, a test fake and the Claude client.

Rules:

- A client turns one system prompt and one user message into the model's full text reply. Failed calls
  raise ``LLMError`` (an ``AdviceError``); ``LLMError.retryable`` tells the caller whether trying again
  later may help.
- ``anthropic`` is an optional dependency (``tdp-core[advisor]``) and is imported only inside
  ``AnthropicClient``, so importing this module never loads it.
- ``AnthropicClient`` uses the first-party Claude API only and never sends sampling parameters, a thinking
  configuration or an assistant prefill: current models reject non-default sampling and prefill, and
  adaptive thinking is their default. Its key is the ``api_key`` argument or ``ANTHROPIC_API_KEY``, passed
  explicitly so the SDK never falls back to a personal login, a profile or federation.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from tdp_core.advisor.schema import AdviceError

if TYPE_CHECKING:
    import anthropic

__all__ = [
    "ANTHROPIC_FALLBACK_BETA",
    "AnthropicClient",
    "Effort",
    "FakeLLMClient",
    "LLMCall",
    "LLMClient",
    "LLMError",
]

logger = logging.getLogger(__name__)

ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMError(AdviceError):
    """A model call failed: network, HTTP status, refusal, or a missing API key or package.

    ``retryable`` is True when the same call may succeed later (rate limits, overload, server errors,
    timeouts, connection errors), after the SDK's own retries were used up.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMClient(Protocol):
    """Anything that can answer one system prompt and one user message with text."""

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """The model's full text reply to ``system`` and one ``user`` message.

        ``max_tokens`` caps the generated tokens, thinking included. Raises ``LLMError`` when the call fails.
        """
        ...


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One recorded call to ``FakeLLMClient.complete``."""

    system: str
    user: str
    max_tokens: int


class FakeLLMClient:
    """An ``LLMClient`` for tests: returns queued replies in order and records every call.

    Each queued item is a reply string, or an exception that ``complete`` raises instead. ``calls`` holds one
    ``LLMCall`` per call, recorded before the reply is taken, so a call that raises is recorded too.
    """

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        """Queue ``responses`` (copied). Raises ``TypeError`` for a bare string, which is a common mistake."""
        if isinstance(responses, str):
            raise TypeError("responses must be a sequence of replies, not a single string")
        self._queue: deque[str | Exception] = deque(responses)
        self.calls: list[LLMCall] = []

    @property
    def remaining(self) -> int:
        """Number of queued replies not used yet."""
        return len(self._queue)

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """Record the call, then return the next queued reply or raise the next queued exception.

        Raises ``RuntimeError`` (deliberately not an ``AdviceError``) when the queue is empty, because that
        is a mistake in the test setup.
        """
        self.calls.append(LLMCall(system, user, max_tokens))
        if not self._queue:
            raise RuntimeError(f"FakeLLMClient has no queued response for call {len(self.calls)}")
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class AnthropicClient:
    """An ``LLMClient`` backed by the Claude API (first party; Bedrock and Vertex need other classes).

    The server-side refusal fallback is on by default (``fallbacks=True``): when a safety classifier
    declines the request, the API retries it on another model inside the same call, which is recommended
    for ``claude-sonnet-5-5``. On a model without fallback support the ``"default"`` form runs the
    requested model alone. ``fallbacks=False`` sends neither the beta header nor the field. ``effort`` sets
    ``output_config.effort``; ``None`` keeps the model's default.

    Requests are not streamed and use the client ``timeout_s`` (seconds) per attempt; the SDK retries
    ``max_retries`` times on 408, 409, 429, 5xx and connection errors before ``complete`` raises.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        timeout_s: float = 300.0,
        max_retries: int = 2,
        effort: Effort | None = None,
        fallbacks: bool = True,
    ) -> None:
        """Create the SDK client for ``model``.

        The key is ``api_key`` or, when it is ``None``, the ``ANTHROPIC_API_KEY`` environment variable. Raises
        ``LLMError`` when the ``anthropic`` package is missing or the key is empty, and ``ValueError`` when
        ``model`` is empty.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("the anthropic package is not installed; install tdp-core[advisor]") from exc
        key = (api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        if not key:
            raise LLMError("no Anthropic API key: pass api_key or set ANTHROPIC_API_KEY")
        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model
        self._effort = effort
        self._fallbacks = fallbacks
        self._sdk: anthropic.Anthropic = anthropic.Anthropic(
            api_key=key, timeout=timeout_s, max_retries=max_retries
        )

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """The reply's text blocks, concatenated (thinking and fallback blocks are skipped; may be ``""``).

        Raises ``LLMError``: retryable for HTTP 408, 409, 429 and 5xx, timeouts and connection errors; not
        retryable for other HTTP errors, other client errors and refusals (the message names the refusal
        category when the API gives one). A reply cut off by ``max_tokens`` or the context window is logged
        as a warning and still returned, so the caller's parse and retry path handles it.
        """
        import anthropic
        from anthropic.types.beta import BetaTextBlock

        try:
            message = self._sdk.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self._effort} if self._effort is not None else anthropic.omit,
                betas=[ANTHROPIC_FALLBACK_BETA] if self._fallbacks else anthropic.omit,
                fallbacks="default" if self._fallbacks else anthropic.omit,
            )
        except anthropic.APIStatusError as exc:
            rid = f" (request {exc.request_id})" if exc.request_id else ""
            status = exc.status_code
            raise LLMError(
                f"Anthropic API returned HTTP {status}{rid}: {exc.message}",
                retryable=status in (408, 409, 429) or status >= 500,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Anthropic API: {exc}", retryable=True) from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic client error: {exc}") from exc

        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details is not None else None
            detail = f" (category {category})" if category else ""
            raise LLMError(f"the model declined the request{detail}", retryable=False)
        if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            logger.warning(
                "Anthropic reply stopped early (%s) after %d output tokens",
                message.stop_reason,
                message.usage.output_tokens,
            )
        return "".join(block.text for block in message.content if isinstance(block, BetaTextBlock))

    def close(self) -> None:
        """Close the SDK client's HTTP connections."""
        self._sdk.close()
