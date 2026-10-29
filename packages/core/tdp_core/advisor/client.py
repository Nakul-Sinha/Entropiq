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
- ``ClaudeCLIClient`` runs the Claude Code CLI (``claude -p``) instead, so the request uses the person's own
  Claude login and needs no API key. It turns off every tool, MCP server, setting file, slash command and
  saved session, runs in an empty temporary folder, and never passes ``ANTHROPIC_API_KEY`` on, so the CLI
  answers the one prompt with the login and nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from tdp_core.advisor.schema import AdviceError

if TYPE_CHECKING:
    import anthropic

__all__ = [
    "ANTHROPIC_FALLBACK_BETA",
    "ENV_CLAUDE_CLI",
    "AnthropicClient",
    "ClaudeCLIClient",
    "Effort",
    "FakeLLMClient",
    "LLMCall",
    "LLMClient",
    "LLMError",
    "find_claude_cli",
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


ENV_CLAUDE_CLI: Final = "ENTROPIQ_CLAUDE_CLI"
"""Environment variable naming the Claude CLI executable, for when ``claude`` is not on ``PATH``."""

_NPM_NATIVE: Final = ("node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")
_RETRYABLE_WORDS: Final = (
    "rate limit",
    "rate_limit",
    "overloaded",
    "too many requests",
    "timed out",
    "timeout",
    "connection",
    "network",
    "temporarily",
    "529",
    "503",
    "502",
    "500",
)
_TAIL_CHARS: Final = 400


def find_claude_cli(environ: Mapping[str, str] | None = None) -> list[str] | None:
    """The command that starts the Claude Code CLI, or ``None`` when it is not installed.

    ``ENTROPIQ_CLAUDE_CLI`` wins when it names an existing file. Otherwise ``claude`` is looked up on the
    ``PATH`` of ``environ`` (default ``os.environ``). On Windows the npm shim ``claude.cmd`` is replaced by
    the ``claude.exe`` it starts, so the arguments never pass through ``cmd.exe`` and its quoting rules.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(ENV_CLAUDE_CLI, "").strip()
    if explicit:
        return [explicit] if Path(explicit).is_file() else None
    found = shutil.which("claude", path=env.get("PATH"))
    if found is None:
        return None
    path = Path(found)
    if path.suffix.lower() in {".cmd", ".bat"}:
        native = path.parent.joinpath(*_NPM_NATIVE)
        if native.is_file():
            return [str(native)]
    return [str(path)]


def _tail(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _TAIL_CHARS else "..." + flat[-_TAIL_CHARS:]


def _looks_retryable(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _RETRYABLE_WORDS)


class ClaudeCLIClient:
    """An ``LLMClient`` that asks the Claude Code CLI (``claude -p``), using the person's own Claude login.

    Every call starts ``command`` in print mode with JSON output and ``model``, the system prompt in a file
    and the user message on standard input. Tools, MCP servers, setting files, slash commands and session
    saving are all turned off, the process runs in an empty temporary folder, and ``ANTHROPIC_API_KEY`` is
    removed from its environment so the login, not a key, pays for the request. ``max_tokens`` becomes
    ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` unless that is already set. Each call may take ``timeout_s`` seconds.
    """

    def __init__(self, model: str, command: Sequence[str], *, timeout_s: float = 300.0) -> None:
        """Use ``command`` (for example the result of ``find_claude_cli()``) for ``model``.

        Raises ``ValueError`` when ``model`` or ``command`` is empty or ``timeout_s`` is not positive.
        """
        if not model.strip():
            raise ValueError("model must not be empty")
        if not command:
            raise ValueError("command must not be empty")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.model = model
        self._command = [str(part) for part in command]
        self._timeout_s = timeout_s

    def argv(self, system_prompt_file: Path) -> list[str]:
        """The full command line of one call, with the system prompt read from ``system_prompt_file``."""
        return [
            *self._command,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--system-prompt-file",
            str(system_prompt_file),
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        """The CLI's text reply (may be ``""``).

        Raises ``LLMError``: retryable for timeouts and for errors that read like rate limits, overload or
        connection problems; not retryable when the CLI cannot start, is not logged in, or fails otherwise.
        """
        env = {name: value for name, value in os.environ.items() if name.upper() != "ANTHROPIC_API_KEY"}
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", str(max_tokens))
        with tempfile.TemporaryDirectory(prefix="entropiq-advice-") as folder:
            prompt_file = Path(folder) / "system-prompt.txt"
            prompt_file.write_text(system, encoding="utf-8")
            try:
                done = subprocess.run(
                    self.argv(prompt_file),
                    input=user,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=folder,
                    env=env,
                    timeout=self._timeout_s,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise LLMError(f"the Claude CLI was not found at {self._command[0]}") from exc
            except subprocess.TimeoutExpired as exc:
                raise LLMError(
                    f"the Claude CLI did not answer within {self._timeout_s:.0f} s", retryable=True
                ) from exc
            except OSError as exc:
                raise LLMError(f"could not start the Claude CLI: {exc}") from exc
        return _cli_reply(done.returncode, done.stdout, done.stderr)


def _cli_reply(code: int, stdout: str, stderr: str) -> str:
    """The ``result`` text of one ``claude -p --output-format json`` run; raises ``LLMError`` on failure."""
    data: object = None
    text = stdout.strip()
    if text:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        result = data.get("result")
        failed = bool(data.get("is_error")) or data.get("subtype", "success") != "success" or code != 0
        if failed:
            detail = str(result or data.get("subtype") or stderr.strip() or f"exit code {code}")
            raise LLMError(
                f"the Claude CLI reported an error: {_tail(detail)}", retryable=_looks_retryable(detail)
            )
        return result if isinstance(result, str) else ""
    detail = stderr.strip() or text or "no output"
    raise LLMError(
        f"the Claude CLI failed (exit code {code}): {_tail(detail)}", retryable=_looks_retryable(detail)
    )
