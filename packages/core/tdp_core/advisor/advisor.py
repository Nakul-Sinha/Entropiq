"""The fix advisor: asks a language model for a fix plan for one file and keeps only grounded steps.

Rules:

- ``Advisor.advise`` sends the system prompt and the file's user prompt once. A reply that is not valid
  advice is retried exactly once, with the validation error appended to the user prompt (the bad reply is
  not repeated); a second invalid reply raises ``AdviceError``. A failed call (``LLMError``) is never
  retried or wrapped here: the SDK already retried what can be retried, and the caller decides with
  ``LLMError.retryable``.
- Grounding (``ground_advice``): a step is kept only when it has at least one target and every target,
  after ``normalize_target``, is in ``valid_targets``: a dotted suffix of a function's qualified name or of
  a scope that encloses a function (so ``Parser.parse``, ``parse`` and ``Parser`` count), the file path or
  its base name. Other steps are dropped and counted in ``dropped_steps``. Each kept step's
  ``effort_hours`` is sorted and clipped into ``[MIN_STEP_HOURS, max(MIN_STEP_HOURS, cost.high)]``, and
  kept steps are renumbered ``1..n`` in the model's order. Advice with no kept step is valid.
- ``advise`` trusts its ``path`` argument and ``syntax.functions``: the static cache is keyed by blob sha,
  so a cached ``FileSyntax`` may carry another path. Logs never contain the prompt or the code.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import PurePosixPath

from tdp_core.advisor.client import LLMClient
from tdp_core.advisor.prompt import (
    PROMPT_VERSION,
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)
from tdp_core.advisor.schema import Advice, AdviceError, AdviceStep, parse_advice
from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import CostEstimate, FileExtra, FileScore, FileSyntax

__all__ = [
    "ADVICE_MAX_TOKENS",
    "MIN_STEP_HOURS",
    "PROMPT_VERSION",
    "Advisor",
    "ground_advice",
    "normalize_target",
    "valid_targets",
]

logger = logging.getLogger(__name__)

# Room for adaptive thinking plus the JSON reply: max_tokens counts both.
ADVICE_MAX_TOKENS = 16_000
MIN_STEP_HOURS = 0.25

_CALL_SUFFIX = re.compile(r"\([^()]*\)$")


def normalize_target(target: str) -> str:
    """``target`` as a plain name.

    Surrounding whitespace, backticks and quotes are removed, then one trailing call suffix (``parse()`` and
    ``parse(data)`` give ``parse``), then a leading ``self.`` or ``cls.``.
    """
    name = target.strip().strip("`'\"").strip()
    name = _CALL_SUFFIX.sub("", name, count=1).strip()
    for prefix in ("self.", "cls."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def valid_targets(path: str, syntax: FileSyntax) -> frozenset[str]:
    """Every name a step may target in the file at ``path``.

    For each function's qualified name, every dotted prefix is a scope (``Parser._scan.keep`` gives
    ``Parser``, ``Parser._scan`` and ``Parser._scan.keep``), and every dotted suffix of a scope is valid
    (``_scan.keep``, ``keep``, ...). The path and its base name are valid too. Classes without functions
    are not known (``FileSyntax`` has no class names). Empty strings are never valid.
    """
    names: set[str] = {path, PurePosixPath(path).name}
    for fn in syntax.functions:
        parts = fn.name.split(".")
        for end in range(1, len(parts) + 1):
            scope = parts[:end]
            for start in range(len(scope)):
                names.add(".".join(scope[start:]))
    names.discard("")
    return frozenset(names)


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def ground_advice(advice: Advice, *, path: str, syntax: FileSyntax, cost: CostEstimate) -> Advice:
    """``advice`` with ungrounded steps dropped, efforts clipped and steps renumbered (see module rules).

    Steps are taken in ``order`` (ties keep the model's order). Targets are normalized, with empty and
    repeated ones removed. ``dropped_steps`` grows by the number of steps dropped now, so grounding twice
    drops nothing more. Empty strings are removed from ``tests_to_add`` and ``risks``. Returns a new
    ``Advice``.
    """
    allowed = valid_targets(path, syntax)
    upper = max(MIN_STEP_HOURS, cost.high)
    kept: list[AdviceStep] = []
    dropped = 0
    ordered = sorted(enumerate(advice.steps), key=lambda item: (item[1].order, item[0]))
    for _, step in ordered:
        targets = list(dict.fromkeys(t for t in map(normalize_target, step.targets) if t))
        if not targets or any(t not in allowed for t in targets):
            dropped += 1
            continue
        low, high = sorted(step.effort_hours)
        kept.append(
            AdviceStep(
                order=len(kept) + 1,
                action=step.action,
                targets=targets,
                effort_hours=(_clip(low, MIN_STEP_HOURS, upper), _clip(high, MIN_STEP_HOURS, upper)),
            )
        )
    return Advice(
        summary=advice.summary,
        steps=kept,
        tests_to_add=[t for t in advice.tests_to_add if t],
        risks=[r for r in advice.risks if r],
        confidence=advice.confidence,
        dropped_steps=advice.dropped_steps + dropped,
    )


class Advisor:
    """Produces grounded ``Advice`` for one file with an ``LLMClient``."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings = DEFAULT_SETTINGS,
        *,
        max_tokens: int = ADVICE_MAX_TOKENS,
    ) -> None:
        """Use ``client`` with ``settings.advisor_max_excerpt_lines`` (0 sends no code) and ``max_tokens`` per
        call. Raises ``ValueError`` when the excerpt cap is negative or ``max_tokens < 1``.
        """
        if settings.advisor_max_excerpt_lines < 0:
            raise ValueError(
                f"advisor_max_excerpt_lines must be >= 0, got {settings.advisor_max_excerpt_lines}"
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._client = client
        self._settings = settings
        self._max_tokens = max_tokens

    def advise(
        self, *, path: str, source: str, syntax: FileSyntax, score: FileScore, extra: FileExtra
    ) -> Advice:
        """Grounded fix advice for the file at repo-relative ``path``.

        ``source`` is the decoded file text whose line numbers match ``syntax``; the step targets are
        checked against ``syntax.functions`` and ``path``, and efforts against ``extra.cost``. Makes one
        call, or two when the first reply is not valid advice. Raises ``LLMError`` when a call fails and
        ``AdviceError`` when the retry is not valid advice either.
        """
        system = build_system_prompt()
        user = build_user_prompt(
            path=path,
            source=source,
            syntax=syntax,
            score=score,
            extra=extra,
            max_excerpt_lines=self._settings.advisor_max_excerpt_lines,
        )
        reply = self._client.complete(system, user, max_tokens=self._max_tokens)
        try:
            draft = parse_advice(reply)
        except AdviceError as first:
            logger.warning("advisor reply for %s was not valid advice (%s); retrying once", path, first)
            retry = self._client.complete(
                system, build_retry_prompt(user, str(first)), max_tokens=self._max_tokens
            )
            try:
                draft = parse_advice(retry)
            except AdviceError as second:
                raise AdviceError(f"no valid advice for {path} after one retry: {second}") from second
        advice = ground_advice(draft, path=path, syntax=syntax, cost=extra.cost)
        logger.info("advice for %s: %d steps kept, %d dropped", path, len(advice.steps), advice.dropped_steps)
        return advice

    @staticmethod
    def cache_key(blob_sha: str, model: str) -> str:
        """Hex SHA-256 of ``"<blob_sha>|<model>|<PROMPT_VERSION>"``: 64 lowercase hex characters."""
        return hashlib.sha256(f"{blob_sha}|{model}|{PROMPT_VERSION}".encode()).hexdigest()
