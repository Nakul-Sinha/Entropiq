"""The advice model returned by the LLM advisor, its errors, and parsing of the model's reply.

Rules:

- ``Advice`` and ``AdviceStep`` match the ``Advice`` type of the HTTP API (plan 3.7):
  ``Advice.model_dump(mode="json")`` gives exactly those keys, with ``effort_hours`` as a list of two
  floats.
- Both models are frozen. Strings are stripped of surrounding whitespace, NaN and infinite numbers are
  rejected, ``summary`` and ``action`` must not be empty, ``confidence`` is matched case-insensitively and
  unknown keys are ignored.
- ``dropped_steps`` is computed by Entropiq (the grounding check in ``tdp_core.advisor.advisor``), never by
  the model: ``response_schema()`` leaves it out and ``parse_advice`` discards any value the model sends.
- ``parse_advice`` raises ``AdviceError`` with a short message that names the failing fields but never
  repeats input values or links, because the message is sent back to the model when the advisor retries.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "Advice",
    "AdviceError",
    "AdviceStep",
    "Confidence",
    "extract_json_object",
    "parse_advice",
    "response_schema",
]

Confidence = Literal["low", "medium", "high"]

_MAX_REPORTED_ERRORS = 10


class AdviceError(RuntimeError):
    """The advisor could not produce valid advice.

    ``tdp_core.advisor.client.LLMError``, a subclass, covers failed model calls.
    """


_MODEL_CONFIG = ConfigDict(frozen=True, str_strip_whitespace=True, allow_inf_nan=False)


class AdviceStep(BaseModel):
    """One step of a fix plan."""

    model_config = _MODEL_CONFIG

    order: int = Field(description="1-based position of the step in the fix order.")
    action: str = Field(min_length=1, description="What to change, in one or two sentences.")
    targets: list[str] = Field(
        description=(
            'Names from the "Functions in this file" list (qualified name such as Parser.parse, or its '
            "last segment), the class that holds such a function, or the file path for a change to the "
            "whole file."
        )
    )
    effort_hours: tuple[float, float] = Field(description="[low, high] estimate in hours for one engineer.")


class Advice(BaseModel):
    """A short fix plan for one file."""

    model_config = _MODEL_CONFIG

    summary: str = Field(
        min_length=1, description="Two or three sentences on what makes the file costly to maintain."
    )
    steps: list[AdviceStep] = Field(description="Concrete steps in the order to do them.")
    tests_to_add: list[str] = Field(
        description="Behaviour that tests should cover before or during the changes."
    )
    risks: list[str] = Field(description="What could break while making the changes.")
    confidence: Confidence = Field(description="How well the facts and the code support the advice.")
    dropped_steps: int = Field(
        default=0, ge=0, description="Steps removed by the grounding check (set by Entropiq)."
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value: object) -> object:
        """Accept ``"High"`` or ``" medium "``: strip and lowercase strings, pass anything else on."""
        if isinstance(value, str):
            return value.strip().lower()
        return value


def response_schema() -> dict[str, Any]:
    """JSON schema of the reply the model must write: ``Advice`` without ``dropped_steps``.

    Returns a fresh dict on every call, so callers may change it.
    """
    schema = Advice.model_json_schema()
    schema["properties"].pop("dropped_steps", None)
    schema["required"] = [name for name in schema.get("required", []) if name != "dropped_steps"]
    return schema


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text``, or ``None`` when there is none.

    Every ``{`` is tried in order as the start of a JSON value; the first one that decodes to an object
    wins. So prose, Markdown fences and text after the object are ignored, and an invalid ``{...}`` before
    the real object is skipped. ``NaN``, ``Infinity`` and ``-Infinity`` are not JSON and make that candidate
    invalid. When the first valid object starts inside an invalid one (for example a reply cut off by the
    token limit), that inner object is returned.
    """
    pos = text.find("{")
    while pos != -1:
        try:
            value, _ = _DECODER.raw_decode(text, pos)
        except (ValueError, RecursionError):
            pass
        else:
            if isinstance(value, dict):
                return value
        pos = text.find("{", pos + 1)
    return None


def _describe(exc: ValidationError) -> str:
    """``"<loc>: <msg>"`` for up to ``_MAX_REPORTED_ERRORS`` errors, joined with ``"; "``.

    Input values, context and documentation links are left out.
    """
    parts = []
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    for error in errors[:_MAX_REPORTED_ERRORS]:
        loc = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{loc}: {error['msg']}")
    if len(errors) > _MAX_REPORTED_ERRORS:
        parts.append(f"and {len(errors) - _MAX_REPORTED_ERRORS} more errors")
    return "; ".join(parts)


def parse_advice(text: str) -> Advice:
    """Validate the model's reply ``text`` into ``Advice`` (with ``dropped_steps == 0``).

    The first JSON object in the text is used (see ``extract_json_object``); a ``dropped_steps`` key in it
    is discarded. Raises ``AdviceError`` when the text holds no JSON object or the object does not match the
    schema; the message lists the failing fields (at most 10) without their values.
    """
    obj = extract_json_object(text)
    if obj is None:
        raise AdviceError("the reply contains no JSON object")
    obj.pop("dropped_steps", None)
    try:
        return Advice.model_validate(obj)
    except ValidationError as exc:
        raise AdviceError("the JSON object does not match the schema: " + _describe(exc)) from exc
