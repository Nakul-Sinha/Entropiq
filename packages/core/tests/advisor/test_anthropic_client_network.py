"""One real call to the Claude API (opt-in: ``uv run pytest -m network`` with ANTHROPIC_API_KEY set)."""

import os

import pytest

from tdp_core.advisor.advisor import Advisor
from tdp_core.advisor.client import AnthropicClient
from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import FileExtra, FileScore, FileSyntax

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"),
]


def test_real_advice_for_the_sample_file(
    sample_path: str,
    sample_source: str,
    sample_syntax: FileSyntax,
    sample_score: FileScore,
    sample_extra: FileExtra,
) -> None:
    client = AnthropicClient(DEFAULT_SETTINGS.advisor_model)
    try:
        advice = Advisor(client).advise(
            path=sample_path,
            source=sample_source,
            syntax=sample_syntax,
            score=sample_score,
            extra=sample_extra,
        )
    finally:
        client.close()
    assert advice.summary
    assert advice.confidence in ("low", "medium", "high")
    assert all(low <= high <= sample_extra.cost.high for low, high in (s.effort_hours for s in advice.steps))
    assert "pwned" not in advice.summary
