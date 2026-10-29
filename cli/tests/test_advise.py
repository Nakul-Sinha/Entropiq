from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tdp_cli.commands import advise
from tdp_cli.errors import CliError, UsageError
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import AnalysisStore, LoadedAnalysis
from tdp_core.advisor.client import FakeLLMClient, LLMError
from tdp_core.advisor.schema import Advice
from tdp_core.config import DEFAULT_SETTINGS

runner = CliRunner()
MODEL = DEFAULT_SETTINGS.advisor_model
REPLY = json.dumps(
    {
        "summary": "Split the long helpers and add tests for negative input.",
        "steps": [
            {
                "order": 1,
                "action": "Merge mu_0 and mu_1 into one helper",
                "targets": ["mu_0", "mu_1"],
                "effort_hours": [0.5, 1.0],
            },
            {
                "order": 2,
                "action": "Rewrite the parser",
                "targets": ["parse_everything"],
                "effort_hours": [1, 2],
            },
        ],
        "tests_to_add": ["negative input for mu_0"],
        "risks": ["callers rely on the sign"],
        "confidence": "medium",
    }
)


def _refuse(model: str, key: str) -> Any:
    raise AssertionError("a test tried to build a real LLM client")


def _refuse_cli(model: str, command: Any) -> Any:
    raise AssertionError("a test tried to build a real Claude CLI client")


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may reach a real model: no key, no Claude CLI, and both factories refuse."""
    monkeypatch.delenv(advise.ENV_API_KEY, raising=False)
    monkeypatch.delenv(advise.ENV_BACKEND, raising=False)
    monkeypatch.setattr(advise, "client_factory", _refuse)
    monkeypatch.setattr(advise, "cli_client_factory", _refuse_cli)
    monkeypatch.setattr(advise, "claude_cli", lambda environ=None: None)


@pytest.fixture
def with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")


@pytest.fixture
def saved_home(dev_home: Path, sample_loaded: LoadedAnalysis) -> Path:
    """The test home with ``sample_loaded`` saved, so ``tdp advise ... <sample_repo>`` reuses it."""
    AnalysisStore(EntropiqHome.from_env().analyses).save(sample_loaded)
    return dev_home


def fake(
    monkeypatch: pytest.MonkeyPatch, *replies: str | Exception
) -> tuple[Callable[[str, str], FakeLLMClient], FakeLLMClient]:
    client = FakeLLMClient(list(replies))
    seen: list[tuple[str, str]] = []

    def factory(model: str, key: str) -> FakeLLMClient:
        seen.append((model, key))
        return client

    monkeypatch.setattr(advise, "client_factory", factory)
    return factory, client


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined]


def _cache_file(session: Session, loaded: LoadedAnalysis, path: str = "pkg/mu.py") -> Path:
    blob = loaded.result.extras[path].blob_sha
    return advise.cache_path(advise.advice_dir(session.home), blob, MODEL)


def test_advice_is_asked_shown_and_saved(
    loaded_session: Session, sample_loaded: LoadedAnalysis, monkeypatch: pytest.MonkeyPatch, with_key: None
) -> None:
    _, client = fake(monkeypatch, REPLY)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert f"Asking {MODEL} for fix advice on pkg/mu.py" in text
    assert "AI suggestions for pkg/mu.py" in text
    assert "1. Merge mu_0 and mu_1 into one helper" in text
    assert "Targets: mu_0, mu_1" in text
    assert "Effort: 0.5-1.0 h" in text
    assert "Tests to add" in text
    assert "negative input for mu_0" in text
    assert "Risks" in text
    assert "Confidence: medium" in text
    assert "1 suggested step was dropped" in text
    assert "review them" in text
    assert "parse_everything" not in text
    assert "Saved" not in text
    assert len(client.calls) == 1
    assert "pkg/mu.py" in client.calls[0].user
    assert "<code_excerpt>" in client.calls[0].user
    cache = _cache_file(loaded_session, sample_loaded)
    assert cache.is_file()
    stored = json.loads(cache.read_text(encoding="utf-8"))
    assert stored["schema"] == advise.CACHE_SCHEMA
    assert stored["model"] == MODEL
    assert stored["path"] == "pkg/mu.py"
    assert stored["created_at"].endswith("Z")

    loaded_session.console.file.truncate(0)  # type: ignore[attr-defined]
    loaded_session.console.file.seek(0)  # type: ignore[attr-defined]
    dispatch(loaded_session, "/advice 1")
    again = _out(loaded_session)
    assert "Saved " in again
    assert "--refresh" in again
    assert len(client.calls) == 1


def test_saved_advice_shows_without_a_key_and_refresh_asks_again(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")
    _, client = fake(monkeypatch, REPLY, REPLY)
    dispatch(loaded_session, "/advise mu.py")
    monkeypatch.delenv(advise.ENV_API_KEY)
    dispatch(loaded_session, "/advise mu.py")
    assert "Saved " in _out(loaded_session)
    assert "AI fix advice is off" not in _out(loaded_session)
    assert len(client.calls) == 1

    dispatch(loaded_session, "/advise mu.py --refresh")
    assert "AI fix advice is off" in _out(loaded_session)
    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")
    dispatch(loaded_session, "/advise mu.py --refresh")
    assert len(client.calls) == 2


def test_no_key_shows_how_to_turn_it_on(loaded_session: Session) -> None:
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "AI fix advice is off" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "run /advise again" in text
    assert f"up to {DEFAULT_SETTINGS.advisor_max_excerpt_lines} lines" in text
    assert "Error" not in text


@pytest.mark.parametrize(
    ("replies", "message", "hint"),
    [
        ([LLMError("busy", retryable=True)], "The advice request failed: busy", advise.RETRY_HINT),
        ([LLMError("bad request")], "The advice request failed: bad request", None),
        (["no json here", "still nothing"], "No usable advice for pkg/mu.py", None),
    ],
)
def test_request_errors(
    loaded_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    with_key: None,
    replies: list[str | Exception],
    message: str,
    hint: str | None,
) -> None:
    fake(monkeypatch, *replies)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert message in text
    if hint is None:
        assert advise.RETRY_HINT not in text
    else:
        assert hint in text


def test_missing_package_gives_the_install_hint(
    loaded_session: Session, monkeypatch: pytest.MonkeyPatch, with_key: None
) -> None:
    def broken(model: str, key: str) -> Any:
        raise LLMError("the anthropic package is not installed; install tdp-core[advisor]")

    monkeypatch.setattr(advise, "client_factory", broken)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "AI advice is unavailable" in text
    assert advise.INSTALL_HINT in text


def test_unreadable_repository(
    loaded_session: Session, sample_loaded: LoadedAnalysis, tmp_path: Path, with_key: None
) -> None:
    meta = dataclasses.replace(sample_loaded.meta, repo_dir=str(tmp_path / "moved"))
    loaded_session.current = LoadedAnalysis(meta, sample_loaded.result)
    dispatch(loaded_session, "/advise pkg/mu.py")
    text = _out(loaded_session)
    assert "Cannot read pkg/mu.py" in text
    assert "run /analyze on it again" in text


def test_read_source_rejects_files_without_a_parser(sample_loaded: LoadedAnalysis) -> None:
    blob = sample_loaded.result.extras["pkg/mu.py"].blob_sha
    with pytest.raises(CliError, match="not a file Entropiq can parse"):
        advise.read_source(sample_loaded, "notes.txt", blob)
    source, syntax = advise.read_source(sample_loaded, "pkg/mu.py", blob)
    assert source.startswith("def mu_0(x):")
    assert len(syntax.functions) == 12


def test_corrupt_cache_is_ignored_and_replaced(
    loaded_session: Session, sample_loaded: LoadedAnalysis, monkeypatch: pytest.MonkeyPatch, with_key: None
) -> None:
    cache = _cache_file(loaded_session, sample_loaded)
    cache.parent.mkdir(parents=True)
    cache.write_text("{not json", encoding="utf-8")
    assert advise.read_cached(cache) is None
    _, client = fake(monkeypatch, REPLY)
    dispatch(loaded_session, "/advise pkg/mu.py")
    assert len(client.calls) == 1
    entry = advise.read_cached(cache)
    assert entry is not None
    assert entry.cached
    assert entry.model == MODEL


def test_read_cached_edge_cases(tmp_path: Path) -> None:
    assert advise.read_cached(tmp_path / "missing.json") is None
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    assert advise.read_cached(wrong) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"schema": 1, "model": "m", "created_at": "2026-01-01T00:00:00"}))
    assert advise.read_cached(partial) is None
    naive = tmp_path / "naive.json"
    advice = Advice.model_validate(json.loads(REPLY))
    body = {"schema": 1, "model": "m", "created_at": "2026-01-01T10:00:00", "advice": advice.model_dump()}
    naive.write_text(json.dumps(body), encoding="utf-8")
    entry = advise.read_cached(naive)
    assert entry is not None
    assert entry.created_at.tzinfo is not None


def test_save_cached_failure_is_not_fatal(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    advice = Advice.model_validate(json.loads(REPLY))
    from datetime import UTC, datetime

    entry = advise.AdviceEntry(advice, "m", datetime(2026, 1, 1, tzinfo=UTC))
    assert not advise.save_cached(blocker / "sub" / "a.json", entry, file_path="a.py", blob_sha="abc")
    assert list(tmp_path.iterdir()) == [blocker]


def test_api_key() -> None:
    assert advise.api_key({"ANTHROPIC_API_KEY": "  "}) is None
    assert advise.api_key({}) is None
    assert advise.api_key({"ANTHROPIC_API_KEY": " k "}) == "k"


def test_anthropic_client_builds_the_platform_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import tdp_core.advisor.client as client_module

    made: list[tuple[str, str | None]] = []

    class Stub:
        def __init__(self, model: str, api_key: str | None = None) -> None:
            made.append((model, api_key))

    monkeypatch.setattr(client_module, "AnthropicClient", Stub)
    assert isinstance(advise.anthropic_client("m", "k"), Stub)
    assert made == [("m", "k")]


def test_usage_errors(loaded_session: Session) -> None:
    with pytest.raises(UsageError, match="say which file"):
        advise.advise_handler(loaded_session, [])
    dispatch(loaded_session, "/advise nothing.py")
    assert "nothing.py is not a ranked file" in _out(loaded_session)


def test_missing_extras(loaded_session: Session, sample_loaded: LoadedAnalysis) -> None:
    extras = {p: e for p, e in sample_loaded.result.extras.items() if p != "pkg/mu.py"}
    result = dataclasses.replace(sample_loaded.result, extras=extras)
    loaded_session.current = LoadedAnalysis(sample_loaded.meta, result)
    dispatch(loaded_session, "/advise pkg/mu.py")
    assert "has no saved details" in _out(loaded_session)


def test_advice_view_without_steps() -> None:
    from datetime import UTC, datetime

    from rich.console import Console

    base = {"summary": "s", "steps": [], "tests_to_add": [], "risks": [], "confidence": "low"}
    when = datetime(2026, 1, 1, tzinfo=UTC)
    for dropped, expected in ((2, advise.ALL_DROPPED), (0, advise.NO_STEPS)):
        advice = Advice.model_validate(base).model_copy(update={"dropped_steps": dropped})
        console = Console(record=True, width=200, color_system=None)
        console.print(advise.advice_view("a.py", advise.AdviceEntry(advice, "m", when)))
        text = console.export_text()
        assert expected in text
        assert "Tests to add" not in text
        assert ("2 suggested steps were dropped" in text) == (dropped == 2)


def test_completion(loaded_session: Session) -> None:
    assert [s.text for s in advise._complete(loaded_session, [], "--r")] == ["--refresh"]
    assert "pkg/mu.py" in [s.text for s in advise._complete(loaded_session, [], "pkg/m")]
    assert list(advise._complete(loaded_session, ["pkg/mu.py"], "pkg")) == []


def test_advise_cli(saved_home: Path, sample_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_app()
    args = ["advise", "pkg/mu.py", str(sample_repo), "--dev-model"]
    off = runner.invoke(app, args)
    assert off.exit_code == 1
    assert off.stdout == ""
    assert "AI fix advice is off" in off.stderr
    assert "run tdp advise again" in off.stderr

    monkeypatch.setenv(advise.ENV_API_KEY, "test-key")
    _, client = fake(monkeypatch, REPLY)
    first = runner.invoke(app, [*args, "--format", "json"])
    assert first.exit_code == 0, first.stderr
    data = json.loads(first.stdout)
    assert data["schema"] == "entropiq.advice.v1"
    assert data["path"] == "pkg/mu.py"
    assert data["model"] == MODEL
    assert data["cached"] is False
    assert len(data["advice"]["steps"]) == 1
    assert data["advice"]["dropped_steps"] == 1

    second = runner.invoke(app, [*args, "-f", "json"])
    assert second.exit_code == 0, second.stderr
    assert json.loads(second.stdout)["cached"] is True
    assert len(client.calls) == 1

    text = runner.invoke(app, ["advise", "1", str(sample_repo), "--dev-model"])
    assert text.exit_code == 0, text.stderr
    assert "AI suggestions for pkg/mu.py" in text.stdout

    bad = runner.invoke(app, ["advise", "nope.py", str(sample_repo), "--dev-model"])
    assert bad.exit_code == 2
    assert "nope.py is not a ranked file" in bad.stderr
