"""LLM advice for one stored file, with a fake LLM client, an in-memory store and a local mirror."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tdp_core.advisor.advisor import Advisor
from tdp_core.advisor.client import FakeLLMClient, LLMClient, LLMError
from tdp_core.contracts import FAMILY_ORDER, FEATURE_INPUT_COLUMNS, CostEstimate, FileExtra, Reason
from tdp_core.parsing import parse_python
from tdp_core.serialize import to_jsonable
from tdp_platform.db.rows import FileResultRow
from tdp_platform.settings import PlatformSettings, get_settings
from tdp_worker import tasks
from tdp_worker.tasks import (
    ADVICE_DISABLED,
    ADVICE_GENERIC_FAILURE,
    ADVICE_MODEL,
    fetch_repo,
    generate_advice,
    make_llm_client,
)
from tdp_worker.testing import MemoryStore, SourceRepo

pytestmark = pytest.mark.usefixtures("eager", "fake_redis", "local_source")

VALID = json.dumps(
    {
        "summary": "Split run into smaller functions.",
        "steps": [{"order": 1, "action": "Extract the negative branch", "targets": ["run"], "effort_hours": [1, 2]}],
        "tests_to_add": ["test_run_negative"],
        "risks": ["Callers rely on the doubling."],
        "confidence": "medium",
    }
)

Factory = Callable[[list[str | Exception]], list[FakeLLMClient]]


def make_row(repo: SourceRepo, path: str = "pkg/core.py", blob_sha: str | None = None) -> FileResultRow:
    blob = blob_sha or repo.blob(path)
    syntax = parse_python(path, (repo.path / path).read_bytes()) if path.endswith(".py") else None
    cost = CostEstimate(hours=2.0, low=1.6, high=2.4, breakdown={"complexity": 60.0})
    extra = FileExtra(
        blob_sha=blob,
        functions=syntax.functions if syntax else (),
        dependents=("pkg/__init__.py",),
        dependencies=("pkg/util.py",),
        test_importers=(),
        commit_activity=(("2026-01", 3),),
        cost=cost,
    )
    reason = Reason("S", "cc_max", 4.0, 95.0, 0.4, "The most complex function is hard to follow")
    return FileResultRow(
        path=path,
        blob_sha=blob,
        priority=96,
        band="Critical",
        risk=0.7,
        risk_low=0.6,
        risk_high=0.8,
        score=1.5,
        cost_low=cost.low,
        cost_high=cost.high,
        debt_types=("complexity",),
        top_reason=reason.text,
        features=dict.fromkeys(FEATURE_INPUT_COLUMNS, 0.0),
        shap={"by_family": dict.fromkeys(FAMILY_ORDER, 0.1), "by_feature": {}},
        reasons={"reasons": [to_jsonable(reason)], "mitigations": []},
        extra=to_jsonable(extra),
    )


@pytest.fixture
def advice_env(worker_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    get_settings.cache_clear()


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch, advice_env: None) -> Factory:
    """``llm(replies)`` installs a fake client factory and returns the list of clients it handed out."""

    def install(replies: list[str | Exception]) -> list[FakeLLMClient]:
        client = FakeLLMClient(replies)
        handed: list[FakeLLMClient] = []

        def factory(settings: PlatformSettings) -> LLMClient:
            handed.append(client)
            return client

        monkeypatch.setattr(tasks, "make_llm_client", factory)
        return handed

    return install


def _stored_file(store: MemoryStore, repo: SourceRepo, *, fetch: bool = True, **row: Any) -> int:
    aid = store.add_analysis(commit_sha=repo.head)
    if fetch:
        assert fetch_repo.apply(args=(str(aid),)).get()
    return store.add_file(aid, make_row(repo, **row))


def test_advice_ready(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    handed = llm([VALID])
    fid = _stored_file(store, source_repo)

    assert generate_advice.apply(args=(fid,)).get() == "ready"

    key = Advisor.cache_key(source_repo.blob("pkg/core.py"), ADVICE_MODEL)
    row = store.advice[key]
    assert (row.status, row.file_result_id, row.model) == ("ready", fid, ADVICE_MODEL)
    assert row.body is not None and row.body["summary"] == "Split run into smaller functions."
    assert row.body["steps"][0]["targets"] == ["run"]
    assert "pkg/core.py" in handed[0].calls[0].user
    assert "def run(x):" in handed[0].calls[0].user


def test_retryable_llm_error_is_retried(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    handed = llm([LLMError("overloaded", retryable=True), VALID])
    fid = _stored_file(store, source_repo)

    assert generate_advice.apply(args=(fid,)).get() == "ready"
    assert len(handed) == 2
    assert len(handed[0].calls) == 2


def test_invalid_replies_fail(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    llm(["no json here", "still nothing"])
    fid = _stored_file(store, source_repo)

    assert generate_advice.apply(args=(fid,)).get() == "failed"
    (row,) = store.advice.values()
    assert row.status == "failed"
    assert row.error is not None and "No valid advice for pkg/core.py" in row.error


def test_non_retryable_llm_error_fails(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    llm([LLMError("the model refused: cyber", retryable=False)])
    fid = _stored_file(store, source_repo)

    assert generate_advice.apply(args=(fid,)).get() == "failed"
    (row,) = store.advice.values()
    assert row.error == "The model refused: cyber."


def test_disabled_without_a_key(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(settings: PlatformSettings) -> LLMClient:
        raise AssertionError("the client must not be built")

    monkeypatch.setattr(tasks, "make_llm_client", factory)
    fid = _stored_file(store, source_repo, fetch=False)

    assert generate_advice.apply(args=(fid,)).get() == "failed"
    (row,) = store.advice.values()
    assert (row.status, row.error) == ("failed", ADVICE_DISABLED)


def test_existing_advice_is_not_generated_again(
    store: MemoryStore, source_repo: SourceRepo, llm: Factory
) -> None:
    handed = llm([VALID])
    fid = _stored_file(store, source_repo)
    assert generate_advice.apply(args=(fid,)).get() == "ready"

    assert generate_advice.apply(args=(fid,)).get() == "ready"
    assert len(handed) == 1


def test_missing_mirror_is_fetched(
    store: MemoryStore, source_repo: SourceRepo, llm: Factory, worker_env: dict[str, Path]
) -> None:
    llm([VALID])
    fid = _stored_file(store, source_repo, fetch=False)
    assert not (worker_env["mirrors"] / "github.com").exists()

    assert generate_advice.apply(args=(fid,)).get() == "ready"
    assert (worker_env["mirrors"] / "github.com" / "acme" / "demo.git").is_dir()


def test_unknown_file(store: MemoryStore, llm: Factory) -> None:
    assert generate_advice.apply(args=(12345,)).get() == "missing"
    assert generate_advice.apply(args=(True,)).get() == "missing"
    assert generate_advice.apply(args=("7",)).get() == "missing"
    assert store.advice == {}


def test_blob_gone_from_the_mirror(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    llm([VALID])
    fid = _stored_file(store, source_repo, blob_sha="f" * 40)

    assert generate_advice.apply(args=(fid,)).get() == "failed"
    (row,) = store.advice.values()
    assert row.error == "The file's source is no longer available in the repository mirror."


def test_only_python_files(store: MemoryStore, source_repo: SourceRepo, llm: Factory) -> None:
    llm([VALID])
    fid = _stored_file(store, source_repo, path="assets/noise.bin")

    assert generate_advice.apply(args=(fid,)).get() == "failed"
    (row,) = store.advice.values()
    assert row.error == "Advice is only available for Python files."


def test_unexpected_error_fails_and_raises(
    store: MemoryStore, source_repo: SourceRepo, llm: Factory
) -> None:
    llm([RuntimeError("socket at C:/secret")])
    fid = _stored_file(store, source_repo)

    result = generate_advice.apply(args=(fid,))

    assert result.state == "FAILURE"
    (row,) = store.advice.values()
    assert (row.status, row.error) == ("failed", ADVICE_GENERIC_FAILURE)


def test_make_llm_client(advice_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_client(*args: Any, **kwargs: Any) -> str:
        built.append((args, kwargs))
        return "client"

    monkeypatch.setattr(tasks, "AnthropicClient", fake_client)

    assert make_llm_client(get_settings()) == "client"  # type: ignore[comparison-overlap]
    assert built == [
        (
            (ADVICE_MODEL,),
            {"api_key": "test-key-not-real", "timeout_s": 180.0, "max_retries": 1},
        )
    ]


def test_make_llm_client_without_a_key(worker_env: dict[str, Path]) -> None:
    with pytest.raises(tasks.AdviceDisabledError, match="disabled"):
        make_llm_client(get_settings())
