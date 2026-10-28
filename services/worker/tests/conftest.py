"""Fixtures for the worker tests: a dev bundle, a scripted source repository, an in-memory store,
fakeredis and Celery eager mode. Nothing here needs Postgres or a real Redis server."""

from __future__ import annotations

import json
import random
import subprocess
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np
import pandas as pd
import pytest

from tdp_core.contracts import (
    EXTRACTOR_VERSION,
    FAMILY_ORDER,
    FEATURE_INPUT_COLUMNS,
    AnalysisResult,
    CostEstimate,
    FileExtra,
    FileScore,
    FileSyntax,
    FunctionInfo,
    GraphData,
    GraphEdge,
    GraphNode,
    Plan,
    PlanStep,
    Reason,
    band_for,
)
from tdp_core.features.cache import StaticCache
from tdp_core.model.bundle import Bundle, load_bundle, make_dev_bundle
from tdp_core.testing import RepoBuilder
from tdp_platform.db.rows import FileResultRow
from tdp_platform.settings import get_settings
from tdp_worker import persist, progress, tasks
from tdp_worker.app import app, reset_process_state
from tdp_worker.persist import AnalysisInfo, FileInfo, ResultPayload, score_from_row
from tdp_worker.progress import ProgressEvent
from tdp_worker.tasks import MirrorSource

SETTINGS_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_SCHEMA",
    "REDIS_URL",
    "REPO_CACHE_DIR",
    "MODEL_DIR",
    "GITHUB_TOKEN",
    "ANTHROPIC_API_KEY",
    "SENTRY_DSN",
    "ALLOWED_GIT_HOSTS",
    "MAX_REPO_KB",
    "ADVISOR_ENABLED",
    "TDP_MODELS_DIR",
    "TDP_DATA_DIR",
)
DEV_VERSION = "dev-worker"
REPOSITORY_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
T0 = datetime(2026, 1, 5, 12, tzinfo=UTC)


def lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


# ---------- model bundle ----------


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model folder with the dev bundle ``dev-worker`` and a production pointer to it."""
    root = tmp_path_factory.mktemp("models")
    make_dev_bundle(root / DEV_VERSION, version=DEV_VERSION)
    pointer = {"version": DEV_VERSION, "uri": None, "sha256": None}
    (root / "production.json").write_text(json.dumps(pointer), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def dev_bundle(models_dir: Path) -> Bundle:
    return load_bundle(models_dir / DEV_VERSION)


# ---------- source repository ----------


@dataclass(frozen=True)
class SourceRepo:
    path: Path
    first: str
    head: str

    def blob(self, rel: str, rev: str | None = None) -> str:
        """The blob sha of ``rel`` at ``rev`` (default the head commit)."""
        out = subprocess.run(
            ["git", "rev-parse", f"{rev or self.head}:{rel}"],
            cwd=self.path,
            check=True,
            capture_output=True,
        )
        return out.stdout.decode().strip()


CORE_V1 = lines("def run(x):", "    if x > 0:", "        return x * 2", "    return 0")
CORE_V2 = lines(
    "from pkg.util import add",
    "",
    "",
    "def run(x):",
    "    if x > 0:",
    "        return x * 2",
    "    if x < -10:",
    "        return add(x, 10)",
    "    return 0",
    "",
    "",
    "def scale(values, factor):",
    "    out = []",
    "    for v in values:",
    "        if v is None:",
    "            continue",
    "        out.append(v * factor)",
    "    return out",
)
CORE_V3 = CORE_V2.replace("return x * 2", "return abs(x) * 2")
UTIL = lines("def add(a, b):", "    # TODO: replace this hack", "    return a + b")
DEEP = lines("def walk(tree):", "    for node in tree:", "        if node:", "            yield node")


@pytest.fixture(scope="session")
def source_repo(tmp_path_factory: pytest.TempPathFactory) -> SourceRepo:
    """A small Python repository with three commits (one a bug fix) and 64 KiB of binary noise."""
    builder = RepoBuilder(tmp_path_factory.mktemp("source") / "demo")
    first = builder.commit(
        {
            "pkg/__init__.py": lines("from pkg.core import run"),
            "pkg/core.py": CORE_V1,
            "pkg/util.py": UTIL,
            "assets/noise.bin": random.Random(7).randbytes(65536),
        },
        "Add the core module",
        T0,
    )
    builder.commit(
        {
            "pkg/core.py": CORE_V2,
            "pkg/sub/__init__.py": lines("from pkg.sub.deep import walk"),
            "pkg/sub/deep.py": DEEP,
            "tests/test_core.py": lines("from pkg.core import run", "", "", "def test_run():", "    run(1)"),
        },
        "Add scaling and tree walking",
        T0 + timedelta(days=20),
    )
    head = builder.commit({"pkg/core.py": CORE_V3}, "fix: handle negative values", T0 + timedelta(days=40))
    return SourceRepo(path=builder.path, first=first, head=head)


# ---------- environment ----------


@pytest.fixture
def worker_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, models_dir: Path
) -> Iterator[dict[str, Path]]:
    """Settings for the tests: mirrors under ``tmp_path``, the dev bundle, unreachable DB and Redis."""
    monkeypatch.chdir(tmp_path)
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    mirrors = tmp_path / "mirrors"
    monkeypatch.setenv("REPO_CACHE_DIR", str(mirrors))
    monkeypatch.setenv("MODEL_DIR", str(models_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://tdp:tdp@127.0.0.1:1/tdp")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    reset_process_state()
    yield {"mirrors": mirrors, "models": models_dir}
    get_settings.cache_clear()
    reset_process_state()


@pytest.fixture
def fake_redis(worker_env: dict[str, Path]) -> fakeredis.FakeRedis:
    """A fakeredis client (own server per test) used as the worker's Redis client."""
    client = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    progress.set_redis(client)
    return client


# ---------- in-memory store ----------


class RecordingCache:
    """A dict-backed ``StaticCache`` that counts puts and hits."""

    def __init__(self) -> None:
        self.data: dict[str, FileSyntax] = {}
        self.puts = 0
        self.hits = 0

    def get(self, blob_sha: str) -> FileSyntax | None:
        found = self.data.get(blob_sha)
        if found is not None:
            self.hits += 1
        return found

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        self.puts += 1
        self.data[blob_sha] = syntax


@dataclass
class AdviceRow:
    file_result_id: int
    model: str
    status: str
    body: dict[str, Any] | None = None
    error: str | None = None


class MemoryStore:
    """An in-memory ``Store`` with the state rules of the platform repositories."""

    def __init__(self) -> None:
        self.analyses: dict[uuid.UUID, dict[str, Any]] = {}
        self.files: dict[uuid.UUID, list[FileResultRow]] = {}
        self.file_rows: dict[int, tuple[uuid.UUID, FileResultRow]] = {}
        self.artifacts: dict[tuple[uuid.UUID, str], Any] = {}
        self.plans: dict[tuple[uuid.UUID, float], Any] = {}
        self.advice: dict[str, AdviceRow] = {}
        self.progress_log: list[tuple[uuid.UUID, str, int]] = []
        self.cache: StaticCache | None = None
        self.closed = False
        self._next_file_id = 1

    # ----- helpers for tests

    def add_analysis(
        self,
        *,
        commit_sha: str,
        model_version: str = DEV_VERSION,
        budget_hours: float = 40.0,
        url: str = "https://github.com/acme/demo",
        host: str = "github.com",
        owner: str = "acme",
        name: str = "demo",
        status: str = "queued",
        repository_id: uuid.UUID = REPOSITORY_ID,
    ) -> uuid.UUID:
        aid = uuid.uuid4()
        self.analyses[aid] = {
            "repository_id": repository_id,
            "url": url,
            "host": host,
            "owner": owner,
            "name": name,
            "commit_sha": commit_sha,
            "model_version": model_version,
            "budget_hours": budget_hours,
            "status": status,
            "stage": None,
            "progress": 0,
            "error": None,
            "summary": None,
            "stats": None,
        }
        return aid

    def add_file(self, analysis_id: uuid.UUID, row: FileResultRow) -> int:
        fid = self._next_file_id
        self._next_file_id += 1
        self.file_rows[fid] = (analysis_id, row)
        self.files.setdefault(analysis_id, []).append(row)
        return fid

    def row(self, analysis_id: uuid.UUID) -> dict[str, Any]:
        return self.analyses[analysis_id]

    def _info(self, analysis_id: uuid.UUID) -> AnalysisInfo:
        a = self.analyses[analysis_id]
        return AnalysisInfo(
            id=analysis_id,
            repository_id=a["repository_id"],
            url=a["url"],
            host=a["host"],
            owner=a["owner"],
            name=a["name"],
            commit_sha=a["commit_sha"],
            model_version=a["model_version"],
            budget_hours=a["budget_hours"],
            status=a["status"],
            stage=a["stage"],
            progress=a["progress"],
        )

    def _transition(self, analysis_id: uuid.UUID, allowed: tuple[str, ...], **values: Any) -> bool:
        a = self.analyses.get(analysis_id)
        if a is None or a["status"] not in allowed:
            return False
        a.update(values)
        return True

    # ----- Store protocol

    def load_analysis(self, analysis_id: uuid.UUID) -> AnalysisInfo | None:
        return self._info(analysis_id) if analysis_id in self.analyses else None

    def set_running(self, analysis_id: uuid.UUID) -> bool:
        return self._transition(analysis_id, ("queued", "running"), status="running", error=None)

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        ok = self._transition(analysis_id, ("queued", "running"), stage=stage, progress=pct)
        if ok:
            self.progress_log.append((analysis_id, stage, pct))
        return ok

    def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        return self._transition(analysis_id, ("queued", "running"), status="failed", error=error)

    def save_results(self, analysis_id: uuid.UUID, payload: ResultPayload) -> bool:
        if self.analyses.get(analysis_id, {}).get("status") != "running":
            return False
        self.files[analysis_id] = list(payload.rows)
        for row in payload.rows:
            self.file_rows[self._next_file_id] = (analysis_id, row)
            self._next_file_id += 1
        self.artifacts[(analysis_id, "graph")] = payload.graph
        self.artifacts[(analysis_id, "treemap")] = payload.treemap
        self.plans[(analysis_id, payload.budget_hours)] = payload.plan
        return self._transition(
            analysis_id,
            ("running",),
            status="succeeded",
            stage="done",
            progress=100,
            summary=payload.summary,
            stats=payload.stats,
        )

    def load_file(self, file_result_id: int) -> FileInfo | None:
        found = self.file_rows.get(file_result_id)
        if found is None:
            return None
        aid, row = found
        from tdp_core.serialize import from_jsonable

        return FileInfo(
            id=file_result_id,
            analysis=self._info(aid),
            path=row.path,
            blob_sha=row.blob_sha,
            score=score_from_row(row),
            extra=from_jsonable(FileExtra, row.extra),
        )

    def begin_advice(self, cache_key: str, *, file_result_id: int, model: str) -> str:
        row = self.advice.get(cache_key)
        if row is None:
            row = AdviceRow(file_result_id=file_result_id, model=model, status="queued")
            self.advice[cache_key] = row
        return row.status

    def advice_ready(self, cache_key: str, body: Mapping[str, Any]) -> bool:
        row = self.advice.get(cache_key)
        if row is None or row.status != "queued":
            return False
        row.status, row.body, row.error = "ready", dict(body), None
        return True

    def advice_failed(self, cache_key: str, error: str) -> bool:
        row = self.advice.get(cache_key)
        if row is None or row.status != "queued":
            return False
        row.status, row.body, row.error = "failed", None, error
        return True

    def static_cache(self) -> StaticCache | None:
        return self.cache

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def store(worker_env: dict[str, Path]) -> Iterator[MemoryStore]:
    memory = MemoryStore()
    persist.set_store(memory)
    yield memory
    persist.set_store(None)


# ---------- Celery and sources ----------


@pytest.fixture
def eager() -> Iterator[None]:
    """Run tasks eagerly; ``task_eager_propagates`` stays False so retries and errbacks run."""
    old = (app.conf.task_always_eager, app.conf.task_eager_propagates)
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = False
    yield
    app.conf.task_always_eager, app.conf.task_eager_propagates = old


@pytest.fixture
def local_source(monkeypatch: pytest.MonkeyPatch, source_repo: SourceRepo) -> SourceRepo:
    """Clone the local source repository instead of the analysis URL."""

    def fake_source(info: AnalysisInfo, settings: Any) -> MirrorSource:
        return MirrorSource(str(source_repo.path), allow_file_protocol=True)

    monkeypatch.setattr(tasks, "mirror_source", fake_source)
    return source_repo


class EventLog:
    """Collects the progress events of one analysis from a fakeredis subscription."""

    def __init__(self, client: fakeredis.FakeRedis, analysis_id: uuid.UUID) -> None:
        self._pubsub = client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(progress.channel_for(analysis_id))
        self._pubsub.get_message(timeout=0.1)  # the subscribe confirmation

    def drain(self) -> list[ProgressEvent]:
        events: list[ProgressEvent] = []
        while True:
            message = self._pubsub.get_message(timeout=0.1)
            if message is None:
                return events
            if message.get("type") == "message":
                events.append(ProgressEvent.from_json(message["data"]))


EventsFactory = Callable[[uuid.UUID], EventLog]


@pytest.fixture
def events(fake_redis: fakeredis.FakeRedis) -> EventsFactory:
    """``events(aid)`` subscribes to an analysis's channel; call ``.drain()`` after the run."""

    def make(analysis_id: uuid.UUID) -> EventLog:
        return EventLog(fake_redis, analysis_id)

    return make


# ---------- a hand-built analysis result ----------

SAMPLE_PRIORITIES = {
    "pkg/core.py": 100,
    "pkg/sub/deep.py": 95,
    "app.py": 10,
    "pkg/util.py": 9,
    "pkg/zero.py": 0,
}


def _reason(feature: str, family: str, text: str) -> Reason:
    return Reason(family=family, feature=feature, value=12.0, percentile=96.0, shap=0.4, text=text)


def make_sample_result(priorities: Mapping[str, int] = SAMPLE_PRIORITIES) -> AnalysisResult:
    """An ``AnalysisResult`` with the given files (89 float32 feature columns, extras, graph, plan)."""
    paths = sorted(priorities, key=lambda p: (-priorities[p], p))
    files: list[FileScore] = []
    extras: dict[str, FileExtra] = {}
    debt_cycle = (("complexity", "test"), ("knowledge",), (), ("admitted",), ())
    for i, path in enumerate(paths):
        priority = priorities[path]
        files.append(
            FileScore(
                path=path,
                score=3.0 - i * 0.5,
                priority=priority,
                band=band_for(priority),
                risk=0.9 - i * 0.1,
                risk_low=0.8 - i * 0.1,
                risk_high=0.95 - i * 0.1,
                reasons=(_reason("cc_max", "S", f"Complex code in {path}"),) if priority >= 50 else (),
                mitigations=(_reason("has_test_file", "T", "Has a test file"),) if i == 1 else (),
                debt_types=debt_cycle[i % len(debt_cycle)],
                shap_by_family={fam: round(0.1 * (k + 1) - i * 0.01, 4) for k, fam in enumerate(FAMILY_ORDER)},
            )
        )
        hours = 4.0 - i * 0.5
        extras[path] = FileExtra(
            blob_sha=f"{i + 1:040x}",
            functions=(FunctionInfo("run", 1, 9, 3, 2, 1, 1),),
            dependents=(),
            dependencies=(),
            test_importers=(),
            commit_activity=(("2026-01", 2),),
            cost=CostEstimate(hours=hours, low=0.8 * hours, high=1.2 * hours, breakdown={"complexity": 60.0}),
        )
    index = sorted(paths)
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        rng.random((len(index), len(FEATURE_INPUT_COLUMNS))).astype(np.float32),
        index=pd.Index(index, name="path"),
        columns=list(FEATURE_INPUT_COLUMNS),
    )
    nodes = tuple(
        GraphNode(id=p, priority=priorities[p], band=band_for(priorities[p]), sloc=10 * (k + 1), dependents=0)
        for k, p in enumerate(paths)
        if p != "pkg/zero.py"
    )
    edges = (GraphEdge(source="pkg/core.py", target="pkg/util.py", kind="import", weight=1.0),) if paths else ()
    steps = (
        (
            PlanStep(
                order=1,
                path=paths[0],
                benefit=0.9,
                cost=extras[paths[0]].cost,
                roi=0.2,
                group=0,
                reason="Riskiest file",
            ),
        )
        if paths
        else ()
    )
    plan = Plan(
        budget_hours=40.0,
        steps=steps,
        total_benefit=0.9 if paths else 0.0,
        total_hours=extras[paths[0]].cost.hours if paths else 0.0,
    )
    return AnalysisResult(
        sha="a" * 40,
        analyzed_at=datetime(2026, 10, 1, 12, tzinfo=UTC),
        model_version=DEV_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        files=tuple(files),
        features=features,
        extras=extras,
        graph=GraphData(nodes=nodes, edges=edges),
        plan=plan,
        stats={"n_files": len(files), "timings_s": {"mine": 0.5}},
    )


@pytest.fixture
def sample_result() -> AnalysisResult:
    return make_sample_result()


def with_status(result: AnalysisResult, **changes: Any) -> AnalysisResult:
    return replace(result, **changes)
