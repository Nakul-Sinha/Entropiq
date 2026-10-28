"""Fixtures for the worker tests: a dev bundle, a scripted source repository, an in-memory store,
fakeredis and Celery eager mode. Nothing here needs Postgres or a real Redis server."""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable, Iterator, Mapping
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
    FunctionInfo,
    GraphData,
    GraphEdge,
    GraphNode,
    Plan,
    PlanStep,
    Reason,
    band_for,
)
from tdp_core.model.bundle import Bundle, load_bundle, make_dev_bundle
from tdp_core.testing import RepoBuilder
from tdp_platform.settings import get_settings
from tdp_worker import persist, progress, tasks
from tdp_worker.app import app, reset_process_state
from tdp_worker.persist import AnalysisInfo
from tdp_worker.tasks import MirrorSource
from tdp_worker.testing import DEFAULT_MODEL_VERSION, EventLog, MemoryStore, SourceRepo

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
DEV_VERSION = DEFAULT_MODEL_VERSION
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


@pytest.fixture
def events(fake_redis: fakeredis.FakeRedis) -> Callable[[uuid.UUID], EventLog]:
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
                shap_by_family={
                    fam: round(0.1 * (k + 1) - i * 0.01, 4) for k, fam in enumerate(FAMILY_ORDER)
                },
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
    edges = (
        (GraphEdge(source="pkg/core.py", target="pkg/util.py", kind="import", weight=1.0),) if paths else ()
    )
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


MakeResult = Callable[[Mapping[str, int]], AnalysisResult]


@pytest.fixture
def make_result() -> MakeResult:
    """``make_result(priorities)`` builds an ``AnalysisResult`` with those files and priorities."""
    return make_sample_result


@pytest.fixture
def sample_result() -> AnalysisResult:
    return make_sample_result()
