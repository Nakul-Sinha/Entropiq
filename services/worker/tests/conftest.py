"""Fixtures for the worker tests.

Tasks run in Celery eager mode against an in-memory store (:class:`tdp_worker.testing.MemoryStore`),
fakeredis and a local RepoBuilder repository cloned through the ``mirror_source`` seam. The settings point
at a dead database and Redis, so anything that escapes the fakes fails loudly.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
from tdp_core.mining.tree import list_files
from tdp_core.model.bundle import Bundle, load_bundle, make_dev_bundle
from tdp_core.testing import RepoBuilder
from tdp_platform.settings import get_settings
from tdp_worker import persist, progress, tasks
from tdp_worker.app import app, reset_process_state
from tdp_worker.progress import ProgressEvent
from tdp_worker.tasks import MirrorSource
from tdp_worker.testing import MemoryStaticCache, MemoryStore

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
T0 = datetime(2023, 3, 6, 9, tzinfo=UTC)
ANA = ("Ana", "ana@example.com")
BOB = ("Bob", "bob@example.com")


def _lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


CORE_V1 = _lines("def run(x):", "    if x > 0:", "        return x * 2", "    return 0")
CORE_V2 = _lines(
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
    "def scale(values, k):",
    "    out = []",
    "    for v in values:",
    "        if v > k:",
    "            out.append(v * k)",
    "        elif v < -k:",
    "            out.append(-v)",
    "        else:",
    "            out.append(v)",
    "    return out",
)
CORE_V3 = CORE_V2.replace("return x * 2", "return abs(x) * 2")
UTIL = _lines(
    "def add(a, b):",
    "    # TODO: replace this hack with a real implementation",
    "    return a + b",
)
SUB_INIT = _lines("from pkg.sub.deep import walk")
DEEP = _lines(
    "from pkg.core import run",
    "",
    "",
    "def walk(items):",
    "    total = 0",
    "    for item in items:",
    "        if item:",
    "            total += run(item)",
    "    return total",
)
TEST_CORE = _lines("from pkg.core import run", "", "", "def test_run():", "    assert run(1) == 2")


@dataclass(frozen=True)
class SourceRepo:
    """The local repository the mirrors are cloned from."""

    path: Path
    first: str
    head: str

    def blob(self, path: str, sha: str | None = None) -> str:
        """The blob sha of ``path`` at ``sha`` (default head)."""
        return list_files(self.path, sha or self.head)[path]


@pytest.fixture(scope="session")
def source_repo(tmp_path_factory: pytest.TempPathFactory) -> SourceRepo:
    """A small Python repository with three commits, a bug fix, a sub package, tests and a binary blob."""
    repo = RepoBuilder(tmp_path_factory.mktemp("source") / "demo")
    first = repo.commit(
        {
            "pkg/__init__.py": "",
            "pkg/core.py": CORE_V1,
            "pkg/util.py": UTIL,
            "tests/test_core.py": TEST_CORE,
        },
        "Add the core package",
        T0,
        ANA,
    )
    repo.commit(
        {
            "pkg/core.py": CORE_V2,
            "pkg/sub/__init__.py": SUB_INIT,
            "pkg/sub/deep.py": DEEP,
            "assets/noise.bin": random.Random(7).randbytes(65536),
        },
        "Add the sub package and scaling",
        T0 + timedelta(days=20),
        BOB,
    )
    head = repo.commit({"pkg/core.py": CORE_V3}, "fix: handle negative values", T0 + timedelta(days=45), ANA)
    return SourceRepo(path=repo.path, first=first, head=head)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model folder with the dev bundle ``dev-worker`` and a ``production.json`` naming it."""
    root = tmp_path_factory.mktemp("models")
    make_dev_bundle(root / DEV_VERSION, version=DEV_VERSION)
    (root / "production.json").write_text(
        json.dumps({"version": DEV_VERSION, "uri": None, "sha256": None}), encoding="utf-8"
    )
    return root


@pytest.fixture(scope="session")
def dev_bundle(models_dir: Path) -> Bundle:
    """The loaded dev bundle (a separate object from the worker's cached one)."""
    return load_bundle(models_dir / DEV_VERSION)


@pytest.fixture
def worker_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> Iterator[Path]:
    """Settings for the tests: mirrors under ``tmp_path/mirrors``, the test models, dead DB and Redis URLs.

    Yields the mirror folder. Process state (store, Redis client, bundles) is reset before and after.
    """
    monkeypatch.chdir(tmp_path)
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    mirrors = tmp_path / "mirrors"
    monkeypatch.setenv("REPO_CACHE_DIR", str(mirrors))
    monkeypatch.setenv("MODEL_DIR", str(models_dir))
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    reset_process_state()
    yield mirrors
    get_settings.cache_clear()
    reset_process_state()


@pytest.fixture
def fake_redis(worker_env: Path) -> fakeredis.FakeRedis:
    """A private fakeredis server installed as the worker's Redis client."""
    client = fakeredis.FakeRedis(server=fakeredis.FakeServer())
    progress.set_redis(client)
    return client


@pytest.fixture
def store(worker_env: Path) -> MemoryStore:
    """An in-memory store with a counting static cache, installed as the worker's store."""
    memory = MemoryStore(cache=MemoryStaticCache())
    persist.set_store(memory)
    return memory


@pytest.fixture
def eager() -> Iterator[None]:
    """Run tasks eagerly. ``task_eager_propagates`` stays False, so eager retries and error callbacks run."""
    saved = (app.conf.task_always_eager, app.conf.task_eager_propagates)
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = False
    yield
    app.conf.task_always_eager, app.conf.task_eager_propagates = saved


@pytest.fixture
def local_source(monkeypatch: pytest.MonkeyPatch, source_repo: SourceRepo) -> None:
    """Clone mirrors from the local source repository (file protocol allowed in tests only)."""
    monkeypatch.setattr(
        tasks, "mirror_source", lambda info, s: MirrorSource(str(source_repo.path), allow_file_protocol=True)
    )


Listen = Callable[[uuid.UUID], Callable[[], list[ProgressEvent]]]


@pytest.fixture
def listen(fake_redis: fakeredis.FakeRedis) -> Iterator[Listen]:
    """``listen(aid)`` subscribes to an analysis channel now and returns a function draining its events."""
    subscriptions: list[Any] = []

    def subscribe(aid: uuid.UUID) -> Callable[[], list[ProgressEvent]]:
        pubsub = fake_redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(progress.channel_for(aid))
        subscriptions.append(pubsub)

        def drain() -> list[ProgressEvent]:
            events: list[ProgressEvent] = []
            while (message := pubsub.get_message(timeout=0.1)) is not None:
                events.append(ProgressEvent.from_json(message["data"]))
            return events

        return drain

    yield subscribe
    for pubsub in subscriptions:
        pubsub.close()


# ---------- a hand-built analysis result ----------

SAMPLE_PRIORITIES = {
    "pkg/core.py": 100,
    "pkg/sub/deep.py": 95,
    "app.py": 10,
    "pkg/util.py": 9,
    "pkg/zeta.py": 0,
}
SAMPLE_SLOC = {"pkg/core.py": 120, "pkg/sub/deep.py": 40, "app.py": 12, "pkg/util.py": 7}  # zeta has no node
SAMPLE_DEBT = {
    "pkg/core.py": ("complexity", "test"),
    "pkg/sub/deep.py": ("complexity",),
    "app.py": ("knowledge",),
}
SAMPLE_HOURS = {
    "pkg/core.py": 6.0,
    "pkg/sub/deep.py": 2.25,
    "app.py": 1.0,
    "pkg/util.py": 0.5,
    "pkg/zeta.py": 0.25,
}


def _reason(feature: str, family: str, text: str) -> Reason:
    return Reason(family=family, feature=feature, value=31.0, percentile=97.5, shap=0.42, text=text)


def make_sample_result() -> AnalysisResult:
    """Five files (priorities 100, 95, 10, 9, 0) with extras, an 89 column float32 frame, graph and plan."""
    paths = sorted(SAMPLE_PRIORITIES, key=lambda p: (-SAMPLE_PRIORITIES[p], p))
    files: list[FileScore] = []
    extras: dict[str, FileExtra] = {}
    for i, path in enumerate(paths):
        priority = SAMPLE_PRIORITIES[path]
        files.append(
            FileScore(
                path=path,
                score=2.0 - 0.5 * i,
                priority=priority,
                band=band_for(priority),
                risk=0.9 - 0.1 * i,
                risk_low=0.8 - 0.1 * i,
                risk_high=0.95 - 0.1 * i,
                reasons=(_reason("cc_max", "S", f"Very complex code in {path}"),) if priority >= 50 else (),
                mitigations=(_reason("has_test_file", "T", "Has a test file"),) if i == 0 else (),
                debt_types=SAMPLE_DEBT.get(path, ()),
                shap_by_family={fam: 0.1 * (k + 1) - 0.05 * i for k, fam in enumerate(FAMILY_ORDER)},
            )
        )
        hours = SAMPLE_HOURS[path]
        extras[path] = FileExtra(
            blob_sha=f"{i + 1:040x}",
            functions=(FunctionInfo("run", 1, 9, 4, 3, 2, 1),),
            dependents=(),
            dependencies=(),
            test_importers=(),
            commit_activity=(("2023-03", 2),),
            cost=CostEstimate(
                hours=hours, low=0.8 * hours, high=1.2 * hours, breakdown={"complexity": 60 * hours}
            ),
        )
    index = sorted(paths)
    values = np.arange(len(index) * len(FEATURE_INPUT_COLUMNS), dtype=np.float32).reshape(len(index), -1)
    features = pd.DataFrame(
        values / 7, index=pd.Index(index, name="path"), columns=list(FEATURE_INPUT_COLUMNS)
    )
    nodes = tuple(
        GraphNode(
            id=p,
            priority=SAMPLE_PRIORITIES[p],
            band=band_for(SAMPLE_PRIORITIES[p]),
            sloc=SAMPLE_SLOC[p],
            dependents=0,
        )
        for p in paths
        if p in SAMPLE_SLOC
    )
    edges = (GraphEdge(source="pkg/sub/deep.py", target="pkg/core.py", kind="import", weight=1.0),)
    step = PlanStep(
        order=1,
        path="pkg/core.py",
        benefit=3.5,
        cost=extras["pkg/core.py"].cost,
        roi=0.57,
        group=0,
        reason="High",
    )
    plan = Plan(budget_hours=40.0, steps=(step,), total_benefit=3.5, total_hours=6.0)
    return AnalysisResult(
        sha="a" * 40,
        analyzed_at=datetime(2026, 10, 1, tzinfo=UTC),
        model_version=DEV_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        files=tuple(files),
        features=features,
        extras=extras,
        graph=GraphData(nodes=nodes, edges=edges),
        plan=plan,
        stats={"n_files": len(paths), "timings_s": {"mine": 0.5}},
    )


@pytest.fixture
def sample_result() -> AnalysisResult:
    """A fresh hand-built result (see :func:`make_sample_result`)."""
    return make_sample_result()
