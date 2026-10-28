"""Fixtures for the platform tests.

Postgres tests read ``TEST_DATABASE_URL`` and skip without it (CI provides a Postgres service). They share
one temporary schema per test session, emptied before every test.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

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
from tdp_core.parsing import get_plugin
from tdp_platform.db.rows import FileResultRow
from tdp_platform.db.session import create_async_db_engine, create_sync_db_engine, make_session_factory
from tdp_platform.db.testing import temporary_schema, truncate_all

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

MakeRow = Callable[..., FileResultRow]


def blob_sha_for(path: str) -> str:
    """A stable fake git blob sha for ``path``."""
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


# ---------- environment ----------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every environment variable the platform settings read."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------- Postgres ----------


@pytest.fixture(scope="session")
def pg_url() -> str:
    """URL of the test database; skips the test when ``TEST_DATABASE_URL`` is not set."""
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set; Postgres tests run in CI")
    return url


@pytest.fixture(scope="session")
def pg_schema(pg_url: str) -> Iterator[str]:
    """A temporary schema migrated to head, dropped after the test session."""
    with temporary_schema(pg_url, prefix="tdp_platform_test") as schema:
        yield schema


@pytest.fixture(scope="session")
def sync_engine(pg_url: str, pg_schema: str) -> Iterator[Engine]:
    """A psycopg engine bound to the test schema."""
    engine = create_sync_db_engine(pg_url, schema=pg_schema)
    yield engine
    engine.dispose()


@pytest.fixture
def clean_db(sync_engine: Engine) -> None:
    """Empty every table of the test schema before the test."""
    with sync_engine.begin() as conn:
        truncate_all(conn)


@pytest_asyncio.fixture
async def async_engine(pg_url: str, pg_schema: str, clean_db: None) -> AsyncIterator[AsyncEngine]:
    """An asyncpg engine (no pooling) bound to the emptied test schema."""
    engine = create_async_db_engine(pg_url, schema=pg_schema, null_pool=True)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(async_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory of the test engine (for tests that need several sessions)."""
    return make_session_factory(async_engine)


@pytest_asyncio.fixture
async def session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """An async session on the emptied test schema, closed after the test."""
    async with session_factory() as s:
        yield s


# ---------- sample data ----------

_PRIORITIES = {"pkg/core.py": 95, "pkg/util.py": 80, "app.py": 40, "pkg/zeta.py": 40}


def _reason(feature: str, family: str, value: float, text: str) -> Reason:
    return Reason(family=family, feature=feature, value=value, percentile=97.5, shap=0.42, text=text)


@pytest.fixture
def analysis_result() -> AnalysisResult:
    """A small hand-built analysis of four files (``files`` sorted by priority desc, then path)."""
    paths = sorted(_PRIORITIES, key=lambda p: (-_PRIORITIES[p], p))
    files: list[FileScore] = []
    extras: dict[str, FileExtra] = {}
    for i, path in enumerate(paths):
        priority = _PRIORITIES[path]
        reasons = (
            ()
            if path == "pkg/zeta.py"
            else (
                _reason("cc_max", "S", 31.0, f"Very complex code in {path}"),
                _reason("commits_90d", "E", 12.0, "Changed often lately"),
            )
        )
        files.append(
            FileScore(
                path=path,
                score=2.5 - i * 0.75,
                priority=priority,
                band=band_for(priority),
                risk=0.9 - i * 0.2,
                risk_low=0.8 - i * 0.2,
                risk_high=0.95 - i * 0.2,
                reasons=reasons,
                mitigations=(_reason("has_test_file", "T", 1.0, "Has a test file"),) if i == 1 else (),
                debt_types=("complexity", "test") if i == 0 else (("knowledge",) if i == 1 else ()),
                shap_by_family={
                    fam: round(0.1 * (k + 1) - i * 0.05, 4) for k, fam in enumerate(FAMILY_ORDER)
                },
            )
        )
        hours = 6.0 - i
        extras[path] = FileExtra(
            blob_sha=blob_sha_for(path),
            functions=(
                FunctionInfo(
                    name="Thing.run",
                    start_line=3,
                    end_line=40,
                    cyclomatic=12,
                    cognitive=20,
                    max_nesting=3,
                    params=2,
                ),
            ),
            dependents=("pkg/util.py",) if path == "pkg/core.py" else (),
            dependencies=("pkg/core.py",) if path == "pkg/util.py" else (),
            test_importers=("tests/test_core.py",) if path == "pkg/core.py" else (),
            commit_activity=(("2026-08", 3), ("2026-09", 0), ("2026-10", 5)),
            cost=CostEstimate(
                hours=hours, low=0.8 * hours, high=1.2 * hours, breakdown={"complexity": 60.0 * hours}
            ),
        )
    index = sorted(paths)  # deliberately not in ranking order
    values = (
        np.arange(len(index) * len(FEATURE_INPUT_COLUMNS), dtype=np.float64).reshape(len(index), -1) / 7.0
    )
    features = pd.DataFrame(
        values.astype(np.float32), index=pd.Index(index, name="path"), columns=FEATURE_INPUT_COLUMNS
    )
    graph = GraphData(
        nodes=tuple(
            GraphNode(id=p, priority=_PRIORITIES[p], band=band_for(_PRIORITIES[p]), sloc=100, dependents=1)
            for p in paths
        ),
        edges=(GraphEdge(source="pkg/util.py", target="pkg/core.py", kind="import", weight=1.0),),
    )
    step_cost = extras["pkg/core.py"].cost
    plan = Plan(
        budget_hours=40.0,
        steps=(
            PlanStep(
                order=1,
                path="pkg/core.py",
                benefit=0.9,
                cost=step_cost,
                roi=0.15,
                group=0,
                reason="Most risky file",
            ),
        ),
        total_benefit=0.9,
        total_hours=step_cost.hours,
    )
    return AnalysisResult(
        sha="a" * 40,
        analyzed_at=datetime(2026, 10, 1, 12, tzinfo=UTC),
        model_version="dev-test",
        extractor_version=EXTRACTOR_VERSION,
        files=tuple(files),
        features=features,
        extras=extras,
        graph=graph,
        plan=plan,
        stats={"n_files": len(files), "timings_s": {"mine": 1.5}},
    )


@pytest.fixture
def make_row() -> MakeRow:
    """Factory of valid ``FileResultRow`` objects; ``cost`` is ``cost_high`` in hours."""

    def make(
        path: str,
        priority: int = 50,
        risk: float = 0.5,
        cost: float = 1.0,
        debt_types: Sequence[str] = (),
        band: str | None = None,
    ) -> FileResultRow:
        return FileResultRow(
            path=path,
            blob_sha=blob_sha_for(path),
            priority=priority,
            band=band or band_for(priority),
            risk=risk,
            risk_low=max(0.0, risk - 0.25),
            risk_high=min(1.0, risk + 0.25),
            score=priority / 8,
            cost_low=cost * 0.5,
            cost_high=cost,
            debt_types=tuple(debt_types),
            top_reason=f"Reason for {path}",
            features={"sloc": 120.0, "cc_max": 7.0},
            shap={"by_family": {"S": 0.25}, "by_feature": {}},
            reasons={"reasons": [], "mitigations": []},
            extra={"blob_sha": blob_sha_for(path)},
        )

    return make


@pytest.fixture
def syntax_sample() -> FileSyntax:
    """A real parse of a small Python module with a comment, a class, functions and imports."""
    source = (
        b"import os\n"
        b"from collections import OrderedDict, defaultdict\n"
        b"\n"
        b"# TODO: remove this hack\n"
        b"class Thing:\n"
        b"    def run(self, x):\n"
        b"        if x > 1:\n"
        b"            return os.sep\n"
        b"        return 'ok'\n"
        b"\n"
        b"def helper(a, b):\n"
        b"    return a or b\n"
    )
    plugin = get_plugin("pkg/a.py")
    assert plugin is not None
    return plugin.parse("pkg/a.py", source)
