"""Storage for the worker tasks: what they read and write, and how results are shaped for the database.

The tasks talk to a :class:`Store`, a small synchronous interface with the state machine of
``tdp_platform.db.repos`` (P5.1): ``set_running`` from queued or running, ``set_progress`` while queued or
running, ``set_failed`` from queued or running, and success (``save_results``) only from running. Every
transition returns False when the row was not in an allowed state. :class:`PostgresStore` implements it on
the platform repositories (async SQLAlchemy on asyncpg, bridged with :func:`run_coro`); the worker tests use
an in-memory store with the same rules.

The pure builders turn an ``AnalysisResult`` into the stored bodies of plan section 3.7: the summary
(:func:`build_summary`), the treemap (:func:`build_treemap`) and the file rows with per-feature SHAP values
(:func:`build_payload`). :func:`score_from_row` rebuilds a ``FileScore`` from a stored row for the advisor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from tdp_core.contracts import (
    BANDS,
    DEBT_TYPES,
    AnalysisResult,
    Band,
    FileExtra,
    FileScore,
    Reason,
)
from tdp_core.features.cache import StaticCache
from tdp_core.model.explain import Explainer
from tdp_core.serialize import from_jsonable, to_jsonable
from tdp_platform.db.models import Analysis, FileResult
from tdp_platform.db.repos import (
    AdviceRepo,
    AnalysisRepo,
    ArtifactRepo,
    FileResultRepo,
    PlanRepo,
    PostgresStaticCache,
)
from tdp_platform.db.rows import FileResultRow, rows_from_result
from tdp_platform.db.session import create_async_db_engine, make_session_factory
from tdp_platform.settings import PlatformSettings, get_settings

if TYPE_CHECKING:  # importing the bundle module loads xgboost
    from tdp_core.model.bundle import Bundle

logger = logging.getLogger(__name__)

ACTIVE_STATUSES: Final = ("queued", "running")
HISTOGRAM_BUCKETS: Final = 10


# ---------- records ----------


@dataclass(frozen=True, slots=True)
class AnalysisInfo:
    """What the tasks need to know about an analysis row and its repository.

    ``url`` is the canonical repository URL (``repositories.url``), ``budget_hours`` the plan budget in
    hours, ``progress`` the stored percentage 0..100 and ``stage`` the last recorded stage (or None).
    """

    id: uuid.UUID
    repository_id: uuid.UUID
    url: str
    host: str
    owner: str
    name: str
    commit_sha: str
    model_version: str
    budget_hours: float
    status: str
    stage: str | None
    progress: int


@dataclass(frozen=True, slots=True)
class FileInfo:
    """A stored file result with its analysis, rebuilt for the advisor (``score`` and ``extra``)."""

    id: int
    analysis: AnalysisInfo
    path: str
    blob_sha: str
    score: FileScore
    extra: FileExtra


@dataclass(frozen=True, slots=True)
class ResultPayload:
    """Everything stored for a successful analysis, in database shapes (JSON-able values).

    ``rows`` are in ``AnalysisResult.files`` order, ``graph`` and ``plan`` are ``to_jsonable`` of the
    result's ``GraphData`` and ``Plan``, ``budget_hours`` is the analysis budget the plan was made for.
    """

    rows: tuple[FileResultRow, ...]
    graph: dict[str, Any]
    treemap: dict[str, Any]
    plan: dict[str, Any]
    budget_hours: float
    summary: dict[str, Any]
    stats: dict[str, Any]


# ---------- pure builders ----------


def build_summary(result: AnalysisResult) -> dict[str, Any]:
    """The ``analyses.summary`` object of plan section 3.7.

    Keys in order: ``files`` (ranked files), ``critical``, ``high``, ``medium``, ``low`` (band counts),
    ``critical_hours`` (estimated fix hours of the Critical files, rounded to 0.01), ``debt_types`` (every
    entry of ``DEBT_TYPES`` in order with the number of files carrying it, zeros included) and
    ``priority_histogram`` (10 counts; bucket ``min(priority // 10, 9)``, so 90..100 share the last one).
    An empty result gives zeros everywhere.
    """
    bands = {name: 0 for name, _ in BANDS}
    debt = dict.fromkeys(DEBT_TYPES, 0)
    histogram = [0] * HISTOGRAM_BUCKETS
    critical_hours = 0.0
    for fs in result.files:
        bands[fs.band] = bands.get(fs.band, 0) + 1
        for kind in dict.fromkeys(fs.debt_types):
            if kind in debt:
                debt[kind] += 1
        histogram[min(max(int(fs.priority), 0) // 10, HISTOGRAM_BUCKETS - 1)] += 1
        if fs.band == "Critical":
            extra = result.extras.get(fs.path)
            if extra is not None:
                critical_hours += float(extra.cost.hours)
    return {
        "files": len(result.files),
        "critical": bands["Critical"],
        "high": bands["High"],
        "medium": bands["Medium"],
        "low": bands["Low"],
        "critical_hours": round(critical_hours, 2),
        "debt_types": debt,
        "priority_histogram": histogram,
    }


@dataclass
class _TreeDir:
    name: str
    path: str
    dirs: dict[str, _TreeDir] = field(default_factory=dict)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        children: list[dict[str, Any]] = [self.dirs[k].to_json() for k in sorted(self.dirs)]
        children.extend(self.files[k] for k in sorted(self.files))
        return {"name": self.name, "path": self.path, "children": children}


def build_treemap(result: AnalysisResult, *, root_name: str = "") -> dict[str, Any]:
    """The treemap artifact: nested directories built from the ranked file paths.

    The root is ``{"name": root_name, "path": "", "children": [...]}``; a directory is ``{"name", "path"
    (POSIX prefix such as "pkg/sub"), "children"}`` and a leaf is ``{"name", "path", "sloc", "priority",
    "band"}``. Only ``result.files`` appear; ``sloc`` comes from the graph nodes (0 when a file has none).
    Children list directories first, then files, each sorted by name in code point order. An empty result
    gives a root without children.
    """
    sloc = {node.id: int(node.sloc) for node in result.graph.nodes}
    root = _TreeDir(name=root_name, path="")
    for fs in result.files:
        parts = fs.path.split("/")
        node = root
        for i, part in enumerate(parts[:-1]):
            child = node.dirs.get(part)
            if child is None:
                child = _TreeDir(name=part, path="/".join(parts[: i + 1]))
                node.dirs[part] = child
            node = child
        node.files[parts[-1]] = {
            "name": parts[-1],
            "path": fs.path,
            "sloc": sloc.get(fs.path, 0),
            "priority": int(fs.priority),
            "band": fs.band,
        }
    return root.to_json()


def shap_by_feature(bundle: Bundle, features: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-feature SHAP values of every row of ``features`` (index = path), relative to the repository.

    Recomputes ``Explainer(bundle).explain(features)`` exactly as ``analyze`` does, which gives the same
    numbers; each value holds the 46 base features (raw plus ``pct_`` contribution, ranker-margin units).
    An empty frame gives ``{}`` without building an explainer.
    """
    if len(features) == 0:
        return {}
    explanations = Explainer(bundle).explain(features)
    return {
        str(path): {str(k): float(v) for k, v in e.shap_by_feature.items()}
        for path, e in zip(features.index, explanations, strict=True)
    }


def build_payload(
    result: AnalysisResult, bundle: Bundle, *, budget_hours: float, root_name: str = ""
) -> ResultPayload:
    """Everything :meth:`Store.save_results` writes for ``result`` (see :class:`ResultPayload`).

    ``budget_hours`` is the analysis budget the plan was computed for and ``root_name`` the repository
    name shown at the treemap root. Raises ``ValueError`` from ``rows_from_result`` for inconsistent results.
    """
    rows = rows_from_result(result, shap_by_feature=shap_by_feature(bundle, result.features))
    return ResultPayload(
        rows=tuple(rows),
        graph=to_jsonable(result.graph),
        treemap=build_treemap(result, root_name=root_name),
        plan=to_jsonable(result.plan),
        budget_hours=float(budget_hours),
        summary=build_summary(result),
        stats=dict(result.stats),
    )


def score_from_row(row: FileResult | FileResultRow) -> FileScore:
    """Rebuild the ``FileScore`` of a stored file result (an ORM row or a ``FileResultRow``).

    Reasons and mitigations come from ``reasons["reasons"]`` and ``reasons["mitigations"]``, the family
    SHAP values from ``shap["by_family"]`` (missing keys give empty values).
    """
    reasons: Mapping[str, Any] = row.reasons or {}
    shap: Mapping[str, Any] = row.shap or {}
    return FileScore(
        path=row.path,
        score=float(row.score),
        priority=int(row.priority),
        band=cast(Band, row.band),
        risk=float(row.risk),
        risk_low=float(row.risk_low),
        risk_high=float(row.risk_high),
        reasons=tuple(from_jsonable(Reason, r) for r in reasons.get("reasons", [])),
        mitigations=tuple(from_jsonable(Reason, r) for r in reasons.get("mitigations", [])),
        debt_types=tuple(row.debt_types),
        shap_by_family={str(k): float(v) for k, v in shap.get("by_family", {}).items()},
    )


# ---------- async bridge ----------


def run_coro[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion from synchronous code and return its result.

    Uses ``asyncio.run`` when no event loop is running in this thread. Inside a running loop (an eager task
    called from async API code) it runs ``asyncio.run`` in a helper thread and waits for it, so the caller's
    loop is blocked for that time but never re-entered.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tdp-worker-db") as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------- the store ----------


class Store(Protocol):
    """Synchronous storage used by the tasks (state machine of ``tdp_platform.db.repos``)."""

    def load_analysis(self, analysis_id: uuid.UUID) -> AnalysisInfo | None:
        """The analysis with its repository, or None."""
        ...

    def set_running(self, analysis_id: uuid.UUID) -> bool:
        """Queued or running to running; False otherwise."""
        ...

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        """Record the stage and percentage while queued or running; False otherwise."""
        ...

    def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        """Queued or running to failed with a user-safe message; False otherwise."""
        ...

    def save_results(self, analysis_id: uuid.UUID, payload: ResultPayload) -> bool:
        """Store rows, artifacts and the plan and mark the analysis succeeded, all or nothing.

        False (and nothing stored) unless the analysis is running.
        """
        ...

    def load_file(self, file_result_id: int) -> FileInfo | None:
        """A stored file result with its analysis, or None."""
        ...

    def begin_advice(self, cache_key: str, *, file_result_id: int, model: str) -> str:
        """Create queued advice unless the key exists; return the row's status afterwards."""
        ...

    def advice_ready(self, cache_key: str, body: Mapping[str, Any]) -> bool:
        """Queued advice to ready with its JSON body; False otherwise."""
        ...

    def advice_failed(self, cache_key: str, error: str) -> bool:
        """Queued advice to failed with a user-safe message; False otherwise."""
        ...

    def static_cache(self) -> StaticCache | None:
        """The parse cache to give ``analyze`` (None for no cache)."""
        ...

    def close(self) -> None:
        """Release connections."""
        ...


def _analysis_info(row: Analysis) -> AnalysisInfo:
    repo = row.repository
    return AnalysisInfo(
        id=row.id,
        repository_id=row.repository_id,
        url=repo.url,
        host=repo.host,
        owner=repo.owner,
        name=repo.name,
        commit_sha=row.commit_sha,
        model_version=row.model_version,
        budget_hours=float(row.budget_hours),
        status=row.status,
        stage=row.stage,
        progress=int(row.progress),
    )


class PostgresStore:
    """:class:`Store` on the P5.1 repositories, one asyncpg engine without pooling per store.

    Without pooling every call opens a fresh connection, so the engine works across the separate event
    loops of :func:`run_coro`. Reads use a plain session; each state change runs in its own short
    transaction. Construction does not connect. Use a store only in the process that created it.
    """

    def __init__(self, database_url: str, *, schema: str | None = None) -> None:
        """A store for ``database_url`` (any Postgres driver name) with tables in ``schema`` (or public)."""
        self._url = database_url
        self._schema = schema
        self._engine = create_async_db_engine(database_url, schema=schema, null_pool=True)
        self._factory = make_session_factory(self._engine)
        self._static_cache: PostgresStaticCache | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: PlatformSettings) -> PostgresStore:
        """A store for ``DATABASE_URL`` and ``DATABASE_SCHEMA``."""
        return cls(settings.database_url, schema=settings.database_schema)

    def _write(self, action: Callable[[AsyncSession], Awaitable[bool]]) -> bool:
        async def go() -> bool:
            async with self._factory.begin() as session:
                return await action(session)

        return run_coro(go())

    def load_analysis(self, analysis_id: uuid.UUID) -> AnalysisInfo | None:
        async def go() -> AnalysisInfo | None:
            async with self._factory() as session:
                row = await AnalysisRepo(session).get(analysis_id)
                return None if row is None else _analysis_info(row)

        return run_coro(go())

    def set_running(self, analysis_id: uuid.UUID) -> bool:
        return self._write(lambda s: AnalysisRepo(s).set_running(analysis_id))

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        return self._write(lambda s: AnalysisRepo(s).set_progress(analysis_id, stage, pct))

    def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        return self._write(lambda s: AnalysisRepo(s).set_failed(analysis_id, error))

    def save_results(self, analysis_id: uuid.UUID, payload: ResultPayload) -> bool:
        async def go() -> bool:
            # One explicit transaction: a manual rollback inside begin() would break its commit on exit.
            async with self._factory() as session:
                await FileResultRepo(session).bulk_insert(analysis_id, payload.rows)
                artifacts = ArtifactRepo(session)
                await artifacts.put(analysis_id, "graph", payload.graph)
                await artifacts.put(analysis_id, "treemap", payload.treemap)
                await PlanRepo(session).put(analysis_id, payload.budget_hours, payload.plan)
                ok = await AnalysisRepo(session).set_succeeded(analysis_id, payload.summary, payload.stats)
                if ok:
                    await session.commit()
                else:
                    await session.rollback()
                return ok

        return run_coro(go())

    def load_file(self, file_result_id: int) -> FileInfo | None:
        async def go() -> FileInfo | None:
            async with self._factory() as session:
                row = await FileResultRepo(session).get(file_result_id)
                if row is None:
                    return None
                analysis = await AnalysisRepo(session).get(row.analysis_id)
                if analysis is None:  # pragma: no cover - the foreign key keeps the analysis
                    return None
                return FileInfo(
                    id=row.id,
                    analysis=_analysis_info(analysis),
                    path=row.path,
                    blob_sha=row.blob_sha,
                    score=score_from_row(row),
                    extra=from_jsonable(FileExtra, row.extra),
                )

        return run_coro(go())

    def begin_advice(self, cache_key: str, *, file_result_id: int, model: str) -> str:
        async def go() -> str:
            async with self._factory.begin() as session:
                row, _ = await AdviceRepo(session).create_queued(
                    cache_key, file_result_id=file_result_id, model=model
                )
                return str(row.status)

        return run_coro(go())

    def advice_ready(self, cache_key: str, body: Mapping[str, Any]) -> bool:
        return self._write(lambda s: AdviceRepo(s).set_ready(cache_key, dict(body)))

    def advice_failed(self, cache_key: str, error: str) -> bool:
        return self._write(lambda s: AdviceRepo(s).set_failed(cache_key, error))

    def static_cache(self) -> StaticCache | None:
        """The ``static_cache`` table through a psycopg engine, created on first use."""
        with self._lock:
            if self._static_cache is None:
                self._static_cache = PostgresStaticCache.from_url(self._url, schema=self._schema)
            return self._static_cache

    def close(self) -> None:
        """Dispose the static cache engine and the asyncpg engine."""
        with self._lock:
            cache, self._static_cache = self._static_cache, None
        if cache is not None:
            cache.close()
        run_coro(self._engine.dispose())


# ---------- per-process store ----------

_store: Store | None = None
_store_pid: int | None = None
_store_lock = threading.Lock()


def get_store() -> Store:
    """The store of this process: a :class:`PostgresStore` from the settings, rebuilt after a fork."""
    global _store, _store_pid
    with _store_lock:
        pid = os.getpid()
        if _store is None or _store_pid != pid:
            _store = PostgresStore.from_settings(get_settings())
            _store_pid = pid
        return _store


def set_store(store: Store | None) -> None:
    """Use ``store`` in this process (tests), or None to build a new one from the settings on next use.

    The previous store is dropped, not closed: after a fork its connections belong to the parent process.
    """
    global _store, _store_pid
    with _store_lock:
        _store = store
        _store_pid = os.getpid() if store is not None else None
