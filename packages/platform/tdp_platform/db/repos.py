"""Repositories: every read and write of the platform database goes through these classes.

Async repositories (``RepositoryRepo``, ``ModelVersionRepo``, ``AnalysisRepo``, ``FileResultRepo``,
``ArtifactRepo``, ``PlanRepo``, ``AdviceRepo``, ``FeedbackRepo``) wrap an :class:`AsyncSession` on the
asyncpg driver. They never commit or roll back: the caller owns the transaction (``async with
factory.begin() as session`` or an explicit ``await session.commit()``), and after a database error the
caller must roll back. Inputs are validated before any SQL runs (``ValueError``). State changes are single
``UPDATE ... WHERE <expected state> RETURNING`` statements and report whether they matched, so concurrent
actors cannot move an analysis into an impossible state. Reads of mutable rows refresh objects already in
the session, and ``Analysis.repository`` is always loaded eagerly.

:class:`PostgresStaticCache` is a synchronous (psycopg) cache of parsed ``FileSyntax`` objects for workers,
implementing the ``StaticCache`` protocol of feature assembly.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self, cast

from sqlalchemy import (
    REAL,
    URL,
    ColumnElement,
    Engine,
    Select,
    Table,
    and_,
    delete,
    func,
    or_,
    select,
    update,
)
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import QueryableAttribute, contains_eager, joinedload, load_only

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import BANDS, DEBT_TYPES, EXTRACTOR_VERSION, FileSyntax
from tdp_core.serialize import from_jsonable, to_jsonable
from tdp_platform.db.models import (
    ARTIFACT_KINDS,
    FEEDBACK_VERDICTS,
    Analysis,
    AnalysisArtifact,
    AnalysisStatus,
    ArtifactKind,
    Feedback,
    FileResult,
    ModelVersion,
    Repository,
    StaticCacheEntry,
    StoredAdvice,
    StoredPlan,
    Verdict,
)
from tdp_platform.db.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_FILE_ID,
    MAX_PAGE_SIZE,
    SORT_KEYS,
    SORT_ORDERS,
    CursorValue,
    SortKey,
    SortOrder,
    decode_cursor,
    encode_cursor,
)
from tdp_platform.db.rows import COPY_COLUMNS, FileResultRow, copy_records
from tdp_platform.db.session import create_sync_db_engine, json_dumps

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS: Final = 2000
DONE_STAGE: Final = "done"
SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
MAX_VERSION_CHARS: Final = 128

_ACTIVE: Final[tuple[AnalysisStatus, ...]] = ("queued", "running")
_BAND_NAMES: Final = frozenset(name for name, _ in BANDS)
_BAD_TEXT = re.compile("[\x00\ud800-\udfff]")  # Postgres text cannot hold NUL or lone surrogates
_REAL_MAX: Final = 3.4028234663852886e38  # largest float4
_REAL_MIN: Final = 1.1754943508222875e-38  # smallest normal float4
_SUMMARY_COLUMNS: Final = (
    FileResult.id,
    FileResult.analysis_id,
    FileResult.path,
    FileResult.blob_sha,
    FileResult.priority,
    FileResult.band,
    FileResult.risk,
    FileResult.risk_low,
    FileResult.risk_high,
    FileResult.score,
    FileResult.cost_low,
    FileResult.cost_high,
    FileResult.debt_types,
    FileResult.top_reason,
)


# ---------- validation helpers ----------


def _check_sha(value: str, name: str = "commit_sha") -> str:
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{name}: expected 40 lower case hex characters, got {value!r}")
    return value


def _check_text(value: str | None, name: str, *, max_chars: int | None = None) -> str:
    if not isinstance(value, str) or not value or _BAD_TEXT.search(value):
        raise ValueError(f"{name}: expected non-empty text without NUL characters, got {value!r}")
    if max_chars is not None and len(value) > max_chars:
        raise ValueError(f"{name}: longer than {max_chars} characters")
    return value


def _check_optional_text(value: str | None, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or _BAD_TEXT.search(value)):
        raise ValueError(f"{name}: expected text without NUL characters")
    return value


def _check_hours(value: float, name: str = "budget_hours") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number of hours, got {value!r}")
    hours = float(value)
    if not math.isfinite(hours) or not _REAL_MIN <= hours <= _REAL_MAX:
        raise ValueError(f"{name}: expected a positive, finite number of hours, got {value!r}")
    return hours


def _check_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_FILE_ID:
        raise ValueError(f"{name}: expected a positive BIGINT id, got {value!r}")
    return value


def _check_json(value: Any, name: str) -> None:
    if value is None:
        raise ValueError(f"{name}: a JSON value is required")
    try:
        json_dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: cannot be stored as JSON ({exc})") from exc


def _error_text(error: str) -> str:
    """``error`` cut to ``MAX_ERROR_CHARS`` (ending in "..." when cut) with unstorable characters replaced."""
    text = _BAD_TEXT.sub("�", str(error))
    if len(text) > MAX_ERROR_CHARS:
        text = text[: MAX_ERROR_CHARS - 3] + "..."
    return text


# ---------- repositories and model versions ----------


class RepositoryRepo:
    """Rows of ``repositories`` (immutable once created)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, repository_id: uuid.UUID) -> Repository | None:
        """The repository with this id, or None."""
        return await self._session.get(Repository, repository_id)

    async def get_by_url(self, url: str) -> Repository | None:
        """The repository with this canonical URL, or None."""
        result = await self._session.execute(select(Repository).where(Repository.url == url))
        return result.scalar_one_or_none()

    async def get_or_create(self, *, url: str, host: str, owner: str, name: str) -> Repository:
        """The repository for ``url``, inserted if missing; an existing row is returned unchanged.

        Safe under concurrency (``INSERT ... ON CONFLICT DO NOTHING``). Empty fields raise ``ValueError``.
        """
        for field_name, value in (("url", url), ("host", host), ("owner", owner), ("name", name)):
            _check_text(value, field_name)
        stmt = (
            pg_insert(Repository)
            .values(url=url, host=host, owner=owner, name=name)
            .on_conflict_do_nothing(index_elements=["url"])
        )
        await self._session.execute(stmt)
        repository = await self.get_by_url(url)
        if repository is None:  # pragma: no cover - the row exists after the insert
            raise RuntimeError(f"repository {url!r} vanished after insert")
        return repository


class ModelVersionRepo:
    """Rows of ``model_versions``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, version: str) -> ModelVersion | None:
        """The model version row, or None."""
        stmt = (
            select(ModelVersion)
            .where(ModelVersion.version == version)
            .execution_options(populate_existing=True)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def ensure(self, version: str, metrics: Mapping[str, Any] | None = None) -> ModelVersion:
        """Make sure ``version`` exists and return its fresh row.

        ``version`` is non-empty text of at most ``MAX_VERSION_CHARS`` characters. Without ``metrics`` an
        existing row is left alone (a new one gets ``{}``); with ``metrics`` they replace the stored ones.
        ``promoted_at`` never changes after the insert.
        """
        _check_text(version, "version", max_chars=MAX_VERSION_CHARS)
        if metrics is None:
            stmt = (
                pg_insert(ModelVersion)
                .values(version=version)
                .on_conflict_do_nothing(index_elements=["version"])
            )
        else:
            _check_json(dict(metrics), "metrics")
            ins = pg_insert(ModelVersion).values(version=version, metrics=dict(metrics))
            stmt = ins.on_conflict_do_update(
                index_elements=["version"], set_={"metrics": ins.excluded.metrics}
            )
        await self._session.execute(stmt)
        row = await self.get(version)
        if row is None:  # pragma: no cover - the row exists after the upsert
            raise RuntimeError(f"model version {version!r} vanished after insert")
        return row


# ---------- analyses ----------


class AnalysisRepo:
    """Rows of ``analyses`` and their state machine.

    Transitions: queued -> running (``set_running``), queued or running -> failed (``set_failed``),
    running -> succeeded (``set_succeeded``), failed -> queued (``requeue``). Progress may change while
    queued or running. Every transition returns False when the row was not in an allowed state (or does
    not exist), leaving it untouched.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _select() -> Select[Analysis]:
        return (
            select(Analysis)
            .options(joinedload(Analysis.repository, innerjoin=True))
            .execution_options(populate_existing=True)
        )

    async def get(self, analysis_id: uuid.UUID) -> Analysis | None:
        """The analysis (with ``repository`` loaded), or None."""
        stmt = self._select().where(Analysis.id == analysis_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def find(self, url: str, commit_sha: str, model_version: str) -> Analysis | None:
        """The analysis of the repository with canonical ``url`` at ``commit_sha`` with ``model_version``.

        ``commit_sha`` must be a full lower case SHA-1 (``ValueError`` otherwise).
        """
        _check_sha(commit_sha)
        stmt = (
            select(Analysis)
            .join(Analysis.repository)
            .options(contains_eager(Analysis.repository))
            .where(
                Repository.url == url,
                Analysis.commit_sha == commit_sha,
                Analysis.model_version == model_version,
            )
            .execution_options(populate_existing=True)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        repository_id: uuid.UUID,
        commit_sha: str,
        model_version: str,
        budget_hours: float = DEFAULT_SETTINGS.default_budget_hours,
    ) -> tuple[Analysis, bool]:
        """Insert a queued analysis, or return the existing one for the same triple.

        Returns ``(analysis, created)``; ``created`` is False when the (repository, commit, model version)
        triple already existed, in which case nothing changes (not even the budget). Safe under
        concurrency. ``commit_sha`` is a full lower case SHA-1, ``budget_hours`` a positive number of hours.
        An unknown repository or model version raises ``sqlalchemy.exc.IntegrityError``.
        """
        _check_sha(commit_sha)
        _check_text(model_version, "model_version")
        budget = _check_hours(budget_hours)
        stmt = (
            pg_insert(Analysis)
            .values(
                repository_id=repository_id,
                commit_sha=commit_sha,
                model_version=model_version,
                budget_hours=budget,
                status="queued",
            )
            .on_conflict_do_nothing(index_elements=["repository_id", "commit_sha", "model_version"])
            .returning(Analysis.id)
        )
        new_id = (await self._session.execute(stmt)).scalar_one_or_none()
        created = new_id is not None
        if new_id is None:
            existing = select(Analysis.id).where(
                Analysis.repository_id == repository_id,
                Analysis.commit_sha == commit_sha,
                Analysis.model_version == model_version,
            )
            new_id = (await self._session.execute(existing)).scalar_one()
        analysis = await self.get(new_id)
        if analysis is None:  # pragma: no cover - the row exists after the insert
            raise RuntimeError(f"analysis {new_id} vanished after insert")
        return analysis, created

    async def _transition(
        self, analysis_id: uuid.UUID, allowed: Sequence[AnalysisStatus], values: dict[str, Any]
    ) -> bool:
        stmt = (
            update(Analysis)
            .where(Analysis.id == analysis_id, Analysis.status.in_(list(allowed)))
            .values(**values)
            .returning(Analysis.id)
            .execution_options(synchronize_session=False)
        )
        return (await self._session.execute(stmt)).first() is not None

    async def set_running(self, analysis_id: uuid.UUID) -> bool:
        """Mark a queued or running analysis as running.

        Sets ``started_at`` on the first call only and clears ``finished_at`` and ``error``; stage and
        progress are kept, so a redelivered task can call it again. False if the analysis is finished.
        """
        values = {
            "status": "running",
            "started_at": func.coalesce(Analysis.started_at, func.now()),
            "finished_at": None,
            "error": None,
        }
        return await self._transition(analysis_id, _ACTIVE, values)

    async def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        """Record the ``stage`` and progress ``pct`` (clamped to 0..100) of a queued or running analysis."""
        _check_text(stage, "stage")
        if isinstance(pct, bool) or not isinstance(pct, (int, float)) or not math.isfinite(pct):
            raise ValueError(f"pct: expected a number 0..100, got {pct!r}")
        progress = max(0, min(100, int(pct)))
        return await self._transition(analysis_id, _ACTIVE, {"stage": stage, "progress": progress})

    async def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        """Mark a queued or running analysis as failed with a user-safe ``error`` message.

        The message is cut to ``MAX_ERROR_CHARS`` characters (the last three become "..."). ``finished_at``
        is set to the transaction time.
        """
        values = {"status": "failed", "error": _error_text(error), "finished_at": func.now()}
        return await self._transition(analysis_id, _ACTIVE, values)

    async def set_succeeded(
        self, analysis_id: uuid.UUID, summary: Mapping[str, Any], stats: Mapping[str, Any]
    ) -> bool:
        """Mark a running analysis as succeeded with its ``summary`` and ``stats`` (JSON objects).

        Sets stage ``DONE_STAGE`` and progress 100. Only allowed from running: False means another actor
        changed the state, and the caller should roll back the transaction that stored the results.
        """
        _check_json(dict(summary), "summary")
        _check_json(dict(stats), "stats")
        values = {
            "status": "succeeded",
            "stage": DONE_STAGE,
            "progress": 100,
            "summary": dict(summary),
            "stats": dict(stats),
            "error": None,
            "finished_at": func.now(),
        }
        return await self._transition(analysis_id, ("running",), values)

    async def requeue(self, analysis_id: uuid.UUID) -> bool:
        """Put a failed analysis back in the queue so it can be retried.

        Clears stage, error, summary, stats, ``started_at`` and ``finished_at``, sets progress 0 and deletes
        the analysis's file results (with their advice and feedback), artifacts and plans. ``created_at``
        is kept. False (and nothing changes) unless the analysis is failed.
        """
        values = {
            "status": "queued",
            "stage": None,
            "progress": 0,
            "error": None,
            "summary": None,
            "stats": None,
            "started_at": None,
            "finished_at": None,
        }
        if not await self._transition(analysis_id, ("failed",), values):
            return False
        for stmt in (
            delete(FileResult).where(FileResult.analysis_id == analysis_id),
            delete(AnalysisArtifact).where(AnalysisArtifact.analysis_id == analysis_id),
            delete(StoredPlan).where(StoredPlan.analysis_id == analysis_id),
        ):
            await self._session.execute(stmt.execution_options(synchronize_session=False))
        return True

    async def list_for_repository(
        self, repository_id: uuid.UUID, *, status: AnalysisStatus | None = None
    ) -> list[Analysis]:
        """Analyses of a repository, oldest first (``created_at``, then id), optionally of one ``status``."""
        stmt = self._select().where(Analysis.repository_id == repository_id)
        if status is not None:
            stmt = stmt.where(Analysis.status == status)
        stmt = stmt.order_by(Analysis.created_at, Analysis.id)
        return list((await self._session.execute(stmt)).scalars().all())


# ---------- file results ----------


@dataclass(frozen=True)
class FilePage:
    """One page of ranked files.

    ``items`` hold only the summary columns (touching ``features``, ``shap``, ``reasons`` or ``extra``
    raises); ``next_cursor`` is None on the last page; ``total`` counts every file matching the filters.
    """

    items: tuple[FileResult, ...]
    next_cursor: str | None
    total: int


_SortColumn = ColumnElement[Any] | QueryableAttribute[Any]


def _check_sort(sort: SortKey, order: SortOrder, limit: int | None = None) -> None:
    if sort not in SORT_KEYS:
        raise ValueError(f"sort: expected one of {SORT_KEYS}, got {sort!r}")
    if order not in SORT_ORDERS:
        raise ValueError(f"order: expected one of {SORT_ORDERS}, got {order!r}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE
    ):
        raise ValueError(f"limit: expected 1..{MAX_PAGE_SIZE}, got {limit!r}")


def _sort_column(sort: SortKey) -> _SortColumn:
    if sort == "priority":
        return FileResult.priority
    if sort == "risk":
        return FileResult.risk
    if sort == "cost":
        return FileResult.cost_high
    # Code point order, like Python's sorted(); the server's default collation may differ.
    return FileResult.path.collate("C")


def _sort_value(item: FileResult, sort: SortKey) -> CursorValue:
    if sort == "priority":
        return item.priority
    if sort == "risk":
        return item.risk
    if sort == "cost":
        return item.cost_high
    return item.path


def _file_filters(
    analysis_id: uuid.UUID, bands: Sequence[str] | None, debt_type: str | None, path_prefix: str | None
) -> list[ColumnElement[bool]]:
    clauses: list[ColumnElement[bool]] = [FileResult.analysis_id == analysis_id]
    if bands is not None:
        if isinstance(bands, str):
            raise ValueError("bands: expected a sequence of band names, got a string")
        unknown = [b for b in bands if b not in _BAND_NAMES]
        if unknown:
            raise ValueError(f"bands: unknown bands {unknown!r}")
        if bands:
            clauses.append(FileResult.band.in_(list(bands)))
    if debt_type is not None:
        if debt_type not in DEBT_TYPES:
            raise ValueError(f"debt_type: expected one of {DEBT_TYPES}, got {debt_type!r}")
        clauses.append(FileResult.debt_types.contains([debt_type]))
    if path_prefix:
        _check_text(path_prefix, "path_prefix")
        clauses.append(FileResult.path.startswith(path_prefix, autoescape=True))
    return clauses


def ranked_files_query(
    analysis_id: uuid.UUID,
    *,
    sort: SortKey,
    order: SortOrder,
    bands: Sequence[str] | None,
    debt_type: str | None,
    path_prefix: str | None,
    after: tuple[CursorValue, int] | None,
    limit: int,
) -> Select[FileResult]:
    """SELECT of up to ``limit + 1`` summary rows of an analysis's files, for keyset paging.

    Sorting: ``priority``, ``risk``, ``cost`` (``cost_high``, hours) or ``path`` (code point order), in
    ``order``, ties broken by id ascending in both orders. ``after`` is the ``(value, id)`` of the last row
    of the previous page. Filters: ``bands`` (any of, empty or None means all), ``debt_type`` (files
    carrying it), ``path_prefix`` (literal prefix; ``%`` and ``_`` match themselves). Only summary columns
    are loaded; the JSON columns raise if touched. Raises ``ValueError`` for an unknown sort, order, band
    or debt type, or ``limit`` outside 1..``MAX_PAGE_SIZE``.
    """
    _check_sort(sort, order, limit)
    column = _sort_column(sort)
    stmt = (
        select(FileResult)
        .options(load_only(*_SUMMARY_COLUMNS, raiseload=True))
        .where(*_file_filters(analysis_id, bands, debt_type, path_prefix))
    )
    if after is not None:
        value, last_id = after
        beyond = column < value if order == "desc" else column > value
        stmt = stmt.where(or_(beyond, and_(column == value, FileResult.id > last_id)))
    ordered = column.desc() if order == "desc" else column.asc()
    return stmt.order_by(ordered, FileResult.id.asc()).limit(limit + 1)


def ranked_files_count(
    analysis_id: uuid.UUID, *, bands: Sequence[str] | None, debt_type: str | None, path_prefix: str | None
) -> Select[int]:
    """SELECT of the number of an analysis's files matching the filters of :func:`ranked_files_query`."""
    return (
        select(func.count())
        .select_from(FileResult)
        .where(*_file_filters(analysis_id, bands, debt_type, path_prefix))
    )


class FileResultRepo:
    """Rows of ``file_results``: bulk storage with COPY, ranked pages and full rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(
        self, analysis_id: uuid.UUID, rows: Iterable[FileResultRow], *, replace: bool = True
    ) -> int:
        """Store an analysis's files with one COPY and return how many were written.

        Rows are validated first (duplicate paths or unstorable JSON raise ``ValueError`` before anything
        touches the database); an unknown analysis raises ``LookupError``. With ``replace`` (the default)
        the analysis's existing files are deleted first, together with their advice and feedback, so a
        retried task can store its results again: pass all rows of an analysis in one call. With
        ``replace=False`` a duplicate path raises asyncpg's ``UniqueViolationError`` (COPY errors are not
        wrapped by SQLAlchemy). Rows get ids in the order given. The COPY runs inside the session's
        transaction; the session must use the asyncpg driver (``TypeError`` otherwise).
        """
        records = copy_records(analysis_id, list(rows))
        conn = await self._session.connection()
        if conn.dialect.driver != "asyncpg":
            raise TypeError(f"bulk_insert needs the asyncpg driver, not {conn.dialect.driver!r}")
        # Any statement through the session makes the driver adapter issue BEGIN before the raw COPY.
        found = await self._session.execute(select(Analysis.id).where(Analysis.id == analysis_id))
        if found.first() is None:
            raise LookupError(f"analysis {analysis_id} does not exist")
        if replace:
            await self._session.execute(
                delete(FileResult)
                .where(FileResult.analysis_id == analysis_id)
                .execution_options(synchronize_session=False)
            )
        if records:
            raw = await conn.get_raw_connection()
            driver = raw.driver_connection
            if driver is None:  # pragma: no cover - a checked out connection always has one
                raise RuntimeError("no driver connection")
            await driver.copy_records_to_table("file_results", records=records, columns=list(COPY_COLUMNS))
        return len(records)

    async def list_ranked(
        self,
        analysis_id: uuid.UUID,
        *,
        sort: SortKey = "priority",
        order: SortOrder = "desc",
        bands: Sequence[str] | None = None,
        debt_type: str | None = None,
        path_prefix: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> FilePage:
        """One page of an analysis's files (see :func:`ranked_files_query` for sorting and filters).

        ``cursor`` is the ``next_cursor`` of the previous page with the same sort and order; a malformed
        or mismatched cursor raises :class:`~tdp_platform.db.pagination.InvalidCursorError`.
        """
        _check_sort(sort, order, limit)
        after = None if cursor is None else decode_cursor(cursor, sort, order)
        stmt = ranked_files_query(
            analysis_id,
            sort=sort,
            order=order,
            bands=bands,
            debt_type=debt_type,
            path_prefix=path_prefix,
            after=after,
            limit=limit,
        )
        found = list((await self._session.execute(stmt)).scalars().all())
        count = ranked_files_count(analysis_id, bands=bands, debt_type=debt_type, path_prefix=path_prefix)
        total = (await self._session.execute(count)).scalar_one()
        next_cursor = None
        if len(found) > limit:
            found = found[:limit]
            last = found[-1]
            next_cursor = encode_cursor(sort, order, _sort_value(last, sort), last.id)
        return FilePage(items=tuple(found), next_cursor=next_cursor, total=total)

    async def get(self, file_id: int, *, analysis_id: uuid.UUID | None = None) -> FileResult | None:
        """The full row of a file (all JSON columns), or None; ``analysis_id`` restricts the match."""
        if isinstance(file_id, bool) or not isinstance(file_id, int) or not 1 <= file_id <= MAX_FILE_ID:
            return None
        stmt = select(FileResult).where(FileResult.id == file_id).execution_options(populate_existing=True)
        if analysis_id is not None:
            stmt = stmt.where(FileResult.analysis_id == analysis_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_all(self, analysis_id: uuid.UUID) -> list[FileResult]:
        """Full rows of every file of an analysis, priority descending then id ascending."""
        stmt = (
            select(FileResult)
            .where(FileResult.analysis_id == analysis_id)
            .order_by(FileResult.priority.desc(), FileResult.id.asc())
            .execution_options(populate_existing=True)
        )
        return list((await self._session.execute(stmt)).scalars().all())


# ---------- artifacts and plans ----------


def _check_kind(kind: ArtifactKind) -> None:
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"kind: expected one of {ARTIFACT_KINDS}, got {kind!r}")


class ArtifactRepo:
    """Rows of ``analysis_artifacts`` (the ``graph`` and ``treemap`` JSON of an analysis)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def put(self, analysis_id: uuid.UUID, kind: ArtifactKind, body: Any) -> None:
        """Store (or replace) an artifact; ``body`` is JSON-able, for example a ``GraphData`` dataclass."""
        _check_kind(kind)
        _check_json(body, "body")
        ins = pg_insert(AnalysisArtifact).values(analysis_id=analysis_id, kind=kind, body=to_jsonable(body))
        stmt = ins.on_conflict_do_update(
            index_elements=["analysis_id", "kind"], set_={"body": ins.excluded.body}
        )
        await self._session.execute(stmt)

    async def get(self, analysis_id: uuid.UUID, kind: ArtifactKind) -> dict[str, Any] | None:
        """The stored JSON of an artifact, or None."""
        _check_kind(kind)
        stmt = select(AnalysisArtifact.body).where(
            AnalysisArtifact.analysis_id == analysis_id, AnalysisArtifact.kind == kind
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


class PlanRepo:
    """Rows of ``plans``: fix plans of an analysis keyed by budget in hours.

    Budgets are stored as REAL (float4), so lookups compare in float4: ``get(aid, 0.1)`` finds the plan
    stored with ``0.1``.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def put(self, analysis_id: uuid.UUID, budget_hours: float, body: Any) -> None:
        """Store (or replace) the plan for a budget; ``body`` is JSON-able, for example a ``Plan``.

        Replacing also resets ``created_at`` to the transaction time. ``budget_hours`` must be positive
        and finite (``ValueError``).
        """
        budget = _check_hours(budget_hours)
        _check_json(body, "body")
        ins = pg_insert(StoredPlan).values(
            analysis_id=analysis_id, budget_hours=budget, body=to_jsonable(body)
        )
        stmt = ins.on_conflict_do_update(
            index_elements=["analysis_id", "budget_hours"],
            set_={"body": ins.excluded.body, "created_at": func.now()},
        )
        await self._session.execute(stmt)

    async def get(self, analysis_id: uuid.UUID, budget_hours: float) -> dict[str, Any] | None:
        """The stored plan JSON for a budget, or None."""
        budget = _check_hours(budget_hours)
        stmt = select(StoredPlan.body).where(
            StoredPlan.analysis_id == analysis_id,
            # asyncpg binds Python floats as float8; compare in float4 like the stored value.
            StoredPlan.budget_hours == sql_cast(budget, REAL),
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


# ---------- advice and feedback ----------


class AdviceRepo:
    """Rows of ``advice``: LLM advice keyed by cache key, with states queued, ready and failed.

    Transitions: queued -> ready (``set_ready``), queued -> failed (``set_failed``), failed -> queued
    (``requeue``); each returns False when the row was not in the expected state. ``created_at`` never
    changes.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, cache_key: str) -> StoredAdvice | None:
        """The advice row, or None."""
        stmt = (
            select(StoredAdvice)
            .where(StoredAdvice.cache_key == cache_key)
            .execution_options(populate_existing=True)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create_queued(
        self, cache_key: str, *, file_result_id: int, model: str
    ) -> tuple[StoredAdvice, bool]:
        """Insert queued advice, or return the existing row for ``cache_key``; ``(row, created)``.

        A missing file raises ``sqlalchemy.exc.IntegrityError``.
        """
        _check_text(cache_key, "cache_key")
        _check_text(model, "model")
        _check_id(file_result_id, "file_result_id")
        stmt = (
            pg_insert(StoredAdvice)
            .values(cache_key=cache_key, file_result_id=file_result_id, model=model, status="queued")
            .on_conflict_do_nothing(index_elements=["cache_key"])
            .returning(StoredAdvice.cache_key)
        )
        created = (await self._session.execute(stmt)).first() is not None
        row = await self.get(cache_key)
        if row is None:  # pragma: no cover - the row exists after the insert
            raise RuntimeError(f"advice {cache_key!r} vanished after insert")
        return row, created

    async def _transition(self, cache_key: str, expected: str, values: dict[str, Any]) -> bool:
        stmt = (
            update(StoredAdvice)
            .where(StoredAdvice.cache_key == cache_key, StoredAdvice.status == expected)
            .values(**values)
            .returning(StoredAdvice.cache_key)
            .execution_options(synchronize_session=False)
        )
        return (await self._session.execute(stmt)).first() is not None

    async def set_ready(self, cache_key: str, body: Any) -> bool:
        """Store the advice ``body`` (JSON; pass pydantic models as ``model_dump(mode="json")``)."""
        _check_json(body, "body")
        values = {"status": "ready", "body": to_jsonable(body), "error": None}
        return await self._transition(cache_key, "queued", values)

    async def set_failed(self, cache_key: str, error: str) -> bool:
        """Mark queued advice as failed; ``error`` is cut to ``MAX_ERROR_CHARS`` characters."""
        return await self._transition(
            cache_key, "queued", {"status": "failed", "body": None, "error": _error_text(error)}
        )

    async def requeue(self, cache_key: str) -> bool:
        """Put failed advice back in the queue (body and error cleared)."""
        return await self._transition(cache_key, "failed", {"status": "queued", "body": None, "error": None})


class FeedbackRepo:
    """Rows of ``feedback``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        file_result_id: int,
        verdict: Verdict,
        comment: str | None = None,
        client_id: str | None = None,
    ) -> int:
        """Store a verdict on a file and return the new feedback id.

        ``verdict`` must be one of ``FEEDBACK_VERDICTS`` and texts free of NUL characters (``ValueError``);
        a missing file raises ``sqlalchemy.exc.IntegrityError``.
        """
        if verdict not in FEEDBACK_VERDICTS:
            raise ValueError(f"verdict: expected one of {FEEDBACK_VERDICTS}, got {verdict!r}")
        _check_id(file_result_id, "file_result_id")
        _check_optional_text(comment, "comment")
        _check_optional_text(client_id, "client_id")
        stmt = (
            pg_insert(Feedback)
            .values(file_result_id=file_result_id, verdict=verdict, comment=comment, client_id=client_id)
            .returning(Feedback.id)
        )
        return int((await self._session.execute(stmt)).scalar_one())


# ---------- static cache (sync) ----------

_STATIC_CACHE = cast(Table, StaticCacheEntry.__table__)


def syntax_is_storable(syntax: FileSyntax) -> bool:
    """False when any text in ``syntax`` holds a NUL character or a lone surrogate.

    JSONB cannot store ``\\u0000`` and lone surrogates cannot be encoded as UTF-8, so such parses are not
    cached (they are simply parsed again next time).
    """
    parts: list[str] = [syntax.path, syntax.language]
    parts.extend(c.text for c in syntax.comments)
    parts.extend(f.name for f in syntax.functions)
    for imp in syntax.imports:
        parts.append(imp.module)
        parts.extend(imp.names)
    parts.extend(syntax.tokens)
    return _BAD_TEXT.search("\n".join(parts)) is None


class PostgresStaticCache:
    """Synchronous cache of parsed files in ``static_cache``, keyed by blob sha and extractor version.

    Implements the ``StaticCache`` protocol (``get``/``put``). ``get`` returns the ``FileSyntax`` exactly as
    stored, including the ``path`` of the file it was first parsed from: when the same blob sits at another
    path, replace the path with ``dataclasses.replace``. Rows that no longer decode count as misses (with a
    warning); parses that JSONB cannot hold are skipped. Use a cache only in the process that created it
    (database engines are not fork safe).
    """

    def __init__(self, engine: Engine, *, extractor_version: str = EXTRACTOR_VERSION) -> None:
        """Wrap ``engine`` (a psycopg engine the caller owns and disposes)."""
        self._engine = engine
        self._extractor_version = _check_text(extractor_version, "extractor_version")
        self._owns_engine = False

    @classmethod
    def from_url(
        cls,
        database_url: str | URL,
        *,
        schema: str | None = None,
        extractor_version: str = EXTRACTOR_VERSION,
    ) -> PostgresStaticCache:
        """A cache with its own pooled psycopg engine for ``database_url``; ``close()`` disposes it."""
        cache = cls(create_sync_db_engine(database_url, schema=schema), extractor_version=extractor_version)
        cache._owns_engine = True
        return cache

    @property
    def extractor_version(self) -> str:
        """The extractor version this cache reads and writes."""
        return self._extractor_version

    def get(self, blob_sha: str) -> FileSyntax | None:
        """The cached parse of a blob, or None (``ValueError`` unless ``blob_sha`` is 40 lower case hex)."""
        _check_sha(blob_sha, "blob_sha")
        stmt = select(_STATIC_CACHE.c.syntax).where(
            _STATIC_CACHE.c.blob_sha == blob_sha,
            _STATIC_CACHE.c.extractor_version == self._extractor_version,
        )
        with self._engine.connect() as conn:
            data = conn.execute(stmt).scalar_one_or_none()
        if data is None:
            return None
        try:
            return from_jsonable(FileSyntax, data)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            logger.warning("ignoring undecodable static cache row for blob %s: %s", blob_sha, exc)
            return None

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        """Cache the parse of a blob; an existing entry is kept. Unstorable parses are skipped."""
        _check_sha(blob_sha, "blob_sha")
        if not syntax_is_storable(syntax):
            logger.debug("not caching blob %s: its text cannot be stored in JSONB", blob_sha)
            return
        stmt = (
            pg_insert(_STATIC_CACHE)
            .values(blob_sha=blob_sha, extractor_version=self._extractor_version, syntax=to_jsonable(syntax))
            .on_conflict_do_nothing(index_elements=["blob_sha", "extractor_version"])
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def close(self) -> None:
        """Dispose the engine if this cache created it (``from_url``); otherwise do nothing."""
        if self._owns_engine:
            self._engine.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
