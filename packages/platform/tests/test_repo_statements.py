"""Repositories driven through a recording session: the SQL they send and how they read results.

These run without a database; the Postgres behaviour itself is covered by the ``test_pg_*`` modules in CI.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import make_url
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Executable

from tdp_platform.db.models import FileResult
from tdp_platform.db.pagination import decode_cursor
from tdp_platform.db.repos import (
    DONE_STAGE,
    AdviceRepo,
    AnalysisRepo,
    ArtifactRepo,
    FeedbackRepo,
    FileResultRepo,
    ModelVersionRepo,
    PlanRepo,
    RepositoryRepo,
)
from tdp_platform.db.rows import COPY_COLUMNS, FileResultRow

pytestmark = pytest.mark.asyncio

MakeRow = Callable[..., FileResultRow]
DIALECT = make_url("postgresql+asyncpg://").get_dialect()()
AID = uuid.UUID("12345678-1234-5678-1234-567812345678")
SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self) -> Any:
        return self.value

    def scalar_one(self) -> Any:
        assert self.value is not None
        return self.value

    def first(self) -> Any:
        return None if self.value is None else (self.value,)

    def scalars(self) -> FakeResult:
        return self

    def all(self) -> list[Any]:
        return list(self.value or [])


class FakeDriver:
    def __init__(self, session: RecordingSession) -> None:
        self.session = session
        self.copies: list[tuple[str, list[tuple[Any, ...]], list[str], int]] = []

    async def copy_records_to_table(
        self, table: str, *, records: list[tuple[Any, ...]], columns: list[str]
    ) -> None:
        self.copies.append((table, records, columns, len(self.session.statements)))


class FakeConnection:
    def __init__(self, session: RecordingSession, driver: str) -> None:
        self.dialect = SimpleNamespace(driver=driver)
        self.raw = SimpleNamespace(driver_connection=FakeDriver(session))

    async def get_raw_connection(self) -> Any:
        return self.raw


class RecordingSession:
    """Records statements; each ``execute`` returns the next queued value (None when the queue is empty)."""

    def __init__(self, *results: Any, driver: str = "asyncpg") -> None:
        self.statements: list[Executable] = []
        self.results = list(results)
        self.conn = FakeConnection(self, driver)

    async def execute(self, statement: Executable, params: Any = None) -> FakeResult:
        self.statements.append(statement)
        return FakeResult(self.results.pop(0) if self.results else None)

    async def connection(self) -> FakeConnection:
        return self.conn

    async def get(self, model: Any, ident: Any) -> Any:
        self.statements.append(cast(Executable, ("get", model, ident)))
        return self.results.pop(0) if self.results else None

    def sql(self, index: int) -> str:
        return " ".join(str(self.compiled(index)).split())

    def params(self, index: int) -> dict[str, Any]:
        return dict(self.compiled(index).params)

    def compiled(self, index: int) -> Any:
        statement = self.statements[index]
        return statement.compile(dialect=DIALECT)  # type: ignore[attr-defined]


def _session(recording: RecordingSession) -> AsyncSession:
    return cast(AsyncSession, recording)


async def test_set_succeeded_only_from_running() -> None:
    rec = RecordingSession(AID)
    assert await AnalysisRepo(_session(rec)).set_succeeded(AID, {"critical": 1}, {"n_files": 2})
    sql = rec.sql(0)
    assert sql.startswith("UPDATE analyses SET status=")
    assert "WHERE analyses.id = $" in sql and "analyses.status IN" in sql
    assert sql.endswith("RETURNING analyses.id")
    params = rec.params(0)
    assert ["running"] in params.values()
    assert {"succeeded", DONE_STAGE, 100} <= {v for v in params.values() if not isinstance(v, (list, dict))}
    assert {"critical": 1} in params.values() and {"n_files": 2} in params.values()

    rec = RecordingSession(None)
    assert not await AnalysisRepo(_session(rec)).set_succeeded(AID, {}, {})


async def test_transitions_name_their_allowed_states() -> None:
    rec = RecordingSession(AID, AID, AID)
    analyses = AnalysisRepo(_session(rec))
    assert await analyses.set_running(AID)
    assert await analyses.set_progress(AID, "parse", 250)
    assert await analyses.set_failed(AID, "e" * 3000)
    assert "coalesce(analyses.started_at, now())" in rec.sql(0)
    for index in range(3):
        assert ["queued", "running"] in rec.params(index).values()
    assert 100 in rec.params(1).values()  # clamped
    error = next(v for v in rec.params(2).values() if isinstance(v, str) and v.startswith("e"))
    assert len(error) == 2000 and error.endswith("...")
    with pytest.raises(ValueError):
        await analyses.set_progress(AID, "parse", float("nan"))  # type: ignore[arg-type]


async def test_requeue_deletes_results_only_when_it_matched() -> None:
    rec = RecordingSession(None)
    assert not await AnalysisRepo(_session(rec)).requeue(AID)
    assert len(rec.statements) == 1

    rec = RecordingSession(AID)
    assert await AnalysisRepo(_session(rec)).requeue(AID)
    assert ["failed"] in rec.params(0).values()
    assert [rec.sql(i).split(" WHERE ")[0] for i in (1, 2, 3)] == [
        "DELETE FROM file_results",
        "DELETE FROM analysis_artifacts",
        "DELETE FROM plans",
    ]


async def test_create_inserts_or_reuses() -> None:
    existing = object()
    rec = RecordingSession(None, AID, existing)
    analysis, created = await AnalysisRepo(_session(rec)).create(
        repository_id=AID, commit_sha=SHA, model_version="v1"
    )
    assert analysis is existing and not created
    conflict = "ON CONFLICT (repository_id, commit_sha, model_version) DO NOTHING RETURNING analyses.id"
    assert conflict in rec.sql(0)
    assert rec.sql(1).startswith("SELECT analyses.id FROM analyses WHERE")
    assert "JOIN repositories" in rec.sql(2)
    assert "queued" in rec.params(0).values() and 40.0 in rec.params(0).values()

    rec = RecordingSession(AID, existing)
    analyses = AnalysisRepo(_session(rec))
    _, created = await analyses.create(repository_id=AID, commit_sha=SHA, model_version="v1")
    assert created and len(rec.statements) == 2


async def test_find_joins_on_url() -> None:
    rec = RecordingSession(None)
    assert await AnalysisRepo(_session(rec)).find("https://github.com/a/b", SHA, "v1") is None
    sql = rec.sql(0)
    assert "JOIN repositories ON repositories.id = analyses.repository_id" in sql
    assert "repositories.url = $" in sql


async def test_list_for_repository_orders_by_creation() -> None:
    rec = RecordingSession([])
    assert await AnalysisRepo(_session(rec)).list_for_repository(AID, status="succeeded") == []
    assert rec.sql(0).endswith("ORDER BY analyses.created_at, analyses.id")
    assert "succeeded" in rec.params(0).values()


async def test_repository_and_model_version_upserts() -> None:
    row: Any = object()
    rec = RecordingSession(None, row)
    repo = await RepositoryRepo(_session(rec)).get_or_create(url="u", host="h", owner="o", name="n")
    assert repo is row
    assert "ON CONFLICT (url) DO NOTHING" in rec.sql(0)

    rec = RecordingSession(None, row, None, row)
    versions = ModelVersionRepo(_session(rec))
    assert await versions.ensure("v1") is row
    assert await versions.ensure("v1", {"ndcg": 0.5}) is row
    assert "ON CONFLICT (version) DO NOTHING" in rec.sql(0)
    assert "ON CONFLICT (version) DO UPDATE SET metrics = excluded.metrics" in rec.sql(2)


async def test_artifact_and_plan_upserts() -> None:
    rec = RecordingSession(None, {"nodes": []})
    artifacts = ArtifactRepo(_session(rec))
    await artifacts.put(AID, "graph", {"nodes": [], "edges": []})
    assert await artifacts.get(AID, "graph") == {"nodes": []}
    assert "ON CONFLICT (analysis_id, kind) DO UPDATE SET body = excluded.body" in rec.sql(0)

    rec = RecordingSession(None, None)
    plans = PlanRepo(_session(rec))
    await plans.put(AID, 0.1, {"steps": []})
    assert await plans.get(AID, 0.1) is None
    assert "ON CONFLICT (analysis_id, budget_hours) DO UPDATE SET body = excluded.body" in rec.sql(0)
    assert "plans.budget_hours = CAST($" in rec.sql(1) and "AS REAL)" in rec.sql(1)


async def test_advice_and_feedback_statements() -> None:
    stored: Any = object()
    rec = RecordingSession(None, stored, "k", None, "k")
    advice = AdviceRepo(_session(rec))
    row, created = await advice.create_queued("k", file_result_id=5, model="m")
    assert row is stored and not created
    assert await advice.set_ready("k", {"summary": "s"})
    assert not await advice.set_failed("k", "boom")
    assert await advice.requeue("k")
    assert "ON CONFLICT (cache_key) DO NOTHING RETURNING advice.cache_key" in rec.sql(0)
    assert "queued" in rec.params(2).values() and "ready" in rec.params(2).values()
    assert "failed" in rec.params(4).values() and "queued" in rec.params(4).values()

    rec = RecordingSession(17)
    assert await FeedbackRepo(_session(rec)).create(file_result_id=5, verdict="agree") == 17
    assert rec.sql(0).endswith("RETURNING feedback.id")


async def test_bulk_insert_selects_before_copying(make_row: MakeRow) -> None:
    rec = RecordingSession(AID)
    rows = [make_row("b.py", priority=90), make_row("a.py", priority=10)]
    assert await FileResultRepo(_session(rec)).bulk_insert(AID, rows) == 2
    assert rec.sql(0).startswith("SELECT analyses.id FROM analyses WHERE analyses.id = $")
    assert rec.sql(1).startswith("DELETE FROM file_results WHERE file_results.analysis_id = $")
    driver = rec.conn.raw.driver_connection
    [(table, records, columns, statements_before)] = driver.copies
    assert (table, columns, statements_before) == ("file_results", list(COPY_COLUMNS), 2)
    assert [r[1] for r in records] == ["b.py", "a.py"] and records[0][0] == AID


async def test_bulk_insert_edge_cases(make_row: MakeRow) -> None:
    rec = RecordingSession(AID)
    assert await FileResultRepo(_session(rec)).bulk_insert(AID, [make_row("a.py")], replace=False) == 1
    assert len(rec.statements) == 1  # no DELETE

    rec = RecordingSession(AID)
    assert await FileResultRepo(_session(rec)).bulk_insert(AID, []) == 0
    assert rec.conn.raw.driver_connection.copies == []

    rec = RecordingSession(None)
    with pytest.raises(LookupError):
        await FileResultRepo(_session(rec)).bulk_insert(AID, [make_row("a.py")])

    rec = RecordingSession(AID, driver="psycopg")
    with pytest.raises(TypeError):
        await FileResultRepo(_session(rec)).bulk_insert(AID, [make_row("a.py")])

    rec = RecordingSession(AID)
    with pytest.raises(ValueError):
        await FileResultRepo(_session(rec)).bulk_insert(AID, [make_row("a.py"), make_row("a.py")])
    assert rec.statements == []


async def test_list_ranked_builds_the_next_cursor() -> None:
    items = [
        FileResult(id=3, path="b.py", priority=90, risk=0.5, cost_high=2.0),
        FileResult(id=7, path="a.py", priority=80, risk=0.25, cost_high=1.0),
        FileResult(id=9, path="c.py", priority=80, risk=0.75, cost_high=4.0),
    ]
    rec = RecordingSession(items, 12)
    page = await FileResultRepo(_session(rec)).list_ranked(AID, sort="risk", order="asc", limit=2)
    assert [i.id for i in page.items] == [3, 7] and page.total == 12
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor, "risk", "asc") == (0.25, 7)
    assert "LIMIT $" in rec.sql(0) and 3 in rec.params(0).values()
    assert rec.sql(1).startswith("SELECT count(*) AS count_1 FROM file_results")

    rec = RecordingSession(items[:2], 2)
    page = await FileResultRepo(_session(rec)).list_ranked(AID, limit=2, cursor=None)
    assert page.next_cursor is None


async def test_file_get_ignores_out_of_range_ids() -> None:
    rec = RecordingSession()
    files = FileResultRepo(_session(rec))
    for file_id in (0, -1, 2**63, True):
        assert await files.get(file_id) is None
    assert rec.statements == []
    assert await files.get(5, analysis_id=AID) is None
    assert "file_results.analysis_id = $" in rec.sql(0)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: RepositoryRepo(s).get_or_create(url="u", host="", owner="o", name="n"),
        lambda s: ModelVersionRepo(s).ensure("v" * 129),
        lambda s: AnalysisRepo(s).create(repository_id=AID, commit_sha="abc", model_version="v"),
        lambda s: AnalysisRepo(s).create(
            repository_id=AID, commit_sha=SHA, model_version="v", budget_hours=-1
        ),
        lambda s: AnalysisRepo(s).create(repository_id=AID, commit_sha=SHA, model_version="", budget_hours=1),
        lambda s: PlanRepo(s).put(AID, True, {}),
        lambda s: PlanRepo(s).put(AID, 1e39, {}),
        lambda s: AnalysisRepo(s).set_progress(AID, "", 5),
        lambda s: ArtifactRepo(s).put(AID, "graph", {"x": float("inf")}),
        lambda s: ArtifactRepo(s).put(AID, "graph", None),
        lambda s: ArtifactRepo(s).put(AID, "heatmap", {}),  # type: ignore[arg-type]
        lambda s: PlanRepo(s).put(AID, 0.0, {}),
        lambda s: PlanRepo(s).get(AID, float("nan")),
        lambda s: AdviceRepo(s).create_queued("", file_result_id=1, model="m"),
        lambda s: AdviceRepo(s).create_queued("k", file_result_id=0, model="m"),
        lambda s: AdviceRepo(s).set_ready("k", {"v": float("nan")}),
        lambda s: FeedbackRepo(s).create(file_result_id=1, verdict="maybe"),  # type: ignore[arg-type]
        lambda s: FeedbackRepo(s).create(file_result_id=1, verdict="agree", client_id="c\x00"),
        lambda s: FileResultRepo(s).list_ranked(AID, sort="size"),  # type: ignore[arg-type]
        lambda s: FileResultRepo(s).list_ranked(AID, limit=501),
        lambda s: FileResultRepo(s).list_ranked(AID, cursor="!!!"),
    ],
)
async def test_invalid_input_is_rejected_before_any_sql(call: Callable[[AsyncSession], Any]) -> None:
    rec = RecordingSession()
    with pytest.raises(ValueError):
        await call(_session(rec))
    assert rec.statements == []
