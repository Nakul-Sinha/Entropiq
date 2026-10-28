"""File results, ranked pages, artifacts and plans against Postgres (CI only)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tdp_core.contracts import AnalysisResult, FileExtra, GraphData, Plan
from tdp_core.serialize import from_jsonable
from tdp_platform.db.models import FileResult, StoredPlan
from tdp_platform.db.pagination import SORT_KEYS, SORT_ORDERS, CursorValue, InvalidCursorError, SortKey
from tdp_platform.db.repos import (
    AnalysisRepo,
    ArtifactRepo,
    FileResultRepo,
    ModelVersionRepo,
    PlanRepo,
    RepositoryRepo,
)
from tdp_platform.db.rows import FileResultRow, rows_from_result

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MakeRow = Callable[..., FileResultRow]
FLOAT_FIELDS = ("risk", "risk_low", "risk_high", "score", "cost_low", "cost_high")

# (path, priority, risk, cost_high, debt types). Ties on every numeric key; paths whose code point order
# differs from a natural-language collation ("B.py" < "Z.py" < "_x.py" < "a.py").
RANKED = (
    ("a.py", 90, 0.5, 2.0, ("complexity",)),
    ("B.py", 90, 0.75, 2.0, ("test",)),
    ("_x.py", 75, 0.5, 1.0, ("complexity", "test")),
    ("pkg/a_%.py", 75, 0.1, 4.0, ("duplication",)),
    ("pkg/ab.py", 50, 0.1, 1.0, ()),
    ("pkg/aX.py", 50, 0.25, 0.5, ("complexity",)),
    ("z.py", 10, 0.75, 3.0, ()),
    ("Z.py", 10, 0.25, 3.0, ()),
)


async def _analysis_id(session: AsyncSession) -> uuid.UUID:
    repository = await RepositoryRepo(session).get_or_create(
        url="https://github.com/acme/widgets", host="github.com", owner="acme", name="widgets"
    )
    await ModelVersionRepo(session).ensure("dev-test")
    analysis, _ = await AnalysisRepo(session).create(
        repository_id=repository.id, commit_sha="0" * 40, model_version="dev-test"
    )
    return analysis.id


async def _file_count(session: AsyncSession) -> int:
    return int((await session.execute(select(func.count()).select_from(FileResult))).scalar_one())


async def test_bulk_insert_round_trips_every_column(
    session: AsyncSession, analysis_result: AnalysisResult
) -> None:
    aid = await _analysis_id(session)
    rows = rows_from_result(analysis_result, shap_by_feature={"pkg/core.py": {"cc_max": 0.125}})
    files = FileResultRepo(session)
    assert await files.bulk_insert(aid, rows) == 4
    await session.commit()

    stored = await files.list_all(aid)
    assert [f.path for f in stored] == [r.path for r in rows]
    ids = [f.id for f in stored]
    assert ids == sorted(ids)
    for f, r in zip(stored, rows, strict=True):
        assert f.analysis_id == aid
        assert (f.blob_sha, f.priority, f.band, f.top_reason) == (
            r.blob_sha,
            r.priority,
            r.band,
            r.top_reason,
        )
        for name in FLOAT_FIELDS:
            assert getattr(f, name) == pytest.approx(getattr(r, name), rel=1e-6, abs=1e-6), name
        assert f.debt_types == list(r.debt_types)
        assert f.features == dict(r.features)
        assert f.shap == dict(r.shap)
        assert f.reasons == dict(r.reasons)
        assert f.extra == dict(r.extra)
        assert from_jsonable(FileExtra, f.extra) == analysis_result.extras[f.path]
    assert stored[0].shap["by_feature"] == {"cc_max": 0.125}

    assert await files.bulk_insert(aid, rows[:2]) == 2  # replaces the earlier rows
    await session.commit()
    assert [f.path for f in await files.list_all(aid)] == [r.path for r in rows[:2]]


async def test_bulk_insert_validates_before_writing(session: AsyncSession, make_row: MakeRow) -> None:
    aid = await _analysis_id(session)
    files = FileResultRepo(session)
    await files.bulk_insert(aid, [make_row("a.py")])
    await session.commit()

    with pytest.raises(ValueError, match="duplicate"):
        await files.bulk_insert(aid, [make_row("x.py"), make_row("x.py")])
    await session.rollback()
    assert [f.path for f in await files.list_all(aid)] == ["a.py"]

    with pytest.raises(LookupError):
        await files.bulk_insert(uuid.uuid4(), [make_row("b.py")])
    await session.rollback()

    assert await files.bulk_insert(aid, [], replace=False) == 0
    await session.commit()
    assert await _file_count(session) == 1
    assert await files.bulk_insert(aid, []) == 0
    await session.commit()
    assert await _file_count(session) == 0


async def test_bulk_insert_without_replace_rejects_duplicates(
    session: AsyncSession, make_row: MakeRow
) -> None:
    aid = await _analysis_id(session)
    files = FileResultRepo(session)
    assert await files.bulk_insert(aid, [make_row("a.py")], replace=False) == 1
    await session.commit()
    with pytest.raises(Exception, match="file_results_analysis_id_path_key") as excinfo:
        await files.bulk_insert(aid, [make_row("a.py")], replace=False)
    assert type(excinfo.value).__name__ == "UniqueViolationError"
    await session.rollback()
    assert await _file_count(session) == 1


async def test_bulk_insert_rolls_back_with_the_transaction(
    session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], make_row: MakeRow
) -> None:
    aid = await _analysis_id(session)
    await session.commit()
    await FileResultRepo(session).bulk_insert(aid, [make_row("a.py"), make_row("b.py")])
    async with session_factory() as other:
        assert await _file_count(other) == 0  # not committed yet
    await session.rollback()
    assert await _file_count(session) == 0


async def _ranked(session: AsyncSession, make_row: MakeRow) -> tuple[uuid.UUID, dict[str, int]]:
    aid = await _analysis_id(session)
    rows = [make_row(p, priority=pr, risk=r, cost=c, debt_types=d) for p, pr, r, c, d in RANKED]
    files = FileResultRepo(session)
    await files.bulk_insert(aid, rows)
    await session.commit()
    ids = {f.path: f.id for f in await files.list_all(aid)}
    return aid, ids


def _expected(sort: SortKey, order: str, ids: dict[str, int]) -> list[str]:
    def value(path: str) -> CursorValue:
        row = next(r for r in RANKED if r[0] == path)
        values: dict[str, CursorValue] = {"priority": row[1], "risk": row[2], "cost": row[3], "path": path}
        return values[sort]

    by_id = sorted(ids, key=ids.__getitem__)
    return sorted(by_id, key=value, reverse=order == "desc")  # stable: ties stay in id order


async def test_paging_reproduces_every_order(session: AsyncSession, make_row: MakeRow) -> None:
    aid, ids = await _ranked(session, make_row)
    files = FileResultRepo(session)
    for sort in SORT_KEYS:
        for order in SORT_ORDERS:
            seen: list[str] = []
            cursor: str | None = None
            for _ in range(10):
                page = await files.list_ranked(aid, sort=sort, order=order, limit=2, cursor=cursor)
                assert page.total == len(RANKED)
                seen.extend(item.path for item in page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
            assert seen == _expected(sort, order, ids), (sort, order)


async def test_filters(session: AsyncSession, make_row: MakeRow) -> None:
    aid, _ = await _ranked(session, make_row)
    files = FileResultRepo(session)

    async def paths(**kw: Any) -> tuple[list[str], int]:
        page = await files.list_ranked(aid, **kw)
        return [item.path for item in page.items], page.total

    assert await paths(bands=["High"]) == (["_x.py", "pkg/a_%.py"], 2)
    assert await paths(bands=["Critical", "Low"]) == (["a.py", "B.py", "z.py", "Z.py"], 4)
    assert await paths(bands=[]) == ([r[0] for r in sorted(RANKED, key=lambda r: -r[1])], 8)
    assert await paths(debt_type="complexity") == (["a.py", "_x.py", "pkg/aX.py"], 3)
    assert await paths(debt_type="knowledge") == ([], 0)
    assert await paths(path_prefix="pkg/a_%") == (["pkg/a_%.py"], 1)
    assert await paths(path_prefix="pkg/a_") == (["pkg/a_%.py"], 1)
    assert await paths(path_prefix="pkg/a") == (["pkg/a_%.py", "pkg/ab.py", "pkg/aX.py"], 3)
    assert await paths(bands=["Medium"], debt_type="complexity") == (["pkg/aX.py"], 1)

    first = await files.list_ranked(aid, bands=["Critical", "Low"], limit=3)
    assert [i.path for i in first.items] == ["a.py", "B.py", "z.py"] and first.next_cursor is not None
    second = await files.list_ranked(aid, bands=["Critical", "Low"], limit=3, cursor=first.next_cursor)
    assert [i.path for i in second.items] == ["Z.py"] and second.next_cursor is None
    assert first.total == second.total == 4


async def test_cursor_errors(session: AsyncSession, make_row: MakeRow) -> None:
    aid, _ = await _ranked(session, make_row)
    files = FileResultRepo(session)
    first = await files.list_ranked(aid, limit=2)
    assert first.next_cursor is not None
    with pytest.raises(InvalidCursorError):
        await files.list_ranked(aid, sort="risk", limit=2, cursor=first.next_cursor)
    with pytest.raises(InvalidCursorError):
        await files.list_ranked(aid, order="asc", limit=2, cursor=first.next_cursor)
    with pytest.raises(InvalidCursorError):
        await files.list_ranked(aid, cursor="!!!")
    with pytest.raises(ValueError):
        await files.list_ranked(aid, limit=0)


async def test_pages_hold_summary_columns_only(
    session: AsyncSession, session_factory: async_sessionmaker[AsyncSession], make_row: MakeRow
) -> None:
    aid, ids = await _ranked(session, make_row)
    async with session_factory() as fresh:
        files = FileResultRepo(fresh)
        page = await files.list_ranked(aid, limit=1)
        item = page.items[0]
        assert (item.path, item.priority, item.band) == ("a.py", 90, "Critical")
        assert item.top_reason == "Reason for a.py" and item.debt_types == ["complexity"]
        with pytest.raises(InvalidRequestError):
            _ = item.features
        full = await files.get(item.id)
        assert full is item
        assert full.features == {"sloc": 120.0, "cc_max": 7.0}

    files = FileResultRepo(session)
    file_id = ids["a.py"]
    found = await files.get(file_id, analysis_id=aid)
    assert found is not None and found.path == "a.py"
    assert await files.get(file_id, analysis_id=uuid.uuid4()) is None
    for missing in (10**9, 0, -1, 2**63):
        assert await files.get(missing) is None


async def test_artifacts(session: AsyncSession, analysis_result: AnalysisResult) -> None:
    aid = await _analysis_id(session)
    artifacts = ArtifactRepo(session)
    await artifacts.put(aid, "graph", analysis_result.graph)
    await artifacts.put(aid, "treemap", {"name": "", "path": "", "children": []})
    await session.commit()
    body = await artifacts.get(aid, "graph")
    assert body is not None and from_jsonable(GraphData, body) == analysis_result.graph
    assert await artifacts.get(aid, "treemap") == {"name": "", "path": "", "children": []}

    await artifacts.put(aid, "graph", {"nodes": [], "edges": []})
    await session.commit()
    assert await artifacts.get(aid, "graph") == {"nodes": [], "edges": []}
    assert await artifacts.get(uuid.uuid4(), "graph") is None
    with pytest.raises(ValueError):
        await artifacts.put(aid, "heatmap", {})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await artifacts.get(aid, "heatmap")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await artifacts.put(aid, "graph", None)
    with pytest.raises(ValueError):
        await artifacts.put(aid, "graph", {"weight": float("nan")})


async def test_plans_are_found_by_float4_budget(
    session: AsyncSession, analysis_result: AnalysisResult
) -> None:
    aid = await _analysis_id(session)
    plans = PlanRepo(session)
    budgets = (40.0, 0.1, 37.3)
    for budget in budgets:
        await plans.put(aid, budget, analysis_result.plan)
    await session.commit()
    for budget in budgets:
        body = await plans.get(aid, budget)
        assert body is not None, budget
        assert from_jsonable(Plan, body) == analysis_result.plan
    assert await plans.get(aid, 41.0) is None
    assert await plans.get(uuid.uuid4(), 40.0) is None

    await plans.put(aid, 0.1, {"steps": []})
    await session.commit()
    assert await plans.get(aid, 0.1) == {"steps": []}
    assert int((await session.execute(select(func.count()).select_from(StoredPlan))).scalar_one()) == 3
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            await plans.put(aid, bad, {"steps": []})
        with pytest.raises(ValueError):
            await plans.get(aid, bad)
