"""Repositories, model versions and the analysis state machine against Postgres (CI only)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from tdp_platform.db.models import (
    Analysis,
    AnalysisArtifact,
    Feedback,
    FileResult,
    ModelVersion,
    Repository,
    StoredAdvice,
    StoredPlan,
)
from tdp_platform.db.repos import (
    DONE_STAGE,
    MAX_ERROR_CHARS,
    AdviceRepo,
    AnalysisRepo,
    ArtifactRepo,
    FeedbackRepo,
    FileResultRepo,
    ModelVersionRepo,
    PlanRepo,
    RepositoryRepo,
)
from tdp_platform.db.rows import FileResultRow

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MakeRow = Callable[..., FileResultRow]
URL = "https://github.com/acme/widgets"
SHA = "0123456789abcdef0123456789abcdef01234567"
VERSION = "dev-test"


async def _repository(session: AsyncSession) -> Repository:
    return await RepositoryRepo(session).get_or_create(
        url=URL, host="github.com", owner="acme", name="widgets"
    )


async def _analysis(session: AsyncSession, sha: str = SHA) -> Analysis:
    repository = await _repository(session)
    await ModelVersionRepo(session).ensure(VERSION)
    analysis, _ = await AnalysisRepo(session).create(
        repository_id=repository.id, commit_sha=sha, model_version=VERSION
    )
    return analysis


async def _get(session: AsyncSession, analysis_id: uuid.UUID) -> Analysis:
    analysis = await AnalysisRepo(session).get(analysis_id)
    assert analysis is not None
    return analysis


async def _count(session: AsyncSession, model: Any) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_repository_get_or_create_is_idempotent(session: AsyncSession) -> None:
    repos = RepositoryRepo(session)
    first = await repos.get_or_create(url=URL, host="github.com", owner="acme", name="widgets")
    await session.commit()
    second = await repos.get_or_create(url=URL, host="elsewhere", owner="x", name="y")
    assert second.id == first.id
    assert (second.host, second.owner, second.name) == ("github.com", "acme", "widgets")
    assert first.created_at.tzinfo is not None
    by_id = await repos.get(first.id)
    by_url = await repos.get_by_url(URL)
    assert by_id is not None and by_url is not None and by_id.id == by_url.id == first.id
    assert await repos.get(uuid.uuid4()) is None
    assert await repos.get_by_url("https://github.com/acme/other") is None
    with pytest.raises(ValueError):
        await repos.get_or_create(url="", host="github.com", owner="acme", name="widgets")
    assert await _count(session, Repository) == 1


async def test_model_version_ensure(session: AsyncSession) -> None:
    versions = ModelVersionRepo(session)
    row = await versions.ensure("v1")
    assert row.metrics == {}
    promoted = row.promoted_at
    assert promoted.tzinfo is not None
    await session.commit()
    row = await versions.ensure("v1", {"ndcg@10": 0.61, "folds": [1, 2]})
    assert row.metrics == {"ndcg@10": 0.61, "folds": [1, 2]}
    assert row.promoted_at == promoted
    await session.commit()
    row = await versions.ensure("v1")
    assert row.metrics == {"ndcg@10": 0.61, "folds": [1, 2]}
    assert await versions.get("v2") is None
    for bad in ("", "x" * 129):
        with pytest.raises(ValueError):
            await versions.ensure(bad)
    with pytest.raises(ValueError):
        await versions.ensure("v1", {"bad": float("nan")})
    assert await _count(session, ModelVersion) == 1


async def test_create_find_and_reuse(session: AsyncSession) -> None:
    repository = await _repository(session)
    await ModelVersionRepo(session).ensure(VERSION)
    analyses = AnalysisRepo(session)
    analysis, created = await analyses.create(
        repository_id=repository.id, commit_sha=SHA, model_version=VERSION
    )
    assert created
    assert (analysis.status, analysis.progress, analysis.budget_hours) == ("queued", 0, 40.0)
    assert analysis.stage is None and analysis.error is None and analysis.summary is None
    assert analysis.started_at is None and analysis.finished_at is None
    assert analysis.created_at.tzinfo is not None
    assert analysis.commit_sha == SHA
    assert analysis.repository.url == URL
    await session.commit()

    again, created_again = await analyses.create(
        repository_id=repository.id, commit_sha=SHA, model_version=VERSION, budget_hours=10
    )
    assert again.id == analysis.id and not created_again
    assert again.budget_hours == 40.0
    found = await analyses.find(URL, SHA, VERSION)
    assert found is not None and found.id == analysis.id and found.repository.url == URL
    assert await analyses.find(URL, "f" * 40, VERSION) is None
    assert await analyses.find("https://github.com/acme/other", SHA, VERSION) is None
    assert await analyses.find(URL, SHA, "other-version") is None
    custom, _ = await analyses.create(
        repository_id=repository.id, commit_sha="1" * 40, model_version=VERSION, budget_hours=12.5
    )
    assert custom.budget_hours == 12.5
    await session.commit()

    for kwargs in (
        {"commit_sha": "abc"},
        {"commit_sha": SHA.upper()},
        {"budget_hours": 0},
        {"budget_hours": float("nan")},
        {"model_version": ""},
    ):
        args: dict[str, Any] = {"repository_id": repository.id, "commit_sha": SHA, "model_version": VERSION}
        args.update(kwargs)
        with pytest.raises(ValueError):
            await analyses.create(**args)
    with pytest.raises(ValueError):
        await analyses.find(URL, SHA.upper(), VERSION)

    with pytest.raises(IntegrityError):
        await analyses.create(repository_id=repository.id, commit_sha="2" * 40, model_version="unknown")
    await session.rollback()
    with pytest.raises(IntegrityError):
        await analyses.create(repository_id=uuid.uuid4(), commit_sha="2" * 40, model_version=VERSION)
    await session.rollback()
    assert await _count(session, Analysis) == 2


async def test_successful_lifecycle(session: AsyncSession) -> None:
    analysis = await _analysis(session)
    await session.commit()
    aid = analysis.id
    analyses = AnalysisRepo(session)

    assert await analyses.set_progress(aid, "fetch", 5)
    current = await _get(session, aid)
    assert (current.status, current.stage, current.progress) == ("queued", "fetch", 5)

    assert await analyses.set_running(aid)
    await session.commit()
    current = await _get(session, aid)
    started = current.started_at
    assert current.status == "running" and started is not None and current.progress == 5

    assert await analyses.set_running(aid)  # a redelivered task marks it running again
    await session.commit()
    current = await _get(session, aid)
    assert current.started_at == started and current.progress == 5 and current.stage == "fetch"

    assert await analyses.set_progress(aid, "parse", 150)
    assert (await _get(session, aid)).progress == 100
    assert await analyses.set_progress(aid, "parse", -5)
    assert (await _get(session, aid)).progress == 0
    with pytest.raises(ValueError):
        await analyses.set_progress(aid, "", 10)

    summary = {
        "files": 4,
        "critical": 1,
        "high": 1,
        "medium": 0,
        "low": 2,
        "critical_hours": 6.0,
        "debt_types": {"test": 1},
        "priority_histogram": [0, 0, 0, 0, 2, 0, 0, 0, 1, 1],
    }
    stats = {"n_files": 4, "skipped": {}, "timings_s": {"mine": 1.5}}
    assert await analyses.set_succeeded(aid, summary, stats)
    await session.commit()
    done = await _get(session, aid)
    assert (done.status, done.stage, done.progress, done.error) == ("succeeded", DONE_STAGE, 100, None)
    assert done.summary == summary and done.stats == stats
    assert done.started_at is not None and done.finished_at is not None
    assert done.finished_at >= done.started_at

    assert not await analyses.set_failed(aid, "too late")
    assert not await analyses.set_progress(aid, "parse", 50)
    assert not await analyses.set_running(aid)
    assert not await analyses.set_succeeded(aid, summary, stats)
    assert not await analyses.requeue(aid)
    await session.commit()
    assert (await _get(session, aid)).status == "succeeded"
    assert not await analyses.set_running(uuid.uuid4())


async def test_failure_and_requeue(session: AsyncSession, make_row: MakeRow) -> None:
    analysis = await _analysis(session)
    aid = analysis.id
    created_at = analysis.created_at
    analyses = AnalysisRepo(session)
    assert not await analyses.set_succeeded(aid, {}, {})  # only from running
    assert await analyses.set_running(aid)
    files = FileResultRepo(session)
    await files.bulk_insert(aid, [make_row("a.py"), make_row("b.py")])
    await ArtifactRepo(session).put(aid, "graph", {"nodes": [], "edges": []})
    await PlanRepo(session).put(aid, 40.0, {"steps": []})
    file_id = (await files.list_all(aid))[0].id
    await AdviceRepo(session).create_queued("key", file_result_id=file_id, model="model")
    await FeedbackRepo(session).create(file_result_id=file_id, verdict="agree")
    assert await analyses.set_failed(aid, "x" * 5000)
    await session.commit()

    failed = await _get(session, aid)
    assert failed.status == "failed" and failed.finished_at is not None
    assert failed.error is not None and len(failed.error) == MAX_ERROR_CHARS and failed.error.endswith("...")
    assert not await analyses.set_succeeded(aid, {}, {})
    assert not await analyses.set_failed(aid, "again")

    assert await analyses.requeue(aid)
    await session.commit()
    queued = await _get(session, aid)
    assert (queued.status, queued.stage, queued.progress, queued.error) == ("queued", None, 0, None)
    assert queued.summary is None and queued.stats is None
    assert queued.started_at is None and queued.finished_at is None
    assert queued.created_at == created_at
    for model in (FileResult, AnalysisArtifact, StoredPlan, StoredAdvice, Feedback):
        assert await _count(session, model) == 0, model
    assert not await analyses.requeue(aid)

    assert await analyses.set_failed(aid, "bad\x00byte")
    await session.commit()
    assert (await _get(session, aid)).error == "bad�byte"


async def test_cleared_json_columns_are_sql_null(session: AsyncSession) -> None:
    analysis = await _analysis(session)
    analyses = AnalysisRepo(session)
    assert await analyses.set_running(analysis.id)
    assert await analyses.set_succeeded(analysis.id, {"critical": 1}, {"n_files": 1})
    await session.commit()
    sql = text("SELECT summary IS NULL AND stats IS NULL FROM analyses WHERE id = :id")
    assert (await session.execute(sql, {"id": analysis.id})).scalar_one() is False
    # Force the failed state so that requeue clears a stored summary and stats.
    force_failed = (
        update(Analysis)
        .where(Analysis.id == analysis.id)
        .values(status="failed")
        .execution_options(synchronize_session=False)
    )
    await session.execute(force_failed)
    assert await analyses.requeue(analysis.id)
    await session.commit()
    assert (await session.execute(sql, {"id": analysis.id})).scalar_one() is True


async def test_list_for_repository(session: AsyncSession) -> None:
    repository = await _repository(session)
    await ModelVersionRepo(session).ensure(VERSION)
    await session.commit()
    analyses = AnalysisRepo(session)
    ids: list[uuid.UUID] = []
    for sha in ("1" * 40, "2" * 40, "3" * 40):
        analysis, _ = await analyses.create(
            repository_id=repository.id, commit_sha=sha, model_version=VERSION
        )
        await session.commit()
        ids.append(analysis.id)
    listed = await analyses.list_for_repository(repository.id)
    assert [a.id for a in listed] == ids
    assert all(a.repository.url == URL for a in listed)
    assert await analyses.set_running(ids[1])
    assert await analyses.set_succeeded(ids[1], {"critical": 1, "high": 2, "critical_hours": 3.5}, {})
    await session.commit()
    done = await analyses.list_for_repository(repository.id, status="succeeded")
    assert [a.id for a in done] == [ids[1]]
    assert done[0].summary == {"critical": 1, "high": 2, "critical_hours": 3.5}
    assert [a.id for a in await analyses.list_for_repository(repository.id, status="queued")] == [
        ids[0],
        ids[2],
    ]
    assert await analyses.list_for_repository(uuid.uuid4()) == []


async def test_cascades(session: AsyncSession, make_row: MakeRow) -> None:
    analysis = await _analysis(session)
    aid = analysis.id
    files = FileResultRepo(session)
    await files.bulk_insert(aid, [make_row("a.py"), make_row("b.py")])
    file_id = (await files.list_all(aid))[0].id
    await ArtifactRepo(session).put(aid, "treemap", {"name": "", "path": "", "children": []})
    await PlanRepo(session).put(aid, 40.0, {"steps": []})
    await AdviceRepo(session).create_queued("key", file_result_id=file_id, model="model")
    await FeedbackRepo(session).create(file_result_id=file_id, verdict="should_be_higher", comment="really")
    await session.commit()

    with pytest.raises(IntegrityError):
        await session.execute(delete(ModelVersion).where(ModelVersion.version == VERSION))
    await session.rollback()

    await session.execute(delete(Repository).where(Repository.url == URL))
    await session.commit()
    for model in (Repository, Analysis, FileResult, AnalysisArtifact, StoredPlan, StoredAdvice, Feedback):
        assert await _count(session, model) == 0, model
    assert await _count(session, ModelVersion) == 1


async def test_status_check_constraint(session: AsyncSession) -> None:
    repository = await _repository(session)
    await ModelVersionRepo(session).ensure(VERSION)
    await session.commit()
    stmt = insert(Analysis).values(
        repository_id=repository.id, commit_sha=SHA, model_version=VERSION, status="paused"
    )
    with pytest.raises(IntegrityError):
        await session.execute(stmt)
    await session.rollback()


async def test_another_session_sees_committed_changes(
    session: AsyncSession, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    analysis = await _analysis(session)
    await session.commit()
    async with session_factory() as other:
        seen = await AnalysisRepo(other).get(analysis.id)
        assert seen is not None and seen.status == "queued"
        assert await AnalysisRepo(session).set_running(analysis.id)
        await session.commit()
        seen = await AnalysisRepo(other).get(analysis.id)
        assert seen is not None and seen.status == "running"
