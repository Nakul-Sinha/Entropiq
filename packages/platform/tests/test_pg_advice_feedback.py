"""Advice and feedback against Postgres (CI only)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from tdp_platform.db.models import Feedback, FileResult, StoredAdvice
from tdp_platform.db.repos import (
    MAX_ERROR_CHARS,
    AdviceRepo,
    AnalysisRepo,
    FeedbackRepo,
    FileResultRepo,
    ModelVersionRepo,
    RepositoryRepo,
)
from tdp_platform.db.rows import FileResultRow

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MakeRow = Callable[..., FileResultRow]
ADVICE = {
    "summary": "Split the long function.",
    "steps": [
        {"order": 1, "action": "Extract helpers", "targets": ["Thing.run"], "effort_hours": [1.0, 2.0]}
    ],
    "tests_to_add": ["test_run_edge_cases"],
    "risks": [],
    "confidence": "medium",
    "dropped_steps": 0,
}


async def _file_id(session: AsyncSession, make_row: MakeRow) -> int:
    repository = await RepositoryRepo(session).get_or_create(
        url="https://github.com/acme/widgets", host="github.com", owner="acme", name="widgets"
    )
    await ModelVersionRepo(session).ensure("dev-test")
    analysis, _ = await AnalysisRepo(session).create(
        repository_id=repository.id, commit_sha="0" * 40, model_version="dev-test"
    )
    files = FileResultRepo(session)
    await files.bulk_insert(analysis.id, [make_row("pkg/a.py"), make_row("pkg/b.py")])
    file_id = (await files.list_all(analysis.id))[0].id
    await session.commit()
    return file_id


async def _get(session: AsyncSession, key: str) -> StoredAdvice:
    row = await AdviceRepo(session).get(key)
    assert row is not None
    return row


async def test_advice_success_path(session: AsyncSession, make_row: MakeRow) -> None:
    file_id = await _file_id(session, make_row)
    advice = AdviceRepo(session)
    row, created = await advice.create_queued("k1", file_result_id=file_id, model="claude-sonnet-5-5")
    assert created and row.status == "queued" and row.body is None and row.error is None
    assert row.file_result_id == file_id and row.model == "claude-sonnet-5-5"
    created_at = row.created_at
    await session.commit()

    again, created_again = await advice.create_queued("k1", file_result_id=file_id, model="other")
    assert not created_again and again.model == "claude-sonnet-5-5"

    assert await advice.set_ready("k1", ADVICE)
    await session.commit()
    ready = await _get(session, "k1")
    assert (ready.status, ready.body, ready.error) == ("ready", ADVICE, None)
    assert ready.created_at == created_at
    assert not await advice.set_failed("k1", "late failure")
    assert not await advice.requeue("k1")
    assert not await advice.set_ready("k1", ADVICE)
    assert await advice.get("missing") is None


async def test_advice_failure_path(session: AsyncSession, make_row: MakeRow) -> None:
    file_id = await _file_id(session, make_row)
    advice = AdviceRepo(session)
    await advice.create_queued("k2", file_result_id=file_id, model="m")
    assert await advice.set_failed("k2", "y" * 3000)
    await session.commit()
    failed = await _get(session, "k2")
    assert failed.status == "failed" and failed.error is not None and len(failed.error) == MAX_ERROR_CHARS
    is_null = text("SELECT body IS NULL FROM advice WHERE cache_key = :key")
    assert (await session.execute(is_null, {"key": "k2"})).scalar_one() is True
    assert not await advice.set_ready("k2", ADVICE)

    assert await advice.requeue("k2")
    await session.commit()
    queued = await _get(session, "k2")
    assert (queued.status, queued.body, queued.error) == ("queued", None, None)
    assert await advice.set_ready("k2", ADVICE)
    await session.commit()
    assert (await _get(session, "k2")).status == "ready"


async def test_advice_validation(session: AsyncSession, make_row: MakeRow) -> None:
    file_id = await _file_id(session, make_row)
    advice = AdviceRepo(session)
    with pytest.raises(ValueError):
        await advice.create_queued("", file_result_id=file_id, model="m")
    with pytest.raises(ValueError):
        await advice.create_queued("k", file_result_id=file_id, model="")
    with pytest.raises(ValueError):
        await advice.create_queued("k", file_result_id=0, model="m")
    await advice.create_queued("k", file_result_id=file_id, model="m")
    with pytest.raises(ValueError):
        await advice.set_ready("k", {"confidence": float("nan")})
    await session.rollback()
    with pytest.raises(IntegrityError):
        await advice.create_queued("k3", file_result_id=10**9, model="m")
    await session.rollback()


async def test_feedback(session: AsyncSession, make_row: MakeRow) -> None:
    file_id = await _file_id(session, make_row)
    feedback = FeedbackRepo(session)
    first = await feedback.create(file_result_id=file_id, verdict="agree", comment="Spot on", client_id="c1")
    second = await feedback.create(file_result_id=file_id, verdict="not_a_problem")
    assert isinstance(first, int) and second > first
    await session.commit()
    stored = (await session.execute(select(Feedback).where(Feedback.id == first))).scalar_one()
    assert (stored.verdict, stored.comment, stored.client_id) == ("agree", "Spot on", "c1")
    assert stored.created_at.tzinfo is not None

    with pytest.raises(ValueError):
        await feedback.create(file_result_id=file_id, verdict="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await feedback.create(file_result_id=file_id, verdict="agree", comment="nul\x00")
    with pytest.raises(IntegrityError):
        await feedback.create(file_result_id=10**9, verdict="agree")
    await session.rollback()


async def test_deleting_a_file_removes_its_advice_and_feedback(
    session: AsyncSession, make_row: MakeRow
) -> None:
    file_id = await _file_id(session, make_row)
    await AdviceRepo(session).create_queued("k", file_result_id=file_id, model="m")
    await FeedbackRepo(session).create(file_result_id=file_id, verdict="should_be_higher")
    await session.commit()
    await session.execute(delete(FileResult).where(FileResult.id == file_id))
    await session.commit()
    for model in (StoredAdvice, Feedback):
        count = (await session.execute(select(func.count()).select_from(model))).scalar_one()
        assert count == 0, model
    assert await AdviceRepo(session).get("k") is None
