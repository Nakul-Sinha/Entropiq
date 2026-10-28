"""Stored shapes (summary, treemap, payload), score rebuilding, the async bridge and the store accessors."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from tdp_core.contracts import DEBT_TYPES, AnalysisResult, GraphData, Plan
from tdp_core.model.bundle import Bundle
from tdp_core.serialize import to_jsonable
from tdp_platform.db.models import FileResult
from tdp_platform.db.repos import DONE_STAGE, PostgresStaticCache
from tdp_platform.db.rows import rows_from_result
from tdp_worker import persist, progress
from tdp_worker.persist import (
    PostgresStore,
    build_payload,
    build_summary,
    build_treemap,
    get_store,
    run_coro,
    score_from_row,
    set_store,
    shap_by_feature,
)
from tdp_worker.testing import MemoryStore

MakeResult = Callable[[Mapping[str, int]], AnalysisResult]


def test_done_stage_matches_the_platform() -> None:
    assert progress.DONE_STAGE == DONE_STAGE


def test_build_summary(sample_result: AnalysisResult) -> None:
    summary = build_summary(sample_result)
    assert summary == {
        "files": 5,
        "critical": 2,
        "high": 0,
        "medium": 0,
        "low": 3,
        "critical_hours": 7.5,
        "debt_types": {
            "complexity": 1,
            "duplication": 0,
            "test": 1,
            "architecture": 0,
            "knowledge": 1,
            "admitted": 1,
            "defect_hotspot": 0,
        },
        "priority_histogram": [2, 1, 0, 0, 0, 0, 0, 0, 0, 2],
    }
    assert list(summary) == [
        "files",
        "critical",
        "high",
        "medium",
        "low",
        "critical_hours",
        "debt_types",
        "priority_histogram",
    ]
    assert list(summary["debt_types"]) == list(DEBT_TYPES)


def test_build_summary_bands_and_buckets(make_result: MakeResult) -> None:
    summary = build_summary(make_result({"a.py": 89, "b.py": 75, "c.py": 74, "d.py": 50, "e.py": 49}))
    assert (summary["critical"], summary["high"], summary["medium"], summary["low"]) == (0, 2, 2, 1)
    assert summary["priority_histogram"] == [0, 0, 0, 0, 1, 1, 0, 2, 1, 0]
    assert summary["critical_hours"] == 0.0


def test_empty_result(make_result: MakeResult) -> None:
    empty = make_result({})
    summary = build_summary(empty)
    assert summary["files"] == 0
    assert summary["priority_histogram"] == [0] * 10
    assert set(summary["debt_types"].values()) == {0}
    assert build_treemap(empty, root_name="x") == {"name": "x", "path": "", "children": []}


def test_build_treemap(sample_result: AnalysisResult) -> None:
    assert build_treemap(sample_result, root_name="demo") == {
        "name": "demo",
        "path": "",
        "children": [
            {
                "name": "pkg",
                "path": "pkg",
                "children": [
                    {
                        "name": "sub",
                        "path": "pkg/sub",
                        "children": [
                            {
                                "name": "deep.py",
                                "path": "pkg/sub/deep.py",
                                "sloc": 20,
                                "priority": 95,
                                "band": "Critical",
                            }
                        ],
                    },
                    {"name": "core.py", "path": "pkg/core.py", "sloc": 10, "priority": 100, "band": "Critical"},
                    {"name": "util.py", "path": "pkg/util.py", "sloc": 40, "priority": 9, "band": "Low"},
                    {"name": "zero.py", "path": "pkg/zero.py", "sloc": 0, "priority": 0, "band": "Low"},
                ],
            },
            {"name": "app.py", "path": "app.py", "sloc": 30, "priority": 10, "band": "Low"},
        ],
    }


def test_build_payload(sample_result: AnalysisResult, dev_bundle: Bundle) -> None:
    payload = build_payload(sample_result, dev_bundle, budget_hours=12.5, root_name="demo")
    assert [r.path for r in payload.rows] == [f.path for f in sample_result.files]
    assert all(len(r.shap["by_feature"]) == 46 for r in payload.rows)
    assert payload.graph == to_jsonable(sample_result.graph)
    assert payload.plan == to_jsonable(sample_result.plan)
    assert payload.treemap == build_treemap(sample_result, root_name="demo")
    assert payload.summary == build_summary(sample_result)
    assert payload.budget_hours == 12.5
    assert payload.stats == sample_result.stats
    assert payload.stats is not sample_result.stats


def test_shap_by_feature_empty_frame(sample_result: AnalysisResult, dev_bundle: Bundle) -> None:
    assert shap_by_feature(dev_bundle, sample_result.features.iloc[0:0]) == {}
    empty = replace(
        sample_result,
        files=(),
        features=sample_result.features.iloc[0:0],
        extras={},
        graph=GraphData((), ()),
        plan=Plan(40.0, (), 0.0, 0.0),
    )
    assert build_payload(empty, dev_bundle, budget_hours=40.0).rows == ()


def test_score_from_row_round_trips(sample_result: AnalysisResult) -> None:
    rows = rows_from_result(sample_result)
    for score, row in zip(sample_result.files, rows, strict=True):
        assert score_from_row(row) == score
        orm = FileResult(
            path=row.path,
            blob_sha=row.blob_sha,
            priority=row.priority,
            band=row.band,
            risk=row.risk,
            risk_low=row.risk_low,
            risk_high=row.risk_high,
            score=row.score,
            cost_low=row.cost_low,
            cost_high=row.cost_high,
            debt_types=list(row.debt_types),
            top_reason=row.top_reason,
            features=dict(row.features),
            shap=to_jsonable(dict(row.shap)),
            reasons=to_jsonable(dict(row.reasons)),
            extra=dict(row.extra),
        )
        assert score_from_row(orm) == score


def test_run_coro_outside_and_inside_a_loop() -> None:
    async def answer(value: int) -> int:
        await asyncio.sleep(0)
        return value

    assert run_coro(answer(1)) == 1

    async def caller() -> int:
        return run_coro(answer(2))

    assert asyncio.run(caller()) == 2


def test_get_store_builds_a_postgres_store_lazily(worker_env: dict[str, Path]) -> None:
    set_store(None)
    built = get_store()
    assert isinstance(built, PostgresStore)
    assert get_store() is built
    cache = built.static_cache()
    assert isinstance(cache, PostgresStaticCache)
    assert built.static_cache() is cache
    built.close()
    memory = MemoryStore()
    set_store(memory)
    assert get_store() is memory
    assert persist.get_store() is memory


def test_postgres_store_from_settings(worker_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from tdp_platform.settings import get_settings

    monkeypatch.setenv("DATABASE_SCHEMA", "tdp_worker_unit")
    get_settings.cache_clear()
    built = PostgresStore.from_settings(get_settings())
    assert built._schema == "tdp_worker_unit"
    built.close()


def test_memory_store_rules(sample_result: AnalysisResult, dev_bundle: Bundle) -> None:
    memory = MemoryStore()
    aid = memory.add_analysis(commit_sha="a" * 40)
    payload = build_payload(sample_result, dev_bundle, budget_hours=40.0)
    assert not memory.save_results(aid, payload)  # only from running
    assert memory.set_progress(aid, "mine", 25)
    assert memory.set_running(aid)
    assert memory.save_results(aid, payload)
    assert not memory.set_failed(aid, "late")
    assert not memory.set_running(aid)
    assert memory.load_analysis(uuid.uuid4()) is None
    assert memory.load_file(10_000) is None
    fid = next(iter(memory.file_rows))
    info = memory.load_file(fid)
    assert info is not None and info.score == sample_result.files[0]
