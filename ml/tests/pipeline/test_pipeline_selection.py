"""Fix selection, the snapshot budget and the issues stage of the dataset pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from tdp_core.labeling.github_issues import issues_table
from tdp_ml.pipeline import stages
from tdp_ml.pipeline.layout import DataLayout, PipelineError, write_table
from tdp_ml.pipeline.manifest import ManifestRepo
from tdp_ml.pipeline.stages import StageOptions, eligible_fixes, keep_newest, newest_fixes


def _commits(rows: list[tuple[str, int, bool]]) -> pa.Table:
    return pa.table(
        {
            "sha": pa.array([r[0] for r in rows], pa.string()),
            "author_time": pa.array([r[1] for r in rows], pa.int64()),
            "is_bulk": pa.array([r[2] for r in rows], pa.bool_()),
        }
    )


def _bugfix(rows: list[tuple[str, bool | None]]) -> pa.Table:
    return pa.table(
        {
            "sha": pa.array([r[0] for r in rows], pa.string()),
            "is_bugfix": pa.array([r[1] for r in rows], pa.bool_()),
        }
    )


def test_keep_newest_fits_the_budget() -> None:
    needs = [{"a", "b"}, {"b", "c"}, {"d", "e"}]
    assert keep_newest(needs, None) == 3
    assert keep_newest(needs, 5) == 3
    assert keep_newest(needs, 4) == 2
    assert keep_newest(needs, 3) == 2
    assert keep_newest(needs, 2) == 1
    assert keep_newest(needs, 1) == 1  # the newest snapshot is kept even above the budget
    assert keep_newest([], 10) == 0


def test_eligible_fixes_follow_the_szz_rule() -> None:
    commits = _commits([("a", 100, False), ("b", 200, True), ("c", 300, False), ("d", 400, False)])
    bugfix = _bugfix([("a", True), ("b", True), ("c", None), ("d", True), ("d", False), ("x", True)])
    assert eligible_fixes(bugfix, commits) == {"a": 100, "d": 400}


def test_newest_fixes_order_and_limit() -> None:
    eligible = {"a": 100, "b": 300, "c": 300, "d": 200, "e": 500}
    within = {"a", "b", "c", "d"}
    assert newest_fixes(eligible, within, None) == ["b", "c", "d", "a"]
    assert newest_fixes(eligible, within, 2) == ["b", "c"]
    assert newest_fixes(eligible, set(), 5) == []


def _repo(issues: bool = True) -> ManifestRepo:
    return ManifestRepo(
        id=1,
        name="Owner/Name",
        slug="owner__name",
        url="https://github.com/Owner/Name.git",
        sha="0" * 40,
        default_branch="main",
        issues=issues,
    )


def test_issues_stage_fetches_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = DataLayout(tmp_path)
    calls: list[tuple[str, str]] = []
    items = [
        {
            "number": 1,
            "created_at": "2022-01-01T00:00:00+00:00",
            "closed_at": None,
            "labels": [{"name": "bug"}],
        },
        {"number": 2, "created_at": "2022-01-02T00:00:00+00:00", "closed_at": None, "labels": []},
    ]

    def fake_fetch(owner: str, name: str, token: str | None, *, cache_path: Path, **_: Any) -> pa.Table:
        calls.append((owner, name))
        table = issues_table(items)
        write_table(cache_path, table)
        return table

    monkeypatch.setattr(stages, "github_token", lambda: "secret")
    monkeypatch.setattr(stages, "fetch_issues", fake_fetch)
    counts = stages.stage_issues(layout, None, _repo(), StageOptions())  # type: ignore[arg-type]
    assert counts == {"issues": 2, "bug_issues": 1}
    assert calls == [("Owner", "Name")]
    assert layout.issues("owner__name").is_file()
    assert stages.stage_issues(layout, None, _repo(), StageOptions()) is None  # type: ignore[arg-type]
    assert stages.stage_issues(layout, None, _repo(False), StageOptions(force=True)) is None  # type: ignore[arg-type]
    assert len(calls) == 1


def test_issues_stage_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stages, "github_token", lambda: None)
    with pytest.raises(PipelineError, match="no GitHub token"):
        stages.stage_issues(DataLayout(tmp_path), None, _repo(), StageOptions())  # type: ignore[arg-type]
