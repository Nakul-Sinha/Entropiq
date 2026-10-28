"""Configs, folder layout, history mining and dataset helpers of the dataset pipeline."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from tdp_core.config import Settings
from tdp_core.contracts import FEATURE_INPUT_COLUMNS, LABEL_COLUMNS
from tdp_core.mining.gitlog import mine_history
from tdp_core.testing.repo_builder import RepoBuilder
from tdp_ml.pipeline.config import ConfigError, Criteria, load_config
from tdp_ml.pipeline.dataset import DATASET_COLUMNS, check_criteria, qid_for, validate_dataset
from tdp_ml.pipeline.history import head_sha, is_ancestor, mine_at
from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json, write_json, write_table
from tdp_ml.pipeline.leakage import compare_rows

KEYWORD_ONLY = {"python-poetry/poetry", "celery/celery", "pytest-dev/pytest", "pylint-dev/pylint", "pypa/pip"}
LABELS = "settings: {history_window_days: null}\n"


# ---------- configs ----------


def test_shipped_configs_load() -> None:
    config = load_config()
    assert len(config.repos) == 23
    assert len({r.id for r in config.repos}) == 23
    assert len({r.slug for r in config.repos}) == 23
    assert {r.name for r in config.repos if not r.issues} == KEYWORD_ONLY
    assert config.max_snapshots == 8
    assert config.szz_max_fixes == 1500
    assert config.max_repos is None
    assert config.criteria.min_source_files == 20
    assert config.settings.history_window_days is None


def _write(tmp_path: Path, repos: str, labels: str = LABELS) -> tuple[Path, Path]:
    (tmp_path / "repos.yaml").write_text(repos, encoding="utf-8")
    (tmp_path / "labels.yaml").write_text(labels, encoding="utf-8")
    return tmp_path / "repos.yaml", tmp_path / "labels.yaml"


@pytest.mark.parametrize(
    ("repos", "labels", "message"),
    [
        ("repos: [{id: 1, name: a/b}]\nextra: 1\n", LABELS, "unknown key"),
        ("repos: [{id: 1, name: a/b}, {id: 1, name: c/d}]\n", LABELS, "used twice"),
        ("repos: [{id: 1, name: a/b, url: 'http://github.com/a/b.git'}]\n", LABELS, "url"),
        ("repos: [{id: 1, name: a/b}]\n", "settings: {horizon_days: soon}\n", "horizon_days"),
    ],
)
def test_bad_configs_are_rejected(tmp_path: Path, repos: str, labels: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(*_write(tmp_path, repos, labels))


# ---------- layout ----------


def test_layout_paths_and_writers(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    assert layout.lineage("a__b", 3) == tmp_path / "features" / "a__b" / "3.lineage.parquet"
    for bad in ("", "a/b", "..", "a\\b"):
        with pytest.raises(ValueError):
            layout.features(bad, 0)
    write_json(layout.manifest, {"x": [1, 2]})
    write_table(layout.commits("a__b"), pa.table({"x": [1]}))
    assert read_json(layout.manifest) == {"x": [1, 2]}
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(PipelineError, match="does not exist"):
        read_json(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(PipelineError, match="cannot be read"):
        read_json(tmp_path / "bad.json")


# ---------- history ----------


def test_mine_at_equals_mine_history_without_mailmap(tmp_path: Path) -> None:
    repo = RepoBuilder(tmp_path / "r")
    t0 = datetime(2023, 1, 1, tzinfo=UTC)
    first = repo.commit({"a.py": "x = 1\n"}, "Add a", t0, ("Ann", "ann@example.com"))
    repo.commit(
        {"a.py": "x = 2\n", "b.py": "y = 1\n"}, "Edit a", t0 + timedelta(days=1), ("Ben", "ben@ex.com")
    )
    sha = repo.head()
    ours, theirs = mine_at(repo.path, sha), mine_history(repo.path, sha)
    assert ours.commits.equals(theirs.commits)
    assert ours.file_changes.equals(theirs.file_changes)
    assert head_sha(repo.path) == sha
    assert is_ancestor(repo.path, first, sha)
    assert not is_ancestor(repo.path, sha, first)


def test_mine_at_uses_the_mailmap_of_the_mined_commit(source_repo: Any, tmp_path: Path) -> None:
    bare = tmp_path / "bare.git"
    subprocess.run(
        ["git", "clone", "--bare", "--quiet", str(source_repo.path), str(bare)],
        check=True,
        capture_output=True,
    )
    before = mine_at(bare, source_repo.shas["d110"], Settings())
    assert "bob@old.example" in set(before.commits.column("author_email").to_pylist())
    after = mine_history(bare, source_repo.head)
    emails = set(after.commits.column("author_email").to_pylist())
    assert "bob@old.example" not in emails and "bob@new.example" in emails


# ---------- dataset helpers ----------


def test_qid_for() -> None:
    assert qid_for(30, 7) == 3007
    assert qid_for(1, 0) == 100
    with pytest.raises(ValueError):
        qid_for(1, 100)


def test_check_criteria_is_inclusive_and_waives_none() -> None:
    criteria = Criteria(
        min_history_days=10,
        min_commits=10,
        min_source_files=20,
        min_bug_issues=30,
        max_days_since_last_commit=365,
        min_snapshots=1,
    )
    values: dict[str, int | None] = {
        "history_days": 10,
        "commits": 10,
        "source_files": 20,
        "bug_issues": None,
        "days_since_last_commit": 365,
        "snapshots": 1,
    }
    assert check_criteria(values, criteria) == ()
    values |= {"source_files": 19, "bug_issues": 0, "days_since_last_commit": 366}
    assert check_criteria(values, criteria) == (
        "source_files 19 < 20",
        "bug_issues 0 < 30",
        "days_since_last_commit 366 > 365",
    )


def _features(paths: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "repo": "a__b",
            "snapshot_idx": np.int16(1),
            "snapshot_time": pd.Timestamp("2024-01-01", tz="UTC"),
            "path": paths,
        }
    )
    for column in FEATURE_INPUT_COLUMNS:
        frame[column] = np.float32(1.0)
    return frame


def test_compare_rows() -> None:
    stored = _features(["a.py", "b.py"])
    assert compare_rows(stored, stored.iloc[::-1].copy()) == []
    changed = stored.copy()
    changed.loc[1, "sloc"] = np.float32(2.0)
    assert compare_rows(stored, changed) == ["b.py sloc: stored 1.0 rebuilt 2.0"]
    assert compare_rows(stored, _features(["a.py", "c.py"]))[0].startswith("paths differ")


def test_validate_dataset_rejects_a_duplicate_row() -> None:
    frame = _features(["a.py", "b.py"])
    for column in LABEL_COLUMNS:
        frame[column] = 0
    frame["qid"] = np.int64(101)
    frame = frame[list(DATASET_COLUMNS)]
    validate_dataset(frame)
    duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(PipelineError, match="duplicate"):
        validate_dataset(duplicate)
