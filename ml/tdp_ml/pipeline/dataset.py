"""Inclusion criteria, ``data/dataset.parquet`` assembly and the pipeline statistics.

Criteria are evaluated from exact counts when the dataset is built: a repository that fails (or whose
outputs are incomplete) is reported with the reasons and left out, never deleted. Each passing repository
contributes one query per kept snapshot, ``qid = repo id * 100 + snapshot index``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from tdp_core.contracts import FEATURE_INPUT_COLUMNS, INDEX_COLUMNS, LABEL_COLUMNS
from tdp_core.labeling.github_issues import read_issues
from tdp_ml.pipeline.config import Criteria
from tdp_ml.pipeline.history import read_table
from tdp_ml.pipeline.layout import (
    DataLayout,
    PipelineError,
    dir_bytes,
    read_json,
    write_frame,
    write_json,
)
from tdp_ml.pipeline.manifest import Manifest, ManifestRepo, manifest_sha256
from tdp_ml.pipeline.stages import REPO_STAGES, read_plan, read_snapshots

logger = logging.getLogger(__name__)

DATASET_COLUMNS: tuple[str, ...] = INDEX_COLUMNS + FEATURE_INPUT_COLUMNS + LABEL_COLUMNS + ("qid",)
QID_SNAPSHOTS: Final = 100
GRADES: Final = (0, 1, 2, 3, 4)
_DAY: Final = 86_400


def qid_for(repo_id: int, snapshot_index: int) -> int:
    """The stable query id of a snapshot: ``repo_id * 100 + snapshot_index``."""
    if not 0 <= snapshot_index < QID_SNAPSHOTS:
        raise ValueError(f"snapshot index must be in 0..{QID_SNAPSHOTS - 1}, got {snapshot_index}")
    return int(repo_id) * QID_SNAPSHOTS + int(snapshot_index)


@dataclass(frozen=True, slots=True)
class CriteriaResult:
    slug: str
    passed: bool
    values: dict[str, int | None]
    failures: tuple[str, ...]


def repo_values(layout: DataLayout, manifest: Manifest, repo: ManifestRepo) -> dict[str, int | None]:
    """The measured value of every criterion (``bug_issues`` None when the repository is keyword-only).

    Raises :class:`PipelineError` when an input is missing.
    """
    commits = read_table(layout.commits(repo.slug), "mine")
    times = commits.column("author_time").cast("int64")
    first = pc.min(times).as_py()
    last = pc.max(times).as_py()
    created = int(manifest.created_at.timestamp())
    history_days = 0 if first is None else (last - first) // _DAY
    since_last = None if last is None else (created - last) // _DAY
    bug_issues: int | None = None
    if repo.issues:
        path = layout.issues(repo.slug)
        if not path.is_file():
            raise PipelineError(f"{path} is missing: run tdp-ml issues")
        issues = read_issues(path)
        bug_issues = int(pc.sum(pc.fill_null(issues.column("is_bug"), False).cast("int64")).as_py() or 0)
    snapshots = read_snapshots(layout, repo.slug)
    source_files = 0
    complete = 0
    for snap in snapshots:
        meta = layout.features_meta(repo.slug, snap.index)
        built = meta.is_file() and layout.features(repo.slug, snap.index).is_file()
        if built and layout.labels(repo.slug, snap.index).is_file():
            complete += 1
    if snapshots:
        newest = layout.features_meta(repo.slug, snapshots[-1].index)
        if newest.is_file():
            source_files = int(read_json(newest)["rows"])
    return {
        "history_days": int(history_days),
        "commits": commits.num_rows,
        "source_files": source_files,
        "bug_issues": bug_issues,
        "days_since_last_commit": None if since_last is None else int(since_last),
        "snapshots": complete,
    }


def check_criteria(values: Mapping[str, int | None], criteria: Criteria) -> tuple[str, ...]:
    """The failed criteria as messages (inclusive thresholds; a None value is waived)."""
    minimums = (
        ("history_days", criteria.min_history_days),
        ("commits", criteria.min_commits),
        ("source_files", criteria.min_source_files),
        ("bug_issues", criteria.min_bug_issues),
        ("snapshots", criteria.min_snapshots),
    )
    failures: list[str] = []
    for key, minimum in minimums:
        value = values.get(key)
        if value is not None and value < minimum:
            failures.append(f"{key} {value} < {minimum}")
    since = values.get("days_since_last_commit")
    if since is not None and since > criteria.max_days_since_last_commit:
        failures.append(f"days_since_last_commit {since} > {criteria.max_days_since_last_commit}")
    return tuple(failures)


def evaluate(layout: DataLayout, manifest: Manifest, repo: ManifestRepo) -> CriteriaResult:
    try:
        values = repo_values(layout, manifest, repo)
    except (PipelineError, OSError, ValueError, KeyError) as exc:
        return CriteriaResult(repo.slug, False, {}, (f"incomplete: {exc}",))
    failures = check_criteria(values, manifest.criteria)
    return CriteriaResult(repo.slug, not failures, values, failures)


def _grades(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["grade"].value_counts() if len(frame) else pd.Series(dtype="int64")
    return {str(g): int(counts.get(g, 0)) for g in GRADES}


def _query(layout: DataLayout, repo: ManifestRepo, index: int) -> pd.DataFrame:
    features = pd.read_parquet(layout.features(repo.slug, index))
    labels = pd.read_parquet(layout.labels(repo.slug, index))
    if features["path"].tolist() != labels["path"].tolist():
        raise PipelineError(f"features and labels of {repo.slug} snapshot {index} have different paths")
    merged = features.merge(labels, on=list(INDEX_COLUMNS), how="inner", validate="one_to_one")
    merged["qid"] = np.int64(qid_for(repo.id, index))
    return merged


def build_dataset(layout: DataLayout, manifest: Manifest) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The dataset frame (columns :data:`DATASET_COLUMNS`, sorted by qid then path) and its summary."""
    frames: list[pd.DataFrame] = []
    per_repo: list[dict[str, Any]] = []
    passing = 0
    for repo in manifest.repos:
        result = evaluate(layout, manifest, repo)
        entry: dict[str, Any] = {
            "slug": repo.slug,
            "id": repo.id,
            "sha": repo.sha,
            "passed": result.passed,
            "failures": list(result.failures),
            "values": result.values,
            "queries": 0,
            "empty_queries": 0,
            "rows": 0,
            "grades": {str(g): 0 for g in GRADES},
            "latent_rate": None,
        }
        use = result.passed and (manifest.max_repos is None or passing < manifest.max_repos)
        if result.passed and not use:
            entry["failures"] = ["over max_repos"]
        if use:
            passing += 1
            repo_frames = []
            for snap in read_snapshots(layout, repo.slug):
                query = _query(layout, repo, snap.index)
                if query.empty:
                    entry["empty_queries"] += 1
                    continue
                repo_frames.append(query)
            if repo_frames:
                repo_frame = pd.concat(repo_frames, ignore_index=True)
                frames.append(repo_frame)
                entry["queries"] = len(repo_frames)
                entry["rows"] = len(repo_frame)
                entry["grades"] = _grades(repo_frame)
                entry["latent_rate"] = round(float(repo_frame["has_latent_defect"].mean()), 4)
        per_repo.append(entry)
    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        raise PipelineError("no repository passed the criteria; the dataset would be empty")
    frame = frame.sort_values(["qid", "path"], kind="stable").reset_index(drop=True)
    frame = frame[list(DATASET_COLUMNS)]
    frame["qid"] = frame["qid"].astype("int64")
    validate_dataset(frame)
    summary = {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": manifest_sha256(layout),
        "rows": len(frame),
        "queries": int(frame["qid"].nunique()),
        "repos": int(frame["repo"].nunique()),
        "grades": _grades(frame),
        "per_repo": per_repo,
    }
    return frame, summary


def write_dataset(layout: DataLayout, frame: pd.DataFrame, summary: Mapping[str, Any]) -> None:
    write_frame(layout.dataset, frame)
    write_json(layout.dataset_summary, dict(summary))


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise :class:`PipelineError` unless ``frame`` is a well-formed dataset (see the module docstring)."""
    if tuple(frame.columns) != DATASET_COLUMNS:
        raise PipelineError("dataset columns differ from DATASET_COLUMNS")
    features = frame[list(FEATURE_INPUT_COLUMNS)]
    if not all(dtype == np.float32 for dtype in features.dtypes):
        raise PipelineError("every feature column must be float32")
    if not np.isfinite(features.to_numpy()).all():
        raise PipelineError("the dataset holds non-finite feature values")
    if frame[list(LABEL_COLUMNS)].isna().any().any():
        raise PipelineError("the dataset holds missing labels")
    if frame["qid"].dtype != np.int64:
        raise PipelineError("qid must be int64")
    if frame.duplicated(["qid", "path"]).any():
        raise PipelineError("the dataset holds a duplicate (qid, path)")
    per_qid = frame.groupby("qid")[["repo", "snapshot_idx"]].nunique()
    if (per_qid > 1).any().any():
        raise PipelineError("a qid spans more than one repository or snapshot")


def collect_stats(layout: DataLayout, manifest: Manifest) -> dict[str, Any]:
    """Counts, grade distributions, criteria, runtimes and disk usage of the pipeline outputs."""
    summary: dict[str, Any] = {}
    if layout.dataset_summary.is_file():
        summary = read_json(layout.dataset_summary)
    by_slug = {entry["slug"]: entry for entry in summary.get("per_repo", [])}
    repos: list[dict[str, Any]] = []
    stage_totals = dict.fromkeys(REPO_STAGES, 0.0)
    stage_max = dict.fromkeys(REPO_STAGES, 0.0)
    for repo in manifest.repos:
        runs: dict[str, Any] = {}
        for stage in REPO_STAGES:
            record_path = layout.run_record(stage, repo.slug)
            if record_path.is_file():
                runs[stage] = read_json(record_path)
        seconds = {stage: float(r["seconds"]) for stage, r in runs.items()}
        for stage, value in seconds.items():
            stage_totals[stage] += value
            stage_max[stage] = max(stage_max[stage], value)
        plan: dict[str, Any] = {}
        with contextlib.suppress(PipelineError):
            plan = read_plan(layout, repo.slug)
        entry = by_slug.get(repo.slug, {})
        repos.append(
            {
                "slug": repo.slug,
                "id": repo.id,
                "issues": repo.issues,
                "commits": runs.get("mine", {}).get("counts", {}).get("commits"),
                "bugfix": runs.get("bugfix", {}).get("counts", {}),
                "schedule": plan.get("schedule"),
                "kept": plan.get("kept"),
                "dropped_for_budget": plan.get("dropped_for_budget"),
                "szz_label_fixes": len(plan.get("label_fixes", [])),
                "szz_feature_fixes": len(plan.get("feature_fixes", [])),
                "passed": entry.get("passed"),
                "failures": entry.get("failures"),
                "values": entry.get("values"),
                "queries": entry.get("queries"),
                "rows": entry.get("rows"),
                "grades": entry.get("grades"),
                "latent_rate": entry.get("latent_rate"),
                "seconds": seconds,
            }
        )
    folders = ("mirrors", "mined", "issues", "bugfix", "snapshots", "szz", "features", "labels", "cache")
    return {
        "dataset": {k: summary.get(k) for k in ("rows", "queries", "repos", "grades", "manifest_sha256")},
        "passing_repos": sum(1 for r in repos if r["passed"]),
        "repos": repos,
        "stage_seconds_total": {k: round(v, 1) for k, v in stage_totals.items()},
        "stage_seconds_max": {k: round(v, 1) for k, v in stage_max.items()},
        "disk_bytes": {name: dir_bytes(layout.root / name) for name in folders},
    }


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """``rows`` as a fixed-width text table with a header line."""
    cells = [[str(c) for c in columns]] + [
        ["" if r.get(c) is None else str(r.get(c)) for c in columns] for r in rows
    ]
    widths = [max(len(line[i]) for line in cells) for i in range(len(columns))]
    return "\n".join(
        "  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True)).rstrip() for line in cells
    )
