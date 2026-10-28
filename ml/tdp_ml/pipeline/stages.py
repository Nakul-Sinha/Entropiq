"""The per-repository stages of the dataset pipeline (``mine`` to ``labels``).

Every stage reads the repository, its pinned sha and the settings from the manifest, writes its outputs
atomically, and is skipped when all of its outputs exist (unless ``force``). The last file a stage writes is
its completion marker, so an interrupted stage simply runs again. A stage returns the counts it reports,
or None when it was skipped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from tdp_core.contracts import Snapshot
from tdp_core.features.build import build_features
from tdp_core.features.cache import DiskStaticCache
from tdp_core.features.satd_model import load_default_classifier
from tdp_core.labeling.bugfix import classify_bugfix
from tdp_core.labeling.github_issues import fetch_issues, read_issues
from tdp_core.labeling.labels import make_labels
from tdp_core.labeling.szz import szz_table
from tdp_core.mining.reach import reachable_shas, snapshot_sha
from tdp_core.mining.snapshots import make_snapshots
from tdp_core.serialize import from_jsonable, to_jsonable
from tdp_ml.pipeline.history import is_ancestor, load_history, mine_at, read_table
from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json, write_frame, write_json, write_table
from tdp_ml.pipeline.manifest import Manifest, ManifestRepo
from tdp_ml.tools.github import github_token

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StageOptions:
    """``force`` rebuilds outputs that exist; ``threads`` is the number of SZZ git threads per repository."""

    force: bool = False
    threads: int = 4


@dataclass(frozen=True, slots=True)
class StageResult:
    """The outcome of one stage for one repository."""

    slug: str
    stage: str
    status: Literal["done", "skipped", "failed"]
    seconds: float
    counts: dict[str, int] = field(default_factory=dict)
    message: str = ""


StageFn = Callable[[DataLayout, Manifest, ManifestRepo, StageOptions], dict[str, int] | None]

REPO_STAGES: tuple[str, ...] = ("mine", "issues", "bugfix", "snapshots", "szz", "features", "labels")


def _mirror(layout: DataLayout, repo: ManifestRepo) -> Path:
    mirror = layout.mirror(repo.slug)
    if not mirror.exists():
        raise PipelineError(f"{mirror} is missing: run tdp-ml manifest")
    return mirror


def _done(paths: Sequence[Path], options: StageOptions) -> bool:
    return not options.force and all(path.is_file() for path in paths)


def _count_true(column: pa.ChunkedArray) -> int:
    return int(pc.sum(pc.fill_null(column, False).cast(pa.int64())).as_py() or 0)


def eligible_fixes(bugfix: pa.Table, commits: pa.Table) -> dict[str, int]:
    """``sha -> author time`` (Unix seconds) of the fixes SZZ and the defect-history features count.

    A fix is a row with ``is_bugfix`` true (null is false; a repeated sha uses its first such row) whose sha
    is a non-bulk commit of ``commits``: the rule of ``szz_table`` and ``defect_history_features``.
    """
    times = commits.column("author_time").cast(pa.int64()).to_pylist()
    bulk = commits.column("is_bulk").to_pylist()
    info = dict(zip(commits.column("sha").to_pylist(), zip(times, bulk, strict=True), strict=True))
    out: dict[str, int] = {}
    seen: set[str] = set()
    shas = bugfix.column("sha").to_pylist()
    for sha, flag in zip(shas, bugfix.column("is_bugfix").to_pylist(), strict=True):
        if not flag or sha is None or sha in seen:
            continue
        seen.add(sha)
        commit = info.get(sha)
        if commit is None or commit[0] is None or commit[1]:
            continue
        out[sha] = int(commit[0])
    return out


def newest_fixes(eligible: Mapping[str, int], within: Set[str], limit: int | None) -> list[str]:
    """The fixes of ``eligible`` inside ``within``, newest first (ties by sha), at most ``limit`` of them."""
    chosen = sorted((-t, sha) for sha, t in eligible.items() if sha in within)
    if limit is not None:
        chosen = chosen[:limit]
    return [sha for _, sha in chosen]


def keep_newest(needs: Sequence[Set[str]], cap: int | None) -> int:
    """How many snapshots to keep, newest first, so the union of their fixes fits ``cap``.

    ``needs[0]`` belongs to the newest snapshot. Returns the largest ``k >= 1`` with
    ``|union(needs[:k])| <= cap`` (all of them for ``cap`` None, 0 for no snapshot). The newest snapshot is
    kept even when it alone exceeds the cap (with a warning).
    """
    if not needs:
        return 0
    if cap is None:
        return len(needs)
    union: set[str] = set(needs[0])
    if len(union) > cap:
        logger.warning("the newest snapshot alone needs %d fixes, above the budget of %d", len(union), cap)
        return 1
    k = 1
    for need in needs[1:]:
        union |= need
        if len(union) > cap:
            break
        k += 1
    return k


def read_snapshots(layout: DataLayout, slug: str) -> list[Snapshot]:
    """The kept snapshots of ``slug`` (oldest first) written by the ``snapshots`` stage."""
    path = layout.snapshots(slug)
    if not path.is_file():
        raise PipelineError(f"{path} is missing: run tdp-ml snapshots")
    return [from_jsonable(Snapshot, item) for item in read_json(path)]


def read_plan(layout: DataLayout, slug: str) -> dict[str, Any]:
    """The snapshot plan of ``slug`` (kept snapshots, window ends and the fixes to trace)."""
    path = layout.snapshot_plan(slug)
    if not path.is_file():
        raise PipelineError(f"{path} is missing: run tdp-ml snapshots")
    plan: dict[str, Any] = read_json(path)
    return plan


def lineage_table(lineage: Mapping[str, frozenset[str]]) -> pa.Table:
    """One row per (path, name) of a ``path_lineage`` mapping, sorted."""
    rows = sorted((path, name) for path, names in lineage.items() for name in names)
    return pa.table(
        {
            "path": pa.array([r[0] for r in rows], pa.string()),
            "name": pa.array([r[1] for r in rows], pa.string()),
        }
    )


def read_lineage(path: Path) -> dict[str, frozenset[str]]:
    """The lineage written by ``features`` (``path -> every name it had``)."""
    table = read_table(path, "features")
    names: dict[str, set[str]] = {}
    for p, n in zip(table.column("path").to_pylist(), table.column("name").to_pylist(), strict=True):
        names.setdefault(p, set()).add(n)
    return {p: frozenset(ns) for p, ns in names.items()}


def _filter_shas(table: pa.Table, shas: Sequence[str]) -> pa.Table:
    mask = pc.is_in(table.column("sha"), value_set=pa.array(list(shas), pa.string()))
    return table.filter(mask)


# ---------- stages ----------


def stage_mine(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """The whole history at the manifest sha (with that sha's mailmap): file changes, then commits."""
    outputs = [layout.file_changes(repo.slug), layout.commits(repo.slug)]
    if _done(outputs, options):
        return None
    hist = mine_at(_mirror(layout, repo), repo.sha, manifest.settings)
    write_table(outputs[0], hist.file_changes)
    write_table(outputs[1], hist.commits)
    return {"commits": hist.commits.num_rows, "file_changes": hist.file_changes.num_rows}


def stage_issues(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """The GitHub issues of a crawled repository (cached; keyword-only repositories are skipped)."""
    if not repo.issues:
        return None
    path = layout.issues(repo.slug)
    if _done([path], options):
        return None
    token = github_token()
    if token is None:
        raise PipelineError(
            "no GitHub token (set GITHUB_TOKEN or run gh auth login); "
            f"or copy {path.name} into {path.parent} from a machine that has one"
        )
    owner, name = repo.name.split("/")
    table = fetch_issues(owner, name, token, cache_path=path, refresh=True)
    return {"issues": table.num_rows, "bug_issues": _count_true(table.column("is_bug"))}


def stage_bugfix(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """The keyword-only bug fixes (features) and the issue-aware ones (labels)."""
    kw_path, label_path = layout.bugfix(repo.slug, keyword=True), layout.bugfix(repo.slug)
    if _done([kw_path, label_path], options):
        return None
    hist = load_history(layout, repo.slug)
    keyword = classify_bugfix(hist.commits, hist.file_changes, None, repo=repo.name)
    if repo.issues:
        issues_path = layout.issues(repo.slug)
        if not issues_path.is_file():
            raise PipelineError(
                f"{issues_path} is missing: run tdp-ml issues on a machine with a GitHub token "
                "and copy it here"
            )
        issues = read_issues(issues_path)
        labels = classify_bugfix(hist.commits, hist.file_changes, issues, repo=repo.name)
    else:
        labels = keyword
    write_table(kw_path, keyword)
    write_table(label_path, labels)
    return {
        "keyword_fixes": _count_true(keyword.column("is_bugfix")),
        "label_fixes": _count_true(labels.column("is_bugfix")),
        "issue_fixes": _count_true(pc.equal(labels.column("source"), "issue")),
    }


def stage_snapshots(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """The kept snapshots and the plan of fixes SZZ must trace for them."""
    plan_path, snaps_path = layout.snapshot_plan(repo.slug), layout.snapshots(repo.slug)
    if _done([plan_path, snaps_path], options):
        return None
    mirror = _mirror(layout, repo)
    settings = manifest.settings
    hist = load_history(layout, repo.slug)
    keyword = read_table(layout.bugfix(repo.slug, keyword=True), "bugfix")
    labels = read_table(layout.bugfix(repo.slug), "bugfix")
    schedule = make_snapshots(repo.slug, hist.commits, settings, repo_dir=mirror)
    candidates = schedule[-manifest.max_snapshots :]
    e_lab = eligible_fixes(labels, hist.commits)
    e_kw = eligible_fixes(keyword, hist.commits)

    newest_first = list(reversed(candidates))
    entries: list[dict[str, Any]] = []
    needs: list[tuple[set[str], set[str]]] = []
    for snap in newest_first:
        end = snapshot_sha(mirror, snap.time + timedelta(days=settings.horizon_days))
        if end is None:
            raise PipelineError(f"snapshot {snap.index} of {repo.slug} has no state at T plus the horizon")
        if not (is_ancestor(mirror, snap.sha, repo.sha) and is_ancestor(mirror, end, repo.sha)):
            raise PipelineError(
                f"snapshot {snap.index} of {repo.slug} is not an ancestor of the manifest sha; "
                "the mirror's default branch moved"
            )
        reach_t = reachable_shas(mirror, snap.sha)
        reach_end = reachable_shas(mirror, end)
        label_k = {sha for sha in e_lab if sha in reach_end and sha not in reach_t}
        feature_k = set(newest_fixes(e_kw, reach_t, settings.inference_szz_max_fixes))
        needs.append((label_k, feature_k))
        entries.append(
            {
                "index": snap.index,
                "sha": snap.sha,
                "end_sha": end,
                "reachable_t": len(reach_t),
                "window_commits": len(reach_end) - len(reach_t),
                "label_fixes": len(label_k),
                "feature_fixes": len(feature_k),
            }
        )
    k = keep_newest([lab | feat for lab, feat in needs], manifest.szz_max_fixes)
    kept = sorted(newest_first[:k], key=lambda s: s.index)
    label_union: set[str] = set()
    feature_union: set[str] = set()
    for lab, feat in needs[:k]:
        label_union |= lab
        feature_union |= feat
    kept_idx = {s.index for s in kept}
    plan = {
        "schedule": len(schedule),
        "candidates": [s.index for s in candidates],
        "kept": [s.index for s in kept],
        "dropped_for_budget": [s.index for s in candidates if s.index not in kept_idx],
        "szz_budget": manifest.szz_max_fixes,
        "snapshots": sorted(entries, key=lambda e: int(e["index"])),
        "label_fixes": sorted(label_union),
        "feature_fixes": sorted(feature_union),
    }
    write_json(plan_path, plan)
    write_json(snaps_path, [to_jsonable(s) for s in kept])
    return {
        "schedule": len(schedule),
        "kept": len(kept),
        "dropped": len(candidates) - len(kept),
        "label_fixes": len(label_union),
        "feature_fixes": len(feature_union),
    }


def stage_szz(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """SZZ for the planned keyword fixes (features) and label fixes (labels), sharing one cache."""
    kw_path, label_path = layout.szz(repo.slug, keyword=True), layout.szz(repo.slug)
    if _done([kw_path, label_path], options):
        return None
    mirror = _mirror(layout, repo)
    plan = read_plan(layout, repo.slug)
    hist = load_history(layout, repo.slug)
    keyword = _filter_shas(
        read_table(layout.bugfix(repo.slug, keyword=True), "bugfix"), plan["feature_fixes"]
    )
    labels = _filter_shas(read_table(layout.bugfix(repo.slug), "bugfix"), plan["label_fixes"])
    cache = layout.szz_cache(repo.slug)
    szz_kw = szz_table(mirror, hist, keyword, jobs=options.threads, cache_dir=cache)
    szz_lab = szz_table(mirror, hist, labels, jobs=options.threads, cache_dir=cache)
    write_table(kw_path, szz_kw)
    write_table(label_path, szz_lab)
    return {
        "label_fixes": len(plan["label_fixes"]),
        "feature_fixes": len(plan["feature_fixes"]),
        "label_rows": szz_lab.num_rows,
        "feature_rows": szz_kw.num_rows,
    }


def _feature_outputs(layout: DataLayout, slug: str, index: int) -> list[Path]:
    return [layout.features(slug, index), layout.lineage(slug, index), layout.features_meta(slug, index)]


def stage_features(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """``build_features`` per kept snapshot from a history mined at the snapshot (its own mailmap)."""
    snapshots = read_snapshots(layout, repo.slug)
    todo = [s for s in snapshots if not _done(_feature_outputs(layout, repo.slug, s.index), options)]
    if not todo:
        return None
    mirror = _mirror(layout, repo)
    settings = manifest.settings
    keyword = read_table(layout.bugfix(repo.slug, keyword=True), "bugfix")
    szz_kw = read_table(layout.szz(repo.slug, keyword=True), "szz")
    satd = load_default_classifier()
    cache = DiskStaticCache(layout.static_cache)
    rows = 0
    for snap in todo:  # oldest first, so the parse cache warms
        started = time.perf_counter()
        hist_t = mine_at(mirror, snap.sha, settings)
        reach = frozenset(hist_t.commits.column("sha").to_pylist())
        res = build_features(
            mirror,
            hist_t,
            snap,
            bugfix=keyword,
            szz=szz_kw,
            satd_model=satd,
            cache=cache,
            settings=settings,
            jobs=1,
            reachable=reach,
        )
        features_path, lineage_path, meta_path = _feature_outputs(layout, repo.slug, snap.index)
        write_frame(features_path, res.features)
        write_table(lineage_path, lineage_table(res.lineage))
        rows += len(res.features)
        write_json(
            meta_path,
            {
                "snapshot": to_jsonable(snap),
                "rows": len(res.features),
                "tests": len(res.test_paths),
                "skipped": res.skipped,
                "timings_s": res.timings_s,
                "seconds": round(time.perf_counter() - started, 3),
            },
        )
        logger.info("%s snapshot %d: %d rows", repo.slug, snap.index, len(res.features))
    return {"built": len(todo), "skipped_snapshots": len(snapshots) - len(todo), "rows": rows}


def stage_labels(
    layout: DataLayout, manifest: Manifest, repo: ManifestRepo, options: StageOptions
) -> dict[str, int] | None:
    """``make_labels`` per kept snapshot with both reachable sets, for the paths of its feature table."""
    snapshots = read_snapshots(layout, repo.slug)
    todo = [s for s in snapshots if not _done([layout.labels(repo.slug, s.index)], options)]
    if not todo:
        return None
    mirror = _mirror(layout, repo)
    settings = manifest.settings
    plan = read_plan(layout, repo.slug)
    end_shas = {int(e["index"]): str(e["end_sha"]) for e in plan["snapshots"]}
    hist = load_history(layout, repo.slug)
    szz_label = read_table(layout.szz(repo.slug), "szz")
    rows = latent = 0
    for snap in todo:
        features_path = layout.features(repo.slug, snap.index)
        if not features_path.is_file():
            raise PipelineError(f"{features_path} is missing: run tdp-ml features")
        paths = pd.read_parquet(features_path, columns=["path"])["path"].tolist()
        lineage = read_lineage(layout.lineage(repo.slug, snap.index))
        reach_t = reachable_shas(mirror, snap.sha)
        reach_end = reachable_shas(mirror, end_shas[snap.index])
        lab = make_labels(
            hist, snap, lineage, paths, szz_label, settings, reachable_T=reach_t, reachable_end=reach_end
        )
        write_frame(layout.labels(repo.slug, snap.index), lab)
        rows += len(lab)
        latent += int(lab["has_latent_defect"].sum())
    return {"built": len(todo), "rows": rows, "latent_files": latent}


STAGE_FUNCTIONS: dict[str, StageFn] = {
    "mine": stage_mine,
    "issues": stage_issues,
    "bugfix": stage_bugfix,
    "snapshots": stage_snapshots,
    "szz": stage_szz,
    "features": stage_features,
    "labels": stage_labels,
}
