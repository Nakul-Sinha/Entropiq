"""``check-leakage``: rebuild sampled queries in clones that hold nothing after T and compare the rows.

A clone of the mirror is checked out at the snapshot sha on a branch and every other ref is deleted, so
no commit after T is reachable. Features are rebuilt there exactly as at inference: the history mined at
the snapshot (the work tree's ``.mailmap``), keyword bug-fix detection, SZZ for the newest
``inference_szz_max_fixes`` prior fixes, no caches. Any difference from the stored rows beyond the
tolerance is leakage (or nondeterminism) and fails the check.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdp_core.config import Settings
from tdp_core.contracts import FEATURE_INPUT_COLUMNS, Snapshot
from tdp_core.features.build import build_features
from tdp_core.features.satd_model import load_default_classifier
from tdp_core.labeling.bugfix import classify_bugfix
from tdp_core.labeling.szz import szz_table
from tdp_core.mining.git import check_rev, run_git
from tdp_core.mining.gitlog import mine_history
from tdp_ml.pipeline.dataset import qid_for
from tdp_ml.pipeline.layout import DataLayout, PipelineError, write_json
from tdp_ml.pipeline.manifest import read_manifest
from tdp_ml.pipeline.stages import read_snapshots

logger = logging.getLogger(__name__)


def truncated_clone(mirror: Path, sha: str, dest: Path) -> Path:
    """A non-bare clone of ``mirror`` at ``dest`` checked out at ``sha`` with no other ref."""
    check_rev(sha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_git(
        ["clone", "--no-local", "--no-checkout", "--quiet", "--", str(mirror), str(dest)],
        dest.parent,
        allow_file_protocol=True,
    )
    run_git(["checkout", "--quiet", "-B", "snapshot", sha], dest)
    refs = run_git(["for-each-ref", "--format=%(refname)"], dest).decode("utf-8").split()
    doomed = [ref for ref in refs if ref != "refs/heads/snapshot"]
    if doomed:
        commands = "".join(f"delete {ref}\n" for ref in doomed).encode("utf-8")
        run_git(["update-ref", "--no-deref", "--stdin"], dest, input=commands)
    run_git(["remote", "remove", "origin"], dest)
    return dest


def rebuild_features(
    clone: Path, snapshot: Snapshot, repo_name: str, settings: Settings, *, threads: int = 4
) -> pd.DataFrame:
    """The feature rows of ``snapshot`` computed inside ``clone`` from nothing but its own history."""
    hist = mine_history(clone, rev=snapshot.sha, settings=settings)
    keyword = classify_bugfix(hist.commits, hist.file_changes, None, repo=repo_name)
    szz = szz_table(
        clone, hist, keyword, max_fixes=settings.inference_szz_max_fixes, jobs=threads, cache_dir=None
    )
    res = build_features(
        clone,
        hist,
        snapshot,
        bugfix=keyword,
        szz=szz,
        satd_model=load_default_classifier(),
        cache=None,
        settings=settings,
        jobs=1,
    )
    return res.features


def compare_rows(
    stored: pd.DataFrame, rebuilt: pd.DataFrame, *, tol: float = 1e-6, limit: int = 10
) -> list[str]:
    """Differences between two feature tables of one query (empty when they agree within ``tol``)."""
    a = stored.sort_values("path", kind="stable").reset_index(drop=True)
    b = rebuilt.sort_values("path", kind="stable").reset_index(drop=True)
    a_paths, b_paths = a["path"].tolist(), b["path"].tolist()
    if a_paths != b_paths:
        only_a = sorted(set(a_paths) - set(b_paths))[:5]
        only_b = sorted(set(b_paths) - set(a_paths))[:5]
        return [
            f"paths differ: only stored {only_a}, only rebuilt {only_b} "
            f"({len(a_paths)} vs {len(b_paths)} rows)"
        ]
    messages: list[str] = []
    for column in ("repo", "snapshot_idx", "snapshot_time"):
        if not (a[column].to_numpy() == b[column].to_numpy()).all():
            messages.append(f"{column} differs")
    for column in FEATURE_INPUT_COLUMNS:
        x = a[column].to_numpy(dtype=np.float64)
        y = b[column].to_numpy(dtype=np.float64)
        bad = np.flatnonzero(~np.isclose(x, y, rtol=tol, atol=tol))
        for i in bad:
            if len(messages) >= limit:
                return messages
            messages.append(f"{a_paths[i]} {column}: stored {float(x[i])!r} rebuilt {float(y[i])!r}")
    return messages


def _remove_tree(path: Path) -> None:
    def retry(func: Callable[[str], Any], name: str, _exc: BaseException) -> None:
        os.chmod(name, stat.S_IWRITE)
        func(name)

    if path.exists():
        shutil.rmtree(path, onexc=retry)


def check_query(root: str, slug: str, index: int, threads: int) -> dict[str, Any]:
    """Rebuild one query and compare it with ``dataset.parquet``; never raises (errors fail the query)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    layout = DataLayout(Path(root))
    started = time.perf_counter()
    report: dict[str, Any] = {"repo": slug, "snapshot_idx": index, "qid": None, "rows": 0}
    clone = layout.tmp / f"leak-{slug}-{index}-{uuid.uuid4().hex[:8]}"
    try:
        manifest = read_manifest(layout)
        repo = manifest.repo(slug)
        qid = qid_for(repo.id, index)
        report["qid"] = qid
        snap = next(s for s in read_snapshots(layout, slug) if s.index == index)
        stored = pd.read_parquet(layout.dataset, filters=[("qid", "==", qid)])
        report["rows"] = len(stored)
        truncated_clone(layout.mirror(slug), snap.sha, clone)
        rebuilt = rebuild_features(clone, snap, repo.name, manifest.settings, threads=threads)
        mismatches = compare_rows(
            stored[["repo", "snapshot_idx", "snapshot_time", "path", *FEATURE_INPUT_COLUMNS]], rebuilt
        )
    except Exception as exc:
        logger.exception("leakage check of %s snapshot %d failed", slug, index)
        mismatches = [f"error: {type(exc).__name__}: {exc}"]
    finally:
        try:
            _remove_tree(clone)
        except OSError as exc:
            logger.warning("could not remove %s: %s", clone, exc)
    report["passed"] = not mismatches
    report["mismatches"] = mismatches
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def check_leakage(
    layout: DataLayout,
    *,
    sample: int = 50,
    seed: int | None = None,
    slugs: Sequence[str] | None = None,
    jobs: int = 1,
    threads: int = 4,
) -> dict[str, Any]:
    """Check ``sample`` random queries of the dataset and write ``runs/check-leakage.json``."""
    manifest = read_manifest(layout)
    if not layout.dataset.is_file():
        raise PipelineError(f"{layout.dataset} is missing: run tdp-ml dataset")
    wanted = {r.slug for r in manifest.select(slugs)}
    keys = pd.read_parquet(layout.dataset, columns=["repo", "snapshot_idx"]).drop_duplicates()
    keys = keys[keys["repo"].isin(wanted)].sort_values(["repo", "snapshot_idx"]).reset_index(drop=True)
    n = min(sample, len(keys))
    used_seed = manifest.settings.seed if seed is None else seed
    picks = sorted(np.random.default_rng(used_seed).choice(len(keys), n, replace=False).tolist()) if n else []
    queries = [(str(keys.at[i, "repo"]), int(keys.at[i, "snapshot_idx"])) for i in picks]
    root = str(layout.root)
    if jobs <= 1 or len(queries) <= 1:
        results = [check_query(root, slug, index, threads) for slug, index in queries]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(queries)), mp_context=context, max_tasks_per_child=1
        ) as pool:
            futures = [pool.submit(check_query, root, slug, index, threads) for slug, index in queries]
            results = [f.result() for f in futures]
    report = {
        "sample": sample,
        "seed": used_seed,
        "checked": len(results),
        "passed": all(r["passed"] for r in results),
        "queries": results,
    }
    write_json(layout.run_record("check-leakage"), report)
    return report
