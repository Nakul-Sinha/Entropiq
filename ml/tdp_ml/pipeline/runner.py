"""Run one repository stage over the manifest's repositories, in process or in spawned worker processes.

``jobs`` repositories run in parallel, one spawned process per repository (``max_tasks_per_child=1``, so
memory is returned after each repository). A stage that fails for one repository is reported as
``failed`` and never stops the others.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tdp_ml.pipeline.layout import DataLayout, write_json
from tdp_ml.pipeline.manifest import read_manifest
from tdp_ml.pipeline.stages import STAGE_FUNCTIONS, StageOptions, StageResult

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def run_repo_stage(stage: str, root: str, slug: str, options: StageOptions) -> StageResult:
    """Run ``stage`` for ``slug`` in the data folder ``root``; never raises (errors give ``failed``).

    A finished stage writes ``runs/<slug>/<stage>.json`` with its seconds and counts.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    layout = DataLayout(Path(root))
    started = time.perf_counter()
    try:
        manifest = read_manifest(layout)
        repo = manifest.repo(slug)
        counts = STAGE_FUNCTIONS[stage](layout, manifest, repo, options)
    except Exception as exc:
        logger.error("%s %s failed:\n%s", stage, slug, traceback.format_exc())
        return StageResult(
            slug, stage, "failed", time.perf_counter() - started, {}, f"{type(exc).__name__}: {exc}"
        )
    seconds = time.perf_counter() - started
    if counts is None:
        message = "keyword-only" if stage == "issues" and not repo.issues else ""
        return StageResult(slug, stage, "skipped", seconds, {}, message)
    record = {
        "stage": stage,
        "slug": slug,
        "seconds": round(seconds, 3),
        "finished_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": counts,
    }
    write_json(layout.run_record(stage, slug), record)
    return StageResult(slug, stage, "done", seconds, counts)


def run_stage(
    stage: str,
    layout: DataLayout,
    slugs: Sequence[str] | None = None,
    *,
    jobs: int = 1,
    options: StageOptions | None = None,
) -> list[StageResult]:
    """Run ``stage`` for the selected repositories (all for None); results come in manifest order.

    Raises :class:`~tdp_ml.pipeline.layout.PipelineError` before any work when the manifest is missing or a
    slug is unknown, and ``ValueError`` for an unknown stage or ``jobs < 1``.
    """
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage!r}")
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    opts = options or StageOptions()
    repos = read_manifest(layout).select(slugs)
    root = str(layout.root)
    if jobs == 1 or len(repos) <= 1:
        return [run_repo_stage(stage, root, repo.slug, opts) for repo in repos]
    results: dict[str, StageResult] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(jobs, len(repos)), mp_context=context, max_tasks_per_child=1
    ) as pool:
        futures = {pool.submit(run_repo_stage, stage, root, repo.slug, opts): repo.slug for repo in repos}
        for future in as_completed(futures):
            slug = futures[future]
            try:
                results[slug] = future.result()
            except Exception as exc:  # a crashed worker process
                results[slug] = StageResult(slug, stage, "failed", 0.0, {}, f"{type(exc).__name__}: {exc}")
            r = results[slug]
            logger.info("%s %s %s in %.1f s %s", stage, slug, r.status, r.seconds, r.message)
    return [results[repo.slug] for repo in repos]
