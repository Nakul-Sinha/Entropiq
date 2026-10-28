"""Dataset pipeline commands: ``manifest``, the seven repository stages, ``dataset``, ``stats`` and
``check-leakage``.

Every command is idempotent: outputs that exist are skipped unless ``--force``. ``--repo`` (repeatable)
limits a stage to some repositories; ``--jobs`` runs that many repositories in parallel processes and
``--threads`` sets the SZZ git threads per repository.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from tdp_ml.pipeline.config import DEFAULT_LABELS_CONFIG, DEFAULT_REPOS_CONFIG, ConfigError, load_config
from tdp_ml.pipeline.dataset import build_dataset, collect_stats, format_table, write_dataset
from tdp_ml.pipeline.layout import DataLayout, PipelineError, write_json
from tdp_ml.pipeline.leakage import check_leakage
from tdp_ml.pipeline.manifest import build_manifest, read_manifest, write_manifest
from tdp_ml.pipeline.runner import LOG_FORMAT, run_stage
from tdp_ml.pipeline.stages import REPO_STAGES, StageOptions

_STAGE_DOCS = {
    "mine": "Mine the whole history of each repository at its manifest sha.",
    "issues": "Fetch the GitHub issues of each crawled repository (needs a token; cached).",
    "bugfix": "Classify bug fixes: keyword-only (features) and issue-aware (labels).",
    "snapshots": "Choose the kept snapshots and the fixes SZZ must trace for them.",
    "szz": "Run SZZ for the planned fixes of each repository.",
    "features": "Build the feature table of every kept snapshot.",
    "labels": "Compute the labels of every kept snapshot.",
}

RepoOption = Annotated[list[str] | None, typer.Option("--repo", help="Repository slug (repeatable).")]
JobsOption = Annotated[int, typer.Option("--jobs", min=1, help="Repositories processed in parallel.")]
ThreadsOption = Annotated[int, typer.Option("--threads", min=1, help="SZZ git threads per repository.")]
ForceOption = Annotated[bool, typer.Option("--force", help="Rebuild outputs that already exist.")]


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _counts(counts: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in counts.items())


def _stage_command(stage: str) -> Any:
    def command(
        repo: RepoOption = None, jobs: JobsOption = 1, threads: ThreadsOption = 4, force: ForceOption = False
    ) -> None:
        _logging()
        started = time.perf_counter()
        try:
            results = run_stage(
                stage,
                DataLayout.from_env(),
                repo,
                jobs=jobs,
                options=StageOptions(force=force, threads=threads),
            )
        except PipelineError as exc:
            raise _fail(str(exc)) from exc
        for r in results:
            line = f"{r.slug:34s} {r.status:8s} {r.seconds:7.1f}s {_counts(r.counts)}"
            typer.echo(f"{line} {r.message}".rstrip())
        tally = {s: sum(1 for r in results if r.status == s) for s in ("done", "skipped", "failed")}
        typer.echo(
            f"{tally['done']} done, {tally['skipped']} skipped, {tally['failed']} failed "
            f"in {time.perf_counter() - started:.1f} s"
        )
        if tally["failed"]:
            raise typer.Exit(1)

    command.__doc__ = _STAGE_DOCS[stage]
    return command


def manifest(
    repos: Annotated[Path, typer.Option("--repos", help="repos.yaml to read.")] = DEFAULT_REPOS_CONFIG,
    labels: Annotated[Path, typer.Option("--labels", help="labels.yaml to read.")] = DEFAULT_LABELS_CONFIG,
    fetch: Annotated[bool, typer.Option("--fetch", help="Refresh mirrors that already exist.")] = False,
    force: ForceOption = False,
) -> None:
    """Pin every repository of repos.yaml to its mirror's HEAD and store the settings of labels.yaml."""
    _logging()
    layout = DataLayout.from_env()
    if layout.manifest.is_file() and not force:
        typer.echo(f"manifest exists: {layout.manifest} (use --force to rebuild it)")
        return
    try:
        config = load_config(repos, labels)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    built = build_manifest(config, layout, fetch=fetch)
    write_manifest(layout, built)
    for r in built.repos:
        typer.echo(
            f"{r.slug:34s} {r.sha[:12]} {r.default_branch:10s} {'issues' if r.issues else 'keyword-only'}"
        )
    typer.echo(f"{len(built.repos)} repositories written to {layout.manifest}")


def dataset(force: ForceOption = False) -> None:
    """Check the criteria, join features and labels per query and write data/dataset.parquet."""
    _logging()
    layout = DataLayout.from_env()
    if layout.dataset.is_file() and layout.dataset_summary.is_file() and not force:
        typer.echo(f"dataset exists: {layout.dataset} (use --force to rebuild it)")
        return
    try:
        frame, summary = build_dataset(layout, read_manifest(layout))
    except PipelineError as exc:
        raise _fail(str(exc)) from exc
    write_dataset(layout, frame, summary)
    rows = []
    for entry in summary["per_repo"]:
        row = {k: entry[k] for k in ("slug", "passed", "queries", "rows")}
        row.update({f"g{g}": entry["grades"][g] for g in entry["grades"]})
        row["failures"] = "; ".join(entry["failures"])
        rows.append(row)
    typer.echo(
        format_table(rows, ["slug", "passed", "queries", "rows", "g0", "g1", "g2", "g3", "g4", "failures"])
    )
    typer.echo(
        f"{summary['repos']} repositories, {summary['queries']} queries, {summary['rows']} rows, "
        f"grades {summary['grades']} -> {layout.dataset}"
    )


def stats() -> None:
    """Report counts, grade distributions, runtimes and disk usage; writes data/stats.json."""
    _logging()
    layout = DataLayout.from_env()
    try:
        report = collect_stats(layout, read_manifest(layout))
    except PipelineError as exc:
        raise _fail(str(exc)) from exc
    write_json(layout.stats, report)
    rows = [
        {
            "slug": r["slug"],
            "commits": r["commits"],
            "kept": len(r["kept"] or []),
            "szz_fixes": r["szz_label_fixes"] + r["szz_feature_fixes"],
            "queries": r["queries"],
            "rows": r["rows"],
            "passed": r["passed"],
        }
        for r in report["repos"]
    ]
    typer.echo(format_table(rows, ["slug", "commits", "kept", "szz_fixes", "queries", "rows", "passed"]))
    typer.echo(f"dataset: {report['dataset']}")
    typer.echo(f"stage seconds (sum over repositories): {report['stage_seconds_total']}")
    typer.echo(f"disk bytes: {report['disk_bytes']}")


def check_leakage_command(
    sample: Annotated[int, typer.Option("--sample", min=1, help="Queries to check.")] = 50,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Sampling seed (default: settings.seed).")
    ] = None,
    repo: RepoOption = None,
    jobs: JobsOption = 1,
    threads: ThreadsOption = 4,
) -> None:
    """Rebuild sampled queries in clones truncated at the snapshot and compare them with the dataset."""
    _logging()
    try:
        report = check_leakage(
            DataLayout.from_env(), sample=sample, seed=seed, slugs=repo, jobs=jobs, threads=threads
        )
    except PipelineError as exc:
        raise _fail(str(exc)) from exc
    for q in report["queries"]:
        status = "ok" if q["passed"] else "MISMATCH"
        typer.echo(
            f"{q['repo']:34s} {q['snapshot_idx']:3d} {q['rows']:6d} rows {status:8s} {q['seconds']:6.1f}s"
        )
        for message in q["mismatches"]:
            typer.echo(f"    {message}")
    failed = sum(1 for q in report["queries"] if not q["passed"])
    typer.echo(f"{report['checked'] - failed} of {report['checked']} queries match")
    if failed:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    app.command("manifest")(manifest)
    for stage in REPO_STAGES:
        app.command(stage, help=_STAGE_DOCS[stage])(_stage_command(stage))
    app.command("dataset")(dataset)
    app.command("stats")(stats)
    app.command("check-leakage")(check_leakage_command)
