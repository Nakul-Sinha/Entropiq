"""``tdp-ml baselines``: score the dataset with every baseline under both evaluation protocols."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final

import pandas as pd
import typer

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.paths import data_dir, results_dir
from tdp_ml.baselines import (
    BASELINES,
    DEFAULT_JOBS,
    N_SHUFFLES,
    check_baselines,
    order_per_query,
    run_baselines,
)
from tdp_ml.eval import (
    PROTOCOLS,
    RANKING_METRICS,
    load_per_query,
    per_query_path,
    save_per_query,
    summarize,
    summary_markdown,
)

SUMMARY_PREFIX = "baselines_summary"

PROVENANCE_KEYS: Final = ("dataset_sha256", "horizon_days", "seed", "shuffles")
"""Fields of a provenance note that must match before stored baseline rows are reused."""

logger = logging.getLogger(__name__)


def summary_paths(root: Path) -> tuple[Path, Path]:
    """Paths of the summary table: ``<root>/baselines_summary.parquet`` and ``.md``."""
    return root / f"{SUMMARY_PREFIX}.parquet", root / f"{SUMMARY_PREFIX}.md"


def file_sha256(path: Path) -> str:
    """Hex SHA-256 digest of the bytes of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(root: Path, protocol: str) -> Path:
    """Path of the provenance note beside a per-query file: ``<root>/baselines_<protocol>.json``."""
    return per_query_path("baselines", protocol, root=root).with_suffix(".json")


def dataset_provenance(
    path: Path, df: pd.DataFrame, *, horizon_days: int, seed: int, shuffles: int, jobs: int
) -> dict[str, Any]:
    """Describe the dataset and options of a baseline run.

    The fields in :data:`PROVENANCE_KEYS` decide whether stored rows can be reused; ``dataset`` (the
    path), ``rows``, ``queries``, ``repos`` and ``jobs`` are informational only.
    """
    return {
        "dataset": Path(path).as_posix(),
        "dataset_sha256": file_sha256(path),
        "rows": len(df),
        "queries": int(df["qid"].nunique()),
        "repos": int(df["repo"].nunique()),
        "horizon_days": int(horizon_days),
        "seed": int(seed),
        "shuffles": int(shuffles),
        "jobs": int(jobs),
    }


def read_provenance(path: Path) -> dict[str, Any] | None:
    """Read a provenance note; returns ``None`` when it is missing, unreadable or not a JSON object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("could not read provenance note %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("provenance note %s is not a JSON object", path)
        return None
    return {str(key): value for key, value in data.items()}


def write_provenance(path: Path, provenance: Mapping[str, Any]) -> Path:
    """Write a provenance note atomically (through ``<name>.tmp``) and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def same_provenance(stored: Mapping[str, Any] | None, current: Mapping[str, Any]) -> bool:
    """Whether a stored note matches ``current`` on every field of :data:`PROVENANCE_KEYS`."""
    return stored is not None and all(
        key in stored and stored[key] == current[key] for key in PROVENANCE_KEYS
    )


def _provenance_line(protocol: str, note: Mapping[str, Any] | None) -> str:
    """One Markdown list item saying which dataset and options produced a protocol's file."""
    if note is None:
        return f"- {protocol}: provenance unknown"
    sha = str(note.get("dataset_sha256", "?"))[:16]
    return (
        f"- {protocol}: dataset sha256 {sha}, {note.get('rows', '?')} rows, "
        f"{note.get('queries', '?')} queries, "
        f"{note.get('repos', '?')} repositories, horizon {note.get('horizon_days', '?')} days, "
        f"seed {note.get('seed', '?')}, {note.get('shuffles', '?')} shuffles"
    )


def write_summary(
    root: Path,
    *,
    n_boot: int = 1000,
    seed: int = DEFAULT_SETTINGS.seed,
    provenance: Mapping[str, Any] | None = None,
) -> str | None:
    """Summarise every ``baselines_<protocol>.parquet`` under ``root``; returns the Markdown table.

    Writes ``baselines_summary.parquet`` (the :func:`tdp_ml.eval.summarize` frame: mean and 95%
    repository-bootstrap interval per model, protocol and metric) and ``baselines_summary.md`` (the
    provenance of each protocol's file, then the ranking metrics). When ``provenance`` is given, a
    protocol file whose note does not match it (see :func:`same_provenance`) is left out with a warning.
    Returns ``None`` and writes nothing when no per-query file is left.
    """
    frames: list[pd.DataFrame] = []
    lines: list[str] = []
    for proto in PROTOCOLS:
        path = per_query_path("baselines", proto, root=root)
        if not path.exists():
            continue
        note = read_provenance(provenance_path(root, proto))
        if provenance is not None and not same_provenance(note, provenance):
            logger.warning(
                "%s: %s comes from another dataset or other options; left out of the summary", proto, path
            )
            continue
        frames.append(load_per_query(path))
        lines.append(_provenance_line(proto, note))
    if not frames:
        return None
    per_query = order_per_query(pd.concat(frames, ignore_index=True))
    summary = summarize(per_query, n_boot=n_boot, seed=seed)
    table = summary_markdown(summary, metrics=[m for m in RANKING_METRICS if m in per_query.columns])
    parquet_path, md_path = summary_paths(root)
    summary.to_parquet(parquet_path, index=False)
    md_path.write_text(
        "# Baselines\n\nMean over test queries with a 95% bootstrap interval that resamples repositories.\n\n"
        + "\n".join(lines)
        + "\n\n"
        + table,
        encoding="utf-8",
    )
    return table


def baselines_command(
    protocol: Annotated[
        str, typer.Option("--protocol", help="Evaluation protocol: temporal, lopo or all.")
    ] = "all",
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help=f"Baseline to run (repeatable; default all of {', '.join(BASELINES)})."),
    ] = None,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", help="Dataset Parquet (default: data/dataset.parquet).")
    ] = None,
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", help="Results folder (default: TDP_RESULTS_DIR).")
    ] = None,
    jobs: Annotated[
        int, typer.Option("--jobs", help="Threads for rf and xgb_point (-1 = every core).")
    ] = DEFAULT_JOBS,
    shuffles: Annotated[
        int, typer.Option("--shuffles", min=1, help="Seeded shuffles averaged by the random baseline.")
    ] = N_SHUFFLES,
    n_boot: Annotated[
        int, typer.Option("--n-boot", min=1, help="Bootstrap replicates of the summary.")
    ] = 1000,
    horizon_days: Annotated[
        int, typer.Option("--horizon-days", min=0, help="Label horizon used for the temporal embargo.")
    ] = DEFAULT_SETTINGS.horizon_days,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = DEFAULT_SETTINGS.seed,
    force: Annotated[
        bool, typer.Option("--force", help="Rerun baselines already present in the result files.")
    ] = False,
) -> None:
    """Run the baselines and write results/baselines_<protocol>.parquet plus a summary table.

    Each protocol's file gets a JSON note (baselines_<protocol>.json) with the dataset sha256 and the
    options used. Baselines already stored are skipped unless --force, but only when the note shows the
    same dataset sha256 and the same --horizon-days, --seed and --shuffles; otherwise the stale file is
    replaced and every requested baseline reruns. New results replace the rows of the same baseline and
    keep the others.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if protocol == "all":
        protocols = list(PROTOCOLS)
    elif protocol in PROTOCOLS:
        protocols = [protocol]
    else:
        raise typer.BadParameter(f"expected temporal, lopo or all, got {protocol!r}", param_hint="--protocol")
    if jobs == 0 or jobs < -1:
        raise typer.BadParameter("must be -1 or a positive number", param_hint="--jobs")
    try:
        names = check_baselines(model if model else BASELINES)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--model") from exc
    path = dataset if dataset is not None else data_dir() / "dataset.parquet"
    if not path.exists():
        typer.echo(f"Dataset not found: {path} (build it with tdp-ml dataset)", err=True)
        raise typer.Exit(2)
    root = out_dir if out_dir is not None else results_dir()
    df = pd.read_parquet(path)
    current = dataset_provenance(path, df, horizon_days=horizon_days, seed=seed, shuffles=shuffles, jobs=jobs)
    typer.echo(
        f"Loaded {len(df)} rows, {df['qid'].nunique()} queries, {df['repo'].nunique()} repositories "
        f"(sha256 {current['dataset_sha256'][:16]})"
    )

    for proto in protocols:
        out = per_query_path("baselines", proto, root=root)
        note_path = provenance_path(root, proto)
        existing = load_per_query(out) if out.exists() else None
        if existing is not None and not same_provenance(read_provenance(note_path), current):
            typer.echo(
                f"{proto}: {out} comes from another dataset or other options; "
                "rerunning every requested baseline"
            )
            existing = None
        done = set() if existing is None or force else set(existing["model"].astype(str))
        todo = [n for n in names if n not in done]
        if not todo:
            typer.echo(f"{proto}: every requested baseline is already in {out} (pass --force to rerun)")
            continue
        start = time.perf_counter()
        fresh = run_baselines(
            df, proto, todo, n_jobs=jobs, seed=seed, horizon_days=horizon_days, n_shuffles=shuffles
        )
        frames = [fresh]
        if existing is not None:
            kept = existing[~existing["model"].astype(str).isin(todo)]
            if len(kept):
                frames.insert(0, kept)
        note_path.unlink(missing_ok=True)
        save_per_query(order_per_query(pd.concat(frames, ignore_index=True)), out)
        write_provenance(note_path, current)
        typer.echo(f"{proto}: ran {', '.join(todo)} in {time.perf_counter() - start:.1f} s, wrote {out}")

    table = write_summary(root, n_boot=n_boot, seed=seed, provenance=current)
    if table is not None:
        typer.echo(table)
        typer.echo(f"Summary written to {summary_paths(root)[1]}")


def register(app: typer.Typer) -> None:
    """Add the ``baselines`` command to ``app``."""
    app.command("baselines")(baselines_command)
