"""The ``calibrate`` command: turn a model's out-of-fold ranker scores into calibrated risk with ranges.

Reads ``<model>_oof.parquet`` (and ``<model>_scores_temporal.parquet`` when present) from the results
folder, writes the calibrator, a JSON record, reliability tables, a Markdown summary and the reliability
diagram there, and adds ``model/calibrator.json`` plus the calibration metrics to the model's LOPO
training run in MLflow. Heavy libraries are imported inside the command so ``tdp-ml --help`` stays fast.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.paths import results_dir

if TYPE_CHECKING:
    from tdp_ml.calibrate import CalibrationOutcome

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


class BootstrapChoice(StrEnum):
    repo = "repo"
    row = "row"


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _cell(metrics: dict[str, float], key: str) -> str:
    value, low, high = metrics[key], metrics[f"{key}_ci_low"], metrics[f"{key}_ci_high"]
    if value != value:  # NaN
        return "n/a"
    if low != low or high != high:
        return f"{value:.3f}"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def _print_outcome(outcome: CalibrationOutcome, paths: dict[str, Path], run_id: str | None) -> None:
    from tdp_ml.calibrate import ECE_TARGET

    header = ["set", "files", "positive rate", "mean risk", "Brier", "ECE", "AUC-PR", "per-query ECE"]
    rows = [
        [
            e.name,
            f"{e.rows.rows:,}",
            f"{e.rows.positive_rate:.3f}",
            f"{e.metrics['mean_risk']:.3f}",
            _cell(e.metrics, "brier"),
            _cell(e.metrics, "ece"),
            f"{e.metrics['auc_pr']:.3f}",
            _cell(e.metrics, "query_ece"),
        ]
        for e in outcome.evaluations
    ]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    typer.echo(
        f"{outcome.model} calibration on the temporal test set: value [95% repository bootstrap interval]"
    )
    for r in [header, *rows]:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)).rstrip())
    value = outcome.primary.metrics["ece"]
    verdict = "met" if outcome.meets_target else f"missed ({value:.4f})"
    typer.echo(f"ECE target {ECE_TARGET:.2f}: {verdict} (pooled ECE {value:.4f})")
    stats, s = outcome.final_stats, outcome.settings
    typer.echo(
        f"calibrator: {int(stats['thresholds'])} thresholds from {outcome.final_rows.rows:,} rows, "
        f"{s['bootstrap']} bootstrap x {s['n_boot']}, mean band width {stats['mean_band_width']:.3f}, "
        f"{outcome.seconds:.1f} s"
    )
    for key, path in paths.items():
        typer.echo(f"{key}: {path}")
    typer.echo(f"MLflow run {run_id} updated" if run_id else "MLflow tracking off")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def calibrate_command(
    model: Annotated[
        str, typer.Option("--model", help="Model prefix of the score files, for example ranker_monotone.")
    ] = "ranker",
    scores: Annotated[
        Path | None,
        typer.Option("--scores", help="Out-of-fold scores (default: <results>/<model>_oof.parquet)."),
    ] = None,
    temporal_scores: Annotated[
        Path | None,
        typer.Option(
            "--temporal-scores",
            help="Final temporal model test scores for a second check "
            "(default: <results>/<model>_scores_temporal.parquet when it exists).",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id", help="MLflow run to update (default: from <results>/<model>_lopo_run.json)."
        ),
    ] = None,
    bootstrap: Annotated[
        BootstrapChoice, typer.Option("--bootstrap", help="Unit the risk range resamples: repo or row.")
    ] = BootstrapChoice.repo,
    n_boot: Annotated[
        int, typer.Option("--n-boot", min=0, help="Bootstrap refits for the risk range.")
    ] = 200,
    ci_boot: Annotated[
        int, typer.Option("--ci-boot", min=1, help="Repository bootstrap draws for metric intervals.")
    ] = 1000,
    bins: Annotated[
        int, typer.Option("--bins", min=1, help="Equal-width bins for ECE and reliability.")
    ] = 10,
    seed: Annotated[
        int, typer.Option("--seed", help="Random seed of every bootstrap.")
    ] = DEFAULT_SETTINGS.seed,
    horizon_days: Annotated[
        int, typer.Option("--horizon-days", min=0, help="Label horizon used by the temporal embargo.")
    ] = DEFAULT_SETTINGS.horizon_days,
    tracking: Annotated[
        bool,
        typer.Option("--tracking/--no-tracking", help="Add the calibrator and metrics to the MLflow run."),
    ] = True,
) -> None:
    """Fit the isotonic risk calibration with ranges and evaluate it on the temporal test set.

    The evaluation curve is fitted on out-of-fold rows whose label window closes before each repository's
    last snapshot and tested on that snapshot; the shipped curve is refitted on every row. The command
    succeeds even when the ECE target (0.05) is missed: the promotion gate checks it.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    from tdp_ml import calibrate as cal

    try:
        cal.output_paths(Path("."), model)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    results = results_dir()
    scores_path = scores if scores is not None else results / f"{model}_oof.parquet"
    temporal_path = temporal_scores
    if temporal_path is None:
        default = results / f"{model}_scores_temporal.parquet"
        if default.is_file():
            temporal_path = default
        else:
            typer.echo(f"no temporal model scores at {default}; skipping that check")
    elif not temporal_path.is_file():
        raise _fail(f"temporal scores not found: {temporal_path}")

    run: dict[str, str] | None = None
    if tracking:
        try:
            chosen = run_id if run_id is not None else cal.default_run_id(results, model)
            run = cal.find_run(chosen)
        except ValueError as exc:
            raise _fail(str(exc)) from exc

    try:
        frame = cal.load_scores(scores_path)
        temporal = cal.load_scores(temporal_path) if temporal_path is not None else None
        outcome = cal.calibrate(
            frame,
            model=model,
            temporal_scores=temporal,
            horizon_days=horizon_days,
            bootstrap=bootstrap.value,
            n_boot=n_boot,
            ci_boot=ci_boot,
            bins=bins,
            seed=seed,
        )
    except (ValueError, OSError) as exc:
        raise _fail(str(exc)) from exc

    from tdp_ml.train import code_git_sha, file_sha256

    provenance: dict[str, Any] = {
        "created_at": _now(),
        "scores": {"path": scores_path.as_posix(), "sha256": file_sha256(scores_path)},
        "temporal_scores": (
            {"path": temporal_path.as_posix(), "sha256": file_sha256(temporal_path)}
            if temporal_path is not None
            else None
        ),
        "run_id": run["run_id"] if run is not None else None,
        "code_git_sha": code_git_sha(),
    }
    paths = cal.write_outputs(outcome, results, provenance=provenance)
    if run is not None:
        artifacts = {
            "model/calibrator.json": paths["calibrator"],
            f"results/{paths['record'].name}": paths["record"],
            f"results/{paths['reliability'].name}": paths["reliability"],
            f"results/{paths['summary'].name}": paths["summary"],
            f"figures/{paths['figure'].name}": paths["figure"],
        }
        metrics, tags = cal.run_payload(outcome, provenance=provenance)
        cal.log_to_run(
            run["run_id"], artifacts=artifacts, metrics=metrics, tags=tags, uri=run["tracking_uri"]
        )
    _print_outcome(outcome, paths, run["run_id"] if run is not None else None)


def register(app: typer.Typer) -> None:
    """Add the ``calibrate`` command to ``app``."""
    app.command("calibrate")(calibrate_command)
