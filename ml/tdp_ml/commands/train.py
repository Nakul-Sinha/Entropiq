"""Ranker commands: ``train`` (tune, fit and evaluate the LambdaMART ranker) and ``mlflow-relocate``.

Heavy libraries (xgboost, optuna, mlflow) are imported inside the commands so ``tdp-ml --help`` stays
fast.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.paths import results_dir
from tdp_ml.pipeline.config import DEFAULT_CONFIG_DIR
from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json, write_json

if TYPE_CHECKING:
    from tdp_ml.tracking import RunLog
    from tdp_ml.train import TrainOutcome

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_TABLE_METRICS = (
    ("NDCG@10", "ndcg_10"),
    ("P@10", "precision_10"),
    ("Recall@20%LOC", "recall_20loc"),
    ("Popt", "popt"),
)


class ProtocolChoice(StrEnum):
    temporal = "temporal"
    lopo = "lopo"


class DeviceChoice(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _manifest_sha(dataset: Path) -> str:
    try:
        data = read_json(dataset.with_name("dataset.json"))
    except PipelineError:
        return ""
    value = data.get("manifest_sha256") if isinstance(data, dict) else None
    return str(value) if value else ""


def _cell(summary: Any, model: str, metric: str) -> str:
    rows = summary[(summary["model"] == model) & (summary["metric"] == metric)]
    if rows.empty or rows["mean"].isna().all():
        return "n/a"
    r = rows.iloc[0]
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def _p_hotspot(outcome: TrainOutcome) -> float:
    from tdp_ml.train import p_greater

    hotspot = outcome.others.get("hotspot")
    return float("nan") if hotspot is None else p_greater(outcome.result.per_query, hotspot)


def _mean(summary: Any, model: str, metric: str) -> float:
    rows = summary[(summary["model"] == model) & (summary["metric"] == metric)]
    return float("nan") if rows.empty else float(rows["mean"].iloc[0])


def _log_outcome(
    run: RunLog,
    outcome: TrainOutcome,
    *,
    config_path: Path,
    monotone_columns: list[str],
    device: str,
    nthread: int,
    horizon_days: int,
    frame: Any,
) -> list[str]:
    """Log tags, params, metrics and artifacts of ``outcome``; returns the artifact paths."""
    from tdp_ml.eval import RANKING_METRICS

    params = outcome.params
    run.set_tags({"tuned_in_run": str(params.info.get("run_id") or run.run_id)})
    logged: dict[str, object] = dict(params.params)
    logged |= {
        "monotone": bool(monotone_columns),
        "monotone_columns": monotone_columns if monotone_columns else "",
        "device": device,
        "nthread": nthread,
        "horizon_days": horizon_days,
        "training_rows": outcome.result.training_rows,
        "training_queries": outcome.result.training_queries,
        "dataset_rows": len(frame),
        "dataset_queries": int(frame["qid"].nunique()),
    }
    if outcome.tuning is not None:
        info = params.info
        logged |= {
            "tuning.n_trials": info.get("n_trials"),
            "tuning.n_folds": info.get("n_folds"),
            "tuning.seed": info.get("seed"),
            "tuning.rounds_factor": info.get("rounds_factor"),
            "tuning.best_value": params.best_value,
            "tuning.fold_rounds": list(params.fold_rounds),
            "tuning.search": info.get("search"),
        }
        trials = outcome.tuning.trials
        for number, value in zip(trials["number"].tolist(), trials["value"].tolist(), strict=True):
            run.log_metrics({"tuning_ndcg_10": float(value)}, step=int(number))
        run.log_metrics({"tuning_best_ndcg_10": outcome.tuning.best_value})
    elif "params" in outcome.files:
        logged["params_file"] = str(outcome.files["params"])
    run.log_params(logged)

    protocol, summary = outcome.protocol, outcome.summary
    metrics: dict[str, float] = {}
    for m in RANKING_METRICS:
        rows = summary[(summary["model"] == outcome.model) & (summary["metric"] == m)]
        if not rows.empty:
            r = rows.iloc[0]
            metrics |= {
                f"{protocol}_{m}": float(r["mean"]),
                f"{protocol}_{m}_ci_low": float(r["ci_low"]),
                f"{protocol}_{m}_ci_high": float(r["ci_high"]),
            }
    own = _mean(summary, outcome.model, "ndcg_10")
    for name in outcome.others:
        metrics[f"{protocol}_ndcg_10_diff_{name}"] = own - _mean(summary, name, "ndcg_10")
    metrics[f"{protocol}_hotspot_ndcg_10"] = _mean(summary, "hotspot", "ndcg_10")
    metrics[f"{protocol}_ndcg_10_p_greater_hotspot"] = _p_hotspot(outcome)
    metrics |= {f"seconds_{k}": float(v) for k, v in outcome.seconds.items()}
    run.log_metrics(metrics)

    artifacts: list[str] = []
    name = "ranker_temporal.json" if protocol == "temporal" else "ranker.json"
    with tempfile.TemporaryDirectory(prefix="tdp-ranker-") as tmp:
        booster_path = Path(tmp) / name
        outcome.result.booster.save_model(str(booster_path))
        run.log_artifact(booster_path, "model")
    artifacts.append(f"model/{name}")
    for path in outcome.files.values():
        if Path(path).is_file():
            run.log_artifact(Path(path), "results")
            artifacts.append(f"results/{Path(path).name}")
    run.log_artifact(config_path, "config")
    artifacts.append(f"config/{config_path.name}")
    return artifacts


def _print_outcome(outcome: TrainOutcome, run_id: str | None) -> None:
    names = [outcome.model, *outcome.others]
    header = ["model", *(label for label, _ in _TABLE_METRICS)]
    rows = [[n, *(_cell(outcome.summary, n, m) for _, m in _TABLE_METRICS)] for n in names]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    typer.echo(f"{outcome.model} {outcome.protocol}: mean [95% repository bootstrap interval]")
    for r in [header, *rows]:
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)).rstrip())
    diff = _mean(outcome.summary, outcome.model, "ndcg_10") - _mean(outcome.summary, "hotspot", "ndcg_10")
    typer.echo(f"NDCG@10 difference vs hotspot {diff:+.3f}, one-sided Wilcoxon p {_p_hotspot(outcome):.3g}")
    if outcome.tuning is not None:
        t = outcome.tuning
        typer.echo(
            f"tuning: {len(t.trials)} trials, best ndcg@10 {t.best_value:.4f}, fold rounds "
            f"{list(t.fold_rounds)}, n_estimators {t.n_estimators}, {t.seconds:.1f} s"
        )
    typer.echo(f"MLflow run {run_id}" if run_id else "MLflow tracking off")
    for key, path in outcome.files.items():
        typer.echo(f"{key}: {path}")


def train_command(
    protocol: Annotated[
        ProtocolChoice,
        typer.Option("--protocol", help="Evaluation protocol: temporal or lopo.", show_default=False),
    ],
    params: Annotated[
        Path | None, typer.Option("--params", help="Reuse a tuned params file and skip tuning.")
    ] = None,
    trials: Annotated[
        int | None, typer.Option("--trials", min=1, help="Optuna trials (default: config).")
    ] = None,
    monotone: Annotated[
        bool, typer.Option("--monotone", help="Constrain the configured features to raise the score.")
    ] = False,
    device: Annotated[
        DeviceChoice | None, typer.Option("--device", help="XGBoost device (default: config).")
    ] = None,
    jobs: Annotated[
        int | None, typer.Option("--jobs", min=1, help="XGBoost threads (default: config).")
    ] = None,
    config: Annotated[Path, typer.Option("--config", help="Ranker config YAML.")] = DEFAULT_CONFIG_DIR
    / "ranker.yaml",
    dataset: Annotated[
        Path | None,
        typer.Option("--dataset", help="Dataset parquet (default: $TDP_DATA_DIR/dataset.parquet)."),
    ] = None,
    horizon_days: Annotated[
        int, typer.Option("--horizon-days", min=0, help="Label horizon used by the temporal embargo.")
    ] = DEFAULT_SETTINGS.horizon_days,
    tracking: Annotated[
        bool, typer.Option("--tracking/--no-tracking", help="Log the run, booster and results to MLflow.")
    ] = True,
) -> None:
    """Tune, train and evaluate the LambdaMART ranker under one protocol.

    Without --params the run tunes on forward folds of the temporal training set first and writes
    <model>_params.json; pass that file to later runs (lopo, --monotone) to reuse the parameters.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    from tdp_ml import train as tr

    try:
        cfg = tr.load_ranker_config(config)
        dataset_path = dataset if dataset is not None else DataLayout.from_env().dataset
        frame = tr.load_dataset(dataset_path)
        tuned = tr.load_params(params) if params is not None else None
        resolved = tr.resolve_device(device.value if device is not None else cfg.device)
    except (ValueError, RuntimeError) as exc:  # ConfigError is a ValueError, PipelineError a RuntimeError
        raise _fail(str(exc)) from exc
    nthread = jobs if jobs is not None else cfg.nthread
    provenance = {
        "dataset_sha256": tr.file_sha256(dataset_path),
        "manifest_sha256": _manifest_sha(dataset_path),
        "code_git_sha": tr.code_git_sha(),
        "device": resolved,
    }
    results = results_dir()
    results.mkdir(parents=True, exist_ok=True)
    prefix = cfg.model_name + ("_monotone" if monotone else "")
    proto = protocol.value

    def run_train(run_id: str | None) -> TrainOutcome:
        return tr.train(
            frame,
            cfg,
            protocol=proto,
            results_dir=results,
            params=tuned,
            params_path=params,
            n_trials=trials,
            monotone=monotone,
            device=resolved,
            nthread=nthread,
            horizon_days=horizon_days,
            run_id=run_id,
            provenance=provenance,
        )

    if not tracking:
        _print_outcome(run_train(None), None)
        return
    from tdp_ml import tracking as trk

    tags = {
        "protocol": proto,
        "model": prefix,
        "monotone": "true" if monotone else "false",
        "device": resolved,
        **provenance,
    }
    with trk.mlflow_run(cfg.experiment, f"{prefix}-{proto}", tags=tags) as run:
        outcome = run_train(run.run_id)
        artifacts = _log_outcome(
            run,
            outcome,
            config_path=config,
            monotone_columns=list(tr.monotone_columns(cfg.monotone_features)) if monotone else [],
            device=resolved,
            nthread=nthread,
            horizon_days=horizon_days,
            frame=frame,
        )
    record = {
        "run_id": run.run_id,
        "experiment": run.experiment,
        "experiment_id": run.experiment_id,
        "tracking_uri": run.tracking_uri,
        "protocol": proto,
        "model": prefix,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "artifacts": artifacts,
    }
    run_file = write_json(results / f"{prefix}_{proto}_run.json", record)
    _print_outcome(outcome, run.run_id)
    typer.echo(f"run: {run_file}")


def relocate_command(
    root: Annotated[
        Path | None,
        typer.Option("--root", help="MLflow file store folder (default: from MLFLOW_TRACKING_URI)."),
    ] = None,
) -> None:
    """Point the artifact locations of a copied MLflow file store at its current folder."""
    from tdp_ml import tracking as trk

    if root is None:
        uri = trk.tracking_uri()
        root = trk.file_store_root(uri)
        if root is None:
            raise _fail(f"{uri} is not a file store")
    try:
        changed = trk.relocate_file_store(root)
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"relocated {len(changed)} files under {root}")


def register(app: typer.Typer) -> None:
    """Add the ``train`` and ``mlflow-relocate`` commands to ``app``."""
    app.command("train")(train_command)
    app.command("mlflow-relocate")(relocate_command)
