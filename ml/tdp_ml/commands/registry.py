"""Registry commands: ``register`` a trained run as a bundle, score the ``golden`` repositories with it and
``promote`` it to production once every gate passes.

Bundles, tarballs and registry records live in ``$TDP_MODELS_DIR``; golden and promotion reports go to
``$TDP_RESULTS_DIR``. ``promote`` writes the tracked ``models/production.json`` of this checkout and
``$TDP_MODELS_DIR/production.json`` unless ``--production-file`` names other files. Heavy libraries
(MLflow, the analysis pipeline, SHAP) are imported inside the commands so ``tdp-ml --help`` stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from tdp_core.paths import data_dir, models_dir, results_dir

if TYPE_CHECKING:
    from tdp_ml.registry import PromotionOutcome

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Golden repositories YAML (default: ml/configs/golden.yaml)."),
]
MirrorsOption = Annotated[
    Path | None,
    typer.Option("--mirrors", help="Folder of <slug>.git mirrors (default: $TDP_DATA_DIR/mirrors)."),
]
JobsOption = Annotated[int, typer.Option("--jobs", min=1, help="Analysis workers per repository.")]


def _fail(message: str, code: int = 2) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _logging() -> None:
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)


def _metric(metrics: dict[str, Any], key: str, digits: int = 3) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    low, high = metrics.get(f"{key}_ci_low"), metrics.get(f"{key}_ci_high")
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def register_command(
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="LOPO training run (default: from <results>/<model>_lopo_run.json)."),
    ] = None,
    model: Annotated[
        str, typer.Option("--model", help="Model name of the default run, for example ranker_monotone.")
    ] = "ranker",
    version: Annotated[
        str | None,
        typer.Option("--version", help="Version to create (default: the next ranker-YYYY.MM.N)."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild the bundle of a run that is already registered.")
    ] = False,
) -> None:
    """Build the production bundle of a LOPO training run, with its model card, tarball and record."""
    _logging()
    from tdp_ml import registry as reg
    from tdp_ml.calibrate import default_run_id
    from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json

    try:
        chosen = run_id if run_id is not None else default_run_id(results_dir(), model)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    dataset: dict[str, Any] | None = None
    summary = DataLayout.from_env().dataset_summary
    if summary.is_file():
        try:
            loaded = read_json(summary)
        except PipelineError as exc:
            typer.echo(f"ignoring the dataset summary: {exc}", err=True)
        else:
            dataset = loaded if isinstance(loaded, dict) else None
    try:
        files = reg.default_production_files()
        current = reg.read_pointer(files[0]) if files else None
        result = reg.register_run(
            chosen,
            models_dir=models_dir(),
            version=version,
            force=force,
            dataset=dataset,
            production_version=current["version"] if current else None,
        )
        record = reg.read_record(models_dir(), result.version)
    except reg.RegistryError as exc:
        raise _fail(str(exc)) from exc
    if not result.created:
        typer.echo(f"run {chosen} is already registered as {result.version} (use --force to rebuild)")
        return
    from tdp_core.model.bundle import load_bundle

    metrics = load_bundle(result.bundle).metadata.get("metrics", {})
    lopo, calibration = metrics.get("lopo", {}), metrics.get("calibration", {})
    ece = calibration.get("ece")
    typer.echo(f"registered {result.version} from run {chosen} ({record.get('model')})")
    typer.echo(f"bundle {result.bundle.as_posix()}")
    typer.echo(f"tarball {result.tarball.as_posix()} ({result.size_bytes} bytes, sha256 {result.sha256})")
    typer.echo(
        f"LOPO NDCG@10 {_metric(lopo, 'ndcg_10')}, hotspot p "
        f"{_metric(lopo, 'ndcg_10_p_greater_hotspot')}, ECE {'n/a' if ece is None else f'{ece:.4f}'}"
    )
    typer.echo(f"next: tdp-ml golden --version {result.version}; tdp-ml promote --version {result.version}")


def _print_golden(report: dict[str, Any]) -> None:
    for repo in report["repos"]:
        top = repo["top"][0] if repo["top"] else None
        top_text = f"top {top['path']} ({top['band']} {top['priority']})" if top else "no files"
        status = "ok" if repo["passed"] else f"FAILED: {repo['errors'][0] if repo['errors'] else 'unknown'}"
        typer.echo(
            f"{repo['name']} {repo['sha'][:7]}: {repo['files']} files, priority "
            f"{repo['priority_min']}..{repo['priority_max']}, {top_text}, {repo['seconds']:.1f} s, {status}"
        )


def golden_command(
    version: Annotated[
        str | None, typer.Option("--version", help="Registered version in $TDP_MODELS_DIR.")
    ] = None,
    bundle: Annotated[Path | None, typer.Option("--bundle", help="Any bundle folder.")] = None,
    config: ConfigOption = None,
    mirrors: MirrorsOption = None,
    jobs: JobsOption = 4,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Report path (default: $TDP_RESULTS_DIR/golden_<version>.json)."),
    ] = None,
) -> None:
    """Score the pinned golden repositories with a bundle and check every result."""
    _logging()
    if (version is None) == (bundle is None):
        raise _fail("pass exactly one of --version and --bundle")
    from tdp_core.model.bundle import BundleSchemaError
    from tdp_ml import registry as reg
    from tdp_ml.pipeline.config import ConfigError
    from tdp_ml.pipeline.layout import write_json

    sha256: str | None = None
    try:
        repos = reg.load_golden_config(config) if config is not None else reg.load_golden_config()
        if version is not None:
            folder = reg.bundle_dir(models_dir(), version)
            try:
                sha256 = str(reg.read_record(models_dir(), version)["sha256"])
            except reg.RegistryError:
                sha256 = None
        else:
            assert bundle is not None
            folder = bundle
        report = reg.run_golden(
            folder, repos, mirrors=mirrors or data_dir() / "mirrors", jobs=jobs, sha256=sha256
        )
    except (reg.RegistryError, ConfigError, FileNotFoundError, BundleSchemaError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    target = out if out is not None else reg.golden_path(results_dir(), str(report["version"]))
    write_json(target, report)
    _print_golden(report)
    verdict = "passed" if report["passed"] else "FAILED"
    typer.echo(f"golden: {verdict} ({target.as_posix()})")
    if not report["passed"]:
        raise typer.Exit(1)


def _value(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == int(value) and abs(value) >= 1:
        return str(int(value))
    return f"{value:.4f}"


def _print_gates(outcome: PromotionOutcome) -> None:
    header = ["gate", "result", "value", "threshold", "detail"]
    rows = [
        [
            g.name,
            "pass" if g.passed else ("FAIL (waived)" if g.waived else "FAIL"),
            _value(g.value),
            g.threshold,
            g.detail,
        ]
        for g in outcome.gates
    ]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header) - 1)]
    for r in [header, *rows]:
        cells = [cell.ljust(w) for cell, w in zip(r[:-1], widths, strict=True)]
        typer.echo("  ".join([*cells, r[-1]]).rstrip())


def promote_command(
    version: Annotated[str, typer.Option("--version", help="Registered version to promote.")],
    waive: Annotated[
        list[str] | None,
        typer.Option("--waive", help="Failed gate to accept anyway: incumbent, hotspot or calibration."),
    ] = None,
    reason: Annotated[
        str | None, typer.Option("--reason", help="One-line reason, recorded with the waivers.")
    ] = None,
    leakage_report: Annotated[
        Path | None,
        typer.Option(
            "--leakage-report", help="Leakage report (default: $TDP_DATA_DIR/runs/check-leakage.json)."
        ),
    ] = None,
    run_leakage: Annotated[
        bool, typer.Option("--run-leakage", help="Run the leakage check on 10 sampled queries first.")
    ] = False,
    config: ConfigOption = None,
    mirrors: MirrorsOption = None,
    jobs: JobsOption = 4,
    threads: Annotated[int, typer.Option("--threads", min=1, help="Git threads per leakage check.")] = 2,
    production_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--production-file",
            help="Pointer file to write (repeatable; default: models/production.json and "
            "$TDP_MODELS_DIR/production.json).",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Run the gates without publishing.")] = False,
) -> None:
    """Run the promotion gates on a registered bundle and publish it as the production model."""
    _logging()
    from tdp_ml import registry as reg
    from tdp_ml.pipeline.config import ConfigError
    from tdp_ml.pipeline.layout import DataLayout, PipelineError

    try:
        repos = reg.load_golden_config(config) if config is not None else reg.load_golden_config()
        files = list(production_file) if production_file else reg.default_production_files()
        outcome = reg.promote(
            version,
            models_dir=models_dir(),
            results_dir=results_dir(),
            layout=DataLayout.from_env(),
            production_files=files,
            golden=repos,
            mirrors=mirrors or data_dir() / "mirrors",
            waive=waive or (),
            reason=reason,
            leakage_report=leakage_report,
            run_leakage=run_leakage,
            jobs=jobs,
            threads=threads,
            dry_run=dry_run,
        )
    except (reg.RegistryError, ConfigError, PipelineError) as exc:
        raise _fail(str(exc)) from exc
    _print_gates(outcome)
    waived = [g.name for g in outcome.gates if g.waived]
    if outcome.promoted:
        line = f"Promoted {version}: wrote {', '.join(p.as_posix() for p in outcome.written)}"
        if waived:
            line += f" with waived gates: {', '.join(waived)}"
        typer.echo(line)
    elif outcome.dry_run and outcome.accepted:
        suffix = f" (with waived gates: {', '.join(waived)})" if waived else ""
        typer.echo(f"Dry run: all gates pass{suffix}; nothing written")
    elif outcome.dry_run:
        failed = ", ".join(g.name for g in outcome.failed)
        typer.echo(f"Dry run: failed gates: {failed}; nothing written")
    else:
        failed = ", ".join(g.name for g in outcome.failed)
        typer.echo(f"PROMOTION REFUSED: failed gates: {failed}")
    typer.echo(f"report: {outcome.report.as_posix()}")
    if not outcome.accepted:
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Add the ``register``, ``golden`` and ``promote`` commands to ``app``."""
    app.command("register")(register_command)
    app.command("golden")(golden_command)
    app.command("promote")(promote_command)
