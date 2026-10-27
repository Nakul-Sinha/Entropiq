"""``tdp-ml satd``: download the SATD comment dataset and train the shipped comment classifier.

The training module is imported inside the commands so ``tdp-ml --help`` stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from tdp_core.features.satd_model import DEFAULT_MODEL_PATH

F1_TARGET = 0.6


def _setup_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(1)


def download(
    dest: Annotated[
        Path | None, typer.Option(help="Folder for the zip and CSV (default: $TDP_DATA_DIR/satd).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Download again even if a verified zip is present.")] = False,
) -> None:
    """Download the Maldonado, Shihab and Tsantalis SATD dataset (pinned and hash-checked) and extract it."""
    from tdp_ml.satd import train

    _setup_logging()
    try:
        csv_path = train.download_dataset(dest or train.default_dataset_dir(), force=force)
        df = train.load_dataset(csv_path)
    except (OSError, ValueError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"dataset: {csv_path}")
    typer.echo(f"rows: {len(df)}  debt rows: {int(df['is_debt'].sum())}  projects: {df['project'].nunique()}")


def _print_summary(report: dict[str, Any], out: Path, report_path: Path) -> None:
    lopo = report["lopo"]
    typer.echo("Leave-one-project-out (threshold 0.5):")
    for key, row in lopo["by_c"].items():
        typer.echo(
            f"  C={key:<6} macro F1 {row['macro_f1']:.3f}  pooled F1 {row['pooled']['f1']:.3f}  "
            f"macro P {row['macro_precision']:.3f}  macro R {row['macro_recall']:.3f}"
        )
    verdict = "meets" if lopo["macro_f1"] >= F1_TARGET else "misses"
    typer.echo(
        f"selected C={lopo['selected_c']}: macro F1 {lopo['macro_f1']:.3f} "
        f"({verdict} the {F1_TARGET} target), pooled F1 {lopo['pooled_f1']:.3f}"
    )
    base = report["keyword_baseline"]
    typer.echo(f"keyword baseline: macro F1 {base['macro_f1']:.3f}, pooled F1 {base['pooled']['f1']:.3f}")
    cand = report["candidates"]
    typer.echo(
        f"keyword candidates: {cand['n']} comments, {cand['debt']} debt (coverage {cand['coverage']:.3f}); "
        f"classifier F1 {cand['classifier']['f1']:.3f} vs all-debt F1 {cand['all_candidates_debt']['f1']:.3f}"
    )
    export = report["export"]
    capped, uncapped = report["final"]["vocabulary_size"], report["vocabulary_size_uncapped"]
    typer.echo(
        f"vocabulary {capped} terms (uncapped {uncapped}), {export['bytes']:,} bytes, "
        f"parity max diff {export['parity_max_abs_diff']:.2e}"
    )
    selected = lopo["by_c"][repr(float(lopo["selected_c"]))]["per_project"]
    projects = report["dataset"]["projects"]
    width = max([len("project"), *(len(p["name"]) for p in projects)]) + 2
    typer.echo(f"  {'project':<{width}}{'rows':>7}{'debt':>6}{'P':>7}{'R':>7}{'F1':>7}")
    for project in projects:
        row = selected[project["name"]]
        typer.echo(
            f"  {project['name']:<{width}}{project['rows']:>7}{project['debt_rows']:>6}"
            f"{row['precision']:>7.3f}{row['recall']:>7.3f}{row['f1']:>7.3f}"
        )
    typer.echo(f"model: {out}")
    typer.echo(f"report: {report_path}")


def train_command(
    csv: Annotated[
        Path | None,
        typer.Option(help="Dataset CSV (default: $TDP_DATA_DIR/satd/technical_debt_dataset.csv)."),
    ] = None,
    out: Annotated[Path, typer.Option(help="Where to write the model JSON.")] = DEFAULT_MODEL_PATH,
    report: Annotated[
        Path | None, typer.Option(help="Report JSON (default: $TDP_RESULTS_DIR/satd/lopo_report.json).")
    ] = None,
    c: Annotated[
        list[float] | None, typer.Option("--c", help="C value to try (repeatable; default 0.1, 1, 10).")
    ] = None,
    max_features: Annotated[int, typer.Option(help="Vocabulary cap; 0 means no cap.")] = 40_000,
    max_bytes: Annotated[int, typer.Option(help="Largest model file allowed, in bytes.")] = 2_000_000,
    jobs: Annotated[int, typer.Option(help="Thread cap for numeric libraries.")] = 4,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    """Evaluate by leave-one-project-out, pick C, train on all comments and export the JSON weights."""
    from tdp_ml.satd import train

    _setup_logging()
    csv_path = csv or train.default_dataset_dir() / train.CSV_NAME
    if not csv_path.is_file():
        raise _fail(FileNotFoundError(f"{csv_path} not found; run `uv run tdp-ml satd download` first"))
    report_path = report or train.default_report_path()
    try:
        result = train.run_training(
            csv_path,
            out,
            report_path,
            c_grid=tuple(c) if c else train.C_GRID,
            max_features=max_features if max_features > 0 else None,
            max_bytes=max_bytes,
            jobs=jobs,
            seed=seed,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise _fail(exc) from exc
    _print_summary(result, out, report_path)


def register(app: typer.Typer) -> None:
    """Add the ``satd`` command group (``download``, ``train``) to ``app``."""
    satd_app = typer.Typer(
        help="SATD comment classifier: dataset download and training.", no_args_is_help=True
    )
    satd_app.command("download")(download)
    satd_app.command("train")(train_command)
    app.add_typer(satd_app, name="satd")
