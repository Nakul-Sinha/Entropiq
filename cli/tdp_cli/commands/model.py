"""``/model`` and ``entropiq model``: which model scores the files, and how it was trained."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer
from rich.table import Table
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.options import DevModelOption, ModelDirOption, cli_guard, cli_session
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.theme import CYAN, clean, make_panel

if TYPE_CHECKING:
    from tdp_cli.model import ModelInfo
    from tdp_cli.session import Session


def _info(session: Session) -> ModelInfo:
    from tdp_cli.model import read_model_info

    if session.model.source == "dev" and not (session.model.path / "metadata.json").is_file():
        session.bundle()
    return read_model_info(session.model)


def _info_json(info: ModelInfo) -> dict[str, Any]:
    return {
        "version": info.version,
        "kind": info.kind_label,
        "demo": info.demo,
        "source": info.source,
        "reason": info.reason,
        "path": str(info.path),
        "created_at": info.created_at,
        "training_rows": info.training_rows,
        "metrics": info.metrics,
        "n_features": info.n_features,
        "extractor_version": info.extractor_version,
        "code_git_sha": info.code_git_sha,
    }


def _render(session: Session, info: ModelInfo) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")
    kind = Text(
        info.kind_label, style="yellow" if info.demo else "green" if info.source == "production" else CYAN
    )
    grid.add_row("Version", Text(clean(info.version), style="bold"))
    grid.add_row("Kind", kind)
    grid.add_row("Chosen by", Text(clean(info.reason)))
    grid.add_row("Folder", Text(clean(str(info.path))))
    grid.add_row("Created", Text(info.created_at or "unknown"))
    grid.add_row(
        "Training rows", Text(f"{info.training_rows:,}" if info.training_rows is not None else "unknown")
    )
    extractor = f" (extractor {info.extractor_version})" if info.extractor_version else ""
    grid.add_row("Features", Text(f"{info.n_features} inputs{extractor}"))
    if info.code_git_sha:
        grid.add_row("Code commit", Text(clean(info.code_git_sha[:12])))
    if info.metrics:
        metrics = Text()
        for i, (key, value) in enumerate(info.metrics.items()):
            if i:
                metrics.append("\n")
            metrics.append(f"{clean(key)} ", style="dim")
            metrics.append(f"{value:.4g}")
        grid.add_row("Metrics", metrics)
    else:
        grid.add_row("Metrics", Text("none recorded", style="dim"))
    session.console.print(make_panel(grid, "Model"))
    if info.demo:
        session.console.print(
            Text(
                "A demo model is trained on synthetic data: rankings show how Entropiq works, not real risk.",
                style="yellow",
            )
        )


def model_handler(session: Session, args: Sequence[str]) -> None:
    """``/model``: version, kind, training metrics and feature count of the model in use."""
    parse_args(args, max_positionals=0)
    _render(session, _info(session))


def model_cli(
    ctx: typer.Context,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = ("table"),
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Show the model in use and its training metrics."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    with cli_guard(session.err_console):
        info = _info(session)
    if output_format == "json":
        sys.stdout.write(json.dumps(_info_json(info), indent=2) + "\n")
    else:
        _render(session, info)


def register(registry: CommandRegistry) -> None:
    """Register ``/model`` and ``entropiq model``."""
    registry.add(
        Command(
            name="model",
            group="Session",
            help="Show the model in use and its training metrics",
            usage="/model",
            details=(
                "The model is chosen at startup: --model-dir, then TDP_MODEL_DIR, then the published "
                "production model, otherwise the demo model (trained on synthetic data)."
            ),
            handler=model_handler,
            cli=model_cli,
        )
    )
