"""Typer helpers shared by the non-interactive subcommands (``tdp analyze``, ``entropiq history``, ...).

Plug-ins declare parameters with the ``Annotated`` aliases below, get a session with :func:`cli_session`,
an analysis with :func:`cli_analysis`, and wrap their work in :func:`cli_guard` so errors become exit codes:
0 success, 1 runtime failure, 2 bad input, 130 cancelled.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.text import Text

from tdp_cli.errors import CliError
from tdp_cli.theme import error, notice

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis

TargetArgument = Annotated[
    str, typer.Argument(help="Local repository folder or GitHub URL.", show_default=False)
]
RefOption = Annotated[
    str | None,
    typer.Option("--ref", help="Branch, tag or commit to analyse (default HEAD).", show_default=False),
]
TopOption = Annotated[
    int | None,
    typer.Option(
        "--top", min=0, help="Rows to show (0 = all; default from settings, 20).", show_default=False
    ),
]
FreshOption = Annotated[
    bool, typer.Option("--fresh", help="Run the analysis again even if a saved one exists.")
]
ModelDirOption = Annotated[
    Path | None, typer.Option("--model-dir", help="Use the model bundle in this folder.", show_default=False)
]
DevModelOption = Annotated[bool, typer.Option("--dev-model", help="Use the synthetic demo model.")]


@dataclass(frozen=True, slots=True)
class RootOptions:
    """Options given before the subcommand (``entropiq --dev-model history``), kept in ``ctx.obj``."""

    model_dir: Path | None = None
    dev_model: bool = False


@contextmanager
def cli_guard(console: Console | None = None) -> Iterator[None]:
    """Turn :class:`CliError` into an error on stderr and ``typer.Exit(exit_code)``; Ctrl+C exits 130."""
    err = console or Console(stderr=True)
    try:
        yield
    except CliError as exc:
        error(err, exc.message, exc.hint)
        raise typer.Exit(exc.exit_code) from exc
    except KeyboardInterrupt as exc:
        notice(err, "Cancelled.")
        raise typer.Exit(130) from exc


def root_options(ctx: typer.Context | None) -> RootOptions:
    """The root options stored by the main callback (defaults when there are none)."""
    obj = ctx.obj if ctx is not None else None
    return obj if isinstance(obj, RootOptions) else RootOptions()


def cli_session(
    ctx: typer.Context | None, *, model_dir: Path | None = None, dev_model: bool = False
) -> Session:
    """A non-interactive session: output on stdout, progress and notices on stderr.

    Command options win over root options. Exits with code 2 for an unusable model choice.
    """
    from tdp_cli.session import Session

    root = root_options(ctx)
    err = Console(stderr=True)
    with cli_guard(err):
        session = Session.create(
            console=Console(),
            err_console=err,
            model_dir=model_dir if model_dir is not None else root.model_dir,
            dev_model=dev_model or root.dev_model,
            interactive=False,
        )
    if session.model.notice:
        notice(err, session.model.notice)
    return session


def cli_analysis(
    session: Session, target: str, *, ref: str | None = None, fresh: bool = False
) -> LoadedAnalysis:
    """Run (or reuse) the analysis of ``target`` for a subcommand; errors exit through :func:`cli_guard`."""
    from tdp_cli.runner import run_analysis

    with cli_guard(session.err_console):
        loaded, reused = run_analysis(session, target, ref=ref, fresh=fresh)
    if reused:
        when = loaded.meta.analyzed_at.astimezone().strftime("%Y-%m-%d %H:%M")
        session.err_console.print(
            Text(f"Using the saved analysis from {when} (pass --fresh to run it again).", style="dim")
        )
    return loaded
