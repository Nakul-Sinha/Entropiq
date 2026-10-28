"""Entry point of the ``entropiq`` and ``tdp`` commands.

Without a subcommand the interactive shell opens; subcommands (``analyze``, ``history``, ``model`` and the
ones plug-ins add) run once and exit.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from tdp_cli import __version__
from tdp_cli.options import DevModelOption, ModelDirOption, RootOptions, cli_guard
from tdp_cli.registry import CommandRegistry, load_registry

APP_HELP = (
    "Entropiq ranks the files of a Git repository by future maintenance pain. "
    "Run it without a command to open the interactive shell."
)


def _version(value: bool) -> None:
    if value:
        sys.stdout.write(f"entropiq {__version__}\n")
        raise typer.Exit(0)


def run_shell(*, model_dir: Path | None, dev_model: bool) -> int:
    """Open the interactive shell; a bad model choice exits with code 2 before the logo is drawn."""
    from tdp_cli.session import Session
    from tdp_cli.shell import Shell

    with cli_guard():
        session = Session.create(model_dir=model_dir, dev_model=dev_model, interactive=True)
    return Shell(session).run()


def build_app(registry: CommandRegistry | None = None) -> typer.Typer:
    """The Typer app with the root options and every subcommand registered by the command plug-ins."""
    app = typer.Typer(
        name="entropiq",
        help=APP_HELP,
        add_completion=False,
        pretty_exceptions_enable=False,
        no_args_is_help=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.callback(invoke_without_command=True)
    def root(
        ctx: typer.Context,
        version: Annotated[
            bool,
            typer.Option(
                "--version", "-V", callback=_version, is_eager=True, help="Print the version and exit."
            ),
        ] = False,
        model_dir: ModelDirOption = None,
        dev_model: DevModelOption = False,
    ) -> None:
        """Entropiq ranks the files of a Git repository by future maintenance pain."""
        ctx.obj = RootOptions(model_dir=model_dir, dev_model=dev_model)
        if ctx.invoked_subcommand is None:
            raise typer.Exit(run_shell(model_dir=model_dir, dev_model=dev_model))

    commands = registry if registry is not None else load_registry()
    for name, function, help_text, group in commands.cli_entries():
        app.command(name, help=help_text, rich_help_panel=group)(function)
    return app


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and isinstance(stream, io.TextIOWrapper):
            with contextlib.suppress(OSError, ValueError):
                stream.reconfigure(errors="replace")


def main() -> None:
    """Run the command line (console scripts ``entropiq`` and ``tdp``)."""
    _configure_streams()
    from rich.logging import RichHandler

    debug = os.environ.get("ENTROPIQ_DEBUG") == "1"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False, markup=False)],
    )
    build_app()()
