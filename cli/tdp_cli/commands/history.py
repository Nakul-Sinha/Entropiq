"""``/history``, ``/open`` and ``entropiq history``: list and reopen saved analyses."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.completion import complete_history
from tdp_cli.errors import UsageError
from tdp_cli.options import cli_guard, cli_session
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.theme import activity, clean, notice, short_sha, success

if TYPE_CHECKING:
    from tdp_cli.session import Session

EMPTY = "No saved analyses yet. Run /analyze to create one."


def history_handler(session: Session, args: Sequence[str]) -> None:
    """``/history``: the saved analyses, newest first, numbered for ``/open``."""
    from tdp_cli.output import history_table

    parse_args(args, max_positionals=0)
    entries = session.store.entries()
    if not entries:
        session.console.print(Text(EMPTY, style="dim"))
        return
    session.console.print(history_table(entries, glyphs=session.glyphs))
    session.console.print(Text("Type /open <number> to reopen one.", style="dim"))


def open_handler(session: Session, args: Sequence[str]) -> None:
    """``/open <number or sha>``: load a saved analysis without running it again."""
    from tdp_cli.output import summary_card

    parsed = parse_args(args, max_positionals=1)
    if not parsed.positionals:
        raise UsageError("say which analysis to open (a number from /history or a commit sha)")
    meta = session.store.find(parsed.positionals[0])
    with activity(session.err_console, "Loading the saved analysis"):
        loaded = session.store.load(meta)
    session.current = loaded
    success(session.console, f"Opened {clean(meta.repo)} @ {short_sha(meta.sha)}")
    if session.model.version is not None and meta.model_version != session.model.version:
        notice(
            session.console,
            f"This analysis used model {clean(meta.model_version)}; "
            f"the session uses {session.model.version}.",
        )
    session.console.print(summary_card(loaded, top=5, glyphs=session.glyphs, reused=True))


def history_cli(
    ctx: typer.Context,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = ("table"),
    limit: Annotated[int, typer.Option("--limit", min=0, help="Analyses to list (0 = all).")] = 20,
) -> None:
    """List the saved analyses, newest first."""
    from tdp_cli.output import history_table

    session = cli_session(ctx)
    with cli_guard(session.err_console):
        entries = session.store.entries()
        shown = entries[:limit] if limit else entries
        if output_format == "json":
            data = [{"number": i, **m.to_dict()} for i, m in enumerate(shown, start=1)]
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
            return
        if not shown:
            session.console.print(Text("No saved analyses yet. Run tdp analyze to create one.", style="dim"))
            return
        session.console.print(history_table(shown, glyphs=session.glyphs))


def register(registry: CommandRegistry) -> None:
    """Register ``/history``, ``/open`` and ``entropiq history``."""
    registry.add(
        Command(
            name="history",
            group="Analysis",
            help="List saved analyses",
            usage="/history",
            details="Every analysis is saved under the Entropiq folder; /open reloads one instantly.",
            handler=history_handler,
            cli=history_cli,
        )
    )
    registry.add(
        Command(
            name="open",
            aliases=("load",),
            group="Analysis",
            help="Reopen a saved analysis without running it again",
            usage="/open <number or sha>",
            details="Use the number shown by /history or at least 4 characters of the commit sha.",
            handler=open_handler,
            complete=complete_history,
        )
    )
