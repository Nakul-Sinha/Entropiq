"""``/help``, ``/clear`` and ``/quit``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.completion import complete_commands
from tdp_cli.errors import ShellExit, UsageError
from tdp_cli.logo import print_header
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.theme import CYAN, clean, make_panel

if TYPE_CHECKING:
    from tdp_cli.session import Session


def help_handler(session: Session, args: Sequence[str]) -> None:
    """``/help`` lists every command; ``/help <command>`` explains one."""
    from tdp_cli.shell import print_command_list

    parsed = parse_args(args, max_positionals=1)
    if not parsed.positionals:
        print_command_list(session)
        return
    name = parsed.positionals[0]
    command = session.registry.get(name)
    if command is None:
        suggestion = session.registry.suggest(name)
        hint = f" Did you mean /{suggestion}?" if suggestion else ""
        raise UsageError(f"there is no command /{clean(name.lstrip('/'))}.{hint}")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("Usage", Text(command.usage_line, style=CYAN))
    grid.add_row("What", Text(command.help))
    if command.details:
        grid.add_row("Details", Text(command.details))
    if command.aliases:
        grid.add_row("Aliases", Text(", ".join(f"/{a}" for a in command.aliases)))
    grid.add_row("Group", Text(command.group))
    if command.cli is not None:
        grid.add_row("Command line", Text(f"tdp {command.cli_name or command.name}"))
    if command.needs_analysis:
        grid.add_row("Needs", Text("a loaded analysis (/analyze or /open first)"))
    session.console.print(make_panel(grid, f"/{command.name}"))


def clear_handler(session: Session, args: Sequence[str]) -> None:
    """``/clear``: clear the screen and redraw the header (the loaded analysis stays)."""
    parse_args(args, max_positionals=0)
    session.console.clear()
    print_header(session.console, model=session.model)


def quit_handler(session: Session, args: Sequence[str]) -> None:
    """``/quit``: leave the shell."""
    raise ShellExit


def register(registry: CommandRegistry) -> None:
    """Register ``/help``, ``/clear`` and ``/quit``."""
    registry.add(
        Command(
            name="help",
            aliases=("h",),
            group="General",
            help="List commands or explain one",
            usage="/help [COMMAND]",
            handler=help_handler,
            complete=complete_commands,
        )
    )
    registry.add(
        Command(
            name="clear",
            aliases=("cls",),
            group="General",
            help="Clear the screen",
            usage="/clear",
            handler=clear_handler,
        )
    )
    registry.add(
        Command(
            name="quit",
            aliases=("exit", "q"),
            group="General",
            help="Leave Entropiq",
            usage="/quit",
            handler=quit_handler,
        )
    )
