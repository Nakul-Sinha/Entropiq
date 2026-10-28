"""``/settings``: show or change the saved settings (plan budget, rows shown, parallel jobs)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING

from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.completion import complete_pref_keys
from tdp_cli.errors import UsageError
from tdp_cli.home import PREF_HELP, Prefs, set_pref
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.theme import make_table, success

if TYPE_CHECKING:
    from tdp_cli.session import Session


def _show(session: Session) -> None:
    table = make_table("Settings")
    table.add_column("Key", style="bold", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    table.add_column("Default", justify="right", style="dim", no_wrap=True)
    table.add_column("Meaning")
    current, defaults = asdict(session.prefs), asdict(Prefs())
    for key, text in PREF_HELP.items():
        table.add_row(key, f"{current[key]:g}", f"{defaults[key]:g}", text)
    session.console.print(table)
    session.console.print(Text("Change one with /settings <key> <value>, or /settings reset.", style="dim"))


def settings_handler(session: Session, args: Sequence[str]) -> None:
    """``/settings``, ``/settings reset`` or ``/settings <key> <value>``."""
    parsed = parse_args(args, max_positionals=2)
    words = parsed.positionals
    if not words:
        _show(session)
        return
    if len(words) == 1:
        if words[0].lower() == "reset":
            session.set_prefs(Prefs())
            success(session.console, "Settings restored to their defaults.")
            return
        raise UsageError(f"give a value: /settings {words[0]} <value>")
    prefs = set_pref(session.prefs, words[0], words[1])
    session.set_prefs(prefs)
    key = words[0].strip().lower().replace("-", "_")
    success(session.console, f"{key} is now {asdict(prefs)[key]:g}.")


def register(registry: CommandRegistry) -> None:
    """Register ``/settings``."""
    registry.add(
        Command(
            name="settings",
            group="Session",
            help="Show or change settings",
            usage="/settings [KEY VALUE or reset]",
            details=(
                "budget_hours is the time available for fixes when a plan is made, top is the number of rows "
                "tdp analyze shows, jobs is the number of parallel workers. Settings are saved in "
                "settings.json in the Entropiq folder."
            ),
            handler=settings_handler,
            complete=complete_pref_keys,
        )
    )
