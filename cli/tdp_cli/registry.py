"""The command registry: every interactive command and Typer subcommand is a plug-in.

Each module in :mod:`tdp_cli.commands` (except ``_private`` helpers) defines
``register(registry: CommandRegistry) -> None`` and calls :meth:`CommandRegistry.add` for its commands.
:func:`load_registry` discovers the modules automatically, so new commands never edit shared files.
"""

from __future__ import annotations

import difflib
import importlib
import pkgutil
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from tdp_cli.session import Session

GROUP_ORDER: Final = ("Analysis", "Explore", "Plan", "Report", "Session", "General")
"""Display order of command groups. Unknown groups sort after "Report" (alphabetically), "General" is last."""

NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One completion candidate: the ``text`` to insert and a short ``meta`` description."""

    text: str
    meta: str = ""


type Handler = Callable[[Session, Sequence[str]], None]
"""Interactive handler: receives the session and the words typed after the command name."""

type ArgCompleter = Callable[[Session, Sequence[str], str], Iterable[Suggestion]]
"""Argument completer: (session, finished arguments, word being typed) -> suggestions."""


@dataclass(frozen=True, slots=True)
class Command:
    """An interactive command (``/name``) and optionally its non-interactive Typer subcommand.

    ``help`` is one line; ``usage`` starts with ``/name``; ``details`` is shown by ``/help name``.
    ``needs_analysis`` makes the shell refuse the command until an analysis is loaded. ``cli`` is a Typer
    command function exposed as ``entropiq <cli_name or name>``.
    """

    name: str
    help: str
    handler: Handler
    group: str = "General"
    aliases: tuple[str, ...] = ()
    usage: str = ""
    details: str = ""
    complete: ArgCompleter | None = None
    needs_analysis: bool = False
    cli: Callable[..., Any] | None = None
    cli_name: str | None = None

    @property
    def usage_line(self) -> str:
        """The usage text, or ``/name`` when none was given."""
        return self.usage or f"/{self.name}"


@dataclass(frozen=True, slots=True)
class _CliEntry:
    name: str
    function: Callable[..., Any]
    help: str
    group: str
    owner: str


def group_rank(group: str) -> tuple[int, str]:
    """Sort key of a group name following :data:`GROUP_ORDER`."""
    if group == "General":
        return (len(GROUP_ORDER) + 1, "")
    if group == "Session":
        return (len(GROUP_ORDER), "")
    if group in GROUP_ORDER:
        return (GROUP_ORDER.index(group), "")
    return (GROUP_ORDER.index("Report") + 1, group.lower())


def _check_name(name: str, what: str) -> None:
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(
            f"invalid {what} {name!r}: use 1 to 32 lower-case letters, digits or '-', starting with a letter"
        )


def _normalise(name: str) -> str:
    return name.strip().lstrip("/").lower()


class CommandRegistry:
    """Holds the commands of the app, looked up by name or alias (case-insensitive, ``/`` optional)."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._lookup: dict[str, str] = {}
        self._cli: dict[str, _CliEntry] = {}

    def add(self, command: Command) -> None:
        """Register ``command``.

        Raises ``ValueError`` for an invalid name or alias, empty or multi-line help, or a name, alias or
        CLI name that another command already uses.
        """
        _check_name(command.name, "command name")
        for alias in command.aliases:
            _check_name(alias, "alias")
        if not command.help.strip() or "\n" in command.help:
            raise ValueError(f"command {command.name!r} needs a one-line help text")
        words = (command.name, *command.aliases)
        if len(set(words)) != len(words):
            raise ValueError(f"command {command.name!r} repeats a name in its aliases")
        for word in words:
            if word in self._lookup:
                raise ValueError(f"/{word} is already taken by /{self._lookup[word]}")
        cli_name = command.cli_name or command.name
        if command.cli is not None:
            _check_name(cli_name, "CLI name")
            self._check_cli_free(cli_name)
        self._commands[command.name] = command
        for word in words:
            self._lookup[word] = command.name
        if command.cli is not None:
            self._cli[cli_name] = _CliEntry(cli_name, command.cli, command.help, command.group, command.name)

    def add_cli(self, name: str, function: Callable[..., Any], *, help: str, group: str = "General") -> None:
        """Register a non-interactive Typer subcommand ``name`` that has no interactive command.

        Raises ``ValueError`` for an invalid or taken name, or empty or multi-line help.
        """
        _check_name(name, "CLI name")
        if not help.strip() or "\n" in help:
            raise ValueError(f"CLI command {name!r} needs a one-line help text")
        self._check_cli_free(name)
        self._cli[name] = _CliEntry(name, function, help, group, "")

    def _check_cli_free(self, name: str) -> None:
        if name in self._cli:
            owner = self._cli[name].owner
            by = f" by /{owner}" if owner else ""
            raise ValueError(f"CLI command {name!r} is already taken{by}")

    def get(self, name: str) -> Command | None:
        """The command named ``name`` or one of its aliases (``"/Analyze"`` works), else None."""
        key = self._lookup.get(_normalise(name))
        return self._commands[key] if key is not None else None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __len__(self) -> int:
        return len(self._commands)

    def commands(self) -> list[Command]:
        """Every command, ordered by group (see :data:`GROUP_ORDER`) then name."""
        return sorted(self._commands.values(), key=lambda c: (group_rank(c.group), c.name))

    def groups(self) -> list[tuple[str, list[Command]]]:
        """Commands grouped for display: ``[(group, commands), ...]`` in group order."""
        out: list[tuple[str, list[Command]]] = []
        for command in self.commands():
            if out and out[-1][0] == command.group:
                out[-1][1].append(command)
            else:
                out.append((command.group, [command]))
        return out

    def names(self, *, aliases: bool = True) -> list[str]:
        """Sorted command names (and aliases unless ``aliases=False``), without the slash."""
        return sorted(self._lookup) if aliases else sorted(self._commands)

    def suggest(self, word: str) -> str | None:
        """The canonical name of the command closest to ``word`` (a typo), or None when nothing is close."""
        matches = difflib.get_close_matches(_normalise(word), self.names(), n=1, cutoff=0.6)
        return self._lookup[matches[0]] if matches else None

    def cli_entries(self) -> list[tuple[str, Callable[..., Any], str, str]]:
        """Typer subcommands as ``(name, function, help, group)``, sorted by name."""
        return [
            (e.name, e.function, e.help, e.group) for e in sorted(self._cli.values(), key=lambda e: e.name)
        ]


def load_registry(package: str = "tdp_cli.commands") -> CommandRegistry:
    """Import every public module of ``package`` (sorted by name) and call its ``register(registry)``.

    Modules whose name starts with ``_`` are skipped (helpers). Raises ``RuntimeError`` when a public module
    has no callable ``register``; ``ValueError`` from duplicate names propagates.
    """
    registry = CommandRegistry()
    pkg = importlib.import_module(package)
    names = sorted(info.name for info in pkgutil.iter_modules(pkg.__path__) if not info.name.startswith("_"))
    for name in names:
        module = importlib.import_module(f"{package}.{name}")
        register = getattr(module, "register", None)
        if not callable(register):
            raise RuntimeError(
                f"{package}.{name} has no register(registry) function; helper modules must start with '_'"
            )
        register(registry)
    return registry


def command_words(commands: Iterable[Command]) -> list[str]:
    """Every name and alias of ``commands``."""
    return [w for c in commands for w in (c.name, *c.aliases)]
