r"""Argument splitting and parsing for interactive commands.

Interactive input is split like a POSIX shell but with backslash escapes disabled, so Windows paths such as
``C:\Users\me\repo`` survive unchanged; quote arguments that contain spaces. Only long options
(``--ref v1`` or ``--ref=v1``) are supported, and ``--`` ends the options.
"""

from __future__ import annotations

import shlex
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from tdp_cli.errors import UsageError


def _lexer(text: str) -> shlex.shlex:
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    return lexer


def split_args(text: str) -> list[str]:
    """Split ``text`` into words (quotes group words, backslashes are literal).

    Raises :class:`UsageError` when a quote is not closed.
    """
    try:
        return list(_lexer(text))
    except ValueError as exc:
        raise UsageError("unbalanced quotes") from exc


def split_args_lenient(text: str) -> list[str]:
    """Like :func:`split_args` but never raises: a dangling quote is closed at the end (for completion)."""
    try:
        return list(_lexer(text))
    except ValueError:
        pass
    for quote in ('"', "'"):
        try:
            return list(_lexer(text + quote))
        except ValueError:
            continue
    return text.split()


def quote_arg(text: str) -> str:
    """Return ``text`` wrapped in double quotes when it contains whitespace (or is empty), else unchanged."""
    if text == "" or any(c.isspace() for c in text):
        return f'"{text}"'
    return text


def _long(name: str) -> str:
    return name if name.startswith("--") else f"--{name}"


@dataclass(frozen=True, slots=True)
class ParsedArgs:
    """Positional arguments in order and options by long name (``"--ref"``); flags map to ``True``."""

    positionals: tuple[str, ...] = ()
    options: dict[str, str | bool] = field(default_factory=dict)

    def flag(self, name: str) -> bool:
        """True when the flag ``name`` (``"--fresh"`` or ``"fresh"``) was given."""
        return self.options.get(_long(name)) is True

    def value(self, name: str) -> str | None:
        """The value of option ``name`` (``"--ref"`` or ``"ref"``), or None when it was not given."""
        value = self.options.get(_long(name))
        return value if isinstance(value, str) else None


def parse_args(
    args: Sequence[str],
    *,
    values: Collection[str] = (),
    flags: Collection[str] = (),
    max_positionals: int | None = None,
) -> ParsedArgs:
    """Parse ``args`` against the options a command accepts.

    ``values`` are options that take a value, ``flags`` are options without one (long names with or without
    the leading ``--``). Raises :class:`UsageError` for an unknown option, a missing value, a value given to a
    flag, or more than ``max_positionals`` positional arguments. A later repeat of an option wins.
    """
    value_names = {_long(v) for v in values}
    flag_names = {_long(f) for f in flags}
    known = sorted(value_names | flag_names)
    positionals: list[str] = []
    options: dict[str, str | bool] = {}
    only_positionals = False
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if only_positionals or not arg.startswith("-") or arg == "-":
            positionals.append(arg)
            continue
        if arg == "--":
            only_positionals = True
            continue
        name, eq, inline = arg.partition("=")
        if name in value_names:
            if eq:
                options[name] = inline
            elif i < len(args) and not args[i].startswith("--"):
                options[name] = args[i]
                i += 1
            else:
                raise UsageError(f"{name} needs a value")
        elif name in flag_names:
            if eq:
                raise UsageError(f"{name} does not take a value")
            options[name] = True
        else:
            valid = f" (valid: {', '.join(known)})" if known else " (this command takes no options)"
            raise UsageError(f"unknown option {name}{valid}")
    if max_positionals is not None and len(positionals) > max_positionals:
        if max_positionals == 0:
            raise UsageError("this command takes no arguments")
        raise UsageError(f"too many arguments (at most {max_positionals})")
    return ParsedArgs(tuple(positionals), options)
