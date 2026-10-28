"""Exceptions shared by the commands of the Entropiq terminal app.

Commands raise :class:`CliError` (or :class:`UsageError` for bad input) with a message that is safe to show;
the shell prints it and stays open, the non-interactive commands print it on stderr and exit with
``exit_code``. :class:`ShellExit` asks the interactive shell to close.
"""

from __future__ import annotations


class CliError(Exception):
    """A failure with a user-facing ``message``, an optional ``hint`` and a process ``exit_code``.

    Exit codes: 1 for runtime failures (git, network, disk), 2 for bad input (paths, URLs, options).
    """

    def __init__(self, message: str, *, exit_code: int = 1, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.hint = hint


class UsageError(CliError):
    """The command was called with arguments it does not accept (exit code 2)."""

    def __init__(self, message: str, *, exit_code: int = 2, hint: str | None = None) -> None:
        super().__init__(message, exit_code=exit_code, hint=hint)


class ShellExit(Exception):
    """Raised by ``/quit`` to leave the interactive shell."""
