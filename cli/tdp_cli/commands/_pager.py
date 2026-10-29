"""The ``/files`` pager: pages of the ranked table with a small sub-prompt of its own.

Keys: ``n`` next page, ``p`` previous page, a rank number opens that file, Enter or ``q`` goes back, and a
line starting with ``/`` leaves the pager and runs that command. On a terminal the sub-prompt uses
prompt_toolkit; otherwise (pipes, CI, tests) it reads plain lines from stdin and echoes them like the shell.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

from rich.text import Text

from tdp_cli.commands._detail import file_detail
from tdp_cli.commands._views import RankedFile, find_file, ranked_table, view_glyphs
from tdp_cli.errors import CliError, ShellExit
from tdp_cli.theme import clean, error

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis

type LineReader = Callable[[str], str | None]
"""Reads one line for the prompt label; None means the input ended or was cancelled."""

PAGER_LABEL: Final = "files"
_NEXT: Final = frozenset({"n", "next"})
_PREVIOUS: Final = frozenset({"p", "prev", "previous"})
_LEAVE: Final = frozenset({"", "q", "quit", "back"})


def stdin_is_tty() -> bool:
    """True when stdin is an interactive terminal."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _prompt_line(session: Session, label: str) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    from tdp_cli.shell import PROMPT_STYLE

    prompt: PromptSession[str] = PromptSession(
        message=[("class:prompt.name", label), ("class:prompt.chevron", f" {session.glyphs.chevron} ")],
        style=Style.from_dict(PROMPT_STYLE),
    )
    return prompt.prompt()


def read_line(session: Session, label: str) -> str | None:
    """One line typed at the ``label`` sub-prompt, or None when the input ended or Ctrl+C/D was pressed."""
    if stdin_is_tty():
        try:
            return _prompt_line(session, label)
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:  # no usable terminal after all: read a plain line instead
            pass
    try:
        line = sys.stdin.readline()
    except (OSError, ValueError, KeyboardInterrupt):
        return None
    if not line:
        return None
    line = line.rstrip("\r\n")
    session.console.print(Text(f"{label} {session.glyphs.chevron} {clean(line)}", style="bold"))
    return line


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def run_files_pager(
    session: Session,
    loaded: LoadedAnalysis,
    rows: Sequence[RankedFile],
    *,
    page_size: int,
    title: str,
    read: LineReader | None = None,
) -> None:
    """Show ``rows`` ``page_size`` at a time and react to the keys typed at the ``files`` sub-prompt.

    Numbers are priority ranks (the ``#`` column), so they work with any filter or sort. Raises
    :class:`ShellExit` when a command typed here asks the shell to close.
    """
    reader: LineReader = read if read is not None else (lambda label: read_line(session, label))
    console = session.console
    glyphs = session.glyphs
    view = view_glyphs(console)
    size = max(1, page_size)
    pages = max(1, math.ceil(len(rows) / size))
    page = 0

    def show() -> None:
        start = page * size
        chunk = rows[start : start + size]
        console.print(Text(title, style="bold"))
        console.print(ranked_table(chunk, glyphs=glyphs))
        if pages > 1:
            console.print(
                _dim(
                    f"Page {page + 1} of {pages} (files {start + 1} to {start + len(chunk)} of {len(rows)}). "
                    "n next, p previous, a rank opens that file, Enter or q goes back."
                )
            )
        else:
            noun = "file" if len(rows) == 1 else "files"
            console.print(
                _dim(f"{len(rows)} {noun}. Type a rank to open that file, or press Enter to go back.")
            )

    show()
    while True:
        line = reader(PAGER_LABEL)
        if line is None:
            return
        text = line.strip()
        word = text.lower()
        if word in _LEAVE:
            return
        if text.startswith("/"):
            from tdp_cli.shell import dispatch

            if not dispatch(session, text):
                raise ShellExit
            return
        if word in _NEXT:
            if page + 1 >= pages:
                console.print(_dim("This is the last page."))
            else:
                page += 1
                show()
            continue
        if word in _PREVIOUS:
            if page == 0:
                console.print(_dim("This is the first page."))
            else:
                page -= 1
                show()
            continue
        if word.lstrip("#").isdigit():
            try:
                item = find_file(loaded.result, text)
            except CliError as exc:
                error(console, exc.message, exc.hint)
                continue
            console.print(file_detail(loaded, item, glyphs=glyphs, view=view))
            console.print(_dim("Back in the file list: n, p, a rank, or Enter to leave."))
            continue
        console.print(_dim("Type n, p, a rank number, or press Enter to go back."))
