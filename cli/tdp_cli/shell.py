"""The interactive shell: a prompt_toolkit input line with completion, history and a bottom toolbar.

Commands run in the main thread, so Ctrl+C raises ``KeyboardInterrupt`` inside the running command, which
:func:`dispatch` turns into "Cancelled." without leaving the shell. Ctrl+D or ``/quit`` leaves.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Final

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from rich.table import Table
from rich.text import Text

from tdp_cli.args import quote_arg, split_args, split_args_lenient
from tdp_cli.errors import CliError, ShellExit, UsageError
from tdp_cli.logo import print_welcome
from tdp_cli.session import NO_ANALYSIS
from tdp_cli.theme import CYAN, VIOLET, Glyphs, clean, error, notice, short_sha

if TYPE_CHECKING:
    from prompt_toolkit.input import Input
    from prompt_toolkit.output import Output

    from tdp_cli.session import Session

NO_ANALYSIS_HINT: Final = NO_ANALYSIS
ENV_DEBUG: Final = "ENTROPIQ_DEBUG"

PROMPT_STYLE: Final[dict[str, str]] = {
    "prompt.name": f"bold {VIOLET}",
    "prompt.chevron": f"bold {CYAN}",
    "bottom-toolbar": "noreverse bg:#1f2937 #9ca3af",
    "bottom-toolbar.key": "bg:#1f2937 #e5e7eb bold",
    "bottom-toolbar.repo": f"bg:#1f2937 {CYAN}",
    "bottom-toolbar.warn": "bg:#1f2937 #eab308",
    "completion-menu.completion.current": f"bg:{VIOLET} #ffffff",
    "completion-menu.meta.completion.current": f"bg:{VIOLET} #ffffff",
}


class ShellCompleter(Completer):
    """Completes ``/command`` names, then the arguments of the command through its ``complete`` hook."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield completions for the text before the cursor (only lines starting with ``/``)."""
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        head, space, rest = text[1:].partition(" ")
        registry = self.session.registry
        if not space:
            prefix = head.lower()
            seen: set[str] = set()
            for candidate in registry.commands():
                names = (candidate.name, *candidate.aliases)
                if candidate.name not in seen and any(w.startswith(prefix) for w in names):
                    seen.add(candidate.name)
                    yield Completion(
                        f"/{candidate.name}",
                        start_position=-len(text),
                        display=f"/{candidate.name}",
                        display_meta=candidate.help,
                    )
            return
        command = registry.get(head)
        if command is None or command.complete is None:
            return
        words = split_args_lenient(rest)
        if rest == "" or rest[-1].isspace():
            finished, raw_word, word = words, "", ""
        else:
            finished = words[:-1]
            raw_word = _last_raw_word(rest)
            word = words[-1] if words else ""
        try:
            suggestions = list(command.complete(self.session, finished, word))
        except Exception:  # a completer bug must never break typing
            return
        for suggestion in suggestions:
            yield Completion(
                quote_arg(suggestion.text),
                start_position=-len(raw_word),
                display=suggestion.text,
                display_meta=suggestion.meta,
            )


def _last_raw_word(text: str) -> str:
    """The raw text of the word being typed at the end of ``text`` (a leading quote included)."""
    quote: str | None = None
    start = 0
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char.isspace():
            start = i + 1
    return text[start:]


def toolbar_fragments(session: Session) -> StyleAndTextTuples:
    """The bottom toolbar: key hints, the loaded repository and a demo-model marker."""
    keys = [
        ("/", "commands"),
        ("Tab", "complete"),
        ("Up/Down", "history"),
        ("Ctrl+C", "cancel"),
        ("Ctrl+D", "exit"),
    ]
    out: StyleAndTextTuples = [("class:bottom-toolbar", " ")]
    for key, label in keys:
        out += [("class:bottom-toolbar.key", key), ("class:bottom-toolbar", f" {label}  ")]
    current = session.current
    if current is not None:
        repo = f"{clean(current.meta.repo)} @ {short_sha(current.meta.sha)}"
        out.append(("class:bottom-toolbar.repo", f" {repo} "))
    else:
        out.append(("class:bottom-toolbar", " no repository loaded "))
    if session.model.demo:
        out.append(("class:bottom-toolbar.warn", " demo model "))
    return out


def prompt_fragments(glyphs: Glyphs) -> StyleAndTextTuples:
    """The prompt: the name and a chevron (``entropiq > `` on terminals without Unicode)."""
    return [("class:prompt.name", "entropiq"), ("class:prompt.chevron", f" {glyphs.chevron} ")]


def print_command_list(session: Session) -> None:
    """Print every command grouped, with aliases and help."""
    console = session.console
    every = session.registry.commands()
    name_width = max((len(c.name) + 3 for c in every), default=0)
    alias_width = max((len(", ".join(f"/{a}" for a in c.aliases)) for c in every), default=0)
    for group, commands in session.registry.groups():
        console.print(Text(group, style=f"bold {VIOLET}"))
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True, min_width=name_width)
        table.add_column(no_wrap=True, style="dim", min_width=alias_width)
        table.add_column()
        for command in commands:
            aliases = ", ".join(f"/{a}" for a in command.aliases)
            table.add_row(Text(f"  /{command.name}", style=CYAN), aliases, command.help)
        console.print(table)
    console.print(
        Text("Type /help <command> for details. Tab completes commands and arguments.", style="dim")
    )


def dispatch(session: Session, line: str) -> bool:
    """Run one input line. Returns False when the shell should close (``/quit``), True otherwise.

    Every error is printed and swallowed, so the shell survives bad input, failed commands, Ctrl+C and bugs.
    """
    console = session.console
    text = line.strip()
    if not text:
        return True
    if text == "/":
        print_command_list(session)
        return True
    try:
        words = split_args(text)
    except UsageError as exc:
        error(console, exc.message)
        return True
    if not words:
        return True
    first = words[0]
    if first.startswith("/"):
        name = first[1:]
        command = session.registry.get(name)
        if command is None:
            suggestion = session.registry.suggest(name) if name else None
            hint = f" Did you mean /{suggestion}?" if suggestion else ""
            error(console, f"Unknown command /{clean(name)}.{hint} Type / to list every command.")
            return True
    else:
        command = session.registry.get(first)
        if command is None:
            suggestion = session.registry.suggest(first)
            hint = f" Did you mean /{suggestion}?" if suggestion else ""
            error(console, f"Commands start with /.{hint} Type / to list every command.")
            return True
    if command.needs_analysis and session.current is None:
        notice(console, NO_ANALYSIS_HINT)
        return True
    try:
        command.handler(session, words[1:])
    except ShellExit:
        return False
    except UsageError as exc:
        error(console, exc.message, exc.hint)
        console.print(Text(f"Usage: {command.usage_line}", style="dim"))
    except CliError as exc:
        error(console, exc.message, exc.hint)
    except KeyboardInterrupt:
        notice(console, "Cancelled.")
    except Exception as exc:
        error(console, f"Unexpected error: {type(exc).__name__}: {clean(str(exc))}")
        if os.environ.get(ENV_DEBUG) == "1":
            console.print_exception()
        else:
            console.print(Text(f"Set {ENV_DEBUG}=1 to see the traceback.", style="dim"))
    return True


class Shell:
    """The interactive loop. ``input``/``output`` inject prompt_toolkit I/O (tests); ``plain`` forces or
    disables the line-by-line mode used when stdin is not a terminal."""

    def __init__(
        self,
        session: Session,
        *,
        input: Input | None = None,
        output: Output | None = None,
        plain: bool | None = None,
    ) -> None:
        self.session = session
        self._input = input
        self._output = output
        if plain is None:
            plain = input is None and not _stdin_is_tty()
        self.plain = plain

    def run(self) -> int:
        """Show the welcome screen and read commands until ``/quit`` or end of input. Returns 0."""
        session = self.session
        print_welcome(session.console, model=session.model)
        if session.model.notice:
            notice(session.console, session.model.notice)
        prompt = None if self.plain else self._make_prompt()
        if prompt is None:
            self._run_plain()
        else:
            self._run_prompt(prompt)
        session.console.print(Text("Bye.", style="dim"))
        return 0

    def _make_prompt(self) -> PromptSession[str] | None:
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.styles import Style

        session = self.session
        history: FileHistory | InMemoryHistory
        try:
            session.home.history_file.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(session.home.history_file))
        except OSError:
            history = InMemoryHistory()
        try:
            prompt: PromptSession[str] = PromptSession(
                message=prompt_fragments(session.glyphs),
                history=history,
                completer=ShellCompleter(session),
                complete_while_typing=True,
                auto_suggest=AutoSuggestFromHistory(),
                bottom_toolbar=lambda: toolbar_fragments(session),
                style=Style.from_dict(PROMPT_STYLE),
                reserve_space_for_menu=8,
                input=self._input,
                output=self._output,
            )
        except Exception as exc:  # no usable terminal (for example a redirected Windows console)
            notice(session.console, f"Falling back to plain input ({type(exc).__name__}).")
            return None
        return prompt

    def _run_prompt(self, prompt: PromptSession[str]) -> None:
        while True:
            try:
                line = prompt.prompt()
            except KeyboardInterrupt:
                continue
            except EOFError:
                return
            if not dispatch(self.session, line):
                return

    def _run_plain(self) -> None:
        console = self.session.console
        chevron = self.session.glyphs.chevron
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                return
            if not line:
                return
            console.print(Text(f"entropiq {chevron} {clean(line.rstrip())}", style="bold"))
            if not dispatch(self.session, line):
                return


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
