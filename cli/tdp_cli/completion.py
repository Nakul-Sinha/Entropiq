"""Argument completers for commands (pure functions; the shell turns them into prompt_toolkit completions).

A completer is called as ``completer(session, finished_args, word)`` and yields
:class:`~tdp_cli.registry.Suggestion` objects whose ``text`` replaces ``word``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from tdp_cli.registry import ArgCompleter, Suggestion
from tdp_cli.theme import short_sha

if TYPE_CHECKING:
    from tdp_cli.session import Session

MAX_PATH_SUGGESTIONS = 50


def complete_commands(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Command names (without the slash) starting with ``word``; only for the first argument."""
    if args:
        return
    prefix = word.lstrip("/").lower()
    for command in session.registry.commands():
        if command.name.startswith(prefix):
            yield Suggestion(command.name, command.help)


def complete_history(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Numbers and short shas of saved analyses; only for the first argument."""
    if args:
        return
    entries = session.store.entries()
    lowered = word.lower()
    for number, meta in enumerate(entries, start=1):
        meta_text = f"{meta.repo} {short_sha(meta.sha)} {meta.analyzed_at.astimezone():%Y-%m-%d}"
        if str(number).startswith(word):
            yield Suggestion(str(number), meta_text)
    for meta in entries:
        if word and meta.sha.lower().startswith(lowered):
            yield Suggestion(short_sha(meta.sha), f"{meta.repo} {meta.analyzed_at.astimezone():%Y-%m-%d}")


def complete_analysis_paths(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Paths of the files in the loaded analysis: prefix matches first (priority order), then substrings."""
    if session.current is None:
        return
    needle = word.lower()
    prefix: list[Suggestion] = []
    inner: list[Suggestion] = []
    for f in session.current.result.files:
        lowered = f.path.lower()
        if lowered.startswith(needle):
            prefix.append(Suggestion(f.path, f"{f.band} {f.priority}"))
        elif needle in lowered:
            inner.append(Suggestion(f.path, f"{f.band} {f.priority}"))
        if len(prefix) >= MAX_PATH_SUGGESTIONS:
            break
    yield from (prefix + inner)[:MAX_PATH_SUGGESTIONS]


def complete_directories(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Local folders matching ``word`` (a partial path), each ending with the path separator."""
    if word.startswith("-"):
        return
    expanded = os.path.expanduser(word) if word.startswith("~") else word
    if expanded.endswith(("/", "\\")) or expanded == "":
        parent, stem = expanded, ""
    else:
        parent, stem = os.path.split(expanded)
    base = Path(parent) if parent else Path(".")
    shown_parent = word[: len(word) - len(stem)] if stem else word
    fold = os.name == "nt"
    match = stem.lower() if fold else stem
    try:
        entries = sorted(os.scandir(base), key=lambda e: e.name.lower())
    except OSError:
        return
    count = 0
    for entry in entries:
        name = entry.name.lower() if fold else entry.name
        if not name.startswith(match) or (entry.name.startswith(".") and not stem.startswith(".")):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        yield Suggestion(f"{shown_parent}{entry.name}{os.sep}", "folder")
        count += 1
        if count >= MAX_PATH_SUGGESTIONS:
            return


def complete_pref_keys(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Setting names (and ``reset``) for the first argument of ``/settings``."""
    from tdp_cli.home import PREF_HELP

    if args:
        return
    for key, text in [*PREF_HELP.items(), ("reset", "Restore every setting to its default")]:
        if key.startswith(word.lower()):
            yield Suggestion(key, text)


def options_completer(*names: str) -> ArgCompleter:
    """A completer offering the long options ``names`` when the word starts with ``-``."""

    def complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
        if not word.startswith("-"):
            return
        for name in names:
            if name.startswith(word) and name not in args:
                yield Suggestion(name, "option")

    return complete


def chain(*completers: ArgCompleter) -> ArgCompleter:
    """A completer yielding the suggestions of every completer in order."""

    def complete(session: Session, args: Sequence[str], word: str) -> Iterable[Suggestion]:
        out: list[Suggestion] = []
        for completer in completers:
            out.extend(completer(session, args, word))
        return out

    return complete
