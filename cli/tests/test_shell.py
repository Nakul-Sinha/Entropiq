from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from tdp_cli.completion import (
    chain,
    complete_analysis_paths,
    complete_directories,
    complete_history,
    complete_pref_keys,
    options_completer,
)
from tdp_cli.errors import CliError, UsageError
from tdp_cli.registry import Command, Suggestion
from tdp_cli.session import Session
from tdp_cli.shell import Shell, ShellCompleter, dispatch, prompt_fragments, toolbar_fragments
from tdp_cli.store import LoadedAnalysis
from tdp_cli.theme import ASCII_GLYPHS

BUILT_INS = {"analyze", "history", "open", "status", "model", "settings", "help", "clear", "quit"}


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined, no-any-return]


def _complete(session: Session, text: str) -> list[tuple[str, str, int]]:
    completer = ShellCompleter(session)
    return [
        (c.text, c.display_meta_text, c.start_position)
        for c in completer.get_completions(Document(text), CompleteEvent())
    ]


def _add(session: Session, name: str, handler: object, **kwargs: object) -> None:
    session.registry.add(Command(name=name, help=f"Test {name}", handler=handler, **kwargs))  # type: ignore[arg-type]


def test_complete_command_names(session: Session) -> None:
    every = _complete(session, "/")
    assert {text.lstrip("/") for text, _, _ in every} >= BUILT_INS
    assert all(start == -1 for _, _, start in every)
    assert ("/analyze", "Analyse a local repository or a GitHub URL", -3) in _complete(session, "/an")
    assert [t for t, _, _ in _complete(session, "/analyse")] == ["/analyze"]
    assert _complete(session, "help") == []
    assert _complete(session, "/nosuch x") == []
    assert _complete(session, "/status ") == []


def test_complete_arguments(session: Session, sample_loaded: LoadedAnalysis) -> None:
    assert [t for t, _, _ in _complete(session, "/help a")] == ["analyze"]
    assert [t for t, _, _ in _complete(session, "/settings ")][:3] == ["budget_hours", "top", "jobs"]
    assert [t for t, _, _ in _complete(session, "/analyze --f")] == ["--fresh"]
    session.store.save(sample_loaded)
    numbers = _complete(session, "/open ")
    assert numbers[0][0] == "1" and "sample" in numbers[0][1]
    by_sha = _complete(session, f"/open {sample_loaded.meta.sha[:3]}")
    assert by_sha[-1][0] == sample_loaded.meta.sha[:7]


def test_complete_analysis_paths(loaded_session: Session) -> None:
    _add(loaded_session, "file", lambda s, a: None, complete=complete_analysis_paths)
    prefix = _complete(loaded_session, "/file pkg/m")
    assert prefix == [("pkg/mu.py", prefix[0][1], -5)]
    assert prefix[0][1].split()[0] in {"Critical", "High", "Medium", "Low"}
    inner = [t for t, _, _ in _complete(loaded_session, "/file eta")]
    assert set(inner) == {"pkg/beta.py", "pkg/eta.py", "pkg/theta.py", "pkg/zeta.py"}
    everything = _complete(loaded_session, "/file ")
    assert [t for t, _, _ in everything] == [f.path for f in loaded_session.current.result.files]  # type: ignore[union-attr]
    loaded_session.current = None
    assert _complete(loaded_session, "/file p") == []


def test_completion_quotes_spaces(session: Session) -> None:
    def spaced(s: Session, args: Sequence[str], word: str) -> list[Suggestion]:
        return [Suggestion("my file.py", "x")]

    def broken(s: Session, args: Sequence[str], word: str) -> list[Suggestion]:
        raise RuntimeError("bug")

    _add(session, "spaced", lambda s, a: None, complete=spaced)
    _add(session, "broken", lambda s, a: None, complete=broken)
    assert _complete(session, '/spaced "my f') == [('"my file.py"', "x", -5)]
    assert _complete(session, "/broken x") == []


def test_completers(session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "alps").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    monkeypatch.chdir(tmp_path)
    found = [s.text for s in complete_directories(session, [], "al")]
    assert "alps" + __import__("os").sep in found
    assert all(not t.startswith(".") for t in (s.text for s in complete_directories(session, [], "")))
    assert list(complete_directories(session, [], "--x")) == []
    assert list(complete_directories(session, [], "missing/dir/x")) == []
    assert [s.text for s in complete_pref_keys(session, [], "t")] == ["top"]
    assert list(complete_pref_keys(session, ["top"], "")) == []
    combined = chain(options_completer("--ref"), complete_pref_keys)
    assert [s.text for s in combined(session, [], "-")] == ["--ref"]
    assert list(options_completer("--ref")(session, ["--ref"], "--")) == []
    assert list(complete_history(session, [], "")) == []


def test_toolbar_and_prompt(session: Session, sample_loaded: LoadedAnalysis) -> None:
    empty = "".join(text for _, text, *_ in toolbar_fragments(session))
    assert "no repository loaded" in empty and "demo model" in empty and "Ctrl+D" in empty
    session.current = sample_loaded
    loaded = "".join(text for _, text, *_ in toolbar_fragments(session))
    assert f"sample @ {sample_loaded.meta.sha[:7]}" in loaded
    assert prompt_fragments(ASCII_GLYPHS) == [
        ("class:prompt.name", "entropiq"),
        ("class:prompt.chevron", " > "),
    ]


def test_dispatch_basics(session: Session) -> None:
    assert dispatch(session, "   ") is True
    assert dispatch(session, "/") is True
    assert "Type /help <command> for details." in _out(session)
    assert dispatch(session, "/anlyze") is True
    assert "Unknown command /anlyze. Did you mean /analyze? Type / to list every command." in _out(session)
    assert dispatch(session, "/zzzzzz") is True
    assert "Unknown command /zzzzzz. Type / to list every command." in _out(session)
    assert dispatch(session, "help status") is True
    assert "Show the loaded repository, model and settings" in _out(session)
    assert dispatch(session, "what is this") is True
    assert "Commands start with /." in _out(session)
    assert dispatch(session, '/open "1') is True
    assert "unbalanced quotes" in _out(session)
    assert dispatch(session, "/quit") is False
    assert dispatch(session, "exit") is False


def test_dispatch_errors(session: Session) -> None:
    def usage(s: Session, a: Sequence[str]) -> None:
        raise UsageError("bad input", hint="try again")

    def failure(s: Session, a: Sequence[str]) -> None:
        raise CliError("it failed", hint="a hint")

    def interrupted(s: Session, a: Sequence[str]) -> None:
        raise KeyboardInterrupt

    def bug(s: Session, a: Sequence[str]) -> None:
        raise RuntimeError("oops")

    _add(session, "usage", usage, usage="/usage <x>")
    _add(session, "failure", failure)
    _add(session, "interrupted", interrupted)
    _add(session, "bug", bug)
    _add(session, "needs", lambda s, a: None, needs_analysis=True)
    for line in ("/usage", "/failure", "/interrupted", "/bug", "/needs"):
        assert dispatch(session, line) is True
    out = _out(session)
    assert "Error: bad input" in out and "Usage: /usage <x>" in out
    assert "Error: it failed" in out and "a hint" in out
    assert "Cancelled." in out
    assert "Unexpected error: RuntimeError: oops" in out
    assert "No analysis is loaded" in out


def test_builtin_commands(loaded_session: Session) -> None:
    session = loaded_session
    for line in (
        "/status",
        "/model",
        "/settings",
        "/settings top 7",
        "/settings top",
        "/settings colour 1",
        "/help analyze",
        "/help nosuch",
        "/history",
        "/clear",
        "/settings reset",
    ):
        assert dispatch(session, line) is True
    out = _out(session)
    assert "sample" in out and "Session model" in out
    assert "89 inputs" in out
    assert "top is now 7." in out
    assert "give a value" in out
    assert "unknown setting" in out
    assert "/analyze [PATH or URL] [--ref REF] [--fresh]" in out
    assert "tdp analyze" in out
    assert "there is no command /nosuch" in out
    assert "No saved analyses yet." in out
    assert "Settings restored to their defaults." in out
    assert session.prefs.top == 20
    session.current = None
    assert dispatch(session, "/status") is True
    assert "No analysis is loaded" in _out(session)


def test_open_and_history(session: Session, sample_loaded: LoadedAnalysis) -> None:
    session.store.save(sample_loaded)
    assert dispatch(session, "/history") is True
    assert dispatch(session, "/open 1") is True
    assert session.current is not None and session.current.meta == sample_loaded.meta
    assert dispatch(session, "/open") is True
    assert dispatch(session, "/open 9") is True
    out = _out(session)
    assert "Type /open <number> to reopen one." in out
    assert f"Opened sample @ {sample_loaded.meta.sha[:7]}" in out
    assert "say which analysis to open" in out
    assert "There is no analysis number 9" in out


def _run_shell(session: Session, keys: str) -> int:
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        return Shell(session, input=pipe, output=DummyOutput()).run()


def test_shell_prompt_loop(session: Session) -> None:
    code = _run_shell(session, "/help\r/status\rabc\x03/quit\r")
    assert code == 0
    out = _out(session)
    assert "Rank technical debt by future maintenance pain" in out
    assert "Demo model in use" in out
    assert "Show the loaded repository, model and settings" in out
    assert "Bye." in out
    history = session.home.history_file.read_text(encoding="utf-8")
    assert "/help" in history and "/quit" in history and "abc" not in history


def test_shell_ctrl_d_exits(session: Session) -> None:
    assert _run_shell(session, "/help\r\x04") == 0
    assert "Bye." in _out(session)


def test_shell_end_to_end(session: Session, sample_repo: Path) -> None:
    keys = f'/analyze "{sample_repo}"\r/history\r/open 1\r/status\r/anlyze\r/quit\r'
    assert _run_shell(session, keys) == 0
    out = _out(session)
    assert "files analysed" in out
    assert "Estimated Critical debt" in out
    assert "Type /open <number> to reopen one." in out
    assert "Opened sample @" in out
    assert "Did you mean /analyze?" in out
    assert session.current is not None


def test_shell_plain_mode(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("/status\nhelp\n"))
    shell = Shell(session)
    assert shell.plain is True
    assert shell.run() == 0
    out = _out(session)
    assert "entropiq " in out and "/status" in out and "Bye." in out
