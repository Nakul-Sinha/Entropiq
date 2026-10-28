from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tdp_cli.registry import GROUP_ORDER, Command, CommandRegistry, load_registry

BUILT_INS = {"analyze", "history", "open", "status", "model", "settings", "help", "clear", "quit"}


def _noop(session: object, args: object) -> None:
    return None


def _cmd(
    name: str, *, group: str = "General", aliases: tuple[str, ...] = (), help: str = "Does it"
) -> Command:
    return Command(name=name, help=help, handler=_noop, group=group, aliases=aliases)


def test_add_and_get_with_aliases_case_and_slash() -> None:
    registry = CommandRegistry()
    registry.add(_cmd("analyze", aliases=("analyse",)))
    assert registry.get("analyze") is registry.get("/Analyse") is registry.get("  ANALYZE ")
    assert registry.get("nope") is None
    assert "analyse" in registry and "nope" not in registry and 3 not in registry
    assert len(registry) == 1
    assert registry.names() == ["analyse", "analyze"]
    assert registry.names(aliases=False) == ["analyze"]


def test_duplicates_are_rejected() -> None:
    registry = CommandRegistry()
    registry.add(_cmd("analyze", aliases=("a",)))
    with pytest.raises(ValueError, match="already taken by /analyze"):
        registry.add(_cmd("a"))
    with pytest.raises(ValueError, match="already taken by /analyze"):
        registry.add(_cmd("other", aliases=("analyze",)))
    with pytest.raises(ValueError, match="repeats"):
        registry.add(_cmd("dup", aliases=("dup",)))


@pytest.mark.parametrize("name", ["Analyze", "1x", "", "a b", "a" * 33, "/x", "x_y"])
def test_bad_names(name: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        CommandRegistry().add(_cmd(name))


def test_help_must_be_one_line() -> None:
    with pytest.raises(ValueError, match="one-line"):
        CommandRegistry().add(_cmd("x", help="two\nlines"))
    with pytest.raises(ValueError, match="one-line"):
        CommandRegistry().add(_cmd("x", help="  "))


def test_group_order() -> None:
    registry = CommandRegistry()
    for name, group in [
        ("zz", "General"),
        ("st", "Session"),
        ("an", "Analysis"),
        ("ex", "Explore"),
        ("pl", "Plan"),
        ("re", "Report"),
        ("od", "Oddities"),
        ("ab", "Advice"),
    ]:
        registry.add(_cmd(name, group=group))
    groups = [g for g, _ in registry.groups()]
    assert groups == ["Analysis", "Explore", "Plan", "Report", "Advice", "Oddities", "Session", "General"]
    assert GROUP_ORDER[-1] == "General"


def test_suggest() -> None:
    registry = load_registry()
    assert registry.suggest("anlyze") == "analyze"
    assert registry.suggest("/histroy") == "history"
    assert registry.suggest("xyzzyplugh") is None


def test_cli_entries_and_add_cli() -> None:
    registry = CommandRegistry()

    def run() -> None:
        return None

    registry.add(Command(name="report", help="Write a report", handler=_noop, group="Report", cli=run))
    registry.add_cli("diff", run, help="Compare two commits", group="Report")
    assert [e[0] for e in registry.cli_entries()] == ["diff", "report"]
    with pytest.raises(ValueError, match="already taken by /report"):
        registry.add_cli("report", run, help="Again")
    with pytest.raises(ValueError, match="already taken"):
        registry.add(Command(name="other", help="x", handler=_noop, cli=run, cli_name="diff"))
    assert registry.get("other") is None


def test_built_ins() -> None:
    registry = load_registry()
    assert set(registry.names(aliases=False)) >= BUILT_INS
    for command in registry.commands():
        assert "\n" not in command.help
        assert chr(0x2013) not in command.help and chr(0x2014) not in command.help
        assert command.usage_line.startswith(f"/{command.name}")
    cli = {name for name, *_ in registry.cli_entries()}
    assert {"analyze", "history", "model"} <= cli


def test_discovery_from_a_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "fakecmds"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    body = textwrap.dedent(
        """
        from tdp_cli.registry import Command

        def register(registry):
            registry.add(Command(name={name!r}, help="Test command", handler=lambda s, a: None))
        """
    )
    (package / "b_second.py").write_text(body.format(name="second"), encoding="utf-8")
    (package / "a_first.py").write_text(body.format(name="first"), encoding="utf-8")
    (package / "_helper.py").write_text("raise RuntimeError('must not be imported')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = load_registry("fakecmds")
    assert registry.names() == ["first", "second"]

    (package / "c_broken.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in [m for m in sys.modules if m.startswith("fakecmds")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(RuntimeError, match="c_broken has no register"):
        load_registry("fakecmds")
