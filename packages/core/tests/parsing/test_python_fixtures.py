"""Fixture files against hand-checked values in fixtures/expected.json.

Fixtures are Python sources stored as ``*.py.txt`` so ruff and mypy (which sweep every ``.py`` file under
``packages/core``) neither lint nor type-check deliberately odd or broken code.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tdp_core.contracts import FileSyntax
from tdp_core.parsing import PythonPlugin

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED: dict[str, dict[str, Any]] = {
    name: value
    for name, value in json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8")).items()
    if not name.startswith("_")
}


def _parse(name: str) -> FileSyntax:
    return PythonPlugin().parse(f"pkg/{name.removesuffix('.txt')}", (FIXTURES / name).read_bytes())


def test_every_fixture_has_expectations() -> None:
    assert sorted(p.name for p in FIXTURES.glob("*.py.txt")) == sorted(EXPECTED)
    assert 6 <= len(EXPECTED) <= 10


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_matches_expected(name: str) -> None:
    expected = EXPECTED[name]
    syntax = _parse(name)

    assert syntax.path == f"pkg/{name.removesuffix('.txt')}"
    assert syntax.language == "python"
    for key in ("sloc", "module_cc", "classes", "parse_error"):
        assert getattr(syntax, key) == expected[key], key
    got_functions = [
        {
            "name": f.name,
            "start": f.start_line,
            "end": f.end_line,
            "cc": f.cyclomatic,
            "cog": f.cognitive,
            "nest": f.max_nesting,
            "params": f.params,
        }
        for f in syntax.functions
    ]
    assert got_functions == expected["functions"]
    got_imports = [[i.module, list(i.names), i.level, i.line] for i in syntax.imports]
    assert got_imports == expected["imports"]
    if "comments" in expected:
        assert [[c.line, c.text] for c in syntax.comments] == expected["comments"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_tokens_are_consistent(name: str) -> None:
    syntax = _parse(name)
    assert len(syntax.tokens) == len(syntax.token_lines) > 0
    assert list(syntax.token_lines) == sorted(syntax.token_lines)
    assert len(set(syntax.token_lines)) <= syntax.sloc
