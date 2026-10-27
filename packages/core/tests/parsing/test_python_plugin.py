import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tdp_core.contracts import CommentInfo, FileSyntax
from tdp_core.parsing import LanguagePlugin, PythonPlugin, get_plugin, is_test_file, parse_python


def _parse(source: str | bytes) -> FileSyntax:
    data = source.encode() if isinstance(source, str) else source
    return parse_python("mod.py", data)


def test_tokens_are_normalised_with_lines() -> None:
    syntax = _parse('x = foo(1, "a" "b", None)  # c\nif True:\n    y = 2.5\n')
    assert syntax.tokens == (
        "ID", "=", "ID", "(", "LIT", ",", "LIT", "LIT", ",", "LIT", ")",
        "if", "LIT", ":",
        "ID", "=", "LIT",
    )  # fmt: skip
    assert syntax.token_lines == (1,) * 11 + (2,) * 3 + (3,) * 3


def test_string_is_one_token_and_not_descended() -> None:
    syntax = _parse('s = f"{a + b!r} and {c}"\n')
    assert syntax.tokens == ("ID", "=", "LIT")


def test_multiline_string_counts_every_line_it_spans() -> None:
    syntax = _parse('x = """first\n\n   third\n"""\n# done\n')
    assert syntax.sloc == 4
    assert syntax.token_lines == (1, 1, 1)


def test_line_continuation_counts_for_sloc_but_is_not_a_token() -> None:
    syntax = _parse("x = 1 + \\\n    2\n")
    assert syntax.tokens == ("ID", "=", "LIT", "+", "LIT")
    assert syntax.sloc == 2


def test_empty_source() -> None:
    syntax = _parse(b"")
    assert syntax.sloc == 0
    assert syntax.module_cc == 1
    for empty in (syntax.functions, syntax.comments, syntax.imports, syntax.tokens, syntax.token_lines):
        assert empty == ()
    assert syntax.classes == 0
    assert syntax.parse_error is False


def test_comment_text_drops_one_hash_and_strips() -> None:
    syntax = _parse("#!/bin/env python\n##  double  \nx = 1  #   trailing \n#\n")
    assert syntax.comments == (
        CommentInfo(1, "!/bin/env python"),
        CommentInfo(2, "#  double"),
        CommentInfo(3, "trailing"),
        CommentInfo(4, ""),
    )
    assert syntax.sloc == 1


def test_bom_and_crlf_give_the_same_result_as_lf() -> None:
    lf = "# note\ndef f(a):\n    if a:\n        return 1\n    return 2\n"
    crlf = _parse(b"\xef\xbb\xbf" + lf.replace("\n", "\r\n").encode())
    plain = _parse(lf)
    assert crlf.parse_error is False
    assert (crlf.sloc, crlf.functions, crlf.comments, crlf.tokens, crlf.token_lines) == (
        plain.sloc,
        plain.functions,
        plain.comments,
        plain.tokens,
        plain.token_lines,
    )


def test_invalid_utf8_does_not_raise() -> None:
    syntax = _parse(b"# caf\xe9\nx = 1\n")
    assert syntax.comments == (CommentInfo(1, "caf\ufffd"),)
    assert syntax.sloc == 1


def test_function_end_ignores_trailing_comments() -> None:
    source = "def f(a):\n    if a:\n        x = 1\n        # inner tail\n    # outer tail\n\ny = 2\n"
    (func,) = _parse(source).functions
    assert (func.start_line, func.end_line) == (1, 3)


def test_decorated_function_starts_at_def_line() -> None:
    (func,) = _parse("@a\n@b(1)\nasync def f():\n    pass\n").functions
    assert (func.name, func.start_line, func.end_line) == ("f", 3, 4)


def test_function_inside_module_level_block_is_not_qualified() -> None:
    source = (
        "if X:\n    def f():\n        pass\nelse:\n    class C:\n        def m(self):\n            pass\n"
    )
    syntax = _parse(source)
    assert [f.name for f in syntax.functions] == ["f", "C.m"]
    assert syntax.module_cc == 2


def test_params_rules() -> None:
    source = (
        "def a(self, x, /, y=1, *args, z: int = 2, **kw): pass\n"
        "def b(cls): pass\n"
        "def c(x, self): pass\n"
        "def d(*, key): pass\n"
        "def e(): pass\n"
    )
    params = {f.name: f.params for f in _parse(source).functions}
    assert params == {"a": 5, "b": 0, "c": 2, "d": 1, "e": 0}


def test_case_guard_counts_as_case_and_if_clause() -> None:
    source = "def f(x):\n    match x:\n        case 1 if x:\n            pass\n"
    (func,) = _parse(source).functions
    assert (func.cyclomatic, func.cognitive, func.max_nesting) == (3, 1, 1)


def test_every_else_clause_adds_cognitive() -> None:
    source = (
        "def f(a):\n"
        "    for i in a:\n"
        "        pass\n"
        "    else:\n"
        "        pass\n"
        "    try:\n"
        "        pass\n"
        "    except E:\n"
        "        pass\n"
        "    else:\n"
        "        pass\n"
    )
    (func,) = _parse(source).functions
    # for +1, for-else +1, except +1, try-else +1
    assert (func.cyclomatic, func.cognitive, func.max_nesting) == (3, 4, 1)


def test_syntax_error_sets_flag_and_skips_missing_tokens() -> None:
    syntax = _parse("def f(:\n    return 1\n")
    assert syntax.parse_error is True
    assert ")" not in syntax.tokens
    assert [f.name for f in syntax.functions] == ["f"]


def test_deep_expression_does_not_recurse() -> None:
    source = "x = " + " + ".join(["1"] * 5000) + "\n"
    syntax = _parse(source)
    assert syntax.tokens.count("LIT") == 5000
    assert syntax.sloc == 1


def test_line_numbers_beyond_256_are_exact() -> None:
    # tree-sitter 0.26.0's Point.row returns a dangling int above 256; the plugin must not use it.
    source = "\n".join(f"def f{i}(a):\n    return a if a else {i}\n" for i in range(300))
    syntax = _parse(source)
    assert len(syntax.functions) == 300
    assert [f.start_line for f in syntax.functions] == [3 * i + 1 for i in range(300)]
    assert [f.end_line for f in syntax.functions] == [3 * i + 2 for i in range(300)]
    assert all(f.cyclomatic == 2 for f in syntax.functions)
    assert syntax.token_lines[-1] == 899


def _thousand_lines() -> bytes:
    chunk = (
        "class C{i}:\n"
        '    """Doc for C{i}."""\n'
        "\n"
        "    def method_{i}(self, a, b=2, *args, **kw):\n"
        "        # comment {i}\n"
        "        total = 0\n"
        "        for x in range(a):\n"
        "            if x % 2 == 0 and x > b:\n"
        "                total += x if x else 1\n"
        "            elif x in (1, 2, 3):\n"
        "                total -= 1\n"
        "            else:\n"
        "                try:\n"
        "                    total *= [y for y in args if y][0]\n"
        "                except (IndexError, KeyError):\n"
        "                    pass\n"
        "        return f'{{total}}' + 'str'\n"
        "\n"
    )
    return "".join(chunk.format(i=i) for i in range(56)).encode()


def test_parses_1000_lines_fast() -> None:
    source = _thousand_lines()
    assert source.count(b"\n") >= 1000
    parse_python("big.py", source)  # warm up: parser creation
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        syntax = parse_python("big.py", source)
        best = min(best, time.perf_counter() - start)
    assert len(syntax.functions) == 56
    assert best < 0.5


def test_parse_is_thread_safe() -> None:
    source = _thousand_lines()
    expected = parse_python("big.py", source)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: parse_python("big.py", source), range(8)))
    assert all(result == expected for result in results)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("test_mod.py", True),
        ("pkg/test_mod.py", True),
        ("pkg/mod_test.py", True),
        ("conftest.py", True),
        ("src/pkg/conftest.py", True),
        ("tests/helpers.py", True),
        ("pkg/test/data.py", True),
        ("a/tests/b/c.py", True),
        ("pkg/mod.py", False),
        ("pkg/testing.py", False),
        ("pkg/contest.py", False),
        ("testsuite/mod.py", False),
        ("pkg/latest_test_data.py", False),
        ("pkg/Test_mod.py", False),
        ("tests.py", False),
    ],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected
    assert PythonPlugin().is_test_file(path) is expected


def test_get_plugin_by_extension() -> None:
    plugin: LanguagePlugin | None = get_plugin("src/pkg/mod.py")
    assert isinstance(plugin, PythonPlugin)
    assert plugin is get_plugin("other.py")
    assert plugin.name == "python"
    assert plugin.extensions == (".py",)
    assert plugin.parse("a.py", b"x = 1\n").language == "python"
    for path in ("README.md", "mod.pyi", "mod.pyc", "mod.PY", "py", "src/py/mod.txt"):
        assert get_plugin(path) is None
