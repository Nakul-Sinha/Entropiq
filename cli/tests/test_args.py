from __future__ import annotations

import pytest

from tdp_cli.args import parse_args, quote_arg, split_args, split_args_lenient
from tdp_cli.errors import UsageError


def test_split_keeps_windows_paths_and_quoted_spaces() -> None:
    assert split_args(r"/analyze C:\Users\me\repo --ref v1") == [
        "/analyze",
        r"C:\Users\me\repo",
        "--ref",
        "v1",
    ]
    assert split_args("/analyze \"C:\\My Repos\\x\" 'a b'") == ["/analyze", "C:\\My Repos\\x", "a b"]
    assert split_args("a # not a comment") == ["a", "#", "not", "a", "comment"]


def test_split_rejects_open_quote() -> None:
    with pytest.raises(UsageError, match="unbalanced quotes"):
        split_args('/analyze "C:\\x')


def test_lenient_split_closes_quotes() -> None:
    assert split_args_lenient('/analyze "C:\\My Re') == ["/analyze", "C:\\My Re"]
    assert split_args_lenient("/open 'ab") == ["/open", "ab"]
    assert split_args_lenient("plain words") == ["plain", "words"]


def test_quote_arg() -> None:
    assert quote_arg("src/a.py") == "src/a.py"
    assert quote_arg("my repo") == '"my repo"'
    assert quote_arg("") == '""'


def test_parse_values_flags_and_positionals() -> None:
    parsed = parse_args(["repo", "--ref", "v1", "--fresh"], values=("--ref",), flags=("fresh",))
    assert parsed.positionals == ("repo",)
    assert parsed.value("--ref") == "v1"
    assert parsed.value("ref") == "v1"
    assert parsed.flag("--fresh") is True
    assert parsed.flag("other") is False
    assert parse_args(["--ref=main"], values=("--ref",)).value("ref") == "main"
    assert parse_args([], values=("--ref",)).value("--ref") is None


def test_double_dash_ends_options() -> None:
    parsed = parse_args(["--", "--ref", "-x"], values=("--ref",))
    assert parsed.positionals == ("--ref", "-x")
    assert parsed.options == {}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--nope"], "unknown option --nope"),
        (["-x"], "unknown option -x"),
        (["--ref"], "--ref needs a value"),
        (["--ref", "--fresh"], "--ref needs a value"),
        (["--fresh=1"], "does not take a value"),
        (["a", "b"], "too many arguments"),
    ],
)
def test_parse_errors(args: list[str], message: str) -> None:
    with pytest.raises(UsageError, match=message):
        parse_args(args, values=("--ref",), flags=("--fresh",), max_positionals=1)


def test_no_arguments_allowed() -> None:
    with pytest.raises(UsageError, match="takes no arguments"):
        parse_args(["x"], max_positionals=0)
    with pytest.raises(UsageError, match="takes no options"):
        parse_args(["--x"])
