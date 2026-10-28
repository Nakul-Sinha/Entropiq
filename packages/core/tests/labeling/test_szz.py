import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import COMMITS_SCHEMA
from tdp_core.labeling import szz
from tdp_core.labeling.bugfix import BUGFIX_SCHEMA, classify_bugfix, default_is_source
from tdp_core.labeling.github_issues import ISSUES_SCHEMA
from tdp_core.labeling.szz import (
    SZZ_SCHEMA,
    SZZ_VERSION,
    InducingLine,
    _is_ignored_line,
    _parse_blame,
    _parse_diff,
    _unquote_label,
    inducing_lines,
    szz_table,
)
from tdp_core.mining import MinedHistory, mine_history
from tdp_core.mining.git import GitError, TransientGitError, run_git
from tdp_core.testing import RepoBuilder

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_64 = "c" * 64


def _lines(*rows: str) -> str:
    return "\n".join(rows) + "\n"


CALC_V1 = (
    "import math",
    "",
    "",
    "def calc(x):",
    "    y = x + 1",
    "    return y",
    "",
    "",
    "def other():",
    "    return 0",
)


def _calc(**changes: str) -> str:
    """CALC_V1 with line ``l<n>`` (1-based) replaced, for example ``_calc(l6="    return y * 2")``."""
    rows = list(CALC_V1)
    for key, value in changes.items():
        rows[int(key[1:]) - 1] = value
    return _lines(*rows)


def _day(t0: datetime, n: float) -> datetime:
    return t0 + timedelta(days=n)


def _when(t0: datetime, n: float) -> datetime:
    """The author time git stores for a commit made at day ``n`` (whole seconds, UTC)."""
    return datetime.fromtimestamp(int(_day(t0, n).timestamp()), tz=UTC)


def _always(path: str) -> bool:
    return True


# ------------------------------------------------------------------------------------------- diff parsing
_HANDMADE_DIFF = b"\n".join(
    [
        b'diff --git "a/sp ace \\303\\251.py" b/pkg/moved.py',
        b"similarity index 90%",
        b'rename from "sp ace \\303\\251.py"',
        b"rename to pkg/moved.py",
        b"index 1111111..2222222 100644",
        b'--- "a/sp ace \\303\\251.py"\t',
        b"+++ b/pkg/moved.py",
        b"@@ -3,0 +4,2 @@ def f():",
        b"+added one",
        b"+added two",
        b"@@ -4,2 +7,2 @@",
        b"--- x",
        b"-old",
        b"+++ y",
        b"+new",
        b"diff --git a/pkg/b.py b/pkg/b.py",
        b"index 3333333..4444444 100644",
        b"--- a/pkg/b.py",
        b"+++ b/pkg/b.py",
        b"@@ -10 +12 @@ class B:",
        b"-    x = 1",
        b"+    x = 2",
        b"diff --git a/pkg/fresh.py b/pkg/fresh.py",
        b"new file mode 100644",
        b"index 0000000..5555555",
        b"--- /dev/null",
        b"+++ b/pkg/fresh.py",
        b"@@ -0,0 +1,2 @@",
        b"+a = 1",
        b"+b = 2",
        b"diff --git a/pkg/gone.py b/pkg/gone.py",
        b"deleted file mode 100644",
        b"index 6666666..0000000",
        b"--- a/pkg/gone.py",
        b"+++ /dev/null",
        b"@@ -1,2 +0,0 @@",
        b"-a = 1",
        b"-b = 2",
        b"diff --git a/pkg/blob.py b/pkg/blob.py",
        b"index 7777777..8888888 100644",
        b"Binary files a/pkg/blob.py and b/pkg/blob.py differ",
        b"diff --git a/pkg/link.py b/pkg/link.py",
        b"index 9999999..aaaaaaa 120000",
        b"--- a/pkg/link.py",
        b"+++ b/pkg/link.py",
        b"@@ -1 +1 @@",
        b"-target_a.py",
        b"\\ No newline at end of file",
        b"+target_b.py",
        b"\\ No newline at end of file",
        b"diff --git a/README.md b/README.md",
        b"index bbbbbbb..ccccccc 100644",
        b"--- a/README.md",
        b"+++ b/README.md",
        b"@@ -2 +2 @@",
        b"-old text",
        b"+new text",
        b"",
    ]
)


def test_parse_diff_keeps_parent_ranges_of_modified_source_files() -> None:
    ranges = _parse_diff(_HANDMADE_DIFF, lambda p: p.endswith(".py"))
    assert ranges == {"sp ace é.py": [(4, 2)], "pkg/b.py": [(10, 1)]}


def test_parse_diff_mode_lines_and_submodules() -> None:
    raw = b"\n".join(
        [
            b"diff --git a/pkg/mode.py b/pkg/mode.py",
            b"old mode 100644",
            b"new mode 100755",
            b"index 1111111..2222222",
            b"--- a/pkg/mode.py",
            b"+++ b/pkg/mode.py",
            b"@@ -2 +2 @@",
            b"-x = 1",
            b"+x = 2",
            b"diff --git a/pkg/sub.py b/pkg/sub.py",
            b"index 1111111..2222222 160000",
            b"--- a/pkg/sub.py",
            b"+++ b/pkg/sub.py",
            b"@@ -1 +1 @@",
            b"-Subproject commit 1111111",
            b"+Subproject commit 2222222",
            b"diff --git a/pkg/was_link.py b/pkg/was_link.py",
            b"old mode 120000",
            b"new mode 100644",
            b"index 1111111..2222222",
            b"--- a/pkg/was_link.py",
            b"+++ b/pkg/was_link.py",
            b"@@ -1 +1 @@",
            b"-target",
            b"+x = 1",
        ]
    )
    assert _parse_diff(raw, _always) == {"pkg/mode.py": [(2, 1)]}


def test_parse_diff_rejects_a_malformed_hunk_header() -> None:
    raw = b"diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ nonsense @@\n"
    with pytest.raises(GitError, match="hunk header"):
        _parse_diff(raw, _always)


def _pair(old: str, new: str) -> bytes:
    return b"\n".join(
        [
            f"diff --git a/{old} b/{new}".encode(),
            b"similarity index 90%",
            f"rename from {old}".encode(),
            f"rename to {new}".encode(),
            b"index 1111111..2222222 100644",
            f"--- a/{old}".encode(),
            f"+++ b/{new}".encode(),
            b"@@ -1 +1 @@",
            b"-x = 1",
            b"+x = 2",
        ]
    )


def test_is_source_applies_to_either_name_once_per_path() -> None:
    raw = b"\n".join(
        [
            _pair("vendor/x.py", "pkg/x.py"),
            _pair("tests/test_a.py", "tests/test_b.py"),
            _pair("pkg/y.py", "pkg/y.py"),
        ]
    )
    assert _parse_diff(raw, default_is_source) == {"vendor/x.py": [(1, 1)], "pkg/y.py": [(1, 1)]}

    calls: Counter[str] = Counter()

    def counting(path: str) -> bool:
        calls[path] += 1
        return path.startswith("pkg/")

    _parse_diff(raw, counting)
    assert calls and max(calls.values()) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"a/plain name.py\t", "a/plain name.py"),
        (b"a/plain.py", "a/plain.py"),
        (b"/dev/null", None),
        (b'"a/sp ace \\303\\251.py"\t', "a/sp ace é.py"),
        (b'"a/t\\tab\\nnew\\"quote\\\\back.py"', 'a/t\tab\nnew"quote\\back.py'),
        (b'"a/\\a\\b\\v\\f\\r.py"', "a/\a\b\v\f\r.py"),
        (b'"a/odd\\q.py"', "a/odd\\q.py"),
        (b'"a/end\\\\"', "a/end\\"),
        (b'"a/\\377.py"', "a/\ufffd.py"),
    ],
)
def test_unquote_label(raw: bytes, expected: str | None) -> None:
    assert _unquote_label(raw) == expected


# ------------------------------------------------------------------------------------------- blame parsing
def test_parse_blame_line_porcelain() -> None:
    raw = b"\n".join(
        [
            f"{SHA_A} 3 1 2".encode(),
            b"author Ana",
            b"author-mail <ana@example.com>",
            b"author-time 1600000000",
            b"author-tz +0000",
            b"summary initial",
            b"boundary",
            b"filename pkg/a.py",
            b"\tline one",
            f"{SHA_A} 4 2".encode(),
            b"author Ana",
            b"author-time 1600000000",
            b"filename pkg/a.py",
            b"\t",
            b"",
            f"{SHA_64} 7 5 1".encode(),
            b"author-time 1700000000",
            f"previous {SHA_B} pkg/old.py".encode(),
            b"filename pkg/a.py",
            b"\tx = 1\r",
            b"",
        ]
    )
    assert _parse_blame(raw) == [
        (1, SHA_A, 1600000000, b"line one"),
        (2, SHA_A, 1600000000, b""),
        (5, SHA_64, 1700000000, b"x = 1\r"),
    ]
    assert _parse_blame(b"") == []


@pytest.mark.parametrize(
    "raw",
    [
        b"\tcontent without header\n",
        f"{SHA_A} 1 1 1\nfilename x.py\n\tno author time\n".encode(),
        f"{SHA_A} 1 1 1\nauthor-time 12\n".encode(),
        f"{SHA_A} 1 1 1\nauthor-time soon\n\tx\n".encode(),
        b"not a header\n",
    ],
)
def test_parse_blame_rejects_malformed_output(raw: bytes) -> None:
    with pytest.raises(GitError):
        _parse_blame(raw)


@pytest.mark.parametrize(
    ("content", "ignored"),
    [
        (b"", True),
        (b"   ", True),
        (b"\r", True),
        (b"# a comment", True),
        (b"    # indented comment\r", True),
        (b"import os", True),
        (b"    import os.path as p", True),
        (b"from x import y", True),
        (b"from . import z", True),
        (b"from .mod import (a,", True),
        (b"importlib.reload(x)", False),
        (b"from_date = 1", False),
        (b"x = 1  # c", False),
        (b"return importer", False),
        (b"    y = x + 1\r", False),
    ],
)
def test_is_ignored_line(content: bytes, ignored: bool) -> None:
    assert _is_ignored_line(content) is ignored


# ------------------------------------------------------------------------------------------- inducing_lines
def _classic(repo: RepoBuilder, t0: datetime) -> dict[str, str]:
    a = repo.commit({"pkg/calc.py": _calc()}, "initial", _day(t0, 0))
    b = repo.commit({"pkg/calc.py": _calc(l6="    return y * 2")}, "handle x", _day(t0, 5))
    f = repo.commit({"pkg/calc.py": _calc(l6="    return abs(y) * 2")}, "fix: negative x", _day(t0, 10))
    return {"a": a, "b": b, "f": f}


def test_classic_case(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas = _classic(repo_builder, t0)
    got = inducing_lines(repo_builder.path, shas["f"], is_source=default_is_source)
    assert got == [InducingLine("pkg/calc.py", 6, shas["b"], _when(t0, 5))]
    assert got[0].inducing_time.tzinfo is not None


def test_addition_only_fix_yields_nothing(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"pkg/calc.py": _calc()}, "initial", _day(t0, 0))
    fix = repo_builder.commit(
        {"pkg/calc.py": _calc() + "\n\ndef guard(x):\n    return x\n"}, "fix: add guard", _day(t0, 3)
    )
    assert inducing_lines(repo_builder.path, fix, is_source=default_is_source) == []


def test_comment_blank_and_import_lines_are_ignored(repo_builder: RepoBuilder, t0: datetime) -> None:
    v1 = _lines("import os", "# helper", "", "def f(x):", "    return x + 1")
    v2 = _lines("import sys", "# better helper", "VALUE = 3", "def f(x):", "    return x + 2")
    a = repo_builder.commit({"pkg/m.py": v1}, "initial", _day(t0, 0))
    fix = repo_builder.commit({"pkg/m.py": v2}, "fix: off by one", _day(t0, 1))
    got = inducing_lines(repo_builder.path, fix, is_source=default_is_source)
    assert got == [InducingLine("pkg/m.py", 5, a, _when(t0, 0))]


def test_reindent_only_fix_yields_nothing(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"pkg/calc.py": _calc()}, "initial", _day(t0, 0))
    fix = repo_builder.commit(
        {"pkg/calc.py": _calc(l5="    y  =  x + 1", l6="    return  y")}, "fix spacing", _day(t0, 1)
    )
    assert inducing_lines(repo_builder.path, fix, is_source=default_is_source) == []


def test_bulk_shas_and_not_after(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas = _classic(repo_builder, t0)
    repo, fix = repo_builder.path, shas["f"]
    assert inducing_lines(repo, fix, is_source=default_is_source, bulk_shas={shas["b"]}) == []
    kept = inducing_lines(repo, fix, is_source=default_is_source, not_after=_when(t0, 5))
    assert [line.inducing_sha for line in kept] == [shas["b"]]
    early = _when(t0, 5) - timedelta(seconds=1)
    assert inducing_lines(repo, fix, is_source=default_is_source, not_after=early) == []


def test_only_source_files_are_traced(repo_builder: RepoBuilder, t0: datetime) -> None:
    files = {
        "tests/test_calc.py": _lines("def test_a():", "    assert 1"),
        "docs/readme.md": _lines("Some text", "More text"),
        "vendor/lib.py": _lines("def lib():", "    return 1"),
        "pkg/calc.py": _calc(),
    }
    repo_builder.commit(files, "initial", _day(t0, 0))
    fix = repo_builder.commit(
        {
            "tests/test_calc.py": _lines("def test_a():", "    assert 2"),
            "docs/readme.md": _lines("Some text", "Fixed text"),
            "vendor/lib.py": _lines("def lib():", "    return 2"),
            "pkg/calc.py": _calc(l10="    return 1"),
        },
        "fix several things",
        _day(t0, 1),
    )
    traced = inducing_lines(repo_builder.path, fix, is_source=default_is_source)
    assert [(line.path, line.line) for line in traced] == [("pkg/calc.py", 10)]
    everything = inducing_lines(repo_builder.path, fix, is_source=_always)
    assert {line.path for line in everything} == {
        "docs/readme.md",
        "pkg/calc.py",
        "tests/test_calc.py",
        "vendor/lib.py",
    }


def test_renames_before_and_inside_the_fix(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"pkg/old.py": _calc()}, "initial", _day(t0, 0))
    b = repo_builder.commit({"pkg/old.py": _calc(l6="    return y * 2")}, "handle x", _day(t0, 2))
    repo_builder.rename("pkg/old.py", "pkg/new.py", "move module", _day(t0, 4))
    fix = repo_builder.commit({"pkg/new.py": _calc(l6="    return abs(y) * 2")}, "fix sign", _day(t0, 6))
    assert inducing_lines(repo_builder.path, fix, is_source=default_is_source) == [
        InducingLine("pkg/new.py", 6, b, _when(t0, 2))
    ]
    moved_fix = repo_builder.rename(
        "pkg/new.py",
        "pkg/newer.py",
        "fix and move",
        _day(t0, 8),
        new_content=_calc(l6="    return abs(y) * 3"),
    )
    assert inducing_lines(repo_builder.path, moved_fix, is_source=default_is_source) == [
        InducingLine("pkg/new.py", 6, fix, _when(t0, 6))
    ]


def test_deleting_a_file_and_the_root_commit_yield_nothing(repo_builder: RepoBuilder, t0: datetime) -> None:
    root = repo_builder.commit({"pkg/calc.py": _calc(), "pkg/gone.py": _calc()}, "initial", _day(t0, 0))
    delete = repo_builder.commit({"pkg/gone.py": None}, "fix: remove broken module", _day(t0, 1))
    assert inducing_lines(repo_builder.path, delete, is_source=default_is_source) == []
    assert inducing_lines(repo_builder.path, root, is_source=default_is_source) == []


def test_unicode_path_matches_the_mined_path(repo_builder: RepoBuilder, t0: datetime) -> None:
    name = "pkg/sp ace é.py"
    a = repo_builder.commit({name: _calc()}, "initial", _day(t0, 0))
    fix = repo_builder.commit({name: _calc(l5="    y = x + 2")}, "fix offset", _day(t0, 1))
    got = inducing_lines(repo_builder.path, fix, is_source=default_is_source)
    mined = mine_history(repo_builder.path).file_changes
    fix_rows = mined.filter(pc.equal(mined.column("sha"), fix))
    assert fix_rows.column("path").to_pylist() == [name]
    assert got == [InducingLine(name, 5, a, _when(t0, 0))]


def test_lines_sorted_by_path_and_parent_line_numbers(repo_builder: RepoBuilder, t0: datetime) -> None:
    a = repo_builder.commit({"pkg/b.py": _calc(), "pkg/a.py": _calc()}, "initial", _day(t0, 0))
    header = "X = 1\nY = 2\n"
    fix = repo_builder.commit(
        {
            "pkg/b.py": header + _calc(l6="    return -y", l10="    return 7"),
            "pkg/a.py": header + _calc(l5="    y = x - 1", l9="def other(z):"),
        },
        "fix values",
        _day(t0, 1),
    )
    got = inducing_lines(repo_builder.path, fix, is_source=default_is_source)
    assert [(line.path, line.line) for line in got] == [
        ("pkg/a.py", 5),
        ("pkg/a.py", 9),
        ("pkg/b.py", 6),
        ("pkg/b.py", 10),
    ]
    assert {line.inducing_sha for line in got} == {a}


def test_inducing_lines_validation(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas = _classic(repo_builder, t0)
    with pytest.raises(ValueError):
        inducing_lines(repo_builder.path, "-x", is_source=default_is_source)
    with pytest.raises(GitError):
        inducing_lines(repo_builder.path, "0" * 40, is_source=default_is_source)
    with pytest.raises(ValueError, match="aware"):
        inducing_lines(
            repo_builder.path, shas["f"], is_source=default_is_source, not_after=datetime(2022, 1, 1)
        )


def test_blame_ranges_are_chunked(
    repo_builder: RepoBuilder, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [f"v{i} = {i}" for i in range(1, 21)]
    a = repo_builder.commit({"pkg/many.py": _lines(*rows)}, "initial", _day(t0, 0))
    fixed = list(rows)
    for i in (1, 5, 9, 13, 17):
        fixed[i] = f"v{i + 1} = -{i + 1}"
    fix = repo_builder.commit({"pkg/many.py": _lines(*fixed)}, "fix signs", _day(t0, 1))
    expected = inducing_lines(repo_builder.path, fix, is_source=default_is_source)
    assert [line.line for line in expected] == [2, 6, 10, 14, 18]
    assert {line.inducing_sha for line in expected} == {a}

    blame_calls: list[Sequence[str]] = []
    real = run_git

    def counting(args: Sequence[str], cwd: Path | None, **kw: Any) -> bytes:
        if "blame" in args:
            blame_calls.append(args)
        return real(args, cwd, **kw)

    monkeypatch.setattr(szz, "run_git", counting)
    monkeypatch.setattr(szz, "_MAX_RANGES_PER_BLAME", 2)
    assert inducing_lines(repo_builder.path, fix, is_source=default_is_source) == expected
    assert len(blame_calls) == 3


# ------------------------------------------------------------------------------------------- szz_table
def _szz_repo(repo: RepoBuilder, t0: datetime) -> dict[str, str]:
    """initial; B rewrites lines 5 and 6; C line 10; F1 fixes 5, 6, 10; F2 fixes line 10 again."""
    a = repo.commit({"pkg/calc.py": _calc(), "pkg/util.py": _calc()}, "initial", _day(t0, 0))
    b = repo.commit(
        {"pkg/calc.py": _calc(l5="    y = x + 2", l6="    return y * 2")}, "handle x", _day(t0, 5)
    )
    c = repo.commit(
        {"pkg/calc.py": _calc(l5="    y = x + 2", l6="    return y * 2", l10="    return 1")},
        "add helper",
        _day(t0, 10),
    )
    f1 = repo.commit(
        {"pkg/calc.py": _calc(l5="    y = abs(x) + 2", l6="    return y * 3", l10="    return 2")},
        "fix: negative x",
        _day(t0, 20),
    )
    f2 = repo.commit(
        {"pkg/calc.py": _calc(l5="    y = abs(x) + 2", l6="    return y * 3", l10="    return 3")},
        "fix wrong helper value",
        _day(t0, 30),
    )
    return {"a": a, "b": b, "c": c, "f1": f1, "f2": f2}


def _rows(table: pa.Table) -> list[tuple[object, ...]]:
    cols = [table.column(name).to_pylist() for name in table.column_names]
    return list(zip(*cols, strict=True))


def _setup(repo: RepoBuilder, t0: datetime) -> tuple[dict[str, str], MinedHistory, pa.Table]:
    shas = _szz_repo(repo, t0)
    hist = mine_history(repo.path)
    return shas, hist, classify_bugfix(hist.commits, hist.file_changes)


def test_szz_table_rows(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas, hist, bugfix = _setup(repo_builder, t0)
    table = szz_table(repo_builder.path, hist, bugfix, jobs=2)
    assert table.schema == SZZ_SCHEMA
    expected = [
        (shas["f2"], _when(t0, 30), "pkg/calc.py", shas["f1"], _when(t0, 20), 1),
        (shas["f1"], _when(t0, 20), "pkg/calc.py", shas["c"], _when(t0, 10), 1),
        (shas["f1"], _when(t0, 20), "pkg/calc.py", shas["b"], _when(t0, 5), 2),
    ]
    assert _rows(table) == sorted(expected, key=lambda row: (row[1], row[0], row[2], row[3]))


def _issue_bugfix(table: pa.Table, sha: str, created: datetime) -> pa.Table:
    """``table`` with ``sha`` marked as an issue-linked fix whose issue was created at ``created``."""
    rows = table.to_pylist()
    for row in rows:
        if row["sha"] == sha:
            row.update(is_bugfix=True, source="issue", issue_number=7, issue_created_at=created)
    return pa.Table.from_pylist(rows, schema=BUGFIX_SCHEMA)


def test_issue_dates_bulk_commits_and_ignored_rows(repo_builder: RepoBuilder, t0: datetime) -> None:
    a = repo_builder.commit({"pkg/calc.py": _calc()}, "initial", _day(t0, 0))
    b = repo_builder.commit({"pkg/calc.py": _calc(l5="    y = x + 2")}, "handle x", _day(t0, 5))
    bulk = repo_builder.commit(
        {
            "pkg/calc.py": _calc(l5="    y = x + 2", l6="    return y * 2"),
            "docs/a.md": "a\n",
            "docs/b.md": "b\n",
        },
        "reformat everything",
        _day(t0, 8),
    )
    c = repo_builder.commit(
        {"pkg/calc.py": _calc(l5="    y = x + 2", l6="    return y * 2", l10="    return 1")},
        "tune other",
        _day(t0, 10),
    )
    fix = repo_builder.commit(
        {"pkg/calc.py": _calc(l5="    y = x + 3", l6="    return y * 3", l10="    return 2")},
        "Correct the result (#7)",
        _day(t0, 20),
    )
    hist = mine_history(repo_builder.path, settings=with_overrides(DEFAULT_SETTINGS, bulk_commit_files=2))
    assert hist.commits.column("is_bulk").to_pylist().count(True) == 1
    issues = pa.Table.from_pydict(
        {
            "number": [7],
            "created_at": [_day(t0, 7)],
            "closed_at": [None],
            "labels": [["bug"]],
            "is_bug": [True],
        },
        schema=ISSUES_SCHEMA,
    )
    bugfix = classify_bugfix(hist.commits, hist.file_changes, issues)
    assert bugfix.column("source").to_pylist()[-1] == "issue"
    ghost = pa.Table.from_pylist(
        [
            {"sha": "f" * 40, "is_bugfix": True, "source": "keyword"},
            {"sha": a, "is_bugfix": None},
            {"sha": bulk, "is_bugfix": True, "source": "keyword"},
        ],
        schema=BUGFIX_SCHEMA,
    )
    table = szz_table(repo_builder.path, hist, pa.concat_tables([bugfix, ghost]), jobs=1)
    # c (after the issue date) and the bulk commit are dropped; b remains.
    assert _rows(table) == [(fix, _when(t0, 20), "pkg/calc.py", b, _when(t0, 5), 1)]
    without_issue = szz_table(repo_builder.path, hist, classify_bugfix(hist.commits, hist.file_changes))
    assert without_issue.num_rows == 0  # "Correct the result" has no keyword
    keyword = _issue_bugfix(bugfix, fix, _day(t0, 30))
    assert {row[3] for row in _rows(szz_table(repo_builder.path, hist, keyword))} == {b, c}


def test_max_fixes_and_validation(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas, hist, bugfix = _setup(repo_builder, t0)
    newest = szz_table(repo_builder.path, hist, bugfix, max_fixes=1)
    assert set(newest.column("fix_sha").to_pylist()) == {shas["f2"]}
    assert szz_table(repo_builder.path, hist, bugfix, max_fixes=0).num_rows == 0
    with pytest.raises(ValueError, match="max_fixes"):
        szz_table(repo_builder.path, hist, bugfix, max_fixes=-1)
    with pytest.raises(ValueError, match="jobs"):
        szz_table(repo_builder.path, hist, bugfix, jobs=0)
    with pytest.raises(ValueError, match="is_bugfix"):
        szz_table(repo_builder.path, hist, bugfix.select(["sha"]))


def test_job_counts_give_equal_tables(repo_builder: RepoBuilder, t0: datetime) -> None:
    _, hist, bugfix = _setup(repo_builder, t0)
    single = szz_table(repo_builder.path, hist, bugfix, jobs=1)
    assert single.equals(szz_table(repo_builder.path, hist, bugfix, jobs=3))
    no_dates = bugfix.drop_columns(["issue_created_at"])
    assert single.equals(szz_table(repo_builder.path, hist, no_dates, jobs=3))


def _blocking(real: Callable[..., bytes], traced: list[str]) -> Callable[..., bytes]:
    def run(args: Sequence[str], cwd: Path | None, **kw: Any) -> bytes:
        if "diff-tree" in args:
            traced.append(args[-2])  # the fix sha, just before "--"
        return real(args, cwd, **kw)

    return run


def test_cache_round_trip(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shas, hist, bugfix = _setup(repo_builder, t0)
    cache = tmp_path / "szz-cache"
    first = szz_table(repo_builder.path, hist, bugfix, cache_dir=cache)
    for name in ("f1", "f2"):
        path = cache / shas[name][:2] / f"{shas[name]}.v{SZZ_VERSION}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["fix_sha"] == shas[name] and data["version"] == SZZ_VERSION
    assert not list(cache.rglob("*.tmp"))

    def no_tracing(args: Sequence[str], cwd: Path | None, **kw: Any) -> bytes:
        assert "diff-tree" not in args and "blame" not in args
        return real(args, cwd, **kw)

    real = run_git
    monkeypatch.setattr(szz, "run_git", no_tracing)
    assert szz_table(repo_builder.path, hist, bugfix, cache_dir=cache, jobs=2).equals(first)
    dated = _issue_bugfix(bugfix, shas["f1"], _day(t0, 7))
    rows = _rows(szz_table(repo_builder.path, hist, dated, cache_dir=cache))
    assert [row[3] for row in rows] == [shas["b"], shas["f1"]]

    traced: list[str] = []
    monkeypatch.setattr(szz, "run_git", _blocking(real, traced))
    corrupt = cache / shas["f1"][:2] / f"{shas['f1']}.v{SZZ_VERSION}.json"
    for garbage in ("{not json", "[1, 2]", json.dumps({"fix_sha": shas["f2"], "version": SZZ_VERSION})):
        corrupt.write_text(garbage, encoding="utf-8")
        assert szz_table(repo_builder.path, hist, bugfix, cache_dir=cache).equals(first)
    bad_rows = {"fix_sha": shas["f1"], "version": SZZ_VERSION, "lines": [["pkg/calc.py", "6", "x", 1]]}
    corrupt.write_text(json.dumps(bad_rows), encoding="utf-8")
    assert szz_table(repo_builder.path, hist, bugfix, cache_dir=cache).equals(first)
    no_lines = {"fix_sha": shas["f1"], "version": SZZ_VERSION, "lines": None}
    corrupt.write_text(json.dumps(no_lines), encoding="utf-8")
    assert szz_table(repo_builder.path, hist, bugfix, cache_dir=cache).equals(first)
    assert traced.count(shas["f1"]) == 5 and shas["f2"] not in traced


def test_unwritable_cache_is_ignored(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, hist, bugfix = _setup(repo_builder, t0)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file in the way", encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="tdp_core.labeling.szz"):
        table = szz_table(repo_builder.path, hist, bugfix, cache_dir=blocker)
    assert table.num_rows == 3
    assert "could not write cache file" in caplog.text


def test_failed_fix_is_skipped_and_not_cached(
    repo_builder: RepoBuilder,
    t0: datetime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shas, hist, bugfix = _setup(repo_builder, t0)
    real = szz._trace

    def flaky(
        repo_dir: Path, fix_sha: str, is_source: Callable[[str], bool], timeout: float
    ) -> list[szz._RawLine]:
        if fix_sha == shas["f1"]:
            raise TransientGitError("git blame timed out after 300 s")
        return real(repo_dir, fix_sha, is_source, timeout)

    monkeypatch.setattr(szz, "_trace", flaky)
    cache = tmp_path / "cache"
    with caplog.at_level(logging.WARNING, logger="tdp_core.labeling.szz"):
        table = szz_table(repo_builder.path, hist, bugfix, cache_dir=cache, jobs=2)
    assert set(table.column("fix_sha").to_pylist()) == {shas["f2"]}
    assert shas["f1"] in caplog.text and "timed out" in caplog.text
    assert not (cache / shas["f1"][:2] / f"{shas['f1']}.v{SZZ_VERSION}.json").exists()
    assert (cache / shas["f2"][:2] / f"{shas['f2']}.v{SZZ_VERSION}.json").exists()


def test_fix_missing_from_the_repository_fails_fast(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    _, hist, bugfix = _setup(repo_builder, t0)
    other = RepoBuilder(tmp_path / "other")
    other.commit({"x.py": "x = 1\n"}, "initial", t0)
    with pytest.raises(GitError, match="2 bug-fix commits are not commits"):
        szz_table(other.path, hist, bugfix)


def test_no_selected_fixes_runs_no_git(
    repo_builder: RepoBuilder, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, hist, bugfix = _setup(repo_builder, t0)

    def fail(*args: object, **kw: object) -> bytes:
        raise AssertionError("git must not run")

    monkeypatch.setattr(szz, "run_git", fail)
    nothing = pa.Table.from_pydict({"sha": bugfix.column("sha"), "is_bugfix": [False] * bugfix.num_rows})
    table = szz_table(repo_builder.path, hist, nothing)
    assert table.schema == SZZ_SCHEMA and table.num_rows == 0
    empty_hist = MinedHistory(COMMITS_SCHEMA.empty_table(), hist.file_changes)
    assert szz_table(repo_builder.path, empty_hist, bugfix).num_rows == 0


def test_undecodable_names_and_undated_commits_are_skipped(
    repo_builder: RepoBuilder, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    shas, hist, bugfix = _setup(repo_builder, t0)
    diff = b"\n".join(
        [
            b"diff --git a/pkg/bad\xff.py b/pkg/bad\xff.py",
            b"index 1111111..2222222 100644",
            b'--- "a/pkg/bad\377.py"',
            b'+++ "b/pkg/bad\377.py"',
            b"@@ -1 +1 @@",
            b"-x = 1",
            b"+x = 2",
        ]
    )

    def fake(args: Sequence[str], cwd: Path | None, **kw: Any) -> bytes:
        assert "blame" not in args
        return diff

    monkeypatch.setattr(szz, "run_git", fake)
    assert szz._trace(repo_builder.path, shas["f1"], default_is_source, 10.0) == []

    rows = hist.commits.to_pylist()
    for row in rows:
        if row["sha"] == shas["f1"]:
            row["author_time"] = None
    undated = MinedHistory(pa.Table.from_pylist(rows, schema=COMMITS_SCHEMA), hist.file_changes)
    fixes, _ = szz._select_fixes(undated, bugfix, None)
    assert [fix.sha for fix in fixes] == [shas["f2"]]
