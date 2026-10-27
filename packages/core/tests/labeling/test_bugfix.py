import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pyarrow as pa
import pytest

from tdp_core.config import DEFAULT_SETTINGS, with_overrides
from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA
from tdp_core.labeling.bugfix import (
    BUGFIX_SCHEMA,
    BUGFIX_SOURCES,
    classify_bugfix,
    default_is_source,
    effective_message,
    issue_references,
    keyword_bugfix,
    source_commits,
)
from tdp_core.labeling.github_issues import ISSUES_SCHEMA, is_bug_issue
from tdp_core.mining import mine_history
from tdp_core.testing import RepoBuilder

_T0 = datetime(2021, 1, 1, tzinfo=UTC)
_REVERTS = "This reverts commit 0123abcd."


def _commits(rows: Sequence[tuple[str, str, bool]]) -> pa.Table:
    """``(sha, message, is_bulk)`` rows with author times one hour apart."""
    n = len(rows)
    return pa.Table.from_pydict(
        {
            "sha": [r[0] for r in rows],
            "author_name": ["Ana"] * n,
            "author_email": ["ana@example.com"] * n,
            "author_time": [_T0 + timedelta(hours=i) for i in range(n)],
            "parents": [[] for _ in range(n)],
            "subject": [r[1].split("\n", 1)[0] for r in rows],
            "message": [r[1] for r in rows],
            "files_touched": [1] * n,
            "is_bulk": [r[2] for r in rows],
        },
        schema=COMMITS_SCHEMA,
    )


def _change(
    sha: str,
    path: str,
    added: int = 1,
    deleted: int = 0,
    old_path: str | None = None,
    binary: bool = False,
) -> dict[str, Any]:
    return {
        "sha": sha,
        "path": path,
        "old_path": old_path,
        "added": added,
        "deleted": deleted,
        "is_binary": binary,
    }


def _changes(rows: Sequence[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=FILE_CHANGES_SCHEMA)


def _issues(rows: Sequence[tuple[int, datetime, list[str]]]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "number": [r[0] for r in rows],
            "created_at": [r[1] for r in rows],
            "closed_at": [None] * len(rows),
            "labels": [r[2] for r in rows],
            "is_bug": [is_bug_issue(r[2]) for r in rows],
        },
        schema=ISSUES_SCHEMA,
    )


def _sources(table: pa.Table) -> list[str]:
    return [str(s) for s in table.column("source").to_pylist()]


def _single(message: str, issues: pa.Table | None = None, repo: str | None = None) -> dict[str, Any]:
    """Classify one commit that changes ``pkg/mod.py``."""
    table = classify_bugfix(
        _commits([("a1", message, False)]), _changes([_change("a1", "pkg/mod.py")]), issues, repo=repo
    )
    row: dict[str, Any] = table.to_pylist()[0]
    return row


# ------------------------------------------------------------------ keyword rule
@pytest.mark.parametrize(
    "message",
    [
        "Fix crash when body is empty",
        "fixes handling of None",
        "Fixed wrong encoding",
        "Handle regression in 2.1",
        "hotfix for broken redirect",
        "\U0001f41b Allow async class methods as dependencies (#681)",
        ":bug: Handle empty header",
        "\U0001f691\ufe0f Patch release for the import error",
        "fix(client): retry on connection reset",
        f'Revert "Revert "Fix crash on empty body""\n\n{_REVERTS}',
        'Reapply "Fix crash on empty body"',
        'Revert "Add cache"\n\nThe cache caused a regression in redirects.',
        "Fix crash on empty body (#12)\n\n* fix tests\n* fix typo",
        "Update docs\n\nAlso fixes a crash in the parser.",
        "Fix typing.Optional handling in get()",
        "Fix link extraction for relative URLs",
        "Fix broken redirect handling",
        "Incorrect content length for unicode bodies",
    ],
)
def test_keyword_positives(message: str) -> None:
    assert keyword_bugfix(message)
    assert _single(message)["source"] == "keyword"


@pytest.mark.parametrize(
    "message",
    [
        "fix typo",
        "prefix handling",
        "fix ci",
        "Fix a typo in the docs",
        "Fix failing tests",
        "Fix flake8 warnings",
        "Fix docstring of Session.send",
        "docs: fix broken link",
        "feat: add retries and fix crash",
        "chore(deps): bump urllib3 to fix CVE",
        "Add support for X",
        "Default to UTF-8",
        "\U0001f4dd Fix links in docs (#15967)",
        "\u270f\ufe0f Fix crash typo in messages",
        ":memo: Fix crash in docs example",
        f'Revert "Fix crash on empty body"\n\n{_REVERTS}',
        'Revert "Reapply "Fix crash on empty body""',
        "Fix typo\n\nUnrelated: the crash is still there",
        "Update docs\n\n* fix typo\n* fix crash",
        "Fix broken links",
        "Fixing the typos",
        "fix couple of typos",
        "Fix the build on Windows",
        "Fix type hints in sessions",
        "",
    ],
)
def test_keyword_negatives(message: str) -> None:
    assert not keyword_bugfix(message)
    assert _single(message)["source"] == "none"


def test_effective_message() -> None:
    assert effective_message("") == ("", "")
    assert effective_message("Fix x\n\nbody\n") == ("Fix x", "body")
    assert effective_message(f'Revert "Fix x" (#3)\n\n{_REVERTS}\nIt broke y.') == ("", "It broke y.")
    assert effective_message(f'Revert "Revert "Fix x""\n\n{_REVERTS}') == ("Fix x", "")
    assert effective_message('Reapply "Fix x"\n\nagain') == ("Fix x", "again")
    # Only quoted reverts are unwrapped; the "This reverts" line stays for anything else.
    assert effective_message(f"Revert the cache\n\n{_REVERTS}") == ("Revert the cache", _REVERTS)


# ------------------------------------------------------------------ issue references
@pytest.mark.parametrize(
    ("message", "repo", "expected"),
    [
        ("Fix #12 and GH-34", None, (12, 34)),
        ("see psf/requests#99", None, ()),
        ("see psf/requests#99", "psf/requests", (99,)),
        ("see urllib3/urllib3#99", "psf/requests", ()),
        ("use &#39; entity", None, ()),
        ("color #123abc", None, ()),
        ("Handle empty body (#45)", None, (45,)),
        ("fixes #7, closes #8, resolves #7", None, (7, 8)),
        ("https://github.com/psf/requests/issues/77", "psf/requests", (77,)),
        ("https://github.com/psf/requests/issues/77", None, ()),
        ("https://github.com/urllib3/urllib3/issues/5", "psf/requests", ()),
        ("https://github.com/PSF/Requests/issues/77#issuecomment-1", "psf/requests", (77,)),
        (f'Revert "Fix #12"\n\n{_REVERTS}', None, ()),
        ("#0 and gh-5", None, (5,)),
        ("Subject\n\nBody mentions #3", None, (3,)),
    ],
)
def test_issue_references(message: str, repo: str | None, expected: tuple[int, ...]) -> None:
    assert issue_references(message, repo) == expected


@pytest.mark.parametrize("repo", ["psf", "psf/requests/x", "/requests", "psf/"])
def test_issue_references_rejects_bad_repo(repo: str) -> None:
    with pytest.raises(ValueError):
        issue_references("#1", repo)


# ------------------------------------------------------------------ issue linking
def test_earliest_referenced_bug_issue_wins() -> None:
    issues = _issues(
        [
            (9, _T0 - timedelta(days=1), ["enhancement"]),
            (4, _T0 - timedelta(days=30), ["bug"]),
            (6, _T0 - timedelta(days=10), ["type: bug"]),
        ]
    )
    row = _single("Handle redirects\n\nSee #9, #6 and #4", issues)
    assert row["is_bugfix"] is True
    assert row["source"] == "issue"
    assert row["issue_number"] == 4
    assert row["issue_created_at"] == _T0 - timedelta(days=30)


def test_non_bug_issue_falls_back_to_keywords() -> None:
    issues = _issues([(9, _T0, ["enhancement"])])
    row = _single("Fixes #9", issues)
    assert (row["source"], row["issue_number"], row["issue_created_at"]) == ("keyword", None, None)
    assert _single("Add #9 support", issues)["source"] == "none"


def test_bug_issue_link_needs_no_keyword() -> None:
    issues = _issues([(4, _T0, ["bug"])])
    row = _single("Release redirected connections\n\n#4", issues)
    assert (row["source"], row["issue_number"]) == ("issue", 4)


def test_bug_issue_link_overrides_declared_non_fix_subject() -> None:
    issues = _issues([(4, _T0, ["bug"])])
    assert _single("docs: explain the crash (#4)", issues)["source"] == "issue"


def test_not_a_bug_verdict_does_not_link() -> None:
    issues = _issues([(4, _T0, ["type: bug", "resolution: not a bug"])])
    assert _single("Change header order (#4)", issues)["source"] == "none"


def test_same_repo_url_links_only_with_repo() -> None:
    issues = _issues([(77, _T0, ["bug"])])
    message = "Handle headers\n\nhttps://github.com/psf/requests/issues/77"
    assert _single(message, issues, repo="psf/requests")["source"] == "issue"
    assert _single(message, issues)["source"] == "none"


def test_revert_does_not_inherit_issue_link() -> None:
    issues = _issues([(12, _T0, ["bug"])])
    assert _single(f'Revert "Handle #12"\n\n{_REVERTS}', issues)["source"] == "none"


# ------------------------------------------------------------------ eligibility
def test_eligibility_rules() -> None:
    commits = _commits(
        [
            ("test", "Fix crash", False),
            ("bulk", "Fix crash", True),
            ("tlink", "Handle #4", False),
            ("rename", "Fix crash", False),
            ("binary", "Fix crash", False),
            ("moved", "Fix crash", False),
            ("vendor", "Fix crash", False),
            ("docs", "Fix crash", False),
            ("ok", "Fix crash", False),
            ("nochange", "Fix crash", False),
        ]
    )
    changes = _changes(
        [
            _change("test", "tests/test_x.py", 3, 1),
            _change("bulk", "pkg/x.py", 3, 1),
            _change("tlink", "tests/test_x.py"),
            _change("rename", "pkg/new.py", 0, 0, old_path="pkg/old.py"),
            _change("binary", "pkg/x.py", 0, 0, binary=True),
            _change("moved", "pkg/x.py", 2, 2, old_path="tests/x.py"),
            _change("vendor", "pkg/_vendor/x.py", 5, 5),
            _change("docs", "README.md", 5, 5),
            _change("docs", "pkg/x.pyc", 5, 5),
            _change("ok", "README.md", 5, 5),
            _change("ok", "pkg/x.py", 1, 0),
            _change("ghost", "pkg/x.py", 1, 0),
        ]
    )
    assert source_commits(commits, changes) == frozenset({"moved", "ok"})
    issues = _issues([(4, _T0, ["bug"])])
    table = classify_bugfix(commits, changes, issues)
    assert dict(zip(table.column("sha").to_pylist(), _sources(table), strict=True)) == {
        "test": "none",
        "bulk": "none",
        "tlink": "none",
        "rename": "none",
        "binary": "none",
        "moved": "keyword",
        "vendor": "none",
        "docs": "none",
        "ok": "keyword",
        "nochange": "none",
    }


def test_old_path_counts_for_renames_of_source_files() -> None:
    commits = _commits([("r", "Fix crash", False)])
    changes = _changes([_change("r", "docs/notes.txt", 1, 1, old_path="pkg/notes.py")])
    assert source_commits(commits, changes) == frozenset({"r"})


def test_custom_is_source_is_called_once_per_path() -> None:
    seen: list[str] = []

    def only_js(path: str) -> bool:
        seen.append(path)
        return path.endswith(".js")

    commits = _commits([("a", "Fix crash", False), ("b", "Fix crash", False), ("c", "Fix bug", False)])
    changes = _changes(
        [_change("a", "x.js"), _change("b", "x.py"), _change("c", "x.py"), _change("c", "y.js")]
    )
    table = classify_bugfix(commits, changes, None, only_js)
    assert _sources(table) == ["keyword", "none", "keyword"]
    assert sorted(seen) == ["x.js", "x.py", "y.js"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("pkg/a.py", True),
        ("src/pkg/__init__.py", True),
        ("docs/conf.py", True),
        ("setup.py", True),
        ("tests/test_a.py", False),
        ("pkg/conftest.py", False),
        ("pkg/a_test.py", False),
        ("pkg/test/helpers.py", False),
        ("README.md", False),
        ("pkg/a.pyc", False),
        ("pip/_vendor/x.py", False),
        ("third_party/lib/x.py", False),
    ],
)
def test_default_is_source(path: str, expected: bool) -> None:
    assert default_is_source(path) is expected


# ------------------------------------------------------------------ output shape
def test_output_has_one_row_per_commit_in_input_order() -> None:
    issues = _issues([(3, _T0, ["bug"])])
    commits = _commits([("c", "Add feature", False), ("a", "Fix crash", False), ("b", "Handle #3", False)])
    changes = _changes([_change(s, "pkg/x.py") for s in ("a", "b", "c")])
    table = classify_bugfix(commits, changes, issues)
    assert table.schema.equals(BUGFIX_SCHEMA)
    assert table.column("sha").to_pylist() == ["c", "a", "b"]
    assert _sources(table) == ["none", "keyword", "issue"]
    assert table.column("is_bugfix").to_pylist() == [False, True, True]
    assert table.column("issue_number").to_pylist() == [None, None, 3]
    assert table.column("issue_created_at").to_pylist() == [None, None, _T0]
    assert set(_sources(table)) <= set(BUGFIX_SOURCES)


def test_empty_commits_give_empty_table() -> None:
    table = classify_bugfix(_commits([]), _changes([]))
    assert table.num_rows == 0
    assert table.schema.equals(BUGFIX_SCHEMA)


def test_no_issues_equals_empty_issues() -> None:
    commits = _commits([("a", "Fix crash #3", False), ("b", "Handle #4", False)])
    changes = _changes([_change("a", "pkg/x.py"), _change("b", "pkg/x.py")])
    assert classify_bugfix(commits, changes, None).equals(classify_bugfix(commits, changes, _issues([])))


def test_issues_missing_columns_raise() -> None:
    commits = _commits([("a", "Fix crash", False)])
    changes = _changes([_change("a", "pkg/x.py")])
    with pytest.raises(ValueError, match="is_bug"):
        classify_bugfix(commits, changes, pa.table({"number": [1], "created_at": [_T0]}))


# ------------------------------------------------------------------ integration and speed
def test_classify_a_real_history(repo_builder: RepoBuilder, t0: datetime) -> None:
    day = timedelta(days=1)
    code = "def parse(text):\n    return text.split()\n"
    repo_builder.commit(
        {"pkg/parse.py": code, "tests/test_parse.py": "def test_parse():\n    pass\n"}, "Add parser", t0
    )
    fixed = code.replace("return", "if not text:\n        return []\n    return")
    fix_sha = repo_builder.commit({"pkg/parse.py": fixed}, "Fix crash on empty input (#3)", t0 + day)
    repo_builder.commit(
        {"tests/test_parse.py": "def test_parse():\n    assert True\n"}, "Fix flaky test", t0 + 2 * day
    )
    documented = '"""Parse text."""\n' + fixed
    repo_builder.commit({"pkg/parse.py": documented}, "Fix typo in docs", t0 + 3 * day)
    # A bulk commit: more files than bulk_commit_files (lowered to 3 so the test stays fast).
    repo_builder.commit(
        {f"pkg/gen/m{i}.py": f"X = {i}\n" for i in range(4)}, "Fix wrong encoding", t0 + 4 * day
    )
    repo_builder.commit(
        {"pkg/parse.py": '"""Parse text."""\n' + code},
        f'Revert "Fix crash on empty input (#3)"\n\nThis reverts commit {fix_sha}.',
        t0 + 5 * day,
    )
    repo_builder.commit(
        {"pkg/parse.py": '"""Parse text."""\n' + fixed + "HEADERS = None\n"},
        "Handle None headers\n\nfixes #4",
        t0 + 6 * day,
    )
    hist = mine_history(repo_builder.path, settings=with_overrides(DEFAULT_SETTINGS, bulk_commit_files=3))
    assert hist.commits.column("is_bulk").to_pylist() == [False, False, False, False, True, False, False]
    issues = _issues([(3, t0, ["bug"]), (4, t0, ["enhancement"])])
    table = classify_bugfix(hist.commits, hist.file_changes, issues, repo="test/repo")
    assert table.column("sha").to_pylist() == hist.commits.column("sha").to_pylist()
    assert _sources(table) == ["none", "issue", "none", "none", "none", "none", "keyword"]
    assert table.column("issue_number").to_pylist()[1] == 3


def test_classify_is_fast() -> None:
    n = 50_000
    messages = ["Fix crash in parser (#12)", "Add feature", "fix typo", "Handle #7\n\nfixes it"]
    commits = pa.Table.from_pydict(
        {
            "sha": [f"{i:040x}" for i in range(n)],
            "author_name": ["Ana"] * n,
            "author_email": ["ana@example.com"] * n,
            "author_time": [_T0 + timedelta(minutes=i) for i in range(n)],
            "parents": [[] for _ in range(n)],
            "subject": [messages[i % 4].split("\n")[0] for i in range(n)],
            "message": [messages[i % 4] for i in range(n)],
            "files_touched": [4] * n,
            "is_bulk": [False] * n,
        },
        schema=COMMITS_SCHEMA,
    )
    paths = ["pkg/a.py", "tests/test_a.py", "README.md", "pkg/sub/b.py"]
    changes = pa.Table.from_pydict(
        {
            "sha": [f"{i // 4:040x}" for i in range(4 * n)],
            "path": [paths[i % 4] for i in range(4 * n)],
            "old_path": [None] * (4 * n),
            "added": [1] * (4 * n),
            "deleted": [1] * (4 * n),
            "is_binary": [False] * (4 * n),
        },
        schema=FILE_CHANGES_SCHEMA,
    )
    issues = _issues([(n_, _T0, ["bug"] if n_ % 2 else ["feature"]) for n_ in range(1, 2000)])
    start = time.perf_counter()
    table = classify_bugfix(commits, changes, issues)
    elapsed = time.perf_counter() - start
    assert table.num_rows == n
    assert elapsed < 10
    assert _sources(table)[:4] == ["keyword", "none", "none", "issue"]
