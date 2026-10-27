from datetime import datetime, timedelta
from typing import Any

import pytest

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA
from tdp_core.mining import GitError, MinedHistory, mine_history, parse_git_log, run_git
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)


def _row(sha: str, path: str, added: int, deleted: int, **kw: Any) -> dict[str, Any]:
    return {
        "sha": sha,
        "path": path,
        "old_path": kw.get("old_path"),
        "added": added,
        "deleted": deleted,
        "is_binary": kw.get("is_binary", False),
    }


def _commit(history: MinedHistory, sha: str) -> dict[str, Any]:
    return next(c for c in history.commits.to_pylist() if c["sha"] == sha)


def test_two_commit_history_counts(repo_builder: RepoBuilder, t0: datetime) -> None:
    c1 = repo_builder.commit({"a.py": "one\ntwo\nthree\n", "b.py": "x\n"}, "add files", t0)
    c2 = repo_builder.commit({"a.py": "one\nTWO\nthree\nfour\n"}, "edit a", t0 + DAY)

    h = mine_history(repo_builder.path)

    assert h.commits.schema.equals(COMMITS_SCHEMA)
    assert h.file_changes.schema.equals(FILE_CHANGES_SCHEMA)
    assert h.commits.column("sha").to_pylist() == [c1, c2]
    assert h.file_changes.to_pylist() == [
        _row(c1, "a.py", 3, 0),
        _row(c1, "b.py", 1, 0),
        _row(c2, "a.py", 2, 1),
    ]
    first, second = h.commits.to_pylist()
    assert first["author_time"] == t0
    assert second["author_time"] == t0 + DAY
    assert first["parents"] == []
    assert second["parents"] == [c1]
    assert (first["files_touched"], second["files_touched"]) == (2, 1)
    assert first["author_name"] == "Ana"
    assert first["author_email"] == "ana@example.com"


def test_subject_and_full_message(repo_builder: RepoBuilder, t0: datetime) -> None:
    sha = repo_builder.commit({"a.py": "1\n"}, "Fix the parser\n\nLonger body.\nSecond line.\n", t0)
    c = _commit(mine_history(repo_builder.path), sha)
    assert c["subject"] == "Fix the parser"
    assert c["message"] == "Fix the parser\n\nLonger body.\nSecond line."


def test_deletion_is_zero_added(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n2\n3\n"}, "add", t0)
    gone = repo_builder.commit({"a.py": None}, "delete", t0 + DAY)
    h = mine_history(repo_builder.path)
    assert [r for r in h.file_changes.to_pylist() if r["sha"] == gone] == [_row(gone, "a.py", 0, 3)]


def test_rename_keeps_old_path(repo_builder: RepoBuilder, t0: datetime) -> None:
    body = "".join(f"line {i}\n" for i in range(10))
    repo_builder.commit({"a.py": body}, "add", t0)
    sha = repo_builder.rename("a.py", "pkg/b.py", "move", t0 + DAY, new_content=body + "extra\n")

    rows = [r for r in mine_history(repo_builder.path).file_changes.to_pylist() if r["sha"] == sha]

    assert rows == [_row(sha, "pkg/b.py", 1, 0, old_path="a.py")]


def test_paths_with_spaces_and_unicode(repo_builder: RepoBuilder, t0: datetime) -> None:
    paths = ["dir with space/my file.py", "naïve/ünïcödé ✓.py", "日本/モジュール.py"]
    sha = repo_builder.commit(dict.fromkeys(paths, "x = 1\n"), "odd names", t0)

    h = mine_history(repo_builder.path)

    assert sorted(h.file_changes.column("path").to_pylist()) == sorted(paths)
    assert _commit(h, sha)["files_touched"] == 3


def test_binary_file(repo_builder: RepoBuilder, t0: datetime) -> None:
    c1 = repo_builder.commit({"logo.png": b"\x89PNG\x00\x01\x02" * 20, "a.py": "x\n"}, "add", t0)
    c2 = repo_builder.commit({"logo.png": b"\x89PNG\x00\xff" * 30}, "new logo", t0 + DAY)

    rows = mine_history(repo_builder.path).file_changes.to_pylist()

    assert _row(c1, "logo.png", 0, 0, is_binary=True) in rows
    assert _row(c1, "a.py", 1, 0) in rows
    assert _row(c2, "logo.png", 0, 0, is_binary=True) in rows


def test_bulk_flag_above_threshold(repo_builder: RepoBuilder, t0: datetime) -> None:
    limit = DEFAULT_SETTINGS.bulk_commit_files
    big = repo_builder.commit({f"f{i:03}.py": f"{i}\n" for i in range(limit + 1)}, "vendor", t0)
    edge = repo_builder.commit({f"f{i:03}.py": f"{i}!\n" for i in range(limit)}, "reformat", t0 + DAY)

    h = mine_history(repo_builder.path)

    assert (_commit(h, big)["files_touched"], _commit(h, big)["is_bulk"]) == (51, True)
    assert (_commit(h, edge)["files_touched"], _commit(h, edge)["is_bulk"]) == (50, False)


def test_mailmap_merges_two_emails(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0, author=("Ana Lima", "ana@example.com"))
    repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY, author=("ana", "Ana.Lima@OldCorp.COM"))
    repo_builder.write_mailmap(["Ana Lima <ana@example.com> <ana.lima@oldcorp.com>"])

    commits = mine_history(repo_builder.path).commits.to_pylist()

    assert {(c["author_name"], c["author_email"]) for c in commits} == {("Ana Lima", "ana@example.com")}


def test_email_lowercased_without_mailmap(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0, author=("Bob", "Bob.Smith@Example.COM"))
    (c,) = mine_history(repo_builder.path).commits.to_pylist()
    assert c["author_email"] == "bob.smith@example.com"


def test_since_filter(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "old", t0)
    mid = repo_builder.commit({"a.py": "2\n"}, "mid", t0 + 10 * DAY)
    new = repo_builder.commit({"b.py": "3\n"}, "new", t0 + 20 * DAY)

    h = mine_history(repo_builder.path, since=t0 + 5 * DAY)

    assert h.commits.column("sha").to_pylist() == [mid, new]
    assert set(h.file_changes.column("sha").to_pylist()) == {mid, new}


def test_max_commits_keeps_newest(repo_builder: RepoBuilder, t0: datetime) -> None:
    shas = [repo_builder.commit({"a.py": f"{i}\n"}, f"c{i}", t0 + i * DAY) for i in range(4)]

    assert mine_history(repo_builder.path, max_commits=2).commits.column("sha").to_pylist() == shas[2:]
    empty = mine_history(repo_builder.path, max_commits=0)
    assert empty.commits.num_rows == 0
    assert empty.commits.schema.equals(COMMITS_SCHEMA)
    assert empty.file_changes.schema.equals(FILE_CHANGES_SCHEMA)
    with pytest.raises(ValueError):
        mine_history(repo_builder.path, max_commits=-1)


def test_sorted_by_author_time_not_topology(repo_builder: RepoBuilder, t0: datetime) -> None:
    parent = repo_builder.commit({"a.py": "1\n"}, "authored later", t0 + 10 * DAY)
    child = repo_builder.commit({"a.py": "2\n"}, "authored earlier", t0)

    h = mine_history(repo_builder.path)

    assert h.commits.column("sha").to_pylist() == [child, parent]
    assert h.file_changes.column("sha").to_pylist() == [child, parent]


def test_empty_commit_has_no_changes(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "add", t0)
    empty = repo_builder.commit({}, "nothing", t0 + DAY)
    h = mine_history(repo_builder.path)
    assert _commit(h, empty)["files_touched"] == 0
    assert empty not in h.file_changes.column("sha").to_pylist()


def test_merge_commits_excluded(repo_builder: RepoBuilder, t0: datetime) -> None:
    path = repo_builder.path
    base = repo_builder.commit({"a.py": "1\n"}, "base", t0)
    run_git(["checkout", "-q", "-b", "feature"], path)
    feature = repo_builder.commit({"b.py": "1\n"}, "feature work", t0 + DAY)
    run_git(["checkout", "-q", "main"], path)
    main_work = repo_builder.commit({"c.py": "1\n"}, "main work", t0 + 2 * DAY)
    run_git(["merge", "-q", "--no-ff", "-m", "merge feature", "feature"], path)

    h = mine_history(path)

    assert sorted(h.commits.column("sha").to_pylist()) == sorted([base, feature, main_work])
    assert all(len(p) <= 1 for p in h.commits.column("parents").to_pylist())


def test_rev_selects_branch_and_rejects_options(repo_builder: RepoBuilder, t0: datetime) -> None:
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY)

    assert mine_history(repo_builder.path, first).commits.column("sha").to_pylist() == [first]
    with pytest.raises(ValueError):
        mine_history(repo_builder.path, "--output=pwned.txt")
    with pytest.raises(GitError):
        mine_history(repo_builder.path, "no-such-branch")


def test_parse_tolerates_separators_in_message_and_tabs_in_paths() -> None:
    sha1, sha2 = "a" * 40, "b" * 64  # sha-256 repositories use 64 hex digits
    raw = (
        f"\x1e{sha1}\x1fAna\x1fANA@X.COM\x1f1641211200\x1f\x1f"
        "odd \x1e message \x1f here\r\nbody\n\x1f\x00"
        "\n3\t1\t\x00old name.py\x00new\tname.py\x00-\t-\tlogo.png\x002\t0\tweird\tpath.py\x00"
        f"\x1e{sha2}\x1fBo\x1fbo@x\x1f1641211300\x1f{sha1}\x1fsecond\n\x1f\x00"
    ).encode()

    h = parse_git_log(raw)

    first, second = h.commits.to_pylist()
    assert first["subject"] == "odd \x1e message \x1f here"
    assert first["message"] == "odd \x1e message \x1f here\r\nbody"
    assert first["author_email"] == "ana@x.com"
    assert first["files_touched"] == 3
    assert second["parents"] == [sha1]
    assert second["files_touched"] == 0
    assert h.file_changes.to_pylist() == [
        _row(sha1, "new\tname.py", 3, 1, old_path="old name.py"),
        _row(sha1, "logo.png", 0, 0, is_binary=True),
        _row(sha1, "weird\tpath.py", 2, 0),
    ]


def test_parse_rejects_garbage() -> None:
    with pytest.raises(GitError):
        parse_git_log(b"\x1enot a git log record")
    assert parse_git_log(b"").commits.num_rows == 0
