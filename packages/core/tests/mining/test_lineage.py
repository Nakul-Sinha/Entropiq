from datetime import datetime, timedelta

import pytest

from tdp_core.mining import mine_history, path_lineage
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)
BODY = "".join(f"line {i}\n" for i in range(10))


def _lineage(repo: RepoBuilder, until: datetime, since: datetime | None = None) -> dict[str, frozenset[str]]:
    h = mine_history(repo.path, since=since)
    return path_lineage(h.file_changes, h.commits, until)


def test_lineage_through_two_renames(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": BODY, "other.py": "x\n"}, "add", t0)
    repo_builder.rename("a.py", "b.py", "first move", t0 + DAY)
    repo_builder.rename("b.py", "pkg/c.py", "second move", t0 + 2 * DAY, new_content=BODY + "more\n")

    assert _lineage(repo_builder, t0 + 3 * DAY) == {
        "other.py": frozenset({"other.py"}),
        "pkg/c.py": frozenset({"a.py", "b.py", "pkg/c.py"}),
    }
    # Only renames up to `until` count.
    assert _lineage(repo_builder, t0 + DAY)["b.py"] == frozenset({"a.py", "b.py"})
    assert _lineage(repo_builder, t0)["a.py"] == frozenset({"a.py"})


def test_delete_then_recreate_starts_new_lineage(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"old.py": BODY}, "add", t0)
    repo_builder.rename("old.py", "a.py", "rename", t0 + DAY)
    repo_builder.commit({"a.py": None}, "delete", t0 + 2 * DAY)
    repo_builder.commit({"a.py": "brand new\n"}, "recreate", t0 + 3 * DAY)

    assert "a.py" not in _lineage(repo_builder, t0 + 2 * DAY)
    assert _lineage(repo_builder, t0 + 3 * DAY) == {"a.py": frozenset({"a.py"})}


def test_removing_some_lines_keeps_lineage(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": BODY}, "add", t0)
    repo_builder.rename("a.py", "b.py", "rename", t0 + DAY)
    repo_builder.commit({"b.py": "line 0\n"}, "shrink", t0 + 2 * DAY)  # 0 added, 9 deleted, still alive

    assert _lineage(repo_builder, t0 + 3 * DAY) == {"b.py": frozenset({"a.py", "b.py"})}


def test_rename_of_file_older_than_since_window(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": BODY}, "add", t0)
    repo_builder.commit({"a.py": "line 0\nline 1\n"}, "shrink", t0 + 10 * DAY)
    repo_builder.rename("a.py", "b.py", "rename", t0 + 20 * DAY)
    repo_builder.commit({"b.py": "line 0\n"}, "shrink again", t0 + 30 * DAY)

    # The window starts after the file was created, so its line count is unknown and is never "deleted".
    lineage = _lineage(repo_builder, t0 + 40 * DAY, since=t0 + 5 * DAY)

    assert lineage == {"b.py": frozenset({"a.py", "b.py"})}


def test_requires_aware_until(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"a.py": "1\n"}, "add", t0)
    h = mine_history(repo_builder.path)
    with pytest.raises(ValueError):
        path_lineage(h.file_changes, h.commits, t0.replace(tzinfo=None))
