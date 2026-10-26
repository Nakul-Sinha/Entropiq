import subprocess
from datetime import datetime, timedelta

from tdp_core.testing import RepoBuilder


def _log(repo: RepoBuilder, fmt: str) -> list[str]:
    out = subprocess.run(
        ["git", "log", f"--format={fmt}"], cwd=repo.path, check=True, capture_output=True, text=True
    )
    return out.stdout.strip().splitlines()


def test_commits_have_given_dates_and_authors(repo_builder: RepoBuilder, t0: datetime) -> None:
    first = repo_builder.commit({"a.py": "x = 1\n"}, "add a", t0)
    second = repo_builder.commit(
        {"a.py": "x = 2\n"}, "change a", t0 + timedelta(days=3), author=("Bo", "bo@example.com")
    )
    assert first != second
    assert _log(repo_builder, "%an|%ae|%at|%s") == [
        f"Bo|bo@example.com|{int((t0 + timedelta(days=3)).timestamp())}|change a",
        f"Ana|ana@example.com|{int(t0.timestamp())}|add a",
    ]


def test_delete_rename_and_exact_bytes(repo_builder: RepoBuilder, t0: datetime) -> None:
    repo_builder.commit({"pkg/a.py": "a = 1\r\n", "b.py": "b\n"}, "init", t0)
    repo_builder.commit({"b.py": None}, "drop b", t0 + timedelta(days=1))
    repo_builder.rename("pkg/a.py", "pkg/sub/c.py", "move a", t0 + timedelta(days=2))
    assert not (repo_builder.path / "b.py").exists()
    assert (repo_builder.path / "pkg/sub/c.py").read_bytes() == b"a = 1\r\n"
    assert len(_log(repo_builder, "%H")) == 3
