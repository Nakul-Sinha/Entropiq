from datetime import datetime, timedelta

import pytest

from tdp_core.mining import GitError, list_files, read_blobs, run_git
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)


def _blob(repo: RepoBuilder, rev: str, path: str) -> str:
    return run_git(["rev-parse", f"{rev}:{path}"], repo.path).decode().strip()


def test_list_files_keeps_regular_files_only(repo_builder: RepoBuilder, t0: datetime) -> None:
    path = repo_builder.path
    first = repo_builder.commit(
        {"src/a.py": "a\n", "run.sh": "#!/bin/sh\n", "dir x/ü.py": "u\n"}, "files", t0
    )
    target = run_git(["hash-object", "-w", "--stdin"], path, input=b"src/a.py").decode().strip()
    run_git(["update-index", "--add", "--cacheinfo", f"120000,{target},link.py"], path)
    run_git(["update-index", "--add", "--cacheinfo", f"160000,{first},vendor/sub"], path)
    run_git(["update-index", "--chmod=+x", "run.sh"], path)
    sha = repo_builder.commit({}, "symlink, submodule, executable", t0 + DAY)

    files = list_files(path, sha)

    assert sorted(files) == ["dir x/ü.py", "run.sh", "src/a.py"]
    assert files["src/a.py"] == _blob(repo_builder, sha, "src/a.py")
    assert files["run.sh"] == _blob(repo_builder, sha, "run.sh")


def test_list_files_at_older_commit(repo_builder: RepoBuilder, t0: datetime) -> None:
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    repo_builder.commit({"b.py": "2\n", "a.py": None}, "two", t0 + DAY)

    assert list(list_files(repo_builder.path, first)) == ["a.py"]
    assert list(list_files(repo_builder.path, "HEAD")) == ["b.py"]
    with pytest.raises(ValueError):
        list_files(repo_builder.path, "--output=x")
    with pytest.raises(GitError):
        list_files(repo_builder.path, "0" * 40)


def test_read_blobs_returns_exact_bytes(repo_builder: RepoBuilder, t0: datetime) -> None:
    payloads = {
        "bin.dat": bytes(range(256)) * 3,
        "crlf.txt": b"a\r\nb\r\n",
        "no_newline.txt": b"no newline at end",
        "empty.txt": b"",
        "newlines.txt": b"\n\n\n",
        "utf8.py": "s = 'ünïcödé ✓'\n".encode(),
    }
    sha = repo_builder.commit(payloads, "blobs", t0)
    files = list_files(repo_builder.path, sha)
    wanted = [files[name] for name in payloads]

    blobs = read_blobs(repo_builder.path, [*wanted, wanted[0]])  # duplicates are read once

    assert list(blobs) == wanted
    for name, data in payloads.items():
        assert blobs[files[name]] == data


def test_read_blobs_edge_cases(repo_builder: RepoBuilder, t0: datetime) -> None:
    sha = repo_builder.commit({"a.py": "1\n"}, "one", t0)

    assert read_blobs(repo_builder.path, []) == {}
    with pytest.raises(GitError, match="missing"):
        read_blobs(repo_builder.path, ["f" * 40])
    with pytest.raises(GitError, match="not a blob"):
        read_blobs(repo_builder.path, [sha])
    with pytest.raises(ValueError):
        read_blobs(repo_builder.path, ["abc def"])
