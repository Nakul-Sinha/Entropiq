from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tdp_core.mining import GitError, ensure_mirror, mine_history, run_git
from tdp_core.mining.mirror import FETCH_REFSPEC
from tdp_core.mining.reach import default_branch
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)


def _refs(repo: Path) -> set[str]:
    out = run_git(["for-each-ref", "--format=%(refname)"], repo).decode()
    return set(out.split())


def test_clone_then_fetch(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> None:
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    run_git(["branch", "doomed"], repo_builder.path)
    run_git(["tag", "v1"], repo_builder.path)
    dest = tmp_path / "mirrors" / "acme__widget.git"

    assert ensure_mirror(repo_builder.path.as_uri(), dest, allow_file_protocol=True) == dest
    assert run_git(["rev-parse", "--is-bare-repository"], dest).strip() == b"true"
    assert run_git(["rev-parse", "HEAD"], dest).strip() == first.encode()
    assert {"refs/heads/main", "refs/heads/doomed", "refs/tags/v1"} <= _refs(dest)

    second = repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY)
    run_git(["branch", "-D", "doomed"], repo_builder.path)
    ensure_mirror(repo_builder.path.as_uri(), dest, allow_file_protocol=True)

    assert run_git(["rev-parse", "HEAD"], dest).strip() == second.encode()
    assert "refs/heads/doomed" not in _refs(dest)
    assert mine_history(dest).commits.column("sha").to_pylist() == [first, second]


def test_plain_local_path_needs_opt_in(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    dest = tmp_path / "m.git"

    with pytest.raises(GitError, match="not allowed"):
        ensure_mirror(str(repo_builder.path), dest)
    assert not dest.exists()

    ensure_mirror(str(repo_builder.path), dest, allow_file_protocol=True)
    assert (dest / "HEAD").is_file()


def test_non_repo_dest_inside_a_checkout_is_left_alone(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    # dest sits inside another checkout (like data/mirrors/ inside this repo) but is not a mirror itself.
    outer = RepoBuilder(tmp_path / "outer")
    outer_head = outer.commit({"b.py": "1\n"}, "outer", t0)
    run_git(["remote", "add", "origin", repo_builder.path.as_uri()], outer.path)
    dest = outer.path / "data" / "mirrors" / "broken.git"
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(GitError):
        ensure_mirror(repo_builder.path.as_uri(), dest, allow_file_protocol=True)
    assert (dest / "notes.txt").read_text(encoding="utf-8") == "keep me"
    # git did not fall back to fetching into the enclosing checkout.
    assert "refs/remotes/origin/main" not in _refs(outer.path)
    assert run_git(["rev-parse", "HEAD"], outer.path).strip() == outer_head.encode()


def test_url_cannot_be_an_option(tmp_path: Path) -> None:
    dest = tmp_path / "m.git"
    with pytest.raises(GitError):
        ensure_mirror("--upload-pack=touch pwned", dest, allow_file_protocol=True)
    assert not (tmp_path / "pwned").exists()


def _remote_config(dest: Path) -> dict[str, str]:
    out = run_git([f"--git-dir={dest}", "config", "--get-regexp", r"^remote\."], dest).decode()
    return dict(line.split(" ", 1) for line in out.splitlines())


def test_new_mirror_is_lean_and_stays_lean(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> None:
    upstream = repo_builder.path
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    run_git(["branch", "topic"], upstream)
    run_git(["update-ref", "refs/tags/v1", first], upstream)
    run_git(["update-ref", "refs/pull/1/head", first], upstream)
    dest = tmp_path / "lean.git"

    ensure_mirror(upstream.as_uri(), dest, allow_file_protocol=True)

    assert _refs(dest) == {"refs/heads/main", "refs/heads/topic", "refs/tags/v1"}
    assert _remote_config(dest) == {
        "remote.origin.url": upstream.as_uri(),
        "remote.origin.fetch": FETCH_REFSPEC,
    }

    second = repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY)
    run_git(["branch", "-D", "topic"], upstream)
    run_git(["branch", "topic2"], upstream)
    run_git(["update-ref", "refs/pull/2/head", second], upstream)
    ensure_mirror(upstream.as_uri(), dest, allow_file_protocol=True)

    assert _refs(dest) == {"refs/heads/main", "refs/heads/topic2", "refs/tags/v1"}
    assert run_git(["rev-parse", "refs/heads/main"], dest).strip() == second.encode()
    assert default_branch(dest) == "main"


def test_updates_a_mirror_laid_out_like_the_prefetched_ones(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    upstream = repo_builder.path
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    dest = tmp_path / "prefetched.git"
    # Same layout as the prefetched mirrors: bare, branch-only refspec, HEAD on the default branch.
    run_git(["init", "-q", "--bare", str(dest)], tmp_path)
    run_git([f"--git-dir={dest}", "remote", "add", "origin", upstream.as_uri()], dest)
    run_git([f"--git-dir={dest}", "config", "remote.origin.fetch", FETCH_REFSPEC], dest)
    run_git([f"--git-dir={dest}", "symbolic-ref", "HEAD", "refs/heads/main"], dest)
    run_git([f"--git-dir={dest}", "fetch", "-q", "--tags", "origin"], dest, allow_file_protocol=True)

    second = repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY)
    ensure_mirror(upstream.as_uri(), dest, allow_file_protocol=True)

    assert mine_history(dest).commits.column("sha").to_pylist() == [first, second]
    assert _remote_config(dest)["remote.origin.fetch"] == FETCH_REFSPEC
    assert default_branch(dest) == "main"


def test_old_full_mirror_stops_fetching_pull_refs(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    upstream = repo_builder.path
    first = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    run_git(["update-ref", "refs/pull/1/head", first], upstream)
    dest = tmp_path / "full.git"
    run_git(
        ["clone", "-q", "--mirror", "--", upstream.as_uri(), str(dest)], tmp_path, allow_file_protocol=True
    )
    assert "refs/pull/1/head" in _refs(dest)

    second = repo_builder.commit({"a.py": "2\n"}, "two", t0 + DAY)
    run_git(["update-ref", "refs/pull/2/head", second], upstream)
    ensure_mirror(upstream.as_uri(), dest, allow_file_protocol=True)

    refs = _refs(dest)
    assert run_git(["rev-parse", "refs/heads/main"], dest).strip() == second.encode()
    assert "refs/pull/2/head" not in refs
    assert "refs/pull/1/head" in refs  # refs an old mirror already holds are left alone


@pytest.mark.network
def test_github_mirror_is_lean(tmp_path: Path) -> None:
    dest = tmp_path / "delgan__loguru.git"
    for _ in range(2):  # clone, then update
        ensure_mirror("https://github.com/Delgan/loguru.git", dest)
        refs = _refs(dest)
        assert not any(ref.startswith("refs/pull/") for ref in refs)
        assert f"refs/heads/{default_branch(dest)}" in refs
