from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

import pytest

from tdp_core.mining import reach
from tdp_core.mining.git import GitError, run_git
from tdp_core.mining.gitlog import mine_history
from tdp_core.mining.mirror import ensure_mirror
from tdp_core.mining.reach import default_branch, reachable_shas, snapshot_sha
from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)
SECOND = timedelta(seconds=1)
IST = timezone(timedelta(hours=5, minutes=30))


def test_default_branch_names_the_head_branch(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    assert default_branch(repo_builder.path) == "main"

    trunk = RepoBuilder(tmp_path / "trunk", branch="trunk")
    trunk.commit({"a.py": "1\n"}, "one", t0)
    assert default_branch(trunk.path) == "trunk"

    mirror = ensure_mirror(trunk.path.as_uri(), tmp_path / "trunk.git", allow_file_protocol=True)
    assert default_branch(mirror) == "trunk"


def test_default_branch_edge_cases(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No commits yet: the branch is unborn, but HEAD still names it.
    empty = RepoBuilder(tmp_path / "empty")
    assert default_branch(empty.path) == "main"
    with pytest.raises(GitError):
        snapshot_sha(empty.path, t0)

    path = repo_builder.path
    sha = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    run_git(["checkout", "-q", "--detach"], path)
    with pytest.raises(GitError, match="detached"):
        default_branch(path)
    with pytest.raises(GitError, match="detached"):
        snapshot_sha(path, t0)
    assert snapshot_sha(path, t0, "main") == sha

    # HEAD pointing outside refs/heads keeps the full ref name.
    run_git(["update-ref", "refs/remotes/origin/main", sha], path)
    run_git(["symbolic-ref", "HEAD", "refs/remotes/origin/main"], path)
    assert default_branch(path) == "refs/remotes/origin/main"
    assert snapshot_sha(path, t0) == sha

    # Any other git failure is passed through unchanged.
    failure = GitError("git broke", returncode=128, stderr="fatal: broken")

    def broken(*args: object, **kwargs: object) -> bytes:
        raise failure

    monkeypatch.setattr(reach, "run_git", broken)
    with pytest.raises(GitError) as info:
        default_branch(path)
    assert info.value is failure


def test_snapshot_sha_boundaries(repo_builder: RepoBuilder, t0: datetime) -> None:
    c1 = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    c2 = repo_builder.commit({"a.py": "2\n"}, "two", t0 + 10 * DAY)
    path = repo_builder.path

    cases: list[tuple[datetime, str | None]] = [
        (t0 - SECOND, None),
        (t0, c1),
        (t0 + 10 * DAY - SECOND, c1),
        (t0 + 10 * DAY, c2),
        (datetime(2030, 1, 1, tzinfo=UTC), c2),
        ((t0 + 10 * DAY).astimezone(IST), c2),
        # A bare "@<ts>" is misread by git before 1973 and returns the tip; the state must still be None.
        (datetime(1970, 1, 2, tzinfo=UTC), None),
        (datetime(1960, 1, 1, tzinfo=UTC), None),
    ]
    assert [snapshot_sha(path, t) for t, _ in cases] == [want for _, want in cases]
    with pytest.raises(ValueError):
        snapshot_sha(path, datetime(2022, 6, 1))


def test_snapshot_sha_uses_commit_dates_on_the_first_parent_chain(
    repo_builder: RepoBuilder, t0: datetime, leaky_history: dict[str, str]
) -> None:
    h = leaky_history
    path = repo_builder.path

    def day(n: float) -> datetime:
        return t0 + n * DAY

    cases: list[tuple[datetime, str | None]] = [
        (day(0.5), None),  # base was authored on day 0 but only committed on day 1
        (day(1) - SECOND, None),
        (day(1), h["base"]),
        (day(2.5), h["base"]),  # feat is off the first-parent chain until the merge
        (day(3), h["main2"]),
        (day(10) - SECOND, h["main2"]),
        (day(10), h["merge"]),
        (day(19), h["merge"]),  # late was authored on day 4 but landed on day 20
        (day(20), h["late"]),
        (day(30), h["main3"]),
    ]
    got = [snapshot_sha(path, t) for t, _ in cases]
    assert got == [want for _, want in cases]

    # States only move forward along the chain, so the histories behind them are nested.
    reached = [reachable_shas(path, sha) for sha in got if sha is not None]
    assert all(a <= b for a, b in pairwise(reached))


def test_snapshot_sha_branch_argument(
    repo_builder: RepoBuilder, t0: datetime, leaky_history: dict[str, str]
) -> None:
    h = leaky_history
    path = repo_builder.path

    assert snapshot_sha(path, t0 + 30 * DAY, "feature") == h["feat"]
    assert snapshot_sha(path, t0 + 30 * DAY, "refs/heads/feature") == h["feat"]
    assert snapshot_sha(path, t0 + 1.5 * DAY, "feature") == h["base"]

    # A tag named like the branch must not shadow it.
    run_git(["update-ref", "refs/tags/main", h["base"]], path)
    assert snapshot_sha(path, t0 + 30 * DAY) == h["main3"]
    assert snapshot_sha(path, t0 + 30 * DAY, "main") == h["main3"]

    with pytest.raises(GitError):
        snapshot_sha(path, t0 + 30 * DAY, "no-such-branch")
    with pytest.raises(ValueError):
        snapshot_sha(path, t0 + 30 * DAY, "--output=x")


def test_reachable_shas(repo_builder: RepoBuilder, leaky_history: dict[str, str]) -> None:
    h = leaky_history
    path = repo_builder.path
    non_merges = {"base", "feat", "main2", "late", "main3"}

    expected = {
        "base": {"base"},
        "feat": {"base", "feat"},
        "main2": {"base", "main2"},
        "merge": {"base", "feat", "main2"},  # the merge itself is not a history commit
        "late": {"base", "feat", "main2", "late"},
        "main3": non_merges,
    }
    for name, names in expected.items():
        got = reachable_shas(path, h[name])
        assert isinstance(got, frozenset)
        assert got == {h[n] for n in names}
        assert got == set(mine_history(path, h[name]).commits.column("sha").to_pylist())

    with pytest.raises(GitError):
        reachable_shas(path, "0" * 40)
    with pytest.raises(ValueError):
        reachable_shas(path, "-n")


def test_bare_mirror_gives_the_same_states(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path, leaky_history: dict[str, str]
) -> None:
    path = repo_builder.path
    mirror = ensure_mirror(path.as_uri(), tmp_path / "leaky.git", allow_file_protocol=True)

    for d in (0.5, 1, 2.5, 5, 12, 22, 30):
        t = t0 + d * DAY
        assert snapshot_sha(mirror, t) == snapshot_sha(path, t)
    merge = leaky_history["merge"]
    assert reachable_shas(mirror, merge) == reachable_shas(path, merge)
