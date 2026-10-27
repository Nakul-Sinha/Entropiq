"""Fixtures for the mining tests: a history whose author dates and commit dates disagree."""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tdp_core.testing import RepoBuilder

DAY = timedelta(days=1)


def _git(
    repo: Path,
    config: Path,
    *args: str,
    authored: datetime | None = None,
    committed: datetime | None = None,
) -> str:
    """Run git in ``repo`` isolated from the user's config, optionally pinning the author or commit date."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_"))}
    env.update({"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    if authored is not None:
        env["GIT_AUTHOR_DATE"] = authored.isoformat()
    if committed is not None:
        env["GIT_COMMITTER_DATE"] = committed.isoformat()
    result = subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    return result.stdout.decode("utf-8", errors="replace").strip()


@pytest.fixture
def leaky_history(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> dict[str, str]:
    """A history where picking states by author date leaks; returns commit name -> sha.

    Dates are days after ``t0`` as author/commit. ``base`` was amended a day later; ``late`` was authored on
    day 4 but rebased onto ``main`` on day 20 (it keeps its author date)::

        main:     base (0/1) -- main2 (3/3) -- merge (10/10) -- late (4/20) -- main3 (25/25)
                      \\                       /
        feature:       feat (2/2) -----------

    The first-parent chain of ``main`` is base, main2, merge, late, main3; ``feat`` is reachable only
    through ``merge``.
    """
    repo = repo_builder.path
    config = tmp_path / "empty.gitconfig"
    config.touch()

    def day(n: float) -> datetime:
        return t0 + n * DAY

    def restamp(committed: datetime) -> str:
        _git(repo, config, "commit", "-q", "--amend", "--no-edit", committed=committed)
        return repo_builder.head()

    repo_builder.commit({"a.py": "a = 0\n"}, "base", day(0))
    base = restamp(day(1))
    _git(repo, config, "checkout", "-q", "-b", "feature")
    feat = repo_builder.commit({"b.py": "b = 2\n"}, "feat", day(2))
    _git(repo, config, "checkout", "-q", "main")
    main2 = repo_builder.commit({"a.py": "a = 3\n"}, "main2", day(3))
    _git(
        repo, config, "merge", "-q", "--no-ff", "-m", "merge", "feature", authored=day(10), committed=day(10)
    )
    merge = repo_builder.head()
    repo_builder.commit({"c.py": "c = 4\n"}, "late", day(4))
    late = restamp(day(20))
    main3 = repo_builder.commit({"a.py": "a = 25\n"}, "main3", day(25))
    return {"base": base, "feat": feat, "main2": main2, "merge": merge, "late": late, "main3": main3}
