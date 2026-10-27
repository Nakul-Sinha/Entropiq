"""Leakage-safe repository states: the default branch, the state at a time T and the history behind it.

The state of a repository at time T is the newest commit on the **first-parent chain of the default branch**
whose **committer date** is ``<= T``, and "history up to T" is the set of non-merge commits reachable from
that commit. Committer dates are used because they record when a commit landed: a rebased or cherry-picked
commit keeps its old author date, and a branch commit authored before T may only be merged after T, so
picking states by author date can describe a tree that already holds work from after T. Walking the first
parent of the default branch follows what the branch itself looked like over time, so side-branch commits
appear only once they were merged. Ages and decay weights elsewhere still use ``author_time``.

Every git call goes through :func:`tdp_core.mining.git.run_git`. ``repo_dir`` may be a work tree or a bare
mirror; git finds the repository from it the same way as the other mining readers do.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tdp_core.mining.git import GitError, check_rev, run_git
from tdp_core.mining.gitlog import _unix_seconds

_HEADS = "refs/heads/"


def _head_ref(repo_dir: Path) -> str:
    """The full ref ``HEAD`` points at, for example ``refs/heads/main``.

    Raises ``GitError`` when ``HEAD`` is detached (git exits with 1) and re-raises any other ``GitError``
    from git unchanged.
    """
    try:
        out = run_git(["symbolic-ref", "--quiet", "HEAD"], repo_dir)
    except GitError as exc:
        if exc.returncode == 1:
            raise GitError(
                f"HEAD of {repo_dir} is detached, so it names no default branch",
                cmd=exc.cmd,
                returncode=exc.returncode,
                stderr=exc.stderr,
            ) from exc
        raise
    return out.decode("utf-8", errors="replace").strip()


def _qualify(branch: str) -> str:
    """``refs/heads/<branch>`` for a plain branch name; a full ref starting with ``refs/`` is kept as is."""
    return branch if branch.startswith("refs/") else _HEADS + branch


def default_branch(repo_dir: Path) -> str:
    """The default branch of ``repo_dir``: the branch ``HEAD`` points at, for example ``"main"`` or ``"dev"``.

    In a bare mirror ``HEAD`` is the upstream default branch recorded at clone time. The ``refs/heads/``
    prefix is removed; when ``HEAD`` points outside ``refs/heads`` the full ref name is returned. In a
    repository without commits the branch is unborn: its name is still returned, and :func:`snapshot_sha`
    then raises ``GitError``.

    Raises ``GitError`` when ``HEAD`` is detached or git cannot run in ``repo_dir``.
    """
    return _head_ref(repo_dir).removeprefix(_HEADS)


def snapshot_sha(repo_dir: Path, t: datetime, branch: str | None = None) -> str | None:
    """The sha of the repository state at ``t``, or ``None`` when no such state exists yet.

    Walking the first-parent chain back from the tip of ``branch``, this is the first commit with committer
    date ``<= t`` (``git rev-list -1 --first-parent --before``); a commit exactly at ``t`` counts. The result
    can be a merge commit. ``t`` must be timezone-aware and is compared in whole seconds, which is exact
    because git stores committer dates in whole seconds. ``None`` means every commit on the chain is newer
    than ``t`` (a ``t`` before 1970 always gives ``None``).

    ``branch`` defaults to the branch ``HEAD`` points at (see :func:`default_branch`). A plain name such as
    ``"main"`` means ``refs/heads/main``, so a tag with the same name cannot shadow the branch; a full ref
    starting with ``refs/`` is used as is. Other revisions (a sha, ``HEAD``) are not accepted.

    For ``t2 >= t1`` the state at ``t2`` is the state at ``t1`` or a first-parent descendant of it, so the
    sets from :func:`reachable_shas` are nested and the state at ``t2`` exists whenever the one at ``t1``
    does.

    Raises ``ValueError`` for a naive ``t`` or a ``branch`` that is empty or looks like an option, and
    ``GitError`` for an unknown branch, an unborn branch, or a detached ``HEAD`` when ``branch`` is None.
    """
    cutoff = _unix_seconds(t)
    if branch is not None:
        check_rev(branch)
    if cutoff < 0:
        return None
    ref = _head_ref(repo_dir) if branch is None else _qualify(branch)
    # "@<ts> +0000" is git's raw date format and exact for every ts >= 0. A bare "@<ts>" is misread by git
    # for ts before 1973 and silently matches the branch tip.
    out = run_git(
        ["rev-list", "-1", "--first-parent", f"--before=@{cutoff} +0000", "--end-of-options", ref, "--"],
        repo_dir,
    )
    return out.decode("ascii", errors="replace").strip() or None


def reachable_shas(repo_dir: Path, sha: str) -> frozenset[str]:
    """The non-merge commits reachable from ``sha`` (``git rev-list --no-merges``).

    ``sha`` itself is included unless it is a merge commit. This is the same set as the ``sha`` column of
    ``mine_history(repo_dir, sha)`` without ``since`` or ``max_commits``. Use it as "history up to T" with
    ``sha`` from :func:`snapshot_sha`.

    Raises ``ValueError`` for an empty or option-like ``sha`` and ``GitError`` for an unknown one.
    """
    check_rev(sha)
    out = run_git(["rev-list", "--no-merges", "--end-of-options", sha, "--"], repo_dir)
    return frozenset(out.decode("ascii", errors="replace").split())
