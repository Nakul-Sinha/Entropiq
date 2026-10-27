"""Lean bare mirrors of analysed repositories (``data/mirrors/<slug>.git``)."""

from __future__ import annotations

import logging
from pathlib import Path

from tdp_core.mining.git import run_git

logger = logging.getLogger(__name__)

# Branches only (tags come from ``--tags``). Pull request refs (``refs/pull/*``) are never fetched.
FETCH_REFSPEC = "+refs/heads/*:refs/heads/*"


def ensure_mirror(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
    """Create or refresh a lean bare mirror of ``url`` at ``dest`` and return ``dest``.

    The mirror holds branches and tags only, so pull request refs (``refs/pull/*``, often many thousands on
    GitHub) are never downloaded. A new mirror is made with ``git clone --bare --no-hardlinks`` and then gets
    :data:`FETCH_REFSPEC` as its stored ``remote.origin.fetch``, so a plain ``git fetch`` keeps it lean too.

    If ``dest`` is a non-empty directory it is treated as an existing mirror and updated with
    ``git --git-dir=<dest> fetch --prune --tags origin +refs/heads/*:refs/heads/*``. The refspec is given on
    the command line, so the prefetched mirrors, mirrors made here and older ``git clone --mirror`` mirrors
    (whose stored refspec is ``+refs/*:refs/*``) are all refreshed the same way and never fetch pull request
    refs; their stored config is not rewritten. Branches deleted upstream are pruned. Tags are added but never
    pruned. Refs outside ``refs/heads`` and ``refs/tags`` that an old mirror already holds are left alone.
    ``HEAD`` (the default branch) is set at clone time only and is not changed by an update.

    ``--git-dir`` means a broken ``dest`` inside another checkout can never touch that checkout, and ``url``
    is passed after ``--`` so it can never be read as an option. Local paths and ``file://`` URLs need
    ``allow_file_protocol=True``.

    Raises ``GitError`` (or ``TransientGitError`` for network failures) from :func:`run_git`; a non-empty
    ``dest`` that is not a git repository also raises ``GitError`` and is left untouched.
    """
    if dest.is_dir() and any(dest.iterdir()):
        logger.info("updating mirror %s", dest)
        # --git-dir stops git from walking up to an enclosing repository when dest is not a mirror.
        run_git(
            [f"--git-dir={dest}", "fetch", "--prune", "--tags", "origin", FETCH_REFSPEC],
            dest,
            allow_file_protocol=allow_file_protocol,
        )
    else:
        logger.info("cloning mirror into %s", dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            ["clone", "--bare", "--no-hardlinks", "--", url, str(dest)],
            dest.parent,
            allow_file_protocol=allow_file_protocol,
        )
        run_git([f"--git-dir={dest}", "config", "remote.origin.fetch", FETCH_REFSPEC], dest)
    return dest
