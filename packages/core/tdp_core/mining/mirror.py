"""Bare mirrors of analysed repositories (``data/mirrors/<slug>.git``)."""

from __future__ import annotations

import logging
from pathlib import Path

from tdp_core.mining.git import run_git

logger = logging.getLogger(__name__)


def ensure_mirror(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
    """Create or refresh a bare mirror of ``url`` at ``dest`` and return ``dest``.

    If ``dest`` is a non-empty directory it is treated as an existing mirror and updated with
    ``git --git-dir=<dest> fetch --prune --tags origin`` (refs deleted upstream are removed; ``--git-dir``
    means a broken ``dest`` inside another checkout can never touch that checkout). Otherwise the parent
    directory is created and ``git clone --mirror --no-hardlinks`` runs. ``url`` is passed after ``--`` so it
    can never be read as an option. Local paths and ``file://`` URLs need ``allow_file_protocol=True``.

    Raises ``GitError`` (or ``TransientGitError`` for network failures) from :func:`run_git`; a non-empty
    ``dest`` that is not a git repository also raises ``GitError`` and is left untouched.
    """
    if dest.is_dir() and any(dest.iterdir()):
        logger.info("updating mirror %s", dest)
        # --git-dir stops git from walking up to an enclosing repository when dest is not a mirror.
        run_git(
            [f"--git-dir={dest}", "fetch", "--prune", "--tags", "origin"],
            dest,
            allow_file_protocol=allow_file_protocol,
        )
    else:
        logger.info("cloning mirror into %s", dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            ["clone", "--mirror", "--no-hardlinks", "--", url, str(dest)],
            dest.parent,
            allow_file_protocol=allow_file_protocol,
        )
    return dest
