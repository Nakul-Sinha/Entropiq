"""GitHub helpers shared by ``tdp-ml`` commands: the API token and repository slugs."""

from __future__ import annotations

import os
import shutil
import subprocess


def github_token() -> str | None:
    """The GitHub token: ``GITHUB_TOKEN`` if set, else ``gh auth token``; None when neither gives one.

    The token is never printed or logged. ``gh`` gets 20 seconds; a missing or failing ``gh`` gives None.
    """
    env = os.environ.get("GITHUB_TOKEN", "").strip()
    if env:
        return env
    gh = shutil.which("gh")
    if gh is None:
        return None
    try:
        result = subprocess.run(
            [gh, "auth", "token"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    token = (result.stdout or "").strip()
    return token if result.returncode == 0 and token else None


def split_slug(slug: str) -> tuple[str, str]:
    """``"psf__requests"`` -> ``("psf", "requests")``. Raises ``ValueError`` without ``"__"``.

    Slugs are lower-cased (plan section 3.4); GitHub API paths are case insensitive, so they work as is.
    """
    owner, sep, name = slug.partition("__")
    if not sep or not owner or not name:
        raise ValueError(f"repository slug must look like 'owner__name', got {slug!r}")
    return owner, name
