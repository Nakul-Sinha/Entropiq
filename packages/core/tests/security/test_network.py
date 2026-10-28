"""Checks against the real github.com (skipped by default; run with ``-m network``)."""

import os
import re
from pathlib import Path

import pytest

from tdp_core.mining.git import GitError, TransientGitError, run_git
from tdp_core.security.limits import (
    DEFAULT_MAX_REPO_KB,
    RepoNotFoundError,
    RepoTooLargeError,
    check_repo_size,
)
from tdp_core.security.urls import DEFAULT_ALLOWED_HOSTS, validate_repo_url

pytestmark = pytest.mark.network


def test_real_github_url_resolves_and_answers_ls_remote(tmp_path: Path) -> None:
    ref = validate_repo_url("https://github.com/psf/requests.git", DEFAULT_ALLOWED_HOSTS)

    out = run_git(["ls-remote", "--", ref.clone_url, "HEAD"], tmp_path, timeout=60).decode().strip()

    assert re.fullmatch(r"[0-9a-f]{40}\tHEAD", out)


def test_real_github_size_check() -> None:
    ref = validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)
    token = os.environ.get("GITHUB_TOKEN")

    assert check_repo_size(ref, token, DEFAULT_MAX_REPO_KB) > 0
    with pytest.raises(RepoTooLargeError):
        check_repo_size(ref, token, 1)


def test_missing_repository_is_not_found_and_never_prompts(tmp_path: Path) -> None:
    ref = validate_repo_url(
        "https://github.com/psf/tdp-this-repository-does-not-exist", DEFAULT_ALLOWED_HOSTS
    )

    with pytest.raises(RepoNotFoundError):
        check_repo_size(ref, os.environ.get("GITHUB_TOKEN"), DEFAULT_MAX_REPO_KB)
    with pytest.raises(GitError) as info:
        run_git(["ls-remote", "--", ref.clone_url, "HEAD"], tmp_path, timeout=60)
    assert not isinstance(info.value, TransientGitError)
