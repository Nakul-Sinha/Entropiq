from datetime import UTC, datetime
from pathlib import Path

import pytest

from tdp_core.testing import RepoBuilder


@pytest.fixture
def t0() -> datetime:
    """A fixed, timezone-aware reference time for scripted histories."""
    return datetime(2022, 1, 3, 12, tzinfo=UTC)


@pytest.fixture
def repo_builder(tmp_path: Path) -> RepoBuilder:
    """An empty git repository with scripted-history helpers."""
    return RepoBuilder(tmp_path / "repo")
