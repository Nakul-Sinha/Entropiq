"""Fixtures of the exploration view tests: a small linked repository with imports, a cycle and a coupling.

``linked_repo`` has the package ``app`` (core and util import each other, views imports core, the routes of
``app/api`` import views, routes and schema always change together), ``setup.py``, a test importing
``app.core`` and a last group of commits that modifies, deletes, adds and renames files, so ``HEAD~8...HEAD``
has every kind of change.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tdp_cli.session import Session
from tdp_core.testing import RepoBuilder

if TYPE_CHECKING:
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult

T0 = datetime(2023, 1, 2, 12, tzinfo=UTC)
TEST_SOURCE = "from app.core import core_0\n\n\ndef test_core():\n    assert core_0(1) == 1\n"


def body(name: str, n: int, rev: int = 0, imports: tuple[str, ...] = ()) -> str:
    """A module importing ``app.<imports>`` with ``n`` one-branch functions and ``rev`` marker lines."""
    lines = [f"from app import {other}" for other in imports]
    if lines:
        lines.append("")
    for j in range(n):
        lines += [f"def {name}_{j}(x):", "    if x > 0:", f"        return x + {j}", "    return -x", ""]
    lines += [f"# rev {k}" for k in range(rev)]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def linked_build(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """The linked repository and the sha of its first commit (``HEAD~8``)."""
    builder = RepoBuilder(tmp_path_factory.mktemp("linked") / "linked")
    base = builder.commit(
        {
            "app/__init__.py": "",
            "app/core.py": body("core", 3, imports=("util",)),
            "app/util.py": body("util", 2, imports=("core",)),
            "app/views.py": body("views", 4, imports=("core",)),
            "app/legacy.py": body("legacy", 1),
            "app/helpers.py": body("helpers", 2),
            "app/api/__init__.py": "",
            "app/api/routes.py": body("routes", 3, imports=("views",)),
            "app/api/schema.py": body("schema", 2),
            "setup.py": "from setuptools import setup\n\nsetup()\n",
            "README.md": "# linked\n",
            "tests/test_core.py": TEST_SOURCE,
        },
        "Add the app",
        T0,
    )
    when = T0
    for k in range(6):
        when += timedelta(days=1)
        builder.commit(
            {
                "app/api/routes.py": body("routes", 3, k + 1, ("views",)),
                "app/api/schema.py": body("schema", 2, k + 1),
            },
            f"Update the api ({k + 1})",
            when,
        )
    when += timedelta(days=1)
    builder.commit(
        {
            "app/views.py": body("views", 5, 1, ("core",)),
            "app/legacy.py": None,
            "docs.txt": "notes\n",
            "tests/test_core.py": TEST_SOURCE + "    assert core_0(-1) == 1\n",
        },
        "Grow the views and drop the legacy module",
        when,
    )
    when += timedelta(days=1)
    builder.rename("app/helpers.py", "app/api/helpers.py", "Move the helpers into the api", when)
    return builder.path, base


@pytest.fixture(scope="session")
def linked_repo(linked_build: tuple[Path, str]) -> Path:
    """The folder of the linked repository."""
    return linked_build[0]


@pytest.fixture(scope="session")
def linked_base(linked_build: tuple[Path, str]) -> str:
    """The sha of the linked repository's first commit."""
    return linked_build[1]


@pytest.fixture(scope="session")
def linked_result(linked_repo: Path, dev_bundle_dir: Path) -> AnalysisResult:
    """The analysis of ``linked_repo`` at HEAD with the demo bundle."""
    from tdp_core.analyze import analyze
    from tdp_core.model.bundle import load_bundle

    return analyze(linked_repo, "HEAD", load_bundle(dev_bundle_dir), cache=None)


@pytest.fixture
def linked_loaded(linked_repo: Path, linked_result: AnalysisResult) -> LoadedAnalysis:
    """``linked_result`` with its metadata (demo model)."""
    from tdp_cli.repos import local_target
    from tdp_cli.store import LoadedAnalysis, make_meta

    return LoadedAnalysis(make_meta(local_target(linked_repo), linked_result, demo_model=True), linked_result)


@pytest.fixture
def linked_session(session: Session, linked_loaded: LoadedAnalysis) -> Session:
    """The interactive test session with ``linked_loaded`` as the current analysis."""
    session.current = linked_loaded
    return session


@pytest.fixture
def saved_linked(dev_home: Path, linked_loaded: LoadedAnalysis) -> LoadedAnalysis:
    """``linked_loaded`` saved in the test home, so ``tdp <command> <linked_repo> --dev-model`` reuses it."""
    from tdp_cli.home import EntropiqHome
    from tdp_cli.store import AnalysisStore

    AnalysisStore(EntropiqHome.from_env().analyses).save(linked_loaded)
    return linked_loaded
