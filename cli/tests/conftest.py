"""Shared fixtures for the command-line tests.

Every test gets its own Entropiq home folder (``ENTROPIQ_HOME``), so nothing touches ``~/.entropiq``.
The demo bundle is built once per test session and copied into homes that need it, and one small
RepoBuilder repository is analysed once and shared.
"""

from __future__ import annotations

import io
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from tdp_cli.home import EntropiqHome
from tdp_cli.model import clear_model_cache
from tdp_cli.session import Session
from tdp_core.testing import RepoBuilder

if TYPE_CHECKING:
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult

T0 = datetime(2023, 1, 2, 12, tzinfo=UTC)
MODULES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lam", "mu")


def module_source(name: str, n_functions: int, revisions: int = 0, todo: bool = False) -> str:
    """A small module with ``n_functions`` functions (one branch each) and ``revisions`` marker lines."""
    lines: list[str] = []
    if todo:
        lines.append("# TODO: hack until the parser handles negative input")
    for j in range(n_functions):
        lines += [f"def {name}_{j}(x):", "    if x > 0:", f"        return x + {j}", "    return -x", ""]
    lines += [f"# rev {k}" for k in range(revisions)]
    return "\n".join(lines) + "\n"


@pytest.fixture(autouse=True)
def entropiq_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh, empty Entropiq home for every test (also clears env vars that pick a model)."""
    home = tmp_path / "entropiq-home"
    monkeypatch.setenv("ENTROPIQ_HOME", str(home))
    monkeypatch.delenv("TDP_MODEL_DIR", raising=False)
    monkeypatch.delenv("TDP_MODELS_DIR", raising=False)
    monkeypatch.delenv("ENTROPIQ_DEBUG", raising=False)
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):  # rich would treat StringIO as a TTY
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    clear_model_cache()
    yield home
    clear_model_cache()


@pytest.fixture(scope="session")
def dev_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo bundle built once per test session."""
    from tdp_core.model.bundle import make_dev_bundle

    return make_dev_bundle(tmp_path_factory.mktemp("bundle") / "dev")


@pytest.fixture
def dev_home(entropiq_home: Path, dev_bundle_dir: Path) -> Path:
    """The test home with the demo bundle already in ``models/dev`` (so nothing rebuilds it)."""
    target = EntropiqHome.from_env().dev_model
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dev_bundle_dir, target)
    return entropiq_home


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A git repository with 12 modules of growing size and churn, a test file and about 30 commits."""
    builder = RepoBuilder(tmp_path_factory.mktemp("sample") / "sample")
    files: dict[str, str | bytes | None] = {
        "pkg/__init__.py": "",
        "tests/test_alpha.py": (
            "from pkg.alpha import alpha_0\n\n\ndef test_alpha():\n    assert alpha_0(1) == 1\n"
        ),
    }
    for i, name in enumerate(MODULES):
        files[f"pkg/{name}.py"] = module_source(name, i + 1)
    builder.commit(files, "Add the package", T0)
    when = T0
    revisions = {name: 0 for name in MODULES}
    plan = [name for i, name in enumerate(MODULES) for _ in range(i // 2)]
    for step, name in enumerate(plan):
        when += timedelta(days=1)
        revisions[name] += 1
        index = MODULES.index(name)
        todo = step == 3
        message = "fix: handle negative input in mu" if step == len(plan) - 1 else f"Update {name}"
        builder.commit(
            {f"pkg/{name}.py": module_source(name, index + 1, revisions[name], todo=todo)}, message, when
        )
    return builder.path


@pytest.fixture(scope="session")
def sample_result(sample_repo: Path, dev_bundle_dir: Path) -> AnalysisResult:
    """The analysis of ``sample_repo`` at HEAD with the demo bundle."""
    from tdp_core.analyze import analyze
    from tdp_core.model.bundle import load_bundle

    return analyze(sample_repo, "HEAD", load_bundle(dev_bundle_dir), cache=None)


@pytest.fixture
def sample_loaded(sample_repo: Path, sample_result: AnalysisResult) -> LoadedAnalysis:
    """``sample_result`` wrapped with its metadata."""
    from tdp_cli.repos import local_target
    from tdp_cli.store import LoadedAnalysis, make_meta

    return LoadedAnalysis(make_meta(local_target(sample_repo), sample_result, demo_model=True), sample_result)


@pytest.fixture
def console() -> Console:
    """A console that records plain text output (read it with ``console.file.getvalue()``)."""
    return Console(file=io.StringIO(), width=120, color_system=None, legacy_windows=False)


@pytest.fixture
def session(dev_home: Path, console: Console) -> Session:
    """An interactive session on the test home with the demo model."""
    return Session.create(console=console, dev_model=True, interactive=True, repo_root=None)


@pytest.fixture
def loaded_session(session: Session, sample_loaded: LoadedAnalysis) -> Session:
    """``session`` with ``sample_loaded`` as the current analysis."""
    session.current = sample_loaded
    return session
