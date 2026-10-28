from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import tdp_cli

HEAVY = (
    "pandas",
    "pyarrow",
    "numpy",
    "xgboost",
    "shap",
    "networkx",
    "tdp_core.analyze",
    "tdp_core.contracts",
)


def _run(code: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "ENTROPIQ_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env, timeout=300
    )


def test_startup_imports_stay_light(entropiq_home: Path) -> None:
    code = f"""
        import io, sys
        from rich.console import Console
        from tdp_cli.main import build_app
        from tdp_cli.session import Session
        from tdp_cli.logo import print_welcome

        build_app()
        console = Console(file=io.StringIO(), width=120)
        session = Session.create(console=console, dev_model=True, repo_root=None)
        print_welcome(console, model=session.model)
        heavy = [m for m in {HEAVY!r} if m in sys.modules]
        heavy += [m for m in sys.modules if m.startswith("tdp_core.mining")]
        print("HEAVY", heavy)
    """
    result = _run(code, entropiq_home)
    assert result.returncode == 0, result.stderr
    assert "HEAVY []" in result.stdout


def test_python_m_prints_the_version(entropiq_home: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tdp_cli", "--version"], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"entropiq {tdp_cli.__version__}"
