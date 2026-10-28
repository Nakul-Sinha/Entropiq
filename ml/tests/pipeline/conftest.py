"""Shared fixtures of the dataset pipeline tests: a small source repository and one full pipeline run."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner, Result

from tdp_core.labeling.github_issues import issues_table
from tdp_core.testing.repo_builder import RepoBuilder
from tdp_ml.cli import load_commands
from tdp_ml.pipeline.layout import write_table

T0 = datetime(2022, 1, 3, 12, 0, tzinfo=UTC)
ANA = ("Ana", "ana@example.com")
BOB_OLD = ("Bob", "bob@old.example")
BOB_NEW = ("Bob", "bob@new.example")
CY = ("Cy", "cy@example.com")

CORE_V0 = """def a(x):
    if x > 0:
        return x
    return -x


def b(items):
    total = 0
    for item in items:
        if item:
            total += item
    return total
"""


def _day(days: int) -> datetime:
    return T0 + timedelta(days=days)


@dataclass(frozen=True)
class SourceRepo:
    path: Path
    head: str
    shas: dict[str, str]  # commit label -> sha


@pytest.fixture(scope="session")
def source_repo(tmp_path_factory: pytest.TempPathFactory) -> SourceRepo:
    """About 17 commits over 240 days with bug fixes, a rename, SATD and a .mailmap added at day 130."""
    repo = RepoBuilder(tmp_path_factory.mktemp("pipe") / "src")
    shas: dict[str, str] = {}
    core = CORE_V0
    util = "def u(v):\n    return v + 1\n\n\ndef w(v):\n    return v * 2\n"
    io = "def read(p):\n    return open(p).read()\n\n\ndef size(p):\n    return len(read(p))\n"
    shas["d0"] = repo.commit(
        {
            "pkg/__init__.py": "",
            "pkg/core.py": core,
            "pkg/util.py": util,
            "tests/test_core.py": "from pkg.core import a\n\n\ndef test_a():\n    assert a(1) == 1\n",
            "README.md": "# demo\n",
        },
        "Initial import",
        _day(0),
        ANA,
    )
    core += "\n\ndef c(n):\n    return 10 / n\n"
    shas["d10"] = repo.commit({"pkg/core.py": core}, "Add c", _day(10), BOB_OLD)
    shas["d20"] = repo.commit({"pkg/io.py": io}, "Add io helpers", _day(20), ANA)
    core = core.replace("    return 10 / n\n", "    return 10 / n if n else 0\n")
    shas["d35"] = repo.commit({"pkg/core.py": core}, "Fix crash in c (#3)", _day(35), ANA)
    util = util.replace("    return v + 1\n", "    return v + 2\n")
    shas["d50"] = repo.commit({"pkg/util.py": util}, "Tune u", _day(50), BOB_NEW)
    io = io.replace("return len(read(p))", "return len(read(p).strip())")
    shas["d70"] = repo.rename("pkg/io.py", "pkg/files.py", "Rename io to files", _day(70), ANA, io)
    core = core.replace("def a(x):\n", "def a(x):\n    # TODO: hack until the parser is fixed\n")
    shas["d80"] = repo.commit({"pkg/core.py": core}, "Note a hack", _day(80), BOB_NEW)
    util = util.replace("    return v + 2\n", "    return v + 1\n")
    shas["d95"] = repo.commit({"pkg/util.py": util}, "Fix wrong result in util", _day(95), ANA)
    io += "\n\ndef lines(p):\n    return read(p).split()\n\n\ndef first(p):\n    return lines(p)[0]\n"
    shas["d110"] = repo.commit({"pkg/files.py": io}, "Add line helpers", _day(110), CY)
    shas["d130"] = repo.commit(
        {".mailmap": "Bob <bob@new.example> <bob@old.example>\n"}, "Add mailmap", _day(130), ANA
    )
    io = io.replace("    return read(p).split()\n", "    return read(p).splitlines()\n")
    shas["d140"] = repo.commit({"pkg/files.py": io}, "Fix bug in files handling", _day(140), BOB_NEW)
    core = core.replace("        if item:\n", "        if item is not None:\n")
    shas["d152"] = repo.commit({"pkg/core.py": core}, "Handle empty input (#7)", _day(152), ANA)
    core += "\n\ndef d(s):\n    return s.upper()\n"
    shas["d160"] = repo.commit({"pkg/core.py": core}, "Add d", _day(160), CY)
    core = core.replace("    return s.upper()\n", "    return s.upper() if s else s\n")
    shas["d175"] = repo.commit({"pkg/core.py": core}, "Fix regression in core", _day(175), ANA)
    util += "\n\ndef z(v):\n    return v - 1\n"
    shas["d200"] = repo.commit({"pkg/util.py": util}, "Add z", _day(200), BOB_NEW)
    io = io.replace("    return lines(p)[0]\n", "    return (lines(p) or [''])[0]\n")
    shas["d215"] = repo.commit({"pkg/files.py": io}, "fix error in files path", _day(215), CY)
    shas["d240"] = repo.commit({"VERSION": "1.0\n"}, "Bump version", _day(240), ANA)
    return SourceRepo(repo.path, repo.head(), shas)


LABELS_YAML = """settings:
  history_window_days: null
  min_history_days: 60
  snapshot_step_days: 30
  horizon_days: 60
  inference_szz_max_fixes: 3
snapshots:
  max_per_repo: 3
szz:
  max_fixes: 50
"""


def repos_yaml(source: Path) -> str:
    url = source.as_posix()
    return f"""max_repos: null
criteria:
  min_history_days: 100
  min_commits: 5
  min_source_files: 2
  min_bug_issues: 1
  max_days_since_last_commit: 100000
  min_snapshots: 1
repos:
  - {{id: 1, name: local/alpha, url: "{url}", issues: false}}
  - {{id: 2, name: local/beta, url: "{url}"}}
"""


def seed_issues(data: Path) -> None:
    """Bug issues 3 (day 30) and 7 (day 145) and a non-bug issue 5 for local/beta."""
    items = [
        {"number": 3, "created_at": _day(30).isoformat(), "closed_at": None, "labels": [{"name": "bug"}]},
        {"number": 5, "created_at": _day(40).isoformat(), "closed_at": None, "labels": [{"name": "docs"}]},
        {"number": 7, "created_at": _day(145).isoformat(), "closed_at": None, "labels": [{"name": "bug"}]},
    ]
    write_table(data / "issues" / "local__beta.parquet", issues_table(items))


Invoke = Callable[..., Result]


@dataclass(frozen=True)
class PipelineRun:
    data: Path
    source: SourceRepo
    invoke: Invoke


def make_invoke() -> Invoke:
    app = typer.Typer()
    load_commands(target=app)
    runner = CliRunner()

    def invoke(*args: str, code: int | None = 0) -> Result:
        result = runner.invoke(app, list(args))
        if code is not None:
            assert result.exit_code == code, f"{args}: exit {result.exit_code}\n{result.output}"
        return result

    return invoke


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory, source_repo: SourceRepo) -> Iterator[PipelineRun]:
    """Every stage once over two repositories built from ``source_repo``."""
    base = tmp_path_factory.mktemp("run")
    data = base / "data"
    seed_issues(data)
    (base / "repos.yaml").write_text(repos_yaml(source_repo.path), encoding="utf-8")
    (base / "labels.yaml").write_text(LABELS_YAML, encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setenv("TDP_DATA_DIR", str(data))
    try:
        invoke = make_invoke()
        invoke("manifest", "--repos", str(base / "repos.yaml"), "--labels", str(base / "labels.yaml"))
        for stage in (
            "mine",
            "issues",
            "bugfix",
            "snapshots",
            "szz",
            "features",
            "labels",
            "dataset",
            "stats",
        ):
            invoke(stage)
        yield PipelineRun(data, source_repo, invoke)
    finally:
        patch.undo()
