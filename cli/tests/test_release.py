from __future__ import annotations

import io
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from typer.testing import CliRunner

from tdp_cli import output
from tdp_cli.commands import _views
from tdp_cli.commands._views import RankedFile
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.model import HOME_POINTER, installed_pointer, resolve_model
from tdp_cli.session import Session
from tdp_core.contracts import AnalysisResult, FileExtra, FileScore, Reason

ANSI = re.compile(r"\x1b\[[0-9;]*m")
VERSION = "ranker-2026.10.1"
LONG_REASON = "Very large file: 892 lines of code (top 1% in repo)"


def _bundle(folder: Path, version: str, *, synthetic: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {"version": version, "metrics": {"ndcg": {"at10": 0.5}}, "training_rows": 10}
    if synthetic:
        meta["synthetic"] = True
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    schema = {"extractor_version": "1", "columns": [f"c{i}" for i in range(5)]}
    (folder / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return folder


def _pointer(path: Path, version: str | None, uri: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "uri": uri, "sha256": None}), encoding="utf-8")
    return path


@pytest.fixture
def home(tmp_path: Path) -> EntropiqHome:
    return EntropiqHome(tmp_path / "home")


def test_installed_pointer_is_in_home_models(home: EntropiqHome) -> None:
    assert installed_pointer(home) == home.models / "production.json"
    assert HOME_POINTER == "production.json"


def test_home_pointer_finds_bundle_in_home(home: EntropiqHome, tmp_path: Path) -> None:
    bundle = _bundle(home.models / VERSION, VERSION)
    _pointer(installed_pointer(home), VERSION, str(tmp_path / "elsewhere" / VERSION))
    choice = resolve_model(home=home, environ={}, repo_root=None)
    assert (choice.source, choice.version, choice.demo) == ("production", VERSION, False)
    assert (choice.kind_label, choice.notice) == ("production model", None)
    assert choice.path == bundle.resolve()
    assert choice.reason == str(installed_pointer(home))


def test_home_pointer_uri_forms(home: EntropiqHome, tmp_path: Path) -> None:
    _bundle(home.models / VERSION, VERSION)
    shared = _bundle(tmp_path / "shared" / VERSION, VERSION)
    pointer = installed_pointer(home)
    _pointer(pointer, VERSION, str(shared))
    assert resolve_model(home=home, environ={}, repo_root=None).path == shared.resolve()
    _pointer(pointer, VERSION, shared.as_uri())
    assert resolve_model(home=home, environ={}, repo_root=None).path == shared.resolve()
    relative = _bundle(home.root / "bundles" / VERSION, VERSION)
    _pointer(pointer, VERSION, f"bundles/{VERSION}")
    assert resolve_model(home=home, environ={}, repo_root=None).path == relative.resolve()


def test_checkout_pointer_beats_home_pointer(home: EntropiqHome, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    in_repo = _bundle(root / "models" / "ranker-a", "ranker-a")
    _pointer(root / "models" / "production.json", "ranker-a", None)
    _bundle(home.models / "ranker-b", "ranker-b")
    _pointer(installed_pointer(home), "ranker-b", None)
    choice = resolve_model(home=home, environ={}, repo_root=root)
    assert (choice.version, choice.reason, choice.path) == (
        "ranker-a",
        "models/production.json",
        in_repo.resolve(),
    )


def test_home_pointer_fallbacks(home: EntropiqHome) -> None:
    pointer = installed_pointer(home)
    assert resolve_model(home=home, environ={}, repo_root=None).reason == "no models/production.json found"
    _pointer(pointer, None, None)
    choice = resolve_model(home=home, environ={}, repo_root=None)
    assert (choice.source, choice.reason) == ("dev", "no production model has been published yet")
    _pointer(pointer, VERSION, None)
    choice = resolve_model(home=home, environ={}, repo_root=None)
    assert (choice.source, choice.reason) == ("dev", f"production model {VERSION} is not on this machine")
    pointer.write_text("{broken", encoding="utf-8")
    choice = resolve_model(home=home, environ={}, repo_root=None)
    assert (choice.source, choice.reason) == ("dev", f"{pointer} is unreadable")


def test_session_uses_home_pointer(entropiq_home: Path) -> None:
    home = EntropiqHome.from_env()
    _bundle(home.models / VERSION, VERSION)
    _pointer(installed_pointer(home), VERSION, None)
    console = Console(file=io.StringIO(), width=100)
    session = Session.create(console=console, interactive=False, repo_root=None)
    assert session.model.kind_label == "production model"
    assert not session.model.demo
    assert session.model.version == VERSION


def _help_text() -> str:
    result = CliRunner().invoke(build_app(), ["--help"])
    assert result.exit_code == 0, result.output
    return ANSI.sub("", result.stdout)


def test_help_offers_shell_completion() -> None:
    text = _help_text()
    assert "--install-completion" in text
    assert "--show-completion" in text


def test_show_completion_prints_a_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION", "1")
    result = CliRunner().invoke(build_app(), ["--show-completion", "bash"])
    assert result.exit_code == 0, result.output
    assert "_ENTROPIQ_COMPLETE" in result.stdout


def test_help_groups_follow_group_order() -> None:
    panels = re.findall(r"(?:─|-) (Analysis|Explore|Plan|Report|Session) (?:─|-)", _help_text())
    assert panels == ["Analysis", "Explore", "Plan", "Report", "Session"]


def _long_files(result: AnalysisResult) -> tuple[list[FileScore], dict[str, FileExtra]]:
    files: list[FileScore] = []
    extras: dict[str, FileExtra] = {}
    for i, score in enumerate(result.files[:12]):
        changed = replace(
            score,
            path="docs/_themes/some/long/folder/name/" + score.path,
            debt_types=("defect_hotspot", "architecture"),
        )
        if i == 0:
            first = (
                replace(score.reasons[0], text=LONG_REASON)
                if score.reasons
                else Reason("S", "loc", 892.0, 99.0, 1.0, LONG_REASON)
            )
            changed = replace(changed, reasons=(first,))
        files.append(changed)
        extra = result.extras.get(score.path)
        if extra is not None:
            extras[changed.path] = extra
    return files, extras


def _render(table: object, width: int) -> str:
    console = Console(width=width, file=io.StringIO(), color_system=None, legacy_windows=False)
    console.print(table, crop=False)
    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert isinstance(text, str)
    return text


@pytest.mark.parametrize("width", [80, 100, 120])
def test_ranked_tables_fit_the_terminal(sample_result: AnalysisResult, width: int) -> None:
    files, extras = _long_files(sample_result)
    rows = [RankedFile(rank, score, extras.get(score.path)) for rank, score in enumerate(files, start=1)]
    tables = [
        _views.ranked_table(rows),
        _views.ranked_table(rows, changes={files[0].path: "modified"}),
        output.files_table(files, extras),
    ]
    for table in tables:
        text = _render(table, width)
        assert max(cell_len(line) for line in text.splitlines()) <= width
        assert "892" in text
