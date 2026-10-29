from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tdp_cli.commands.trend import commit_times, trend_json, trend_points, trend_view
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.registry import load_registry
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import AnalysisMeta, AnalysisStore, LoadedAnalysis
from tdp_cli.theme import ASCII_GLYPHS

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOW = datetime(2026, 1, 10, 9, tzinfo=UTC)


def _out(session: Session) -> str:
    text = session.console.file.getvalue()  # type: ignore[attr-defined]
    session.console.file.seek(0)  # type: ignore[attr-defined]
    session.console.file.truncate()  # type: ignore[attr-defined]
    return text


def _sha(repo: Path, rev: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", rev], capture_output=True, text=True, check=True
    ).stdout.strip()


def _meta(
    base: AnalysisMeta, sha: str, critical: int, hours: float, minutes: int, model: str
) -> AnalysisMeta:
    counts = {"Critical": critical, "High": 2, "Medium": 3, "Low": base.n_files - critical - 5}
    return dataclasses.replace(
        base,
        sha=sha,
        counts=counts,
        critical_hours=hours,
        analyzed_at=NOW + timedelta(minutes=minutes),
        model_version=model,
    )


@pytest.fixture
def history(dev_home: Path, sample_repo: Path, sample_loaded: LoadedAnalysis) -> list[AnalysisMeta]:
    """Three saved analyses of ``sample_repo`` (HEAD~20, HEAD~10, HEAD), saved in a scrambled time order."""
    base = sample_loaded.meta
    store = AnalysisStore(EntropiqHome.from_env().analyses)
    metas = [
        _meta(base, _sha(sample_repo, "HEAD~20"), 1, 2.5, 30, "dev-1"),
        _meta(base, _sha(sample_repo, "HEAD~10"), 3, 6.0, 10, "dev-1"),
        _meta(base, _sha(sample_repo, "HEAD"), 2, 4.25, 20, "dev-2"),
    ]
    for meta in metas:
        store.save(LoadedAnalysis(meta, sample_loaded.result))
    other = dataclasses.replace(base, slug="other-12345678", repo="other", sha="f" * 40)
    store.save(LoadedAnalysis(other, sample_loaded.result))
    return metas


def test_commit_times(sample_repo: Path, tmp_path: Path) -> None:
    head, older = _sha(sample_repo, "HEAD"), _sha(sample_repo, "HEAD~3")
    times = commit_times(sample_repo, [head, older, head])
    assert times is not None and list(times) == [head, older]
    assert times[older] < times[head] and times[head].tzinfo is UTC
    assert commit_times(sample_repo, []) == {}
    assert commit_times(sample_repo, ["a" * 40]) is None
    assert commit_times(sample_repo, ["HEAD"]) is None
    assert commit_times(tmp_path / "gone", [head]) is None


def test_trend_points_order(history: list[AnalysisMeta], sample_repo: Path, tmp_path: Path) -> None:
    entries = AnalysisStore(EntropiqHome.from_env().analyses).entries()
    slug = history[0].slug
    points = trend_points(entries, slug, repo_dir=sample_repo)
    assert [p.meta.sha for p in points] == [m.sha for m in history]
    assert all(p.committed_at is not None for p in points)
    by_time = trend_points(entries, slug, repo_dir=tmp_path / "gone")
    assert [p.meta.sha for p in by_time] == [history[1].sha, history[2].sha, history[0].sha]
    assert all(p.committed_at is None for p in by_time)
    assert trend_points(entries, slug, repo_dir=None) == by_time
    fake = dataclasses.replace(history[0], sha="0" * 40, analyzed_at=NOW)
    mixed = trend_points(entries, slug, repo_dir=sample_repo, extra=fake)
    assert mixed[0].meta.sha == "0" * 40 and all(p.committed_at is None for p in mixed)
    again = trend_points(entries, slug, repo_dir=sample_repo, extra=history[2])
    assert len(again) == 3
    assert trend_points(entries, "nothing", repo_dir=sample_repo) == []


def test_trend_view(history: list[AnalysisMeta], sample_repo: Path, loaded_session: Session) -> None:
    entries = AnalysisStore(EntropiqHome.from_env().analyses).entries()
    points = trend_points(entries, history[0].slug, repo_dir=sample_repo)
    loaded_session.console.print(trend_view(points, repo="sample", current_sha=history[2].sha))
    out = _out(loaded_session)
    assert "Saved analyses of sample (oldest first, by commit date)" in out
    assert f"{history[2].sha[:7]} loaded" in out
    assert "2023-01-" in out and "dev-2 demo" in out
    assert "Critical files  " in out and "first 1, latest 2, highest 3" in out
    assert "first 2.5 h, latest 4." in out and "highest 6.0 h" in out
    assert "Since the previous analysis: Critical files 3 to 2 (↓ 1), Critical hours 6.0 h to 4." in out
    assert "These analyses used 2 different models" in out
    loaded_session.console.print(trend_view(points[:2], repo="sample", glyphs=ASCII_GLYPHS))
    out = _out(loaded_session)
    assert "Critical files 1 to 3 (^ 2), Critical hours 2.5 h to 6.0 h (^ 3.5 h)" in out
    same = [dataclasses.replace(p, meta=history[0]) for p in points[:2]]
    loaded_session.console.print(trend_view(same, repo="sample"))
    out = _out(loaded_session)
    assert "(no change)" in out and "different models" not in out
    loaded_session.console.print(trend_view(points[:1], repo="sample"))
    assert "Only one saved analysis of this repository" in _out(loaded_session)
    loaded_session.console.print(trend_view([], repo="sample"))
    assert "No saved analyses of sample yet." in _out(loaded_session)


def test_trend_json(history: list[AnalysisMeta], sample_repo: Path) -> None:
    entries = AnalysisStore(EntropiqHome.from_env().analyses).entries()
    points = trend_points(entries, history[0].slug, repo_dir=sample_repo)
    doc = trend_json(points, repo="sample", slug=history[0].slug)
    assert doc["schema"] == "entropiq.trend.v1" and doc["order"] == "commit_time"
    assert [p["number"] for p in doc["points"]] == [1, 2, 3]
    first = doc["points"][0]
    assert first["sha"] == history[0].sha and first["critical"] == 1 and first["critical_hours"] == 2.5
    assert first["committed_at"].endswith("Z") and first["analyzed_at"] == "2026-01-10T09:30:00Z"
    assert set(first) == {
        "number",
        "sha",
        "committed_at",
        "analyzed_at",
        "model_version",
        "demo_model",
        "files",
        "critical",
        "high",
        "medium",
        "low",
        "critical_hours",
    }
    unknown = trend_json(trend_points(entries, history[0].slug, repo_dir=None), repo="sample", slug="s")
    assert unknown["order"] == "analysis_time" and unknown["points"][0]["committed_at"] is None


def test_trend_command(history: list[AnalysisMeta], loaded_session: Session) -> None:
    dispatch(loaded_session, "/trend")
    out = _out(loaded_session)
    assert "Saved analyses of sample" in out and "loaded" in out
    assert "Since the previous analysis" in out
    dispatch(loaded_session, "/trend now")
    assert "Usage: /trend" in _out(loaded_session)


def test_trend_command_with_unsaved_analysis(loaded_session: Session) -> None:
    dispatch(loaded_session, "/trend")
    out = _out(loaded_session)
    assert "Saved analyses of sample" in out and " loaded" in out
    assert "Only one saved analysis" in out


def test_tdp_trend(history: list[AnalysisMeta], sample_repo: Path, tmp_path: Path) -> None:
    app = build_app()
    table = runner.invoke(app, ["trend", str(sample_repo)])
    assert table.exit_code == 0, table.stderr
    assert "Critical hours" in ANSI.sub("", table.stdout)
    data = runner.invoke(app, ["trend", str(sample_repo), "--format", "json"])
    assert data.exit_code == 0, data.stderr
    doc = json.loads(data.stdout)
    assert doc["repo"] == "sample" and len(doc["points"]) == 3 and doc["order"] == "commit_time"
    url = runner.invoke(app, ["trend", "https://github.com/pallets/click", "-f", "json"])
    assert url.exit_code == 0, url.stderr
    assert json.loads(url.stdout)["points"] == [] and json.loads(url.stdout)["slug"] == "pallets__click"
    empty = runner.invoke(app, ["trend", "https://github.com/pallets/click"])
    assert "No saved analyses of pallets/click yet." in ANSI.sub("", empty.stdout)
    missing = runner.invoke(app, ["trend", str(tmp_path / "gone")])
    assert missing.exit_code == 2


def test_view_commands_are_registered() -> None:
    registry = load_registry()
    expected = {
        "files": "Explore",
        "file": "Explore",
        "graph": "Explore",
        "tree": "Explore",
        "diff": "Report",
        "trend": "Analysis",
    }
    cli_names = {name for name, *_ in registry.cli_entries()}
    dashes = (chr(0x2013), chr(0x2014))
    for name, group in expected.items():
        command = registry.get(name)
        assert command is not None and command.group == group and command.needs_analysis
        assert command.usage.startswith(f"/{name}") and command.cli is not None
        assert name in cli_names
        assert not any(d in command.help + command.details + command.usage for d in dashes)
    help_text = ANSI.sub("", runner.invoke(build_app(), ["--help"]).stdout)
    for name in expected:
        assert name in help_text
