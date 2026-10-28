from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest
from rich.console import Console

from tdp_cli.errors import CliError
from tdp_cli.output import (
    analysis_json,
    analysis_markdown,
    band_counts,
    critical_hours,
    critical_range,
    files_table,
    history_table,
    summary_card,
    top_reason,
)
from tdp_cli.runner import STAGE_LABELS, AnalysisProgress, run_analysis
from tdp_cli.session import Session
from tdp_cli.store import LoadedAnalysis
from tdp_core.analyze import ANALYSIS_STAGES
from tdp_core.contracts import CostEstimate, FileExtra, FileScore, Reason


def _render(renderable: object, width: int = 160) -> str:
    console = Console(file=io.StringIO(), width=width, color_system=None)
    console.print(renderable)
    return console.file.getvalue()  # type: ignore[attr-defined, no-any-return]


def _score(path: str, band: str, priority: int, reason: str = "") -> FileScore:
    reasons = (Reason("S", "sloc", 1.0, 99.0, 0.5, reason),) if reason else ()
    return FileScore(path, 1.0, priority, band, 0.5, 0.4, 0.6, reasons, (), (), {})  # type: ignore[arg-type]


def _extra(hours: float) -> FileExtra:
    return FileExtra("b", (), (), (), (), (), CostEstimate(hours, 0.8 * hours, 1.2 * hours, {}))


def test_band_counts_keep_zeros() -> None:
    files = [_score("a.py", "Critical", 95), _score("b.py", "Low", 3), _score("c.py", "Low", 1)]
    assert band_counts(files) == {"Critical": 1, "High": 0, "Medium": 0, "Low": 2}
    assert band_counts([]) == {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}


def test_critical_hours(sample_loaded: LoadedAnalysis) -> None:
    files = (
        _score("a.py", "Critical", 95, "big"),
        _score("b.py", "Critical", 91),
        _score("c.py", "High", 80),
    )
    result = dataclasses.replace(
        sample_loaded.result,
        files=files,
        extras={"a.py": _extra(10), "b.py": _extra(2.5), "c.py": _extra(100)},
    )
    assert critical_hours(result) == pytest.approx(12.5)
    assert critical_range(result) == pytest.approx((10.0, 15.0))
    assert top_reason(files[0]) == "big"
    assert top_reason(files[1]) == ""


def test_summary_card(sample_loaded: LoadedAnalysis) -> None:
    text = _render(summary_card(sample_loaded, top=3))
    result = sample_loaded.result
    assert f"{len(result.files)} files analysed" in text
    for band in ("Critical", "High", "Medium", "Low"):
        assert band in text
    for f in result.files[:3]:
        assert f.path in text
    assert result.files[3].path not in text
    assert "demo" in text
    assert "Estimated Critical debt" in text
    assert "Saved analysis from" in _render(summary_card(sample_loaded, top=0, reused=True))


def test_summary_card_empty(sample_loaded: LoadedAnalysis) -> None:
    empty = LoadedAnalysis(sample_loaded.meta, dataclasses.replace(sample_loaded.result, files=()))
    assert "No Python source files to rank at this commit." in _render(summary_card(empty))


def test_files_table(sample_loaded: LoadedAnalysis) -> None:
    result = sample_loaded.result
    text = _render(files_table(result.files, result.extras, limit=2, start=5, title="Top"))
    assert "Cost" in text and " h" in text
    assert result.files[0].path in text and result.files[2].path not in text
    assert any(line.lstrip().startswith("5 ") for line in text.splitlines())
    plain = _render(files_table(result.files))
    assert "Cost" not in plain
    assert all(f.path in plain for f in result.files)


def test_history_table(sample_loaded: LoadedAnalysis) -> None:
    text = _render(history_table([sample_loaded.meta]))
    assert "sample" in text and sample_loaded.meta.sha[:7] in text and "demo" in text


def test_analysis_json(sample_loaded: LoadedAnalysis) -> None:
    data = analysis_json(sample_loaded, top=3)
    assert data["schema"] == "entropiq.analysis.v1"
    assert data["sha"] == sample_loaded.result.sha
    assert data["analyzed_at"].endswith("Z")
    assert data["summary"]["files"] == len(sample_loaded.result.files)
    assert len(data["files"]) == 3
    row = data["files"][0]
    assert row["rank"] == 1 and row["path"] == sample_loaded.result.files[0].path
    assert set(row) == {
        "rank", "path", "priority", "band", "risk", "risk_low", "risk_high", "cost_hours", "cost_low",
        "cost_high", "debt_types", "top_reason", "reasons",
    }  # fmt: skip
    assert len(analysis_json(sample_loaded, top=0)["files"]) == len(sample_loaded.result.files)


def test_analysis_markdown(sample_loaded: LoadedAnalysis) -> None:
    text = analysis_markdown(sample_loaded, top=2)
    assert text.startswith(f"## Entropiq: sample at {sample_loaded.meta.sha[:7]}")
    assert "| # | File | Priority | Risk | Cost | Top reason |" in text
    assert f"Showing 2 of {len(sample_loaded.result.files)} files." in text
    weird = dataclasses.replace(sample_loaded.result.files[0], path="a|b.py")
    patched = LoadedAnalysis(sample_loaded.meta, dataclasses.replace(sample_loaded.result, files=(weird,)))
    md = analysis_markdown(patched, top=0)
    assert "`a\\|b.py`" in md
    assert "Showing" not in md


def test_progress_plain_lines() -> None:
    console = Console(file=io.StringIO(), width=120, color_system=None)
    with AnalysisProgress(console, title="Analysing x") as progress:
        for stage, pct in ANALYSIS_STAGES:
            progress(stage, pct, f"Done {stage}")
    lines = console.file.getvalue().splitlines()  # type: ignore[attr-defined]
    assert lines[0] == "Analysing x"
    assert lines[1].startswith("[ 25%] Done mine (")
    assert len([ln for ln in lines if ln.startswith("[")]) == len(ANALYSIS_STAGES)
    assert lines[-1].startswith("Analysis finished in")


def test_progress_on_a_terminal() -> None:
    console = Console(file=io.StringIO(), width=120, force_terminal=True)
    with AnalysisProgress(console, title="Analysing x") as progress:
        for stage, pct in ANALYSIS_STAGES:
            progress(stage, pct, f"Done {stage}")
        progress("unknown", 99, "odd")
    assert "Done plan" in console.file.getvalue()  # type: ignore[attr-defined]


def test_progress_skips_finish_line_on_error() -> None:
    console = Console(file=io.StringIO(), width=120, color_system=None)
    with pytest.raises(KeyboardInterrupt), AnalysisProgress(console, title="t") as progress:
        progress("mine", 25, "m")
        raise KeyboardInterrupt
    assert "Analysis finished" not in console.file.getvalue()  # type: ignore[attr-defined]


def test_stage_labels_match_core() -> None:
    assert tuple(STAGE_LABELS) == tuple(name for name, _ in ANALYSIS_STAGES)


def test_run_analysis_saves_and_reuses(
    session: Session, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded, reused = run_analysis(session, str(sample_repo))
    assert reused is False
    assert session.current is loaded
    assert len(session.store.entries()) == 1
    assert loaded.meta.demo_model is True
    assert loaded.meta.model_version == session.model.version
    assert "Analysis finished in" in session.console.file.getvalue()  # type: ignore[attr-defined]

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("analyze must not run again")

    monkeypatch.setattr("tdp_core.analyze.analyze", boom)
    again, reused = run_analysis(session, str(sample_repo / "pkg"))
    assert reused is True
    assert again.meta == loaded.meta
    assert len(again.result.files) == len(loaded.result.files)
    monkeypatch.undo()

    fresh, reused = run_analysis(session, str(sample_repo), fresh=True)
    assert reused is False
    assert fresh.meta.sha == loaded.meta.sha
    assert len(session.store.entries()) == 1


def test_run_analysis_errors(session: Session, sample_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(CliError, match="does not name a commit"):
        run_analysis(session, str(sample_repo), ref="nope")
    with pytest.raises(CliError, match="is not a git repository"):
        run_analysis(session, str(tmp_path))
