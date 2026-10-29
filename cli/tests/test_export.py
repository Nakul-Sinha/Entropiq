from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from tdp_cli.commands import export
from tdp_cli.errors import CliError, UsageError
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import NO_ANALYSIS_HINT, dispatch
from tdp_cli.store import AnalysisStore, LoadedAnalysis
from tdp_cli.theme import short_sha

runner = CliRunner()


@pytest.fixture(autouse=True)
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def saved_home(dev_home: Path, sample_loaded: LoadedAnalysis) -> Path:
    """The test home with ``sample_loaded`` saved, so ``tdp report <sample_repo>`` reuses it."""
    AnalysisStore(EntropiqHome.from_env().analyses).save(sample_loaded)
    return dev_home


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined]


def _words(session: Session) -> str:
    """The console output with line breaks and runs of spaces turned into single spaces."""
    return " ".join(_out(session).split())


def _default(loaded: LoadedAnalysis, ext: str) -> str:
    return f"{loaded.meta.slug}-{short_sha(loaded.meta.sha)}{ext}"


def _meta(slug: str, sha: str = "abcdef1234567") -> Any:
    return SimpleNamespace(slug=slug, sha=sha)


def test_export_markdown(loaded_session: Session, in_tmp: Path) -> None:
    dispatch(loaded_session, "/export md -o out.md")
    text = (in_tmp / "out.md").read_text(encoding="utf-8")
    assert text.startswith("## Technical debt report")
    assert "### Fix plan" in text
    assert "\r" not in text
    out = _words(loaded_session)
    assert "Wrote the Markdown report to" in out
    assert "out.md" in "".join(_out(loaded_session).splitlines())
    assert "all 13 files and the fix plan" in out


def test_export_json_default_name(
    loaded_session: Session, sample_loaded: LoadedAnalysis, in_tmp: Path
) -> None:
    dispatch(loaded_session, "/export JSON")
    path = in_tmp / _default(sample_loaded, ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "tdp-report"
    assert len(data["files"]) == 13
    assert "(13 files)" in _out(loaded_session).replace("\n", "")


def test_export_sarif_into_a_folder(
    loaded_session: Session, sample_loaded: LoadedAnalysis, in_tmp: Path
) -> None:
    (in_tmp / "reports").mkdir()
    dispatch(loaded_session, "/export sarif -o reports")
    data = json.loads((in_tmp / "reports" / _default(sample_loaded, ".sarif")).read_text(encoding="utf-8"))
    assert data["version"] == "2.1.0"
    assert len(data["runs"][0]["results"]) == 3
    assert "(3 results)" in _out(loaded_session).replace("\n", "")

    dispatch(loaded_session, "/export sarif --min-band critical -o=critical.sarif")
    data = json.loads((in_tmp / "critical.sarif").read_text(encoding="utf-8"))
    assert len(data["runs"][0]["results"]) == 1
    assert "(1 result)" in _out(loaded_session).replace("\n", "")


def test_export_top(loaded_session: Session, in_tmp: Path) -> None:
    dispatch(loaded_session, "/export markdown --top 2 -o top.md")
    assert "Showing the top 2 of 13 files" in (in_tmp / "top.md").read_text(encoding="utf-8")
    assert "top 2 of 13 files" in _out(loaded_session).replace("\n", "")
    dispatch(loaded_session, "/export md --top 0 -o all.md")
    assert "Showing the top" not in (in_tmp / "all.md").read_text(encoding="utf-8")


def test_export_to_the_console_and_alias(loaded_session: Session, in_tmp: Path) -> None:
    dispatch(loaded_session, "/report md -o -")
    text = _out(loaded_session)
    assert "## Technical debt report" in text
    assert "### Fix plan" in text
    assert list(in_tmp.iterdir()) == []


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("/export", export.NO_FORMAT),
        ("/export pdf", "unknown format pdf; use md, json or sarif"),
        ("/export md --top -1", export.TOP_ERROR),
        ("/export md --top x", export.TOP_ERROR),
        ("/export sarif --min-band low", export.BAND_ERROR),
        ("/export md -o", "--output"),
    ],
)
def test_export_usage_errors(loaded_session: Session, in_tmp: Path, line: str, message: str) -> None:
    dispatch(loaded_session, line)
    text = _out(loaded_session)
    assert message in text
    assert "Usage: /export md|json|sarif" in text
    assert list(in_tmp.iterdir()) == []


def test_export_without_an_analysis(session: Session) -> None:
    dispatch(session, "/export md")
    assert NO_ANALYSIS_HINT in _out(session).replace("\n", " ")


def test_cannot_write(loaded_session: Session, in_tmp: Path) -> None:
    (in_tmp / "file").write_text("x", encoding="utf-8")
    dispatch(loaded_session, "/export md -o file/sub/out.md")
    assert "Cannot write" in _out(loaded_session)
    with pytest.raises(CliError, match="Cannot write"):
        export.write_report(in_tmp / "file" / "x.md", "text")


def test_expand_short_options() -> None:
    assert export.expand_short_options(["md", "-o", "a.md", "-o=b.md"]) == [
        "md",
        "--output",
        "a.md",
        "--output=b.md",
    ]
    assert export.expand_short_options(["--", "-o"]) == ["--", "-o"]


def test_parsers() -> None:
    assert export.parse_format(" MarkDown ") == "md"
    assert export.parse_format("Sarif") == "sarif"
    assert export.parse_top("0") == 0
    assert export.parse_min_band("CRITICAL") == "Critical"
    with pytest.raises(UsageError):
        export.parse_top("1.5")


def test_default_file_name_and_output_path(tmp_path: Path) -> None:
    assert export.default_file_name(_meta("psf__requests"), "sarif") == "psf__requests-abcdef1.sarif"
    assert export.default_file_name(_meta("my repo/x:y"), "md") == "my-repo-x-y-abcdef1.md"
    assert export.default_file_name(_meta("///"), "json") == "analysis-abcdef1.json"
    meta = _meta("demo")
    assert export.output_path(None, meta, "md", cwd=tmp_path) == tmp_path / "demo-abcdef1.md"
    assert export.output_path("out/", meta, "md", cwd=tmp_path) == tmp_path / "out" / "demo-abcdef1.md"
    assert export.output_path("out\\", meta, "md", cwd=tmp_path) == tmp_path / "out\\" / "demo-abcdef1.md"
    assert export.output_path("r.md", meta, "md", cwd=tmp_path) == tmp_path / "r.md"
    absolute = tmp_path / "abs.md"
    assert export.output_path(str(absolute), meta, "md", cwd=Path("/elsewhere")) == absolute


def test_completion(loaded_session: Session) -> None:
    assert [s.text for s in export._complete(loaded_session, [], "")] == ["md", "json", "sarif"]
    assert [s.text for s in export._complete(loaded_session, [], "s")] == ["sarif"]
    assert list(export._complete(loaded_session, ["md"], "")) == []
    assert [s.text for s in export._complete(loaded_session, ["md"], "--t")] == ["--top"]
    assert [s.text for s in export._complete(loaded_session, ["sarif", "--min-band"], "c")] == ["critical"]
    assert list(export._complete(loaded_session, ["md", "-o"], "")) == []


def test_report_cli(saved_home: Path, sample_repo: Path, sample_loaded: LoadedAnalysis, in_tmp: Path) -> None:
    app = build_app()
    base = ["report", str(sample_repo), "--dev-model"]
    sarif = runner.invoke(app, [*base, "--format", "sarif", "-o", "x.sarif"])
    assert sarif.exit_code == 0, sarif.stderr
    assert sarif.stdout == ""
    assert "Wrote the SARIF report" in sarif.stderr
    assert json.loads((in_tmp / "x.sarif").read_text(encoding="utf-8"))["version"] == "2.1.0"

    critical = runner.invoke(app, [*base, "-f", "sarif", "--min-band", "critical", "-o", "-"])
    assert critical.exit_code == 0, critical.stderr
    assert len(json.loads(critical.stdout)["runs"][0]["results"]) == 1

    as_json = runner.invoke(app, [*base, "--format", "json", "-o", "-"])
    assert as_json.exit_code == 0, as_json.stderr
    assert json.loads(as_json.stdout)["format"] == "tdp-report"

    md = runner.invoke(app, [*base, "--top", "0"])
    assert md.exit_code == 0, md.stderr
    text = (in_tmp / _default(sample_loaded, ".md")).read_text(encoding="utf-8")
    assert text.startswith("## Technical debt report")
    assert "Showing the top" not in text

    bad = runner.invoke(app, [*base, "--format", "pdf"])
    assert bad.exit_code == 2
