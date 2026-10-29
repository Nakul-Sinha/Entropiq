from __future__ import annotations

import dataclasses
import io
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tdp_cli.commands import _pager
from tdp_cli.commands._detail import COST_LABELS, NO_DETAILS, file_detail, file_json, plan_step
from tdp_cli.commands._views import (
    ASCII_VIEW_GLYPHS,
    DEBT_REMEDIES,
    NO_FILES,
    UNICODE_VIEW_GLYPHS,
    FileQuery,
    diverging_bar,
    file_label,
    file_row,
    file_slocs,
    find_file,
    folder_suggestions,
    json_header,
    limited,
    make_query,
    normalise_path,
    parse_bands,
    parse_count,
    parse_debt_types,
    ranked_by_path,
    ranked_files,
    ranked_table,
    select_files,
    sparkline,
    utc_iso,
    view_glyphs,
    view_glyphs_for,
)
from tdp_cli.commands.files import complete_file, complete_files
from tdp_cli.errors import CliError, ShellExit, UsageError
from tdp_cli.main import build_app
from tdp_cli.output import analysis_json
from tdp_cli.registry import load_registry
from tdp_cli.session import Session
from tdp_cli.shell import dispatch
from tdp_cli.store import LoadedAnalysis
from tdp_cli.theme import ASCII_GLYPHS, UNICODE_GLYPHS
from tdp_core.contracts import DEBT_TYPES, AnalysisResult
from tdp_core.planning.cost import COST_KINDS

runner = CliRunner()
LINKED_FILES = (
    "app/__init__.py",
    "app/api/__init__.py",
    "app/api/helpers.py",
    "app/api/routes.py",
    "app/api/schema.py",
    "app/core.py",
    "app/util.py",
    "app/views.py",
    "setup.py",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
DASHES = (chr(0x2013), chr(0x2014))
CHEVRON = UNICODE_GLYPHS.chevron


def _out(session: Session) -> str:
    text = session.console.file.getvalue()  # type: ignore[attr-defined]
    session.console.file.seek(0)  # type: ignore[attr-defined]
    session.console.file.truncate()  # type: ignore[attr-defined]
    return text


def _reader(lines: list[str]) -> Iterator[str | None]:
    yield from lines
    while True:
        yield None


def _plain(text: str) -> str:
    return ANSI.sub("", text)


# ---------------------------------------------------------------- parsing helpers


def test_parse_bands_and_types() -> None:
    assert parse_bands("low,CRITICAL, high") == ("Critical", "High", "Low")
    assert parse_bands("medium,medium") == ("Medium",)
    with pytest.raises(UsageError, match="unknown band 'urgent'"):
        parse_bands("urgent")
    with pytest.raises(UsageError, match="--band needs a band"):
        parse_bands(" , ")
    assert parse_debt_types("Defect-Hotspot,test") == ("test", "defect_hotspot")
    assert parse_debt_types("defect hotspot") == ("defect_hotspot",)
    with pytest.raises(UsageError, match="unknown debt type 'smell'"):
        parse_debt_types("smell")
    with pytest.raises(UsageError, match="--type needs a debt type"):
        parse_debt_types("")


def test_parse_count_and_query() -> None:
    assert parse_count(" 7 ", "--top") == 7
    assert parse_count("3", "--depth", minimum=1, maximum=6) == 3
    with pytest.raises(UsageError, match="--top needs a whole number"):
        parse_count("many", "--top")
    with pytest.raises(UsageError, match="--top must be at least 0"):
        parse_count("-1", "--top")
    with pytest.raises(UsageError, match="--depth must be between 1 and 6"):
        parse_count("9", "--depth", minimum=1, maximum=6)
    query = make_query(band="high", debt_type="test", path=".\\app\\", sort="RISK", top="4")
    assert query == FileQuery(("High",), ("test",), "app/", "risk", 4)
    assert make_query() == FileQuery()
    assert make_query(top=3).top == 3
    with pytest.raises(UsageError, match="--sort takes priority, risk or cost"):
        make_query(sort="name")
    with pytest.raises(UsageError, match="--top must be at least 0"):
        make_query(top=-2)
    assert query.describe() == "Ranked files (band High, debt test, path app/, sorted by risk)"
    assert FileQuery().describe() == "Ranked files"


def test_normalise_path_and_small_helpers() -> None:
    assert normalise_path(" .\\app\\\\api\\ ") == "app/api/"
    assert normalise_path("/./app//core.py") == "app/core.py"
    assert normalise_path("") == ""
    assert limited(["a", "b", "c"], 2) == (["a", "b"], 1)
    assert limited(["a"], 5) == (["a"], 0)
    from datetime import UTC, datetime, timedelta, timezone

    when = datetime(2024, 5, 1, 14, 30, tzinfo=timezone(timedelta(hours=2)))
    assert utc_iso(when) == "2024-05-01T12:30:00Z"
    assert utc_iso(datetime(2024, 5, 1, tzinfo=UTC)) == "2024-05-01T00:00:00Z"


def test_remedies_and_cost_labels_follow_the_contracts() -> None:
    assert tuple(DEBT_REMEDIES) == DEBT_TYPES
    assert tuple(COST_LABELS) == COST_KINDS
    assert not any(d in t for d in DASHES for t in DEBT_REMEDIES.values())


def test_view_glyphs() -> None:
    assert view_glyphs_for("utf-8") is UNICODE_VIEW_GLYPHS
    assert view_glyphs_for("ascii") is ASCII_VIEW_GLYPHS
    assert view_glyphs_for("cp1252") is ASCII_VIEW_GLYPHS
    assert view_glyphs(Console(file=io.StringIO(), legacy_windows=False)) in (
        UNICODE_VIEW_GLYPHS,
        ASCII_VIEW_GLYPHS,
    )
    assert all(c.isascii() for c in "".join(dataclasses.astuple(ASCII_VIEW_GLYPHS)))


def test_sparkline_and_diverging_bar() -> None:
    assert sparkline([]) == ""
    assert sparkline([0, 0, 0]) == "▁▁▁"
    assert sparkline([0, 1, 7, -2]) == "▁▂█▁"
    assert sparkline([1, 100]) == "▂█"
    assert sparkline([0, 4, 8], ASCII_VIEW_GLYPHS) == "_=#"
    zero = diverging_bar(0.0, 1.0, width=4)
    assert zero.plain == "    │    "
    up = diverging_bar(0.5, 1.0, width=4)
    assert up.plain == "    │██  "
    assert any(span.style == "#ef4444" for span in up.spans)
    down = diverging_bar(-2.0, 1.0, width=4)
    assert down.plain == "████│    "
    assert any(span.style == "#14b8a6" for span in down.spans)
    assert diverging_bar(0.001, 1.0, width=4).plain == "    │█   "
    assert diverging_bar(0.5, 0.0, width=3).plain == "   │   "
    assert diverging_bar(-0.5, 1.0, width=2, glyphs=ASCII_VIEW_GLYPHS).plain == " #|  "


# ---------------------------------------------------------------- selection


def test_ranked_files_and_lookups(linked_result: AnalysisResult) -> None:
    rows = ranked_files(linked_result)
    assert [r.rank for r in rows] == list(range(1, len(LINKED_FILES) + 1))
    assert sorted(r.path for r in rows) == list(LINKED_FILES)
    assert all(r.extra is linked_result.extras[r.path] for r in rows)
    by_path = ranked_by_path(linked_result)
    assert set(by_path) == set(LINKED_FILES)
    assert by_path["app/core.py"].rank == [f.path for f in linked_result.files].index("app/core.py") + 1
    slocs = file_slocs(linked_result)
    assert set(slocs) == set(LINKED_FILES)
    assert slocs["app/__init__.py"] == 0 and slocs["app/views.py"] > slocs["app/util.py"] > 0


def test_select_files_filters_and_sorts(linked_result: AnalysisResult) -> None:
    every = select_files(linked_result, FileQuery())
    assert [r.path for r in every] == [f.path for f in linked_result.files]
    api = select_files(linked_result, make_query(path="app/api"))
    assert {r.path for r in api} == {
        "app/api/__init__.py",
        "app/api/helpers.py",
        "app/api/routes.py",
        "app/api/schema.py",
    }
    low = select_files(linked_result, make_query(band="low"))
    assert low and all(r.score.band == "Low" for r in low)
    arch = select_files(linked_result, make_query(debt_type="architecture"))
    assert arch and all("architecture" in r.score.debt_types for r in arch)
    by_risk = select_files(linked_result, make_query(sort="risk"))
    keys = [(-r.score.risk, r.rank) for r in by_risk]
    assert keys == sorted(keys)
    by_cost = select_files(linked_result, make_query(sort="cost"))
    cost_keys = [(-r.extra.cost.hours, r.rank) for r in by_cost if r.extra]
    assert cost_keys == sorted(cost_keys)
    no_extras = dataclasses.replace(
        linked_result, extras={"app/core.py": linked_result.extras["app/core.py"]}
    )
    assert select_files(no_extras, make_query(sort="cost"))[0].path == "app/core.py"
    assert select_files(linked_result, make_query(path="lib/")) == []


def test_find_file(linked_result: AnalysisResult) -> None:
    first, second = linked_result.files[0].path, linked_result.files[1].path
    assert find_file(linked_result, "1").path == first
    assert find_file(linked_result, " #2 ").path == second
    assert find_file(linked_result, "app/core.py").rank == ranked_by_path(linked_result)["app/core.py"].rank
    assert find_file(linked_result, ".\\app\\util.py").path == "app/util.py"
    assert find_file(linked_result, "views.py").path == "app/views.py"
    assert find_file(linked_result, "api/schema.py").path == "app/api/schema.py"
    with pytest.raises(UsageError, match=r"__init__.py matches 2 files"):
        find_file(linked_result, "__init__.py")
    with pytest.raises(CliError, match=r"No ranked file matches iews.py") as missing:
        find_file(linked_result, "iews.py")
    assert missing.value.exit_code == 2
    with pytest.raises(CliError, match="ranks go from 1 to 9"):
        find_file(linked_result, "0")
    with pytest.raises(CliError, match="No file has rank 10"):
        find_file(linked_result, "10")
    with pytest.raises(CliError, match="No ranked file matches"):
        find_file(linked_result, "  ")
    empty = dataclasses.replace(linked_result, files=())
    with pytest.raises(CliError, match=NO_FILES):
        find_file(empty, "1")


def test_find_file_lists_many_matches(linked_result: AnalysisResult) -> None:
    score = linked_result.files[0]
    many = tuple(dataclasses.replace(score, path=f"pkg{i}/mod.py") for i in range(7))
    result = dataclasses.replace(linked_result, files=many, extras={})
    with pytest.raises(UsageError, match=r"mod.py matches 7 files: .* and 2 more"):
        find_file(result, "mod.py")


def test_ranked_table_and_rows(linked_loaded: LoadedAnalysis) -> None:
    rows = ranked_files(linked_loaded.result)
    plain = make_table_headers(ranked_table(rows))
    assert plain == ["#", "File", "Priority", "Risk", "Cost", "Debt", "Top reason"]
    changed = make_table_headers(ranked_table(rows[:2], changes={rows[0].path: "modified"}))
    assert changed == ["#", "File", "Change", "Priority", "Risk", "Cost", "Debt", "Top reason"]
    expected = analysis_json(linked_loaded, top=0)["files"]
    assert [file_row(r) for r in rows] == expected
    bare = dataclasses.replace(rows[0], extra=None)
    assert file_row(bare)["cost_hours"] is None
    console = Console(file=io.StringIO(), width=160, color_system=None, legacy_windows=False)
    console.print(ranked_table([bare], glyphs=ASCII_GLYPHS))
    assert f"* {bare.score.band} {bare.score.priority}" in console.file.getvalue()  # type: ignore[attr-defined]


def make_table_headers(table: object) -> list[str]:
    return [str(column.header) for column in table.columns]  # type: ignore[attr-defined]


def test_json_header_and_file_label(linked_loaded: LoadedAnalysis) -> None:
    header = json_header(linked_loaded, "entropiq.test.v1")
    assert header == {
        "schema": "entropiq.test.v1",
        "repo": "linked",
        "sha": linked_loaded.result.sha,
        "model_version": linked_loaded.result.model_version,
        "demo_model": True,
    }
    item = ranked_by_path(linked_loaded.result)["app/core.py"]
    label = file_label(item)
    assert label.plain == f"app/core.py  ● {item.score.band} {item.score.priority}  #{item.rank}"
    short = file_label(item, glyphs=ASCII_GLYPHS, name="core.py", rank=False)
    assert short.plain == f"core.py  * {item.score.band} {item.score.priority}"


# ---------------------------------------------------------------- /files and its pager


def test_pager_pages_and_opens_files(linked_session: Session) -> None:
    loaded = linked_session.current
    assert loaded is not None
    rows = ranked_files(loaded.result)
    lines = _reader(["p", "n", "n", "p", "3", "junk", "#99", "q"])
    _pager.run_files_pager(
        linked_session, loaded, rows, page_size=5, title="Ranked files", read=lambda _: next(lines)
    )
    out = _out(linked_session)
    assert out.count("Ranked files") == 3
    assert "Page 1 of 2 (files 1 to 5 of 9)" in out
    assert "Page 2 of 2 (files 6 to 9 of 9)" in out
    assert "This is the first page." in out
    assert "This is the last page." in out
    assert "Why it ranks here" in out and rows[2].path in out
    assert "Back in the file list" in out
    assert "Type n, p, a rank number, or press Enter to go back." in out
    assert "Error: No file has rank 99" in out


def test_pager_one_page_and_commands(linked_session: Session) -> None:
    loaded = linked_session.current
    assert loaded is not None
    rows = ranked_files(loaded.result)[:1]
    lines = _reader(["/status"])
    _pager.run_files_pager(
        linked_session, loaded, rows, page_size=20, title="One", read=lambda _: next(lines)
    )
    out = _out(linked_session)
    assert "1 file. Type a rank to open that file, or press Enter to go back." in out
    assert "linked" in out  # /status ran
    ended = _reader([])
    _pager.run_files_pager(linked_session, loaded, rows, page_size=0, title="One", read=lambda _: next(ended))
    with pytest.raises(ShellExit):
        quit_lines = _reader(["/quit"])
        _pager.run_files_pager(
            linked_session, loaded, rows, page_size=5, title="One", read=lambda _: next(quit_lines)
        )


def test_read_line_from_plain_stdin(linked_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\r\nsecond\x1b\n"))
    assert _pager.stdin_is_tty() is False
    assert _pager.read_line(linked_session, "files") == "n"
    assert _pager.read_line(linked_session, "files") == "second\x1b"
    assert _pager.read_line(linked_session, "files") is None
    out = _out(linked_session)
    assert f"files {CHEVRON} n" in out and f"files {CHEVRON} second?" in out
    monkeypatch.setattr(sys, "stdin", None)
    assert _pager.stdin_is_tty() is False


def test_files_command_pages(linked_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    linked_session.prefs = dataclasses.replace(linked_session.prefs, top=5)
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n2\n\n"))
    assert dispatch(linked_session, "/files") is True
    out = _out(linked_session)
    assert "Page 2 of 2" in out
    assert f"files {CHEVRON} 2" in out
    assert "Contribution by feature family" in out
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quit\n"))
    assert dispatch(linked_session, "/files") is False


def test_files_command_filters_and_errors(linked_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    dispatch(linked_session, "/files --path app/api --sort risk --top 2")
    out = _out(linked_session)
    assert "Ranked files (path app/api, sorted by risk)" in out
    assert "2 files. Type a rank" in out
    assert "app/core.py" not in out
    dispatch(linked_session, "/files --path nowhere/")
    assert "No files match these filters." in _out(linked_session)
    dispatch(linked_session, "/files --band urgent")
    out = _out(linked_session)
    assert "Error: unknown band 'urgent'" in out
    assert "Usage: /files [--band B]" in out
    dispatch(linked_session, "/files extra")
    assert "Usage: /files" in _out(linked_session)
    loaded = linked_session.current
    assert loaded is not None
    linked_session.current = dataclasses.replace(loaded, result=dataclasses.replace(loaded.result, files=()))
    dispatch(linked_session, "/files")
    assert NO_FILES in _out(linked_session)


def test_files_needs_an_analysis(session: Session) -> None:
    dispatch(session, "/files")
    assert "No analysis is loaded" in _out(session)


# ---------------------------------------------------------------- /file


def test_file_detail_sections(linked_session: Session) -> None:
    dispatch(linked_session, "/file app/core.py")
    out = _out(linked_session)
    for heading in (
        "Why it ranks here",
        "Contribution by feature family",
        "Most complex functions",
        "Connections",
        "Commit activity, last 24 months",
        "Debt types and usual remedies",
        "Cost breakdown",
    ):
        assert heading in out
    assert "Imported by (2)" in out and "app/util.py, app/views.py" in out
    assert "Tests that import it (1)" in out and "tests/test_core.py" in out
    assert "Static structure" in out and "Self-admitted debt" in out
    assert "core_0" in out and "Cyclomatic" in out
    for label in COST_LABELS.values():
        assert label in out
    assert "lines of code" in out and "linked @" in out and "(demo)" in out
    assert "commits; busiest month" in out or "commit; busiest month" in out


def test_file_detail_without_extras_and_plan(linked_loaded: LoadedAnalysis) -> None:
    result = dataclasses.replace(linked_loaded.result, extras={})
    loaded = dataclasses.replace(linked_loaded, result=result)
    item = find_file(result, "app/core.py")
    console = Console(file=io.StringIO(), width=120, color_system=None, legacy_windows=False)
    console.print(file_detail(loaded, item))
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert out.count(NO_DETAILS) == 4
    assert "No cost estimate stored" in out
    steps = linked_loaded.result.plan.steps
    assert steps and plan_step(linked_loaded.result, steps[0].path) == 1
    assert plan_step(linked_loaded.result, "nowhere.py") is None
    doc = file_json(loaded, item)
    assert doc["extra"] is None and doc["schema"] == "entropiq.file.v1"


def test_file_detail_odd_values(linked_loaded: LoadedAnalysis) -> None:
    from tdp_core.contracts import FileExtra

    result = linked_loaded.result
    item = find_file(result, "app/core.py")
    extra = item.extra
    assert isinstance(extra, FileExtra)
    quiet = dataclasses.replace(
        extra,
        functions=(),
        commit_activity=tuple((month, 0) for month, _ in extra.commit_activity),
        cost=dataclasses.replace(extra.cost, breakdown={**extra.cost.breakdown, "rewrite": 30.0}),
    )
    score = dataclasses.replace(item.score, reasons=(), mitigations=(), debt_types=(), shap_by_family={})
    odd = dataclasses.replace(item, score=score, extra=quiet)
    console = Console(file=io.StringIO(), width=120, color_system=None, legacy_windows=False)
    console.print(file_detail(linked_loaded, odd, glyphs=ASCII_GLYPHS, view=ASCII_VIEW_GLYPHS))
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "No single family stands out" in out
    assert "No functions in this file." in out
    assert "No commits in these 24 months." in out
    assert "No debt type assigned." in out
    assert "rewrite" in out
    empty = dataclasses.replace(odd, extra=dataclasses.replace(quiet, commit_activity=()))
    console.print(file_detail(linked_loaded, empty))
    assert "No commit activity stored" in console.file.getvalue()  # type: ignore[attr-defined]


def test_file_command_errors(linked_session: Session) -> None:
    dispatch(linked_session, "/file")
    out = _out(linked_session)
    assert "Error: say which file" in out and "Usage: /file <path or rank>" in out
    dispatch(linked_session, "/file nowhere.py")
    assert "No ranked file matches nowhere.py" in _out(linked_session)
    dispatch(linked_session, "/file #1")
    assert linked_session.current is not None
    assert linked_session.current.result.files[0].path in _out(linked_session)


# ---------------------------------------------------------------- completion and registration


def test_completion(linked_session: Session) -> None:
    def texts(args: list[str], word: str, completer: object = complete_files) -> list[str]:
        return [s.text for s in completer(linked_session, args, word)]  # type: ignore[operator]

    assert texts([], "--b") == ["--band"]
    assert texts([], "--") == ["--band", "--type", "--path", "--sort", "--top"]
    assert texts(["--band"], "H") == ["high"]
    assert texts(["--type"], "d") == ["duplication", "defect_hotspot"]
    assert texts(["--sort"], "") == ["priority", "risk", "cost"]
    assert texts(["--path"], "app/") == ["app/", "app/api/"]
    assert texts(["--top"], "") == []
    assert texts([], "app/c", complete_file) == ["app/core.py"]
    assert texts(["app/core.py"], "", complete_file) == []
    assert [s.text for s in folder_suggestions(linked_session, "app/a")] == ["app/api/"]
    linked_session.current = None
    assert list(folder_suggestions(linked_session, "")) == []


def test_registered() -> None:
    registry = load_registry()
    for name, group in (("files", "Explore"), ("file", "Explore")):
        command = registry.get(name)
        assert command is not None and command.group == group
        assert command.usage.startswith(f"/{name}")
        assert command.needs_analysis and command.cli is not None
        assert not any(d in command.details for d in DASHES)


# ---------------------------------------------------------------- tdp files / tdp file


def test_tdp_files(saved_linked: LoadedAnalysis, linked_repo: Path) -> None:
    app = build_app()
    table = runner.invoke(app, ["files", str(linked_repo), "--dev-model", "--top", "3"])
    assert table.exit_code == 0, table.stderr
    assert "Using the saved analysis" in table.stderr
    assert "Showing 3 of 9 files. Use --top 0 to see all." in table.stdout
    data = runner.invoke(
        app,
        ["files", str(linked_repo), "--dev-model", "--band", "low,medium", "--sort", "cost", "-f", "json"],
    )
    assert data.exit_code == 0, data.stderr
    doc = json.loads(data.stdout)
    assert doc["schema"] == "entropiq.files.v1" and doc["repo"] == "linked"
    assert doc["query"] == {"band": ["Medium", "Low"], "type": [], "path": "", "sort": "cost", "top": 20}
    assert doc["total"] == len(doc["files"]) and all(f["band"] in ("Medium", "Low") for f in doc["files"])
    none = runner.invoke(app, ["files", str(linked_repo), "--dev-model", "--path", "lib"])
    assert none.exit_code == 0 and "No files match these filters." in none.stdout
    bad_band = runner.invoke(app, ["files", str(linked_repo), "--dev-model", "--band", "urgent"])
    assert bad_band.exit_code == 2 and "unknown band" in bad_band.stderr
    bad_sort = runner.invoke(app, ["files", str(linked_repo), "--dev-model", "--sort", "name"])
    assert bad_sort.exit_code == 2


def test_tdp_file(saved_linked: LoadedAnalysis, linked_repo: Path) -> None:
    app = build_app()
    data = runner.invoke(app, ["file", "app/core.py", str(linked_repo), "--dev-model", "--format", "json"])
    assert data.exit_code == 0, data.stderr
    doc = json.loads(data.stdout)
    assert doc["schema"] == "entropiq.file.v1" and doc["files"] == 9
    assert doc["file"]["path"] == "app/core.py"
    assert doc["extra"]["test_importers"] == ["tests/test_core.py"]
    assert set(doc["remedies"]) == set(doc["file"]["debt_types"])
    table = runner.invoke(app, ["file", "1", str(linked_repo), "--dev-model"])
    assert table.exit_code == 0 and "Why it ranks here" in _plain(table.stdout)
    missing = runner.invoke(app, ["file", "nowhere.py", str(linked_repo), "--dev-model"])
    assert missing.exit_code == 2 and "No ranked file matches nowhere.py" in missing.stderr
