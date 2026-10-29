from __future__ import annotations

import dataclasses
import io
import json
import math
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tdp_cli.commands import plan as plan_cmd
from tdp_cli.errors import UsageError
from tdp_cli.home import EntropiqHome
from tdp_cli.main import build_app
from tdp_cli.session import Session
from tdp_cli.shell import NO_ANALYSIS_HINT, dispatch
from tdp_cli.store import AnalysisStore, LoadedAnalysis
from tdp_cli.theme import format_hour_range
from tdp_core.config import DEFAULT_SETTINGS

runner = CliRunner()
SAVED_ORDER = ["pkg/mu.py", "pkg/lam.py", "pkg/kappa.py", "pkg/eta.py", "pkg/theta.py", "pkg/iota.py"]


@pytest.fixture
def saved_home(dev_home: Path, sample_loaded: LoadedAnalysis) -> Path:
    """The test home with ``sample_loaded`` saved, so ``tdp plan <sample_repo>`` reuses it."""
    AnalysisStore(EntropiqHome.from_env().analyses).save(sample_loaded)
    return dev_home


def _out(session: Session) -> str:
    return session.console.file.getvalue()  # type: ignore[attr-defined]


def test_check_and_parse_budget() -> None:
    assert plan_cmd.check_budget(0) == 0.0
    assert plan_cmd.check_budget(2000) == 2000.0
    assert plan_cmd.parse_budget("10") == 10.0
    assert plan_cmd.parse_budget(" 12.5H ") == 12.5
    for bad in ("-1", "2001", "abc", "nan", "inf", ""):
        with pytest.raises(UsageError, match="--budget must be a number"):
            plan_cmd.parse_budget(bad)
    with pytest.raises(UsageError):
        plan_cmd.check_budget(math.inf)


def test_plan_for_budget(sample_loaded: LoadedAnalysis) -> None:
    from tdp_core.planning.plan import import_graph_from_edges, make_plan

    result = sample_loaded.result
    assert plan_cmd.plan_for_budget(result, result.plan.budget_hours) is result.plan
    rebuilt = plan_cmd.plan_for_budget(result, 12.0)
    expected = make_plan(
        result.files,
        result.features,
        {p: e.cost for p, e in result.extras.items()},
        import_graph_from_edges(result.graph.edges),
        12.0,
        DEFAULT_SETTINGS.min_plan_priority,
    )
    assert rebuilt == expected
    assert rebuilt.total_hours <= 12.0 + 1e-9


def test_candidate_benefit(sample_loaded: LoadedAnalysis) -> None:
    count, total = plan_cmd.candidate_benefit(sample_loaded.result)
    assert count == 6
    assert total == pytest.approx(sample_loaded.result.plan.total_benefit)


def test_benefit_bar() -> None:
    assert plan_cmd.benefit_bar(0).plain == "░" * 10 + "   0%"
    assert plan_cmd.benefit_bar(0.5).plain == "█" * 5 + "░" * 5 + "  50%"
    assert plan_cmd.benefit_bar(1).plain == "█" * 10 + " 100%"
    assert plan_cmd.benefit_bar(1.5).plain == "█" * 10 + " 100%"
    assert plan_cmd.benefit_bar(math.nan).plain.endswith("  0%")
    assert plan_cmd.benefit_bar(0.3, ascii_only=True).plain == "###-------  30%"


def test_plan_shows_the_saved_plan(loaded_session: Session) -> None:
    assert dispatch(loaded_session, "/plan")
    text = _out(loaded_session)
    assert "Fix plan for sample" in text
    assert "budget 40.0 h" in text
    positions = [text.index(path) for path in SAVED_ORDER]
    assert positions == sorted(positions)
    assert "6 files" in text
    assert "31.5 h" in text
    assert "100%" in text
    assert "Return is benefit per estimated hour" in text


def test_plan_with_a_budget(loaded_session: Session) -> None:
    dispatch(loaded_session, "/plan --budget 12")
    text = _out(loaded_session)
    assert "budget 12.0 h" in text
    assert "pkg/mu.py" in text
    assert "pkg/kappa.py" in text
    assert "pkg/lam.py" not in text
    assert "2 files" in text


def test_plan_with_a_tiny_budget(loaded_session: Session) -> None:
    dispatch(loaded_session, "/plan --budget 3")
    text = _out(loaded_session)
    assert "No file fits" in text
    assert "4.2 h" in text
    assert "Raise the budget to at least 5 hours" in text


def test_plan_usage_errors(loaded_session: Session) -> None:
    dispatch(loaded_session, "/plan --budget abc")
    text = _out(loaded_session)
    assert "--budget must be a number" in text
    assert "Usage: /plan [--budget HOURS]" in text
    dispatch(loaded_session, "/plan x")
    assert "this command takes no arguments" in _out(loaded_session)


def test_plan_without_an_analysis(session: Session) -> None:
    dispatch(session, "/plan")
    assert NO_ANALYSIS_HINT in _out(session).replace("\n", " ")


def test_empty_plan_messages(sample_loaded: LoadedAnalysis) -> None:
    result = sample_loaded.result
    empty = dataclasses.replace(result.plan, steps=(), budget_hours=0.0, total_hours=0.0, total_benefit=0.0)
    no_files = dataclasses.replace(result, files=[])
    assert "nothing to plan" in plan_cmd.empty_plan_message(no_files, empty)
    low = dataclasses.replace(result, files=[dataclasses.replace(f, priority=10) for f in result.files])
    assert "no file has priority 50" in plan_cmd.empty_plan_message(low, empty)
    assert "No file fits in a budget of 0 h" in plan_cmd.empty_plan_message(result, empty)
    roomy = dataclasses.replace(empty, budget_hours=100.0)
    assert plan_cmd.empty_plan_message(result, roomy).startswith("Nothing to plan within")


def test_cycle_note_and_narrow_console(sample_loaded: LoadedAnalysis) -> None:
    plan = sample_loaded.result.plan
    first, second, *rest = sorted(plan.steps, key=lambda s: s.order)
    shared = dataclasses.replace(plan, steps=(first, dataclasses.replace(second, group=first.group), *rest))
    wide = Console(file=io.StringIO(), width=120, color_system=None)
    plan_cmd.print_plan(wide, sample_loaded, shared)
    assert plan_cmd.CYCLE_NOTE in wide.file.getvalue()  # type: ignore[attr-defined]

    narrow = Console(file=io.StringIO(), width=70, color_system=None)
    plan_cmd.print_plan(narrow, sample_loaded, plan)
    text = narrow.file.getvalue()  # type: ignore[attr-defined]
    folded = text.replace("\n", "").replace(" ", "")
    for path in SAVED_ORDER:
        assert path in folded
    assert plan_cmd.CYCLE_NOTE not in text


def test_plan_table_keeps_every_column_at_80_columns(sample_loaded: LoadedAnalysis) -> None:
    plan = sample_loaded.result.plan
    long_path = "src/requests/packages/_internal_utils.py"
    reason = "fix before src/requests/sessions.py, which depends on it"
    steps = sorted(plan.steps, key=lambda s: s.order)
    first = dataclasses.replace(steps[0], path=long_path, reason=reason)
    console = Console(file=io.StringIO(), width=80, color_system=None)
    console.print(
        plan_cmd.plan_table(dataclasses.replace(plan, steps=(first, *steps[1:])), reference_benefit=0)
    )
    text = console.file.getvalue()  # type: ignore[attr-defined]
    lines = text.splitlines()
    assert max(len(line) for line in lines) <= 80
    assert "…" not in text
    for step in steps:
        assert format_hour_range(step.cost.low, step.cost.high) in text
    assert " 100%" in text
    starts = [i for i, line in enumerate(lines) if line[:1].isdigit()]
    assert [lines[i].split()[0] for i in starts] == [str(step.order) for step in steps]
    # The first step's cell: the start of its path sits on the row line, the rest on the lines below it.
    below = lines[starts[0] + 1 : starts[1]]
    first_cell = lines[starts[0]].split()[1] + "".join("".join(line.split()) for line in below)
    assert first_cell.startswith(long_path)
    assert first_cell.endswith("".join(reason.split()))


def test_plan_json(sample_loaded: LoadedAnalysis) -> None:
    data = plan_cmd.plan_json(sample_loaded, sample_loaded.result.plan)
    assert data["schema"] == "entropiq.plan.v1"
    assert data["budget_hours"] == 40.0
    assert data["candidate_files"] == 6
    assert [s["path"] for s in data["steps"]] == SAVED_ORDER
    assert set(data["steps"][0]) == {
        "order",
        "path",
        "group",
        "benefit",
        "roi",
        "cost_hours",
        "cost_low",
        "cost_high",
        "reason",
    }
    json.dumps(data)


def test_plan_cli(saved_home: Path, sample_repo: Path) -> None:
    app = build_app()
    table = runner.invoke(app, ["plan", str(sample_repo), "--dev-model"])
    assert table.exit_code == 0, table.stderr
    assert "Fix plan for sample" in table.stdout
    assert "pkg/mu.py" in table.stdout

    as_json = runner.invoke(app, ["plan", str(sample_repo), "--dev-model", "--budget", "12", "-f", "json"])
    assert as_json.exit_code == 0, as_json.stderr
    data = json.loads(as_json.stdout)
    assert data["schema"] == "entropiq.plan.v1"
    assert data["budget_hours"] == 12.0
    assert len(data["steps"]) == 2

    md = runner.invoke(app, ["plan", str(sample_repo), "--dev-model", "--format", "md"])
    assert md.exit_code == 0, md.stderr
    assert "### Fix plan" in md.stdout

    too_big = runner.invoke(app, ["plan", str(sample_repo), "--dev-model", "--budget", "5000"])
    assert too_big.exit_code == 2
