"""``/plan`` and ``tdp plan``: the budgeted fix plan of the loaded analysis.

The plan saved with an analysis is shown as it is when the budget matches; any other budget rebuilds it with
``tdp_core.planning.plan.make_plan`` from the stored scores, features, costs and import edges (fast, no git).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, cast

import typer
from rich.console import Console
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.completion import options_completer
from tdp_cli.errors import CliError, UsageError
from tdp_cli.options import (
    DevModelOption,
    FreshOption,
    ModelDirOption,
    RefOption,
    TargetArgument,
    cli_analysis,
    cli_guard,
    cli_session,
)
from tdp_cli.registry import Command, CommandRegistry, Suggestion
from tdp_cli.theme import (
    CYAN,
    DIM,
    can_encode,
    clean,
    format_hour_range,
    format_hours,
    make_table,
    short_sha,
)
from tdp_core.config import DEFAULT_SETTINGS

if TYPE_CHECKING:
    from rich.table import Table

    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult, Plan

PLAN_SCHEMA: Final = "entropiq.plan.v1"
MAX_BUDGET_HOURS: Final = 2000.0
BAR_WIDTH: Final = 10
BUDGET_ERROR: Final = f"--budget must be a number of hours between 0 and {MAX_BUDGET_HOURS:.0f}"
CYCLE_NOTE: Final = "Files that share a group number are in an import cycle: fix them together."
PLAN_CAPTION: Final = (
    "Return is benefit per estimated hour; "
    "benefit is the file's risk weighted by how many files depend on it."
)

_OPTIONS = options_completer("--budget")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def check_budget(hours: float) -> float:
    """``hours`` as a float when it is finite and between 0 and 2000. Raises :class:`UsageError` otherwise."""
    value = float(hours)
    if not math.isfinite(value) or not 0.0 <= value <= MAX_BUDGET_HOURS:
        raise UsageError(BUDGET_ERROR)
    return value


def parse_budget(raw: str) -> float:
    """The budget typed after ``--budget``. Raises :class:`UsageError` for anything but 0 to 2000 hours."""
    try:
        value = float(raw.strip().lower().removesuffix("h").strip())
    except ValueError as exc:
        raise UsageError(BUDGET_ERROR) from exc
    return check_budget(value)


def plan_for_budget(result: AnalysisResult, budget_hours: float) -> Plan:
    """The saved plan when it was made for ``budget_hours``, else a plan rebuilt for that budget."""
    if abs(budget_hours - result.plan.budget_hours) <= 1e-9:
        return result.plan
    from tdp_core.planning.plan import import_graph_from_edges, make_plan

    try:
        return make_plan(
            result.files,
            result.features,
            {path: extra.cost for path, extra in result.extras.items()},
            import_graph_from_edges(result.graph.edges),
            budget_hours,
            DEFAULT_SETTINGS.min_plan_priority,
        )
    except (KeyError, ValueError) as exc:
        raise CliError(f"Cannot build the plan: {clean(str(exc))}") from exc


def candidate_benefit(
    result: AnalysisResult, min_priority: int = DEFAULT_SETTINGS.min_plan_priority
) -> tuple[int, float]:
    """How many files the planner could choose from and their total benefit (``make_plan``'s own rule)."""
    from tdp_core.planning.plan import benefit

    features = result.features
    count = 0
    total = 0.0
    for score in result.files:
        extra = result.extras.get(score.path)
        if score.priority < min_priority or extra is None or not extra.cost.hours > 0:
            continue
        if score.path not in features.index:
            continue
        count += 1
        dependents = float(cast("float", features.at[score.path, "transitive_dependents"]))
        total += benefit(score.risk, dependents)
    return count, total


def benefit_bar(fraction: float, *, width: int = BAR_WIDTH, ascii_only: bool = False) -> Text:
    """A ``█████░░░░░  50%`` bar (``#####-----`` in ASCII); the fraction is clamped to 0..1."""
    value = min(max(fraction, 0.0), 1.0) if math.isfinite(fraction) else 0.0
    filled = round(value * width)
    full, empty = ("#", "-") if ascii_only else ("█", "░")
    bar = Text(full * filled, style=CYAN)
    bar.append(empty * (width - filled), style=DIM)
    bar.append(f" {round(100 * value):3d}%")
    return bar


def plan_table(plan: Plan, *, reference_benefit: float, ascii_only: bool = False) -> Table:
    """The plan's steps in fix order with a cumulative benefit bar against ``reference_benefit``.

    Each step's reason is a dim second line under its path, and only that cell wraps, so the order, hours,
    return and bar columns stay whole on an 80 column terminal.
    """
    reference = reference_benefit if reference_benefit > 0 else plan.total_benefit
    table = make_table()
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("File and reason", overflow="fold")
    table.add_column("Group", justify="right", no_wrap=True)
    table.add_column("Hours", justify="right", no_wrap=True)
    table.add_column("Return", justify="right", no_wrap=True)
    table.add_column("Benefit so far", no_wrap=True)
    cumulative = 0.0
    for step in sorted(plan.steps, key=lambda s: (s.order, s.path)):
        cumulative += step.benefit
        fraction = cumulative / reference if reference > 0 else 0.0
        cell = Text(clean(step.path))
        cell.append("\n")
        cell.append(clean(step.reason), style="dim")
        table.add_row(
            str(step.order),
            cell,
            str(step.group),
            format_hour_range(step.cost.low, step.cost.high),
            f"{step.roi:.3f}",
            benefit_bar(fraction, ascii_only=ascii_only),
        )
    return table


def empty_plan_message(
    result: AnalysisResult, plan: Plan, min_priority: int = DEFAULT_SETTINGS.min_plan_priority
) -> str:
    """Why a plan has no steps, in one sentence."""
    if not result.files:
        return "No files were ranked, so there is nothing to plan."
    costs = [
        extra.cost.hours
        for score in result.files
        if score.priority >= min_priority
        and (extra := result.extras.get(score.path)) is not None
        and extra.cost.hours > 0
    ]
    if not costs:
        return f"Nothing to plan: no file has priority {min_priority} or more."
    cheapest = min(costs)
    if cheapest <= plan.budget_hours:
        return f"Nothing to plan within a budget of {format_hours(plan.budget_hours)}."
    return (
        f"No file fits in a budget of {format_hours(plan.budget_hours)}; the cheapest candidate needs about "
        f"{format_hours(cheapest)}. Raise the budget to at least {math.ceil(cheapest)} hours."
    )


def print_plan(console: Console, loaded: LoadedAnalysis, plan: Plan) -> None:
    """Print the plan as a table with totals and notes (or one sentence when it is empty)."""
    meta = loaded.meta
    min_priority = DEFAULT_SETTINGS.min_plan_priority
    header = Text("Fix plan for ", style="bold")
    header.append(clean(meta.repo))
    header.append(f" @ {short_sha(meta.sha)}, budget {format_hours(plan.budget_hours)}")
    console.print(header)
    if not plan.steps:
        console.print(Text(empty_plan_message(loaded.result, plan, min_priority), style="yellow"))
        return
    count, reference = candidate_benefit(loaded.result, min_priority)
    ascii_only = not can_encode("█░", console.encoding)
    console.print(plan_table(plan, reference_benefit=reference, ascii_only=ascii_only))
    low = math.fsum(step.cost.low for step in plan.steps)
    high = math.fsum(step.cost.high for step in plan.steps)
    share = round(100 * plan.total_benefit / reference) if reference > 0 else 100
    console.print(
        Text(
            f"{_plural(len(plan.steps), 'file')}, {format_hours(plan.total_hours)} of the "
            f"{format_hours(plan.budget_hours)} budget (estimate {format_hour_range(low, high)}), "
            f"benefit {plan.total_benefit:.2f}, {share}% of the benefit of fixing all "
            f"{_plural(count, 'file')} with priority {min_priority} or more."
        )
    )
    groups = Counter(step.group for step in plan.steps)
    if any(n > 1 for n in groups.values()):
        console.print(Text(CYCLE_NOTE, style="dim"))
    console.print(Text(PLAN_CAPTION, style="dim"))


def plan_json(loaded: LoadedAnalysis, plan: Plan) -> dict[str, Any]:
    """The plan as a JSON-ready dict (schema ``entropiq.plan.v1``)."""
    count, reference = candidate_benefit(loaded.result)
    return {
        "schema": PLAN_SCHEMA,
        "repo": loaded.meta.repo,
        "sha": loaded.meta.sha,
        "model_version": loaded.meta.model_version,
        "budget_hours": plan.budget_hours,
        "min_priority": DEFAULT_SETTINGS.min_plan_priority,
        "total_hours": plan.total_hours,
        "total_benefit": plan.total_benefit,
        "candidate_files": count,
        "candidate_benefit": reference,
        "steps": [
            {
                "order": step.order,
                "path": step.path,
                "group": step.group,
                "benefit": step.benefit,
                "roi": step.roi,
                "cost_hours": step.cost.hours,
                "cost_low": step.cost.low,
                "cost_high": step.cost.high,
                "reason": step.reason,
            }
            for step in sorted(plan.steps, key=lambda s: (s.order, s.path))
        ],
    }


def plan_handler(session: Session, args: Sequence[str]) -> None:
    """``/plan [--budget HOURS]``."""
    parsed = parse_args(args, values=("--budget",), max_positionals=0)
    raw = parsed.value("--budget")
    budget = session.prefs.budget_hours if raw is None else parse_budget(raw)
    loaded = session.require_analysis()
    print_plan(session.console, loaded, plan_for_budget(loaded.result, budget))


def _complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    if args and args[-1] == "--budget":
        return
    yield from _OPTIONS(session, args, word)


def plan_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    budget: Annotated[
        float | None,
        typer.Option(
            "--budget",
            min=0.0,
            max=MAX_BUDGET_HOURS,
            help="Hours available for fixes (default from settings, 40).",
            show_default=False,
        ),
    ] = None,
    output_format: Annotated[
        Literal["table", "md", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    ref: RefOption = None,
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Show the fix plan that fits a time budget."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    with cli_guard(session.err_console):
        hours = session.prefs.budget_hours if budget is None else check_budget(budget)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        plan = plan_for_budget(loaded.result, hours)
        if output_format == "json":
            sys.stdout.write(json.dumps(plan_json(loaded, plan), indent=2) + "\n")
        elif output_format == "md":
            from tdp_core.report.markdown import plan_to_markdown

            sys.stdout.write(plan_to_markdown(plan))
        else:
            print_plan(session.console, loaded, plan)


def register(registry: CommandRegistry) -> None:
    """Register ``/plan`` and ``tdp plan``."""
    registry.add(
        Command(
            name="plan",
            group="Plan",
            help="Show the fix plan that fits a time budget",
            usage="/plan [--budget HOURS]",
            details=(
                "Chooses the files whose fixes give the most benefit within the budget (default: the "
                "budget_hours setting, see /settings) and orders them so a file comes before the files that "
                "depend on it. Group numbers mark files in one import cycle; Hours is the estimated effort "
                "range; Return is benefit per hour; the bar shows how much of the benefit of fixing every "
                "file with priority 50 or more the plan reaches so far."
            ),
            handler=plan_handler,
            complete=_complete,
            needs_analysis=True,
            cli=plan_cli,
        )
    )
