"""The ``/file`` detail screen: why a file ranks where it does and what fixing it involves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from tdp_cli.commands._views import (
    DEBT_REMEDIES,
    RED,
    TEAL,
    UNICODE_VIEW_GLYPHS,
    RankedFile,
    ViewGlyphs,
    diverging_bar,
    file_slocs,
    json_header,
    limited,
    section,
    sparkline,
)
from tdp_cli.theme import (
    CYAN,
    UNICODE_GLYPHS,
    Glyphs,
    band_text,
    clean,
    format_hour_range,
    format_hours,
    format_risk_range,
    make_panel,
    make_table,
    short_sha,
)

if TYPE_CHECKING:
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult, FileExtra

TOP_FUNCTIONS: Final = 5
MAX_NEIGHBOURS: Final = 8
BAR_HALF_WIDTH: Final = 12

COST_LABELS: Final[dict[str, str]] = {
    "complexity": "Simplify complex functions",
    "duplication": "Remove duplicated code",
    "tests": "Write missing tests",
    "cycle": "Break the import cycle",
    "satd": "Resolve admitted debt comments",
}
"""Labels of the cost breakdown kinds, in ``tdp_core.planning.cost.COST_KINDS`` order."""

FILE_SCHEMA: Final = "entropiq.file.v1"
NO_DETAILS: Final = "No details stored for this file."
SHAP_NOTE: Final = (
    "Red bars to the right raise the risk, teal bars to the left lower it "
    "(SHAP values in ranker score units)."
)
COST_NOTE: Final = (
    "Minutes are counted before the blast-radius multiplier (more importers cost more) "
    "and the minimum estimate."
)


def plan_step(result: AnalysisResult, path: str) -> int | None:
    """The order of ``path`` in the analysis' fix plan (1-based), or None when it is not in the plan."""
    for step in result.plan.steps:
        if step.path == path:
            return step.order
    return None


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def _sloc(result: AnalysisResult, path: str) -> int | None:
    return file_slocs(result).get(path)


def _header(loaded: LoadedAnalysis, item: RankedFile, glyphs: Glyphs) -> RenderableType:
    meta, result, score, extra = loaded.meta, loaded.result, item.score, item.extra
    sep = f"  {glyphs.sep}  "
    first = band_text(score.band, score.priority, glyphs=glyphs)
    first.append(f"{sep}rank {item.rank} of {len(result.files)}{sep}")
    first.append(
        f"risk {format_risk_range(score.risk, score.risk_low, score.risk_high)} in the next 12 months"
    )
    if extra is not None:
        cost = Text(f"Estimated fix {format_hour_range(extra.cost.low, extra.cost.high)} ")
        cost.append(f"({format_hours(extra.cost.hours)})", style="dim")
    else:
        cost = Text("No cost estimate stored")
    step = plan_step(result, score.path)
    if step is not None:
        cost.append(f"{sep}fix plan step {step} of {len(result.plan.steps)}")
    else:
        cost.append(f"{sep}not in the fix plan ({format_hours(result.plan.budget_hours)} budget)")
    debt = Text(f"Debt types: {clean(', '.join(score.debt_types)) or 'none'}")
    sloc = _sloc(result, score.path)
    if sloc is not None:
        debt.append(f"{sep}{sloc:,} lines of code")
    model = f"{clean(meta.repo)} @ {short_sha(result.sha)}, model {clean(result.model_version)}"
    footer = _dim(model + (" (demo)" if meta.demo_model else ""))
    return make_panel(Group(first, cost, debt, footer), Text(clean(score.path), style="bold"))


def _reasons(item: RankedFile, glyphs: Glyphs) -> list[RenderableType]:
    score = item.score
    lines: list[RenderableType] = []
    for reason in score.reasons:
        line = Text(f"{glyphs.up} ", style=RED)
        line.append(clean(reason.text), style="")
        lines.append(line)
    for reason in score.mitigations:
        line = Text(f"{glyphs.down} ", style=TEAL)
        line.append(clean(reason.text), style="")
        lines.append(line)
    return lines or [_dim("No single family stands out for this file.")]


def _shap_value(value: float) -> str:
    return "0.00" if round(value, 2) == 0 else f"{value:+.2f}"


def _families(item: RankedFile, view: ViewGlyphs) -> list[RenderableType]:
    from tdp_core.contracts import FAMILY_NAMES, FAMILY_ORDER

    values = {fam: float(item.score.shap_by_family.get(fam, 0.0)) for fam in FAMILY_ORDER}
    scale = max((abs(v) for v in values.values()), default=0.0) or 1.0
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    for fam in FAMILY_ORDER:
        value = values[fam]
        grid.add_row(
            FAMILY_NAMES[fam],
            diverging_bar(value, scale, width=BAR_HALF_WIDTH, glyphs=view),
            _shap_value(value),
        )
    return [grid, _dim(SHAP_NOTE)]


def _functions(extra: FileExtra) -> list[RenderableType]:
    if not extra.functions:
        return [_dim("No functions in this file.")]
    ordered = sorted(extra.functions, key=lambda f: (-f.cyclomatic, -f.cognitive, f.start_line, f.name))
    table = make_table()
    table.add_column("Function", overflow="fold")
    table.add_column("Lines", no_wrap=True)
    table.add_column("Cyclomatic", justify="right", no_wrap=True)
    table.add_column("Cognitive", justify="right", no_wrap=True)
    for func in ordered[:TOP_FUNCTIONS]:
        table.add_row(
            Text(clean(func.name)),
            f"{func.start_line}-{func.end_line}",
            str(func.cyclomatic),
            str(func.cognitive),
        )
    out: list[RenderableType] = [table]
    if len(ordered) > TOP_FUNCTIONS:
        out.append(_dim(f"and {len(ordered) - TOP_FUNCTIONS} more functions"))
    return out


def _names(paths: Sequence[str]) -> Text:
    if not paths:
        return _dim("none")
    shown, more = limited(paths, MAX_NEIGHBOURS)
    text = Text(clean(", ".join(shown)))
    if more:
        text.append(f" and {more} more", style="dim")
    return text


def _connections(extra: FileExtra) -> list[RenderableType]:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row(f"Imported by ({len(extra.dependents)})", _names(extra.dependents))
    grid.add_row(f"Imports ({len(extra.dependencies)})", _names(extra.dependencies))
    grid.add_row(f"Tests that import it ({len(extra.test_importers)})", _names(extra.test_importers))
    return [grid]


def _activity(extra: FileExtra, view: ViewGlyphs) -> list[RenderableType]:
    months = extra.commit_activity
    if not months:
        return [_dim("No commit activity stored for this file.")]
    line = _dim(clean(months[0][0]))
    line.append(" ")
    line.append(sparkline([count for _, count in months], view), style=CYAN)
    line.append(" ")
    line.append(clean(months[-1][0]), style="dim")
    total = sum(count for _, count in months)
    if total <= 0:
        return [line, _dim("No commits in these 24 months.")]
    busiest, most = max(months, key=lambda m: (m[1], m[0]))
    plural = "commit" if total == 1 else "commits"
    return [line, _dim(f"{total} {plural}; busiest month {clean(busiest)} with {most}")]


def _remedies(item: RankedFile) -> list[RenderableType]:
    if not item.score.debt_types:
        return [_dim("No debt type assigned.")]
    lines: list[RenderableType] = []
    for name in item.score.debt_types:
        line = Text(clean(name), style="bold")
        line.append(": " + DEBT_REMEDIES.get(name, "No usual remedy is recorded for this type."))
        lines.append(line)
    return lines


def _cost(extra: FileExtra) -> list[RenderableType]:
    cost = extra.cost
    table = make_table()
    table.add_column("Work")
    table.add_column("Minutes", justify="right", no_wrap=True)
    keys = [k for k in COST_LABELS if k in cost.breakdown]
    keys += sorted(k for k in cost.breakdown if k not in COST_LABELS)
    for key in keys:
        label = COST_LABELS.get(key)
        table.add_row(label if label is not None else Text(clean(key)), f"{cost.breakdown[key]:.0f}")
    estimate = Text(f"Estimate {format_hours(cost.hours)} ({format_hour_range(cost.low, cost.high)}). ")
    estimate.append(COST_NOTE, style="dim")
    return [table, estimate]


def file_detail(
    loaded: LoadedAnalysis,
    item: RankedFile,
    *,
    glyphs: Glyphs = UNICODE_GLYPHS,
    view: ViewGlyphs = UNICODE_VIEW_GLYPHS,
) -> Group:
    """The detail screen of one ranked file: header panel, then seven sections.

    Sections: why it ranks here (reasons and mitigations), contribution by feature family (SHAP bars),
    most complex functions, connections, commit activity of the last 24 months, debt types with their usual
    remedy, and the cost breakdown. Sections that need stored details say so when the file has none.
    """
    extra = item.extra
    parts: list[RenderableType] = [_header(loaded, item, glyphs)]

    def add(title: str, body: list[RenderableType]) -> None:
        parts.append(Text(""))
        parts.append(section(title))
        parts.extend(body)

    missing: list[RenderableType] = [_dim(NO_DETAILS)]
    add("Why it ranks here", _reasons(item, glyphs))
    add("Contribution by feature family", _families(item, view))
    add("Most complex functions", _functions(extra) if extra else missing)
    add("Connections", _connections(extra) if extra else missing)
    add("Commit activity, last 24 months", _activity(extra, view) if extra else missing)
    add("Debt types and usual remedies", _remedies(item))
    add("Cost breakdown", _cost(extra) if extra else missing)
    return Group(*parts)


def file_json(loaded: LoadedAnalysis, item: RankedFile) -> dict[str, Any]:
    """The ``tdp file --format json`` document (schema ``entropiq.file.v1``)."""
    from tdp_core.serialize import to_jsonable

    result = loaded.result
    return {
        **json_header(loaded, FILE_SCHEMA),
        "rank": item.rank,
        "files": len(result.files),
        "file": to_jsonable(item.score),
        "extra": to_jsonable(item.extra) if item.extra is not None else None,
        "plan_step": plan_step(result, item.path),
        "remedies": {t: DEBT_REMEDIES[t] for t in item.score.debt_types if t in DEBT_REMEDIES},
    }
