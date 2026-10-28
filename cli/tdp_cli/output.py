"""Renderings of an analysis shared by the commands: summary card, file tables, JSON and Markdown.

Repository strings are always cleaned and placed in :class:`rich.text.Text` objects, never in markup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC
from typing import TYPE_CHECKING, Any, Final

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tdp_cli.theme import (
    BAND_ORDER,
    UNICODE_GLYPHS,
    Glyphs,
    band_text,
    clean,
    format_hour_range,
    format_hours,
    format_risk,
    make_panel,
    make_table,
    short_sha,
)

if TYPE_CHECKING:
    from tdp_cli.store import AnalysisMeta, LoadedAnalysis
    from tdp_core.contracts import AnalysisResult, FileExtra, FileScore

JSON_SCHEMA: Final = "entropiq.analysis.v1"


def band_counts(files: Sequence[FileScore]) -> dict[str, int]:
    """Files per band in :data:`~tdp_cli.theme.BAND_ORDER` order (every band present, zeros kept)."""
    counts = dict.fromkeys(BAND_ORDER, 0)
    for f in files:
        counts[f.band] = counts.get(f.band, 0) + 1
    return counts


def critical_hours(result: AnalysisResult) -> float:
    """Estimated hours to fix every Critical file (sum of their cost estimates)."""
    return float(
        sum(
            result.extras[f.path].cost.hours
            for f in result.files
            if f.band == "Critical" and f.path in result.extras
        )
    )


def critical_range(result: AnalysisResult) -> tuple[float, float]:
    """Low and high estimates of :func:`critical_hours` in hours."""
    costs = [
        result.extras[f.path].cost for f in result.files if f.band == "Critical" and f.path in result.extras
    ]
    return float(sum(c.low for c in costs)), float(sum(c.high for c in costs))


def top_reason(score: FileScore) -> str:
    """The first reason sentence of a file, or an empty string."""
    return score.reasons[0].text if score.reasons else ""


def _local_time(meta: AnalysisMeta) -> str:
    return meta.analyzed_at.astimezone().strftime("%Y-%m-%d %H:%M")


def files_table(
    files: Sequence[FileScore],
    extras: Mapping[str, FileExtra] | None = None,
    *,
    limit: int | None = None,
    start: int = 1,
    title: str | None = None,
    glyphs: Glyphs = UNICODE_GLYPHS,
) -> Table:
    """A ranked table of ``files``: #, File, Band, Risk, Cost (when ``extras`` is given), Top reason.

    ``limit`` caps the rows (None or 0 shows all); ``start`` is the rank of the first row.
    """
    table = make_table(title)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("File", no_wrap=True, overflow="fold")
    table.add_column("Band", no_wrap=True)
    table.add_column("Risk", justify="right", no_wrap=True)
    if extras is not None:
        table.add_column("Cost", justify="right", no_wrap=True)
    table.add_column("Top reason", overflow="fold", min_width=20)
    shown = files[:limit] if limit else files
    for rank, f in enumerate(shown, start=start):
        row: list[RenderableType] = [
            str(rank),
            Text(clean(f.path)),
            band_text(f.band, f.priority, glyphs=glyphs),
            format_risk(f.risk),
        ]
        if extras is not None:
            extra = extras.get(f.path)
            row.append(format_hour_range(extra.cost.low, extra.cost.high) if extra else "")
        row.append(Text(clean(top_reason(f)), style="dim"))
        table.add_row(*row)
    return table


def _counts_line(counts: Mapping[str, int], glyphs: Glyphs) -> Text:
    line = Text()
    for i, band in enumerate(BAND_ORDER):
        if i:
            line.append("   ")
        line.append_text(band_text(band, glyphs=glyphs))
        line.append(f" {counts.get(band, 0)}")
    return line


def summary_card(
    loaded: LoadedAnalysis, *, top: int = 5, glyphs: Glyphs = UNICODE_GLYPHS, reused: bool = False
) -> Panel:
    """A panel summarising an analysis: counts per band, Critical debt hours, model, and the top files."""
    meta, result = loaded.meta, loaded.result
    title = Text(f"{clean(meta.repo)} @ {short_sha(meta.sha)}", style="bold")
    parts: list[RenderableType] = []
    if reused:
        parts.append(Text(f"Saved analysis from {_local_time(meta)}", style="dim"))
    else:
        total = result.stats.get("total_s")
        took = f" in {float(total):.1f} s" if isinstance(total, (int, float)) else ""
        parts.append(Text(f"{len(result.files)} files analysed{took}", style="dim"))
    if not result.files:
        parts.append(Text("No Python source files to rank at this commit.", style="yellow"))
    else:
        parts.append(_counts_line(band_counts(result.files), glyphs))
        hours = critical_hours(result)
        low, high = critical_range(result)
        debt = Text("Estimated Critical debt: ")
        debt.append(format_hours(hours), style="bold")
        if hours:
            debt.append(f" ({format_hour_range(low, high)})", style="dim")
        parts.append(debt)
    model = Text(f"Model {clean(result.model_version)} {glyphs.sep} ", style="dim")
    model.append("demo" if meta.demo_model else "trained", style="yellow" if meta.demo_model else "green")
    parts.append(model)
    if top > 0 and result.files:
        parts.append(Text(""))
        parts.append(files_table(result.files, limit=top, glyphs=glyphs))
    return make_panel(Group(*parts), title)


def history_table(entries: Sequence[AnalysisMeta], *, glyphs: Glyphs = UNICODE_GLYPHS) -> Table:
    """Saved analyses: #, Repository, Commit, Analysed (local time), Critical, High, Files, Model."""
    table = make_table()
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Repository", overflow="fold")
    table.add_column("Commit", no_wrap=True)
    table.add_column("Analysed", no_wrap=True)
    table.add_column("Critical", justify="right", style="bold #ef4444", no_wrap=True)
    table.add_column("High", justify="right", style="bold #f97316", no_wrap=True)
    table.add_column("Files", justify="right", no_wrap=True)
    table.add_column("Model", overflow="fold")
    for number, meta in enumerate(entries, start=1):
        model = Text(clean(meta.model_version))
        if meta.demo_model:
            model.append(" demo", style="yellow")
        table.add_row(
            str(number),
            Text(clean(meta.repo)),
            short_sha(meta.sha),
            _local_time(meta),
            str(meta.counts.get("Critical", 0)),
            str(meta.counts.get("High", 0)),
            str(meta.n_files),
            model,
        )
    return table


def analysis_json(loaded: LoadedAnalysis, *, top: int) -> dict[str, Any]:
    """The ``tdp analyze --format json`` document (schema ``entropiq.analysis.v1``); ``top=0`` keeps all."""
    meta, result = loaded.meta, loaded.result
    counts = band_counts(result.files)
    files = result.files[:top] if top > 0 else result.files
    rows: list[dict[str, Any]] = []
    for rank, f in enumerate(files, start=1):
        extra = result.extras.get(f.path)
        rows.append(
            {
                "rank": rank,
                "path": f.path,
                "priority": f.priority,
                "band": f.band,
                "risk": f.risk,
                "risk_low": f.risk_low,
                "risk_high": f.risk_high,
                "cost_hours": extra.cost.hours if extra else None,
                "cost_low": extra.cost.low if extra else None,
                "cost_high": extra.cost.high if extra else None,
                "debt_types": list(f.debt_types),
                "top_reason": top_reason(f),
                "reasons": [r.text for r in f.reasons],
            }
        )
    return {
        "schema": JSON_SCHEMA,
        "repo": meta.repo,
        "source": meta.source,
        "kind": meta.kind,
        "ref": meta.ref,
        "sha": result.sha,
        "analyzed_at": result.analyzed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "model_version": result.model_version,
        "demo_model": meta.demo_model,
        "extractor_version": result.extractor_version,
        "summary": {
            "files": len(result.files),
            "critical": counts["Critical"],
            "high": counts["High"],
            "medium": counts["Medium"],
            "low": counts["Low"],
            "critical_hours": round(critical_hours(result), 3),
        },
        "files": rows,
    }


def _md_cell(text: str) -> str:
    return clean(" ".join(text.split())).replace("|", "\\|")


def analysis_markdown(loaded: LoadedAnalysis, *, top: int) -> str:
    """The ``tdp analyze --format md`` report: a heading, a summary sentence and the ranked table."""
    meta, result = loaded.meta, loaded.result
    counts = band_counts(result.files)
    model = f"{result.model_version} (demo)" if meta.demo_model else result.model_version
    lines = [f"## Entropiq: {_md_cell(meta.repo)} at {short_sha(result.sha)}", ""]
    if not result.files:
        lines.append(f"No Python source files to rank at this commit (model {_md_cell(model)}).")
        return "\n".join(lines) + "\n"
    lines.append(
        f"{len(result.files)} files ranked with model {_md_cell(model)}: "
        f"{counts['Critical']} Critical, {counts['High']} High, {counts['Medium']} Medium, "
        f"{counts['Low']} Low. Estimated Critical debt: {format_hours(critical_hours(result))}."
    )
    lines += ["", "| # | File | Priority | Risk | Cost | Top reason |", "|---:|---|---|---:|---|---|"]
    shown = result.files[:top] if top > 0 else result.files
    for rank, f in enumerate(shown, start=1):
        extra = result.extras.get(f.path)
        cost = format_hour_range(extra.cost.low, extra.cost.high) if extra else ""
        lines.append(
            f"| {rank} | `{_md_cell(f.path)}` | {f.band} {f.priority} | {format_risk(f.risk)} | {cost} "
            f"| {_md_cell(top_reason(f))} |"
        )
    if len(shown) < len(result.files):
        lines += ["", f"Showing {len(shown)} of {len(result.files)} files."]
    return "\n".join(lines) + "\n"
