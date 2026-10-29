"""``/trend`` (plus ``tdp trend``): how the Critical files and hours of one repository changed over time.

It reads the saved analyses of the repository from the local cache (``/history`` lists them) and orders them
by commit date with one ``git rev-list`` call, or by analysis time when the repository is not available.
``tdp trend`` never runs an analysis and never touches the network.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group, RenderableType
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.commands._views import (
    RED,
    TEAL,
    UNICODE_VIEW_GLYPHS,
    ViewGlyphs,
    sparkline,
    utc_iso,
    view_glyphs,
)
from tdp_cli.options import TargetArgument, cli_guard, cli_session
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.theme import (
    BAND_STYLES,
    CYAN,
    UNICODE_GLYPHS,
    Glyphs,
    clean,
    format_hours,
    make_table,
    short_sha,
)

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import AnalysisMeta

TREND_SCHEMA: Final = "entropiq.trend.v1"
_SHA: Final = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True, slots=True)
class TrendPoint:
    """One saved analysis and the commit time of its commit (None when unknown)."""

    meta: AnalysisMeta
    committed_at: datetime | None


def commit_times(repo_dir: Path, shas: Sequence[str]) -> dict[str, datetime] | None:
    """The committer time (UTC) of each commit in ``shas``, from one ``git rev-list`` call.

    Returns None when the folder is missing, a sha is not 40 to 64 lower-case hex digits, git fails (an
    unknown commit makes it fail) or its output does not cover every sha. No shas give an empty dict.
    """
    from tdp_core.mining.git import GitError, run_git

    wanted = list(dict.fromkeys(shas))
    if not wanted:
        return {}
    if not repo_dir.is_dir() or not all(_SHA.fullmatch(sha) for sha in wanted):
        return None
    try:
        out = run_git(["rev-list", "--no-walk=unsorted", "--timestamp", *wanted], repo_dir, timeout=60)
        times: dict[str, datetime] = {}
        for line in out.decode("ascii", "replace").splitlines():
            stamp, _, sha = line.strip().partition(" ")
            if sha:
                times[sha] = datetime.fromtimestamp(int(stamp), UTC)
    except (GitError, OSError, ValueError, OverflowError):
        return None
    if any(sha not in times for sha in wanted):
        return None
    return {sha: times[sha] for sha in wanted}


def trend_points(
    entries: Sequence[AnalysisMeta],
    slug: str,
    *,
    repo_dir: Path | None,
    extra: AnalysisMeta | None = None,
) -> list[TrendPoint]:
    """The saved analyses of repository ``slug``, oldest first.

    ``extra`` (the loaded analysis) is added when its commit is not saved. Points are ordered by commit
    time when ``repo_dir`` gives a time for every commit, else by analysis time; ties go by sha.
    """
    metas = [m for m in entries if m.slug == slug]
    known = {m.sha for m in metas}
    if extra is not None and extra.slug == slug and extra.sha not in known:
        metas.append(extra)
    times = commit_times(repo_dir, [m.sha for m in metas]) if repo_dir is not None and metas else None
    if times is not None:
        metas.sort(key=lambda m: (times[m.sha], m.sha))
        return [TrendPoint(m, times[m.sha]) for m in metas]
    metas.sort(key=lambda m: (m.analyzed_at, m.sha))
    return [TrendPoint(m, None) for m in metas]


def _by_commit(points: Sequence[TrendPoint]) -> bool:
    return bool(points) and all(p.committed_at is not None for p in points)


def _critical(meta: AnalysisMeta) -> int:
    return meta.counts.get("Critical", 0)


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def _delta(label: str, before: float, after: float, fmt: Callable[[float], str], glyphs: Glyphs) -> Text:
    line = Text(f"{label} {fmt(before)} to {fmt(after)} ")
    if after > before:
        line.append(f"({glyphs.up} {fmt(after - before)})", style=RED)
    elif after < before:
        line.append(f"({glyphs.down} {fmt(before - after)})", style=TEAL)
    else:
        line.append("(no change)", style="dim")
    return line


def _count(value: float) -> str:
    return str(int(value))


def _hours(value: float) -> str:
    return format_hours(round(value, 1))


def _series(label: str, values: Sequence[float], fmt: Callable[[float], str], view: ViewGlyphs) -> Text:
    line = Text(f"{label}  ")
    line.append(sparkline(values, view), style=CYAN)
    line.append(
        f"  first {fmt(values[0])}, latest {fmt(values[-1])}, highest {fmt(max(values))}", style="dim"
    )
    return line


def trend_view(
    points: Sequence[TrendPoint],
    *,
    repo: str,
    current_sha: str | None = None,
    glyphs: Glyphs = UNICODE_GLYPHS,
    view: ViewGlyphs = UNICODE_VIEW_GLYPHS,
) -> Group:
    """The ``/trend`` screen: a table of the saved analyses, two sparklines and the latest change.

    Times are shown in the local time zone; ``current_sha`` marks the loaded analysis. A notice warns when
    the analyses used different models, since their numbers are then not directly comparable.
    """
    title = Text(f"Saved analyses of {clean(repo)}", style="bold")
    if not points:
        return Group(_dim(f"No saved analyses of {clean(repo)} yet. Run /analyze (or tdp analyze) first."))
    by_commit = _by_commit(points)
    title.append(
        " (oldest first, by commit date)" if by_commit else " (oldest first, by analysis time)", style="dim"
    )
    table = make_table()
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Commit", no_wrap=True)
    table.add_column("Committed", no_wrap=True)
    table.add_column("Analysed", no_wrap=True)
    table.add_column("Critical", justify="right", no_wrap=True, style=BAND_STYLES["Critical"])
    table.add_column("High", justify="right", no_wrap=True, style=BAND_STYLES["High"])
    table.add_column("Files", justify="right", no_wrap=True)
    table.add_column("Critical hours", justify="right", no_wrap=True)
    table.add_column("Model", overflow="fold")
    for number, point in enumerate(points, start=1):
        meta = point.meta
        commit = Text(short_sha(clean(meta.sha)))
        if current_sha is not None and meta.sha == current_sha:
            commit.append(" loaded", style="dim")
        model = Text(clean(meta.model_version))
        if meta.demo_model:
            model.append(" demo", style="yellow")
        table.add_row(
            str(number),
            commit,
            point.committed_at.astimezone().strftime("%Y-%m-%d") if point.committed_at else "",
            meta.analyzed_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            str(_critical(meta)),
            str(meta.counts.get("High", 0)),
            str(meta.n_files),
            format_hours(meta.critical_hours),
            model,
        )
    parts: list[RenderableType] = [title, table]
    if len(points) == 1:
        parts.append(
            _dim(
                "Only one saved analysis of this repository; analyse other commits "
                "(for example /analyze --ref HEAD~50) to see a trend."
            )
        )
    else:
        counts = [float(_critical(p.meta)) for p in points]
        hours = [p.meta.critical_hours for p in points]
        parts.append(_series("Critical files", counts, _count, view))
        parts.append(_series("Critical hours", hours, _hours, view))
        since = Text("Since the previous analysis: ")
        since.append_text(_delta("Critical files", counts[-2], counts[-1], _count, glyphs))
        since.append(", ")
        since.append_text(_delta("Critical hours", round(hours[-2], 1), round(hours[-1], 1), _hours, glyphs))
        parts.append(since)
    models = {p.meta.model_version for p in points}
    if len(models) > 1:
        parts.append(
            Text(
                f"These analyses used {len(models)} different models, so their numbers are not directly "
                "comparable.",
                style="yellow",
            )
        )
    return Group(*parts)


def trend_json(points: Sequence[TrendPoint], *, repo: str, slug: str) -> dict[str, Any]:
    """The ``tdp trend --format json`` document (schema ``entropiq.trend.v1``), points oldest first.

    ``order`` is ``commit_time`` or ``analysis_time``; times are ISO 8601 UTC with ``Z`` (``committed_at``
    is null when unknown); ``critical_hours`` is in hours.
    """
    return {
        "schema": TREND_SCHEMA,
        "repo": repo,
        "slug": slug,
        "order": "commit_time" if _by_commit(points) else "analysis_time",
        "points": [
            {
                "number": number,
                "sha": p.meta.sha,
                "committed_at": utc_iso(p.committed_at) if p.committed_at else None,
                "analyzed_at": utc_iso(p.meta.analyzed_at),
                "model_version": p.meta.model_version,
                "demo_model": p.meta.demo_model,
                "files": p.meta.n_files,
                "critical": _critical(p.meta),
                "high": p.meta.counts.get("High", 0),
                "medium": p.meta.counts.get("Medium", 0),
                "low": p.meta.counts.get("Low", 0),
                "critical_hours": p.meta.critical_hours,
            }
            for number, p in enumerate(points, start=1)
        ],
    }


def trend_handler(session: Session, args: Sequence[str]) -> None:
    """``/trend``: the saved analyses of the loaded repository with their Critical counts and hours."""
    parse_args(args, max_positionals=0)
    meta = session.require_analysis().meta
    points = trend_points(session.store.entries(), meta.slug, repo_dir=Path(meta.repo_dir), extra=meta)
    session.console.print(
        trend_view(
            points,
            repo=meta.repo,
            current_sha=meta.sha,
            glyphs=session.glyphs,
            view=view_glyphs(session.console),
        )
    )


def trend_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
) -> None:
    """Show how Critical files and hours changed across saved analyses (reads the local cache only)."""
    session = cli_session(ctx)
    with cli_guard(session.err_console):
        from tdp_cli.repos import looks_like_url, resolve_local, resolve_url

        raw = target.strip() or "."
        repo = resolve_url(raw, None, session.home)[0] if looks_like_url(raw) else resolve_local(raw)
        points = trend_points(session.store.entries(), repo.slug, repo_dir=repo.repo_dir)
        if output_format == "json":
            document = trend_json(points, repo=repo.display, slug=repo.slug)
            sys.stdout.write(json.dumps(document, indent=2) + "\n")
            return
        session.console.print(
            trend_view(points, repo=repo.display, glyphs=session.glyphs, view=view_glyphs(session.console))
        )


def register(registry: CommandRegistry) -> None:
    """Register ``/trend`` and ``tdp trend``."""
    registry.add(
        Command(
            name="trend",
            group="Analysis",
            help="Show how Critical files and hours changed across saved analyses",
            usage="/trend",
            details=(
                "Lists the saved analyses of the loaded repository, oldest commit first, with their Critical "
                "and High counts and the estimated hours to fix the Critical files, draws both as "
                "sparklines and says what changed since the previous analysis. Analyse older commits with "
                "/analyze --ref (for example HEAD~50) to build up a trend; analyses made with different "
                "models are flagged because their numbers are not directly comparable."
            ),
            handler=trend_handler,
            needs_analysis=True,
            cli=trend_cli,
        )
    )
