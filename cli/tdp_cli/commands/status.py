"""``/status``: the loaded repository, the model and the settings at a glance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.registry import Command, CommandRegistry
from tdp_cli.session import NO_ANALYSIS
from tdp_cli.theme import band_text, clean, format_hours, make_panel, short_sha

if TYPE_CHECKING:
    from tdp_cli.session import Session


def status_handler(session: Session, args: Sequence[str]) -> None:
    """``/status``."""
    parse_args(args, max_positionals=0)
    glyphs = session.glyphs
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")
    current = session.current
    if current is None:
        grid.add_row("Repository", Text(NO_ANALYSIS, style="yellow"))
    else:
        meta = current.meta
        grid.add_row("Repository", Text(f"{clean(meta.repo)} ({clean(meta.source)})"))
        grid.add_row("Commit", Text(f"{short_sha(meta.sha)} (ref {clean(meta.ref)})"))
        grid.add_row("Analysed", Text(meta.analyzed_at.astimezone().strftime("%Y-%m-%d %H:%M")))
        counts = Text(f"{meta.n_files} files   ")
        for band in ("Critical", "High", "Medium", "Low"):
            counts.append_text(band_text(band, glyphs=glyphs))
            counts.append(f" {meta.counts.get(band, 0)}   ")
        grid.add_row("Files", counts)
        grid.add_row("Critical debt", Text(format_hours(meta.critical_hours)))
        demo = " (demo)" if meta.demo_model else ""
        grid.add_row("Analysed with", Text(f"model {clean(meta.model_version)}{demo}"))
    model = session.model
    model_text = Text(f"{model.version or 'not built yet'} ")
    model_text.append(model.kind_label, style="yellow" if model.demo else "green")
    model_text.append(f" ({clean(model.reason)})", style="dim")
    grid.add_row("Session model", model_text)
    grid.add_row("Entropiq folder", Text(clean(str(session.home.root))))
    prefs = session.prefs
    grid.add_row("Settings", Text(f"budget_hours {prefs.budget_hours:g}, top {prefs.top}, jobs {prefs.jobs}"))
    session.console.print(make_panel(grid, "Status"))


def register(registry: CommandRegistry) -> None:
    """Register ``/status``."""
    registry.add(
        Command(
            name="status",
            group="Session",
            help="Show the loaded repository, model and settings",
            usage="/status",
            handler=status_handler,
        )
    )
