"""Colours, glyphs, number formats and table styles shared by every view of the terminal app.

Rules every view follows: a band colour is always paired with the band word (colour is never the only
signal), risk is a percentage, cost is an hour range, and strings that come from an analysed repository
(paths, reasons, names) are passed through :func:`clean` and shown as :class:`rich.text.Text`, never
interpolated into rich markup.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

VIOLET: Final = "#8b5cf6"
CYAN: Final = "#06b6d4"
DIM: Final = "#9ca3af"

BAND_ORDER: Final = ("Critical", "High", "Medium", "Low")
"""Band names from the most to the least urgent (the names of ``tdp_core.contracts.BANDS``)."""

BAND_STYLES: Final[dict[str, str]] = {
    "Critical": "bold #ef4444",
    "High": "bold #f97316",
    "Medium": "#eab308",
    "Low": "#22c55e",
}


@dataclass(frozen=True, slots=True)
class Glyphs:
    """Small symbols used in the views, with an ASCII set for terminals that cannot show Unicode."""

    bullet: str
    check: str
    cross: str
    chevron: str
    sep: str
    up: str
    down: str


UNICODE_GLYPHS: Final = Glyphs("●", "✓", "✗", "\u276f", "·", "↑", "↓")
ASCII_GLYPHS: Final = Glyphs("*", "+", "x", ">", "|", "^", "v")


def can_encode(text: str, encoding: str | None) -> bool:
    """True when ``text`` can be written in ``encoding`` (None or an unknown codec means UTF-8)."""
    try:
        text.encode(encoding or "utf-8")
    except UnicodeEncodeError:
        return False
    except LookupError:
        return True
    return True


def glyphs_for(encoding: str | None) -> Glyphs:
    """The Unicode glyphs when ``encoding`` can show them, else the ASCII ones."""
    sample = "".join(
        (
            UNICODE_GLYPHS.bullet,
            UNICODE_GLYPHS.check,
            UNICODE_GLYPHS.cross,
            UNICODE_GLYPHS.chevron,
            UNICODE_GLYPHS.sep,
            UNICODE_GLYPHS.up,
            UNICODE_GLYPHS.down,
        )
    )
    return UNICODE_GLYPHS if can_encode(sample, encoding) else ASCII_GLYPHS


def console_glyphs(console: Console) -> Glyphs:
    """The glyph set that ``console``'s encoding can show."""
    return glyphs_for(console.encoding)


def clean(text: str) -> str:
    """Replace non-printable characters (control codes, ESC, newlines, tabs) with ``?``.

    Use it on every repository-derived string before display, so a file name or commit text cannot inject
    terminal escape sequences.
    """
    if text.isprintable():
        return text
    return "".join(c if c.isprintable() else "?" for c in text)


def band_style(band: str) -> str:
    """The rich style of ``band`` (an empty style for unknown names)."""
    return BAND_STYLES.get(band, "")


def band_text(band: str, priority: int | None = None, *, glyphs: Glyphs = UNICODE_GLYPHS) -> Text:
    """A coloured ``"● Critical 97"`` label: the colour always comes with the band word."""
    label = f"{glyphs.bullet} {band}" if priority is None else f"{glyphs.bullet} {band} {priority}"
    return Text(label, style=band_style(band))


def format_risk(risk: float) -> str:
    """Risk probability as a whole percentage: ``"0%"``, ``"<1%"`` when it rounds to zero, else ``"42%"``."""
    if risk <= 0:
        return "0%"
    percent = round(100 * risk)
    return "<1%" if percent == 0 else f"{percent}%"


def format_risk_range(risk: float, low: float, high: float) -> str:
    """Risk with its range, such as ``"42% (35-49%)"``."""
    return f"{format_risk(risk)} ({format_risk(low).rstrip('%')}-{format_risk(high)})"


def _hours(value: float) -> str:
    return "0" if value == 0 else f"{value:.1f}"


def format_hours(hours: float) -> str:
    """Hours with one decimal (``"12.4 h"``), ``"0 h"`` for zero."""
    return f"{_hours(hours)} h"


def format_hour_range(low: float, high: float) -> str:
    """An hour range such as ``"3.2-4.8 h"`` (uses a hyphen, never a dash)."""
    return f"{_hours(low)}-{_hours(high)} h"


def short_sha(sha: str) -> str:
    """The first seven characters of a commit sha."""
    return sha[:7]


def make_table(title: str | None = None, *, caption: str | None = None) -> Table:
    """A table in the shared style (simple header rule, bold headers, left-aligned title)."""
    return Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAD,
        header_style="bold",
        title_justify="left",
        title_style="bold",
        caption_justify="left",
        show_edge=False,
        pad_edge=False,
    )


def make_panel(
    body: RenderableType, title: str | Text | None = None, *, subtitle: str | None = None
) -> Panel:
    """A violet-bordered panel with a left-aligned title (pass repository names as ``Text``)."""
    return Panel(
        body,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="left",
        border_style=VIOLET,
        padding=(0, 1),
    )


def notice(console: Console, text: str) -> None:
    """Print a yellow ``! text`` notice (``text`` is plain, never markup)."""
    console.print(Text(f"! {text}", style="yellow"))


def error(console: Console, text: str, hint: str | None = None) -> None:
    """Print a red ``Error: text`` line and an optional dim hint (both plain text)."""
    console.print(Text(f"Error: {text}", style="bold red"))
    if hint:
        console.print(Text(hint, style="dim"))


def success(console: Console, text: str) -> None:
    """Print a green check line (plain text)."""
    console.print(Text(f"{console_glyphs(console).check} {text}", style="green"))


@contextmanager
def activity(console: Console, text: str) -> Iterator[None]:
    """Show a spinner with ``text`` while the block runs (terminals), or print ``text`` once otherwise."""
    if not console.is_terminal:
        console.print(Text(f"{text}...", style="dim"))
        yield
        return
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(style=CYAN),
        TextColumn("{task.description}", markup=False),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(text, total=None)
        yield
