"""The startup screen: the gradient ENTROPIQ logo, the tagline, the version line and the welcome tips."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from tdp_cli import __version__
from tdp_cli.theme import CYAN, VIOLET, can_encode, console_glyphs, make_panel

if TYPE_CHECKING:
    from tdp_cli.model import ModelChoice

BLOCK_LOGO: Final[tuple[str, ...]] = (
    "███████╗███╗   ██╗████████╗██████╗  ██████╗ ██████╗ ██╗ ██████╗ ",
    "██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗██║██╔═══██╗",
    "█████╗  ██╔██╗ ██║   ██║   ██████╔╝██║   ██║██████╔╝██║██║   ██║",
    "██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██║   ██║██╔═══╝ ██║██║▄▄ ██║",
    "███████╗██║ ╚████║   ██║   ██║  ██║╚██████╔╝██║     ██║╚██████╔╝",
    "╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝ ╚══▀▀═╝ ",
)
"""ENTROPIQ in figlet "ANSI Shadow" letters (6 rows of 64 columns)."""

_ASCII_LETTERS: Final[dict[str, tuple[str, ...]]] = {
    "E": (" _____ ", "| ____|", "|  _|  ", "| |___ ", "|_____|"),
    "N": (" _   _ ", "| \\ | |", "|  \\| |", "| |\\  |", "|_| \\_|"),
    "T": (" _____ ", "|_   _|", "  | |  ", "  | |  ", "  |_|  "),
    "R": (" ____  ", "|  _ \\ ", "| |_) |", "|  _ < ", "|_| \\_\\"),
    "O": ("  ___  ", " / _ \\ ", "| | | |", "| |_| |", " \\___/ "),
    "P": (" ____  ", "|  _ \\ ", "| |_) |", "|  __/ ", "|_|    "),
    "I": (" ___ ", "|_ _|", " | | ", " | | ", "|___|"),
    "Q": ("  ___  ", " / _ \\ ", "| | | |", "| |_| |", " \\__\\_\\"),
}

ASCII_LOGO: Final[tuple[str, ...]] = tuple(
    " ".join(_ASCII_LETTERS[letter][row] for letter in "ENTROPIQ") for row in range(5)
)
"""ENTROPIQ in figlet "standard" letters (pure ASCII, 5 rows of 61 columns)."""

COMPACT_LOGO: Final = ("ENTROPIQ",)

TAGLINE: Final = "Rank technical debt by future maintenance pain"

_TIPS: Final = (
    ("/analyze <path or GitHub URL>", "Rank the files of a repository"),
    ("/help", "List every command"),
    ("Ctrl+D", "Exit"),
)


def lerp_color(start: str, end: str, t: float) -> str:
    """The colour a fraction ``t`` (clamped to 0..1) of the way from ``start`` to ``end`` (``#rrggbb``)."""
    t = min(max(t, 0.0), 1.0)
    a = [int(start.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    b = [int(end.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(x + (y - x) * t) for x, y in zip(a, b, strict=True)]
    return "#" + "".join(f"{v:02x}" for v in mixed)


def gradient_text(rows: Sequence[str], start: str = VIOLET, end: str = CYAN) -> Text:
    """``rows`` joined by newlines, each non-space character coloured by its column (left to right)."""
    width = max((len(r) for r in rows), default=0)
    text = Text(no_wrap=True, overflow="crop")
    for index, row in enumerate(rows):
        if index:
            text.append("\n")
        for x, char in enumerate(row):
            if char == " ":
                text.append(char)
            else:
                text.append(char, style=f"bold {lerp_color(start, end, x / max(width - 1, 1))}")
    return text


def logo_rows(encoding: str | None, width: int) -> tuple[str, ...]:
    """The logo that fits a terminal ``width`` columns wide whose output uses ``encoding``.

    Block letters when they fit (64 columns) and can be encoded; the ASCII letters otherwise (61 columns);
    the one-word compact logo on narrower terminals.
    """
    block_width = max(len(r) for r in BLOCK_LOGO)
    ascii_width = max(len(r) for r in ASCII_LOGO)
    if width >= block_width and can_encode("".join(BLOCK_LOGO), encoding):
        return BLOCK_LOGO
    if width >= ascii_width:
        return ASCII_LOGO
    return COMPACT_LOGO


def version_line(console: Console, model: ModelChoice, app_version: str = __version__) -> Text:
    """``entropiq 0.1.0 · model dev-... · demo model`` with the model kind coloured."""
    sep = console_glyphs(console).sep
    kind_style = "green" if model.kind_label == "production model" else "yellow" if model.demo else "cyan"
    line = Text(f"entropiq {app_version} {sep} model ", style="dim")
    line.append(model.version or "not built yet", style="bold")
    line.append(f" {sep} ", style="dim")
    line.append(model.kind_label, style=kind_style)
    return line


def print_welcome(console: Console, *, model: ModelChoice, app_version: str = __version__) -> None:
    """Print the startup screen: logo, tagline, version line and a panel with three quick tips."""
    rows = logo_rows(console.encoding, console.width)
    console.print()
    console.print(gradient_text(rows), no_wrap=True, crop=True)
    console.print(Text(TAGLINE, style="italic dim"))
    console.print(version_line(console, model, app_version))
    console.print()
    tips = Table.grid(padding=(0, 2))
    tips.add_column(style=f"bold {CYAN}", no_wrap=True)
    tips.add_column()
    for key, text in _TIPS:
        tips.add_row(key, text)
    console.print(make_panel(Group(tips), "Welcome"))


def print_header(console: Console, *, model: ModelChoice, app_version: str = __version__) -> None:
    """Print a one-line gradient header and the version line (after ``/clear``)."""
    console.print(gradient_text(COMPACT_LOGO), no_wrap=True, crop=True)
    console.print(version_line(console, model, app_version))
