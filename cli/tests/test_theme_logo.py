from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

import tdp_cli
from tdp_cli.home import EntropiqHome
from tdp_cli.logo import (
    ASCII_LOGO,
    BLOCK_LOGO,
    COMPACT_LOGO,
    TAGLINE,
    gradient_text,
    lerp_color,
    logo_rows,
    print_header,
    print_welcome,
)
from tdp_cli.model import dev_choice
from tdp_cli.theme import (
    ASCII_GLYPHS,
    BAND_ORDER,
    UNICODE_GLYPHS,
    activity,
    band_text,
    can_encode,
    clean,
    error,
    format_hour_range,
    format_hours,
    format_risk,
    format_risk_range,
    glyphs_for,
    make_panel,
    make_table,
    notice,
    short_sha,
    success,
)
from tdp_core.contracts import BANDS


def _console(width: int = 120, **kwargs: object) -> Console:
    return Console(file=io.StringIO(), width=width, color_system=None, **kwargs)  # type: ignore[arg-type]


def test_band_names_match_contracts() -> None:
    assert tuple(name for name, _ in BANDS) == BAND_ORDER


def test_band_text_pairs_colour_with_word() -> None:
    text = band_text("Critical", 97)
    assert text.plain == f"{UNICODE_GLYPHS.bullet} Critical 97"
    assert "#ef4444" in str(text.style)
    assert band_text("Low", glyphs=ASCII_GLYPHS).plain == "* Low"


@pytest.mark.parametrize(
    ("risk", "expected"), [(0.0, "0%"), (0.001, "<1%"), (0.005, "<1%"), (0.42, "42%"), (1.0, "100%")]
)
def test_format_risk(risk: float, expected: str) -> None:
    assert format_risk(risk) == expected


def test_number_formats() -> None:
    assert format_risk_range(0.42, 0.35, 0.49) == "42% (35-49%)"
    assert format_hours(0) == "0 h"
    assert format_hours(12.44) == "12.4 h"
    assert format_hour_range(3.2, 4.8) == "3.2-4.8 h"
    assert format_hour_range(0, 0) == "0-0 h"
    assert short_sha("0123456789abcdef") == "0123456"


def test_glyphs_follow_encoding() -> None:
    assert glyphs_for("cp1252") is ASCII_GLYPHS
    assert glyphs_for("utf-8") is UNICODE_GLYPHS
    assert glyphs_for(None) is UNICODE_GLYPHS
    assert can_encode("abc", "no-such-codec")


def test_clean_removes_control_characters() -> None:
    assert clean("ok/path.py") == "ok/path.py"
    assert clean("evil\x1b[31mred\nx\ty") == "evil?[31mred?x?y"


def test_markup_like_strings_print_literally() -> None:
    console = _console()
    error(console, "[bold red]boom[/] path", hint="[link]x[/link]")
    notice(console, "[i]n[/i]")
    success(console, "[b]done[/b]")
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Error: [bold red]boom[/] path" in out
    assert "[link]x[/link]" in out
    assert "! [i]n[/i]" in out
    assert "[b]done[/b]" in out


def test_tables_and_panels() -> None:
    console = _console()
    table = make_table("Title")
    table.add_column("Column A")
    table.add_row("1")
    console.print(table)
    console.print(make_panel("body", "Head"))
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Title" in out and "Head" in out and "body" in out


def test_activity_prints_once_without_a_terminal() -> None:
    console = _console()
    with activity(console, "Working"):
        pass
    assert console.file.getvalue().strip() == "Working..."  # type: ignore[attr-defined]
    live = Console(file=io.StringIO(), force_terminal=True, width=80)
    with activity(live, "Working"):
        pass


def test_block_logo_shape() -> None:
    assert len(BLOCK_LOGO) == 6
    assert {len(row) for row in BLOCK_LOGO} == {64}
    assert set("".join(BLOCK_LOGO)) <= set("█╗╔═╝║╚▄▀ ")


def test_ascii_logo_shape() -> None:
    assert len(ASCII_LOGO) == 5
    assert {len(row) for row in ASCII_LOGO} == {61}
    assert all(row.isascii() for row in ASCII_LOGO)


def test_gradient_runs_from_violet_to_cyan() -> None:
    assert lerp_color("#000000", "#ffffff", 0.5) == "#808080"
    assert lerp_color("#8B5CF6", "#06B6D4", -1) == "#8b5cf6"
    assert lerp_color("#8b5cf6", "#06b6d4", 2) == "#06b6d4"
    text = gradient_text(BLOCK_LOGO)
    styles = [str(span.style) for span in text.spans]
    assert "#8b5cf6" in styles[0]
    assert any("#06b6d4" in s for s in styles)
    assert all("#06b6d4" not in s for s in styles[:20])
    assert text.plain.split("\n") == list(BLOCK_LOGO)


def test_logo_rows_fall_back() -> None:
    assert logo_rows("cp1252", 120) == ASCII_LOGO
    assert logo_rows("utf-8", 120) == BLOCK_LOGO
    assert logo_rows("utf-8", 40) == COMPACT_LOGO
    assert logo_rows("ascii", 62) == ASCII_LOGO


def test_print_welcome(tmp_path: Path) -> None:
    console = _console()
    model = dev_choice(EntropiqHome(tmp_path), "--dev-model")
    print_welcome(console, model=model)
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert BLOCK_LOGO[0].rstrip() in out
    assert TAGLINE in out
    assert f"entropiq {tdp_cli.__version__}" in out
    assert "not built yet" in out and "demo model" in out
    for tip in ("/analyze <path or GitHub URL>", "/help", "Ctrl+D"):
        assert tip in out
    header = _console()
    print_header(header, model=model)
    assert "ENTROPIQ" in header.file.getvalue()  # type: ignore[attr-defined]


def test_print_welcome_ascii_console(tmp_path: Path) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    console = Console(file=stream, width=120, color_system=None)
    print_welcome(console, model=dev_choice(EntropiqHome(tmp_path), "--dev-model"))
    stream.flush()
    out = raw.getvalue().decode("cp1252")
    assert ASCII_LOGO[1] in out
    assert "█" not in out


def test_no_dashes_in_sources() -> None:
    root = Path(tdp_cli.__file__).parent
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2013) not in text and chr(0x2014) not in text, path
