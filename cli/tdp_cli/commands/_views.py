"""Building blocks shared by the exploration views: file selection, the ranked table, glyphs and text charts.

A file's *rank* is its 1-based position in ``result.files`` (priority order). Every view shows and accepts
that number, so ``/file 3`` names the same file whatever filter or sort a table used.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from tdp_cli.errors import CliError, UsageError
from tdp_cli.output import top_reason
from tdp_cli.registry import Suggestion
from tdp_cli.theme import (
    BAND_ORDER,
    UNICODE_GLYPHS,
    VIOLET,
    Glyphs,
    band_text,
    can_encode,
    clean,
    format_hour_range,
    format_risk,
    make_table,
)

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult, FileExtra, FileScore

TEAL: Final = "#14b8a6"
RED: Final = "#ef4444"
SORT_KEYS: Final = ("priority", "risk", "cost")
NO_FILES: Final = "This analysis has no ranked files."
MAX_FOLDER_SUGGESTIONS: Final = 50

DEBT_REMEDIES: Final[dict[str, str]] = {
    "complexity": "Split the longest and most branched functions into smaller ones and flatten deep nesting.",
    "duplication": "Move the repeated code into one shared function or module and call it from each place.",
    "test": "Add tests that cover the main paths of the public functions before changing them.",
    "architecture": (
        "Break import cycles and hidden couplings by moving shared code into a lower-level module."
    ),
    "knowledge": "Spread knowledge: review changes across the team, pair on fixes and document the module.",
    "admitted": "Resolve the TODO, FIXME and HACK comments, or turn each one into a tracked issue.",
    "defect_hotspot": (
        "Review the recent bug fixes here and add regression tests for the failures they fixed."
    ),
}
"""The usual remedy of each debt type, in ``tdp_core.contracts.DEBT_TYPES`` order."""

_BAND_WORDS: Final = ", ".join(b.lower() for b in BAND_ORDER)
_MANY_SLASHES: Final = re.compile(r"/{2,}")
_RANK_KEY: Final = re.compile(r"^#?(\d+)$")


@dataclass(frozen=True, slots=True)
class ViewGlyphs:
    """Characters of the text charts, with an ASCII set for terminals that cannot show block characters."""

    block: str
    axis: str
    spark: str
    arrow: str
    both: str
    more: str


UNICODE_VIEW_GLYPHS: Final = ViewGlyphs("█", "│", "▁▂▃▄▅▆▇█", "→", "⇄", "…")
ASCII_VIEW_GLYPHS: Final = ViewGlyphs("#", "|", "_.-:=+*#", "->", "<->", "...")


def view_glyphs_for(encoding: str | None) -> ViewGlyphs:
    """The Unicode chart glyphs when ``encoding`` can show all of them, else the ASCII ones."""
    g = UNICODE_VIEW_GLYPHS
    sample = g.block + g.axis + g.spark + g.arrow + g.both + g.more
    return UNICODE_VIEW_GLYPHS if can_encode(sample, encoding) else ASCII_VIEW_GLYPHS


def view_glyphs(console: Console) -> ViewGlyphs:
    """The chart glyphs ``console``'s encoding can show."""
    return view_glyphs_for(console.encoding)


@dataclass(frozen=True, slots=True)
class RankedFile:
    """One ranked file: its priority rank (1-based position in ``result.files``), score and details."""

    rank: int
    score: FileScore
    extra: FileExtra | None

    @property
    def path(self) -> str:
        """The repo-relative POSIX path."""
        return self.score.path


@dataclass(frozen=True, slots=True)
class FileQuery:
    """Filters and order of a file list. Empty fields do not filter; ``top`` 0 means every file."""

    bands: tuple[str, ...] = ()
    debt_types: tuple[str, ...] = ()
    path_prefix: str = ""
    sort: str = "priority"
    top: int = 0

    def matches(self, item: RankedFile) -> bool:
        """True when ``item`` passes every filter (band, any of the debt types, path prefix)."""
        score = item.score
        if self.bands and score.band not in self.bands:
            return False
        if self.debt_types and not any(t in self.debt_types for t in score.debt_types):
            return False
        return item.path.startswith(self.path_prefix)

    def describe(self) -> str:
        """A plain heading such as ``"Ranked files (band Critical/High, sorted by risk)"``."""
        parts: list[str] = []
        if self.bands:
            parts.append(f"band {'/'.join(self.bands)}")
        if self.debt_types:
            parts.append(f"debt {'/'.join(self.debt_types)}")
        if self.path_prefix:
            parts.append(f"path {clean(self.path_prefix)}")
        if self.sort != "priority":
            parts.append(f"sorted by {self.sort}")
        return f"Ranked files ({', '.join(parts)})" if parts else "Ranked files"


def normalise_path(text: str) -> str:
    """A typed path as a repo-relative POSIX path: ``.\\app\\util\\`` becomes ``app/util/``.

    Backslashes become slashes, repeated slashes collapse, and leading ``./`` and ``/`` are dropped; a
    trailing slash is kept.
    """
    path = _MANY_SLASHES.sub("/", text.strip().replace("\\", "/"))
    while path.startswith(("./", "/")):
        path = path[2:] if path.startswith("./") else path[1:]
    return path


def _items(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_bands(text: str) -> tuple[str, ...]:
    """A comma list of band names (any case) in Critical to Low order. Raises :class:`UsageError`."""
    by_word = {b.lower(): b for b in BAND_ORDER}
    chosen: set[str] = set()
    for part in _items(text):
        band = by_word.get(part.lower())
        if band is None:
            raise UsageError(f"unknown band {clean(part)!r}; use {_BAND_WORDS} (or a comma list)")
        chosen.add(band)
    if not chosen:
        raise UsageError(f"--band needs a band: {_BAND_WORDS}")
    return tuple(b for b in BAND_ORDER if b in chosen)


def parse_debt_types(text: str) -> tuple[str, ...]:
    """A comma list of debt types (``defect-hotspot`` and any case work). Raises :class:`UsageError`."""
    chosen: set[str] = set()
    for part in _items(text):
        name = re.sub(r"[-\s]+", "_", part.lower())
        if name not in DEBT_REMEDIES:
            raise UsageError(f"unknown debt type {clean(part)!r}; use {', '.join(DEBT_REMEDIES)}")
        chosen.add(name)
    if not chosen:
        raise UsageError(f"--type needs a debt type: {', '.join(DEBT_REMEDIES)}")
    return tuple(t for t in DEBT_REMEDIES if t in chosen)


def parse_count(text: str, option: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    """A whole number for ``option`` between ``minimum`` and ``maximum``. Raises :class:`UsageError`."""
    try:
        value = int(text.strip())
    except ValueError:
        raise UsageError(f"{option} needs a whole number") from None
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is not None:
            raise UsageError(f"{option} must be between {minimum} and {maximum}")
        raise UsageError(f"{option} must be at least {minimum}")
    return value


def make_query(
    *,
    band: str | None = None,
    debt_type: str | None = None,
    path: str | None = None,
    sort: str | None = None,
    top: int | str | None = None,
) -> FileQuery:
    """Build a :class:`FileQuery` from option values (None means not given). Raises :class:`UsageError`."""
    order = (sort or "priority").strip().lower()
    if order not in SORT_KEYS:
        raise UsageError("--sort takes priority, risk or cost")
    if isinstance(top, str):
        count = parse_count(top, "--top")
    elif top is None:
        count = 0
    elif top < 0:
        raise UsageError("--top must be at least 0")
    else:
        count = top
    return FileQuery(
        bands=parse_bands(band) if band is not None else (),
        debt_types=parse_debt_types(debt_type) if debt_type is not None else (),
        path_prefix=normalise_path(path) if path is not None else "",
        sort=order,
        top=count,
    )


def ranked_files(result: AnalysisResult) -> list[RankedFile]:
    """Every ranked file of ``result`` with its rank, in priority order."""
    return [RankedFile(i, f, result.extras.get(f.path)) for i, f in enumerate(result.files, start=1)]


def select_files(result: AnalysisResult, query: FileQuery) -> list[RankedFile]:
    """The files passing ``query``'s filters in its order (highest first). ``query.top`` is NOT applied.

    ``priority`` keeps rank order, ``risk`` sorts by calibrated risk and ``cost`` by estimated hours (files
    without a cost estimate last); ties keep rank order.
    """
    rows = [item for item in ranked_files(result) if query.matches(item)]
    if query.sort == "risk":
        rows.sort(key=lambda r: (-r.score.risk, r.rank))
    elif query.sort == "cost":
        rows.sort(key=lambda r: (r.extra is None, -(r.extra.cost.hours if r.extra else 0.0), r.rank))
    return rows


def find_file(result: AnalysisResult, key: str) -> RankedFile:
    """The ranked file named by ``key``: a rank (``3`` or ``#3``), a path, or a unique path ending.

    An ending must match whole path components (``views.py`` matches ``app/views.py``, ``iews.py`` does not).
    Raises :class:`UsageError` when several files match and :class:`CliError` (exit code 2) when none does.
    """
    rows = ranked_files(result)
    if not rows:
        raise CliError(NO_FILES, exit_code=2)
    text = key.strip()
    number = _RANK_KEY.match(text)
    if number:
        rank = int(number.group(1))
        if not 1 <= rank <= len(rows):
            raise CliError(f"No file has rank {rank}; ranks go from 1 to {len(rows)}.", exit_code=2)
        return rows[rank - 1]
    wanted = normalise_path(text)
    if wanted:
        for item in rows:
            if item.path == wanted:
                return item
        matches = [item for item in rows if item.path.endswith("/" + wanted)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            listed = ", ".join(clean(m.path) for m in matches[:5])
            more = f" and {len(matches) - 5} more" if len(matches) > 5 else ""
            raise UsageError(
                f"{clean(text)} matches {len(matches)} files: {listed}{more}", hint="Type more of the path."
            )
    raise CliError(
        f"No ranked file matches {clean(text)}.",
        exit_code=2,
        hint="Type /files to list them; tests and skipped files are not ranked.",
    )


def ranked_table(
    rows: Sequence[RankedFile], *, glyphs: Glyphs = UNICODE_GLYPHS, changes: Mapping[str, str] | None = None
) -> Table:
    """The file table: #, File, Change (only with ``changes``), Priority, Risk, Cost, Debt, Top reason."""
    table = make_table()
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("File", overflow="fold")
    if changes is not None:
        table.add_column("Change", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Risk", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)
    table.add_column("Debt", overflow="fold")
    table.add_column("Top reason", overflow="fold")
    for item in rows:
        score, extra = item.score, item.extra
        row: list[RenderableType] = [str(item.rank), Text(clean(item.path))]
        if changes is not None:
            row.append(changes.get(item.path, ""))
        row += [
            band_text(score.band, score.priority, glyphs=glyphs),
            format_risk(score.risk),
            format_hour_range(extra.cost.low, extra.cost.high) if extra else "",
            Text(clean(", ".join(score.debt_types))),
            Text(clean(top_reason(score)), style="dim"),
        ]
        table.add_row(*row)
    return table


def file_row(item: RankedFile) -> dict[str, Any]:
    """One file as a JSON row with the keys of a ``tdp analyze --format json`` row."""
    score, extra = item.score, item.extra
    return {
        "rank": item.rank,
        "path": score.path,
        "priority": score.priority,
        "band": score.band,
        "risk": score.risk,
        "risk_low": score.risk_low,
        "risk_high": score.risk_high,
        "cost_hours": extra.cost.hours if extra else None,
        "cost_low": extra.cost.low if extra else None,
        "cost_high": extra.cost.high if extra else None,
        "debt_types": list(score.debt_types),
        "top_reason": top_reason(score),
        "reasons": [r.text for r in score.reasons],
    }


def sparkline(values: Sequence[float], glyphs: ViewGlyphs = UNICODE_VIEW_GLYPHS) -> str:
    """One character per value, scaled to the largest value (zero and negative values get the lowest)."""
    levels = glyphs.spark
    top = max(values, default=0)
    out: list[str] = []
    for value in values:
        if value <= 0 or top <= 0:
            out.append(levels[0])
        else:
            out.append(levels[min(7, max(1, math.ceil(7 * value / top)))])
    return "".join(out)


def diverging_bar(
    value: float, scale: float, *, width: int, glyphs: ViewGlyphs = UNICODE_VIEW_GLYPHS
) -> Text:
    """A bar around a centre axis, ``2 * width + 1`` cells wide.

    Positive values grow to the right in red (they raise the risk), negative ones to the left in teal; any
    non-zero value shows at least one block, and ``|value| >= scale`` fills its half.
    """
    n = 0
    if value != 0 and scale > 0:
        n = min(width, max(1, round(abs(value) / scale * width)))
    bar = Text()
    if value < 0:
        bar.append(" " * (width - n))
        bar.append(glyphs.block * n, style=TEAL)
        bar.append(glyphs.axis, style="dim")
        bar.append(" " * width)
    else:
        bar.append(" " * width)
        bar.append(glyphs.axis, style="dim")
        bar.append(glyphs.block * n, style=RED)
        bar.append(" " * (width - n))
    return bar


def section(title: str) -> Text:
    """A section heading of a detail screen."""
    return Text(title, style=f"bold {VIOLET}")


def limited(items: Sequence[str], limit: int) -> tuple[list[str], int]:
    """The first ``limit`` items and how many were left out."""
    return list(items[:limit]), max(0, len(items) - limit)


def utc_iso(when: datetime) -> str:
    """``when`` as ISO 8601 in UTC with a ``Z`` suffix."""
    return when.astimezone(UTC).isoformat().replace("+00:00", "Z")


def ranked_by_path(result: AnalysisResult) -> dict[str, RankedFile]:
    """Every ranked file of ``result`` keyed by its path."""
    return {item.path: item for item in ranked_files(result)}


def file_slocs(result: AnalysisResult) -> dict[str, int]:
    """Lines of code per path, from the graph nodes (every ranked file has one; use ``.get(path, 0)``)."""
    return {node.id: node.sloc for node in result.graph.nodes}


def json_header(loaded: LoadedAnalysis, schema: str) -> dict[str, Any]:
    """The first keys of every view's JSON document: schema, repo, sha, model_version and demo_model."""
    return {
        "schema": schema,
        "repo": loaded.meta.repo,
        "sha": loaded.result.sha,
        "model_version": loaded.result.model_version,
        "demo_model": loaded.meta.demo_model,
    }


def file_label(
    item: RankedFile, *, glyphs: Glyphs = UNICODE_GLYPHS, name: str | None = None, rank: bool = True
) -> Text:
    """A file as ``app/core.py  ● High 80  #3``: cleaned path (or ``name``), band label and dim rank."""
    label = Text(clean(name if name is not None else item.path))
    label.append("  ")
    label.append_text(band_text(item.score.band, item.score.priority, glyphs=glyphs))
    if rank:
        label.append(f"  #{item.rank}", style="dim")
    return label


def folder_suggestions(session: Session, word: str) -> Iterator[Suggestion]:
    """Folders (``app/``, ``app/api/``) holding ranked files of the loaded analysis that start with ``word``.

    Sorted by name, at most :data:`MAX_FOLDER_SUGGESTIONS`; nothing when no analysis is loaded.
    """
    if session.current is None:
        return
    seen: set[str] = set()
    for f in session.current.result.files:
        parts = f.path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            folder = "/".join(parts[:depth]) + "/"
            if folder.startswith(word):
                seen.add(folder)
    for folder in sorted(seen)[:MAX_FOLDER_SUGGESTIONS]:
        yield Suggestion(folder, "folder")
