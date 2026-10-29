"""``/export`` (alias ``/report``) and ``tdp report``: write the analysis as Markdown, JSON or SARIF.

The writers are ``tdp_core.report``'s: Markdown is the ranking report followed by the saved fix plan, JSON
is the full ``tdp-report`` document, and SARIF 2.1.0 lists the files at or above a band for code scanning.
Without ``-o`` the file is ``<repo slug>-<short sha>.<ext>`` in the current folder; ``-o -`` prints the
report instead.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

import typer
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
    TopOption,
    cli_analysis,
    cli_guard,
    cli_session,
)
from tdp_cli.registry import Command, CommandRegistry, Suggestion
from tdp_cli.theme import clean, short_sha, success

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import AnalysisMeta
    from tdp_core.contracts import AnalysisResult

FORMATS: Final = MappingProxyType(
    {"md": ("Markdown", ".md"), "json": ("JSON", ".json"), "sarif": ("SARIF", ".sarif")}
)
FORMAT_ALIASES: Final = MappingProxyType({"markdown": "md"})
MIN_BANDS: Final = MappingProxyType({"high": "High", "critical": "Critical"})
STDOUT: Final = "-"
USAGE: Final = "/export md|json|sarif [-o FILE] [--top N] [--min-band high|critical]"
NO_FORMAT: Final = "say which format to write: md, json or sarif"
NO_FORMAT_HINT: Final = "For example /export md -o report.md"
TOP_ERROR: Final = "--top must be a whole number of rows (0 lists every file)"
BAND_ERROR: Final = "--min-band must be high or critical"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_OPTIONS = options_completer("--output", "--top", "--min-band")
_FORMAT_SUGGESTIONS: Final = (
    Suggestion("md", "Markdown report and fix plan"),
    Suggestion("json", "Full analysis as JSON"),
    Suggestion("sarif", "SARIF 2.1.0 for code scanning"),
)


def expand_short_options(args: Sequence[str]) -> list[str]:
    """``args`` with ``-o X`` spelled ``--output X`` and ``-o=X`` as ``--output=X`` (not after ``--``)."""
    out: list[str] = []
    passthrough = False
    for arg in args:
        if passthrough:
            out.append(arg)
        elif arg == "--":
            passthrough = True
            out.append(arg)
        elif arg == "-o":
            out.append("--output")
        elif arg.startswith("-o="):
            out.append("--output=" + arg[3:])
        else:
            out.append(arg)
    return out


def parse_format(raw: str) -> str:
    """``md``, ``json`` or ``sarif`` in any case (``markdown`` too). Raises :class:`UsageError` otherwise."""
    key = raw.strip().lower()
    key = FORMAT_ALIASES.get(key, key)
    if key not in FORMATS:
        raise UsageError(f"unknown format {clean(raw)}; use md, json or sarif")
    return key


def parse_top(raw: str) -> int:
    """The ``--top`` row count: a whole number of 0 or more. Raises :class:`UsageError` otherwise."""
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise UsageError(TOP_ERROR) from exc
    if value < 0:
        raise UsageError(TOP_ERROR)
    return value


def parse_min_band(raw: str) -> str:
    """``High`` or ``Critical`` from either word in any case. Raises :class:`UsageError` otherwise."""
    band = MIN_BANDS.get(raw.strip().lower())
    if band is None:
        raise UsageError(BAND_ERROR)
    return band


def default_file_name(meta: AnalysisMeta, fmt: str) -> str:
    """``<slug>-<short sha><ext>``, unsafe slug characters replaced (``psf__requests-3c7baaa.sarif``)."""
    slug = _UNSAFE.sub("-", meta.slug).strip("-.") or "analysis"
    return f"{slug}-{short_sha(meta.sha)}{FORMATS[fmt][1]}"


def output_path(raw: str | None, meta: AnalysisMeta, fmt: str, *, cwd: Path | None = None) -> Path:
    """Where to write: the default name in ``cwd`` (default: the current folder) when ``raw`` is None.

    A relative ``raw`` is taken from ``cwd`` and ``~`` is expanded. An existing folder, or a ``raw`` ending in
    ``/`` or ``\\``, gets the default name inside it.
    """
    base = Path.cwd() if cwd is None else cwd
    name = default_file_name(meta, fmt)
    if raw is None:
        return base / name
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    if raw.endswith(("/", "\\")) or path.is_dir():
        return path / name
    return path


def render_report(result: AnalysisResult, fmt: str, *, top: int | None = 20, min_band: str = "High") -> str:
    """The report text in ``fmt``; ``top`` limits the Markdown rows (None lists every file).

    Markdown is the ranking report plus the saved fix plan (made with the analysis's budget), JSON is the
    ``tdp-report`` document, SARIF lists the files at or above ``min_band`` (``High`` or ``Critical``).
    """
    if fmt == "md":
        from tdp_core.report.markdown import plan_to_markdown, to_markdown

        return to_markdown(result, top) + "\n" + plan_to_markdown(result.plan)
    if fmt == "json":
        from tdp_core.report.jsonout import to_json

        return to_json(result)
    from tdp_core.report.sarif import SarifBand, to_sarif

    return json.dumps(to_sarif(result, cast("SarifBand", min_band)), indent=2) + "\n"


def report_detail(result: AnalysisResult, fmt: str, content: str, top: int | None) -> str:
    """A few words on what the report holds (``top 20 of 130 files and the fix plan``, ``3 results``)."""
    n = len(result.files)
    files = "1 file" if n == 1 else f"{n} files"
    if fmt == "md":
        if top is not None and top < n:
            return f"top {top} of {files} and the fix plan"
        return f"all {files} and the fix plan"
    if fmt == "json":
        return files
    results = len(json.loads(content)["runs"][0]["results"])
    return "1 result" if results == 1 else f"{results} results"


def write_report(path: Path, content: str) -> Path:
    """Write ``content`` as UTF-8 with ``\\n`` line ends and return the resolved path.

    Raises :class:`CliError` when the folder or file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise CliError(f"Cannot write {clean(str(path))}: {exc.strerror or exc}") from exc
    return path.resolve()


def _wrote(label: str, path: Path, detail: str) -> str:
    return f"Wrote the {label} report to {clean(str(path))} ({detail})"


def export_handler(session: Session, args: Sequence[str]) -> None:
    """``/export md|json|sarif [-o FILE] [--top N] [--min-band high|critical]``."""
    parsed = parse_args(
        expand_short_options(args), values=("--output", "--top", "--min-band"), max_positionals=1
    )
    if not parsed.positionals:
        raise UsageError(NO_FORMAT, hint=NO_FORMAT_HINT)
    fmt = parse_format(parsed.positionals[0])
    raw_top = parsed.value("--top")
    rows = session.prefs.top if raw_top is None else parse_top(raw_top)
    top = rows or None
    raw_band = parsed.value("--min-band")
    band = "High" if raw_band is None else parse_min_band(raw_band)
    loaded = session.require_analysis()
    content = render_report(loaded.result, fmt, top=top, min_band=band)
    target = parsed.value("--output")
    if target == STDOUT:
        session.console.print(Text(content.rstrip("\n")), soft_wrap=True)
        return
    path = write_report(output_path(target, loaded.meta, fmt), content)
    detail = report_detail(loaded.result, fmt, content, top)
    success(session.console, _wrote(FORMATS[fmt][0], path, detail))


def _complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    if args and args[-1] == "--min-band":
        yield from (Suggestion(name, "band") for name in MIN_BANDS if name.startswith(word.lower()))
        return
    if args and args[-1] in ("--output", "-o", "--top"):
        return
    if word.startswith("-"):
        yield from _OPTIONS(session, args, word)
        return
    if not any(a in FORMATS or a in FORMAT_ALIASES for a in args):
        yield from (s for s in _FORMAT_SUGGESTIONS if s.text.startswith(word.lower()))


def report_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    output_format: Annotated[
        Literal["md", "json", "sarif"], typer.Option("--format", "-f", help="Report format.")
    ] = "md",
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="File to write (default <slug>-<sha>.<ext> in the current folder; - for stdout).",
            show_default=False,
        ),
    ] = None,
    top: TopOption = None,
    min_band: Annotated[
        Literal["high", "critical"], typer.Option("--min-band", help="Lowest band in the SARIF output.")
    ] = "high",
    ref: RefOption = None,
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Write the analysis to a Markdown, JSON or SARIF file."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    rows = session.prefs.top if top is None else top
    limit = rows or None
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        content = render_report(loaded.result, output_format, top=limit, min_band=MIN_BANDS[min_band])
        if output == STDOUT:
            sys.stdout.write(content)
            return
        path = write_report(output_path(output, loaded.meta, output_format), content)
        detail = report_detail(loaded.result, output_format, content, limit)
        success(session.err_console, _wrote(FORMATS[output_format][0], path, detail))


def register(registry: CommandRegistry) -> None:
    """Register ``/export`` (alias ``/report``) and ``tdp report``."""
    registry.add(
        Command(
            name="export",
            aliases=("report",),
            group="Report",
            help="Write the analysis to a Markdown, JSON or SARIF file",
            usage=USAGE,
            details=(
                "md writes the ranking report (the top files, see --top; 0 lists every file) and the fix "
                "plan saved with the analysis; json writes the full analysis; sarif writes SARIF 2.1.0 for "
                "code scanning with the High and Critical files (--min-band critical keeps only Critical). "
                "Without -o the file is <repo>-<sha>.<ext> in the current folder; -o - prints it here. On "
                "the command line: tdp report [PATH] --format sarif -o FILE."
            ),
            handler=export_handler,
            complete=_complete,
            needs_analysis=True,
            cli=report_cli,
            cli_name="report",
        )
    )
