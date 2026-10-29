"""``/files`` and ``/file`` (plus ``tdp files`` and ``tdp file``): the ranked list and one file's detail."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.commands import _pager
from tdp_cli.commands._detail import file_detail, file_json
from tdp_cli.commands._views import (
    DEBT_REMEDIES,
    NO_FILES,
    SORT_KEYS,
    file_row,
    find_file,
    folder_suggestions,
    json_header,
    make_query,
    ranked_table,
    select_files,
    view_glyphs,
)
from tdp_cli.completion import complete_analysis_paths, options_completer
from tdp_cli.errors import UsageError
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
from tdp_cli.theme import BAND_ORDER

if TYPE_CHECKING:
    from tdp_cli.session import Session

FILES_SCHEMA = "entropiq.files.v1"
FILES_OPTIONS = ("--band", "--type", "--path", "--sort", "--top")
NO_MATCH = "No files match these filters."
_OPTIONS = options_completer(*FILES_OPTIONS)


def files_handler(session: Session, args: Sequence[str]) -> None:
    """``/files [--band B] [--type T] [--path PREFIX] [--sort KEY] [--top N]``: the paged ranked table."""
    parsed = parse_args(args, values=FILES_OPTIONS, max_positionals=0)
    query = make_query(
        band=parsed.value("--band"),
        debt_type=parsed.value("--type"),
        path=parsed.value("--path"),
        sort=parsed.value("--sort"),
        top=parsed.value("--top"),
    )
    loaded = session.require_analysis()
    if not loaded.result.files:
        session.console.print(Text(NO_FILES, style="dim"))
        return
    rows = select_files(loaded.result, query)
    if query.top:
        rows = rows[: query.top]
    if not rows:
        session.console.print(Text(NO_MATCH, style="dim"))
        return
    _pager.run_files_pager(session, loaded, rows, page_size=session.prefs.top, title=query.describe())


def file_handler(session: Session, args: Sequence[str]) -> None:
    """``/file <path or rank>``: the detail screen of one ranked file."""
    parsed = parse_args(args, max_positionals=1)
    if not parsed.positionals:
        raise UsageError("say which file: a path, a unique ending or a rank from /files")
    loaded = session.require_analysis()
    item = find_file(loaded.result, parsed.positionals[0])
    session.console.print(file_detail(loaded, item, glyphs=session.glyphs, view=view_glyphs(session.console)))


def complete_files(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Completion for ``/files``: option names, then band names, debt types, sort keys or folders."""
    last = args[-1] if args else ""
    if last == "--band":
        yield from (Suggestion(b.lower(), "band") for b in BAND_ORDER if b.lower().startswith(word.lower()))
    elif last == "--type":
        yield from (Suggestion(t, "debt type") for t in DEBT_REMEDIES if t.startswith(word.lower()))
    elif last == "--sort":
        yield from (Suggestion(k, "sort") for k in SORT_KEYS if k.startswith(word.lower()))
    elif last == "--path":
        yield from folder_suggestions(session, word)
    elif last != "--top":
        yield from _OPTIONS(session, args, word)


def complete_file(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Completion for ``/file``: the paths of the loaded analysis, for the first argument only."""
    if not args:
        yield from complete_analysis_paths(session, args, word)


def files_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    ref: RefOption = None,
    band: Annotated[
        str | None,
        typer.Option("--band", help="Only these bands (comma list: critical,high).", show_default=False),
    ] = None,
    debt_type: Annotated[
        str | None,
        typer.Option(
            "--type", help="Only files with one of these debt types (comma list).", show_default=False
        ),
    ] = None,
    path: Annotated[
        str | None, typer.Option("--path", help="Only paths starting with this prefix.", show_default=False)
    ] = None,
    sort: Annotated[
        Literal["priority", "risk", "cost"], typer.Option("--sort", help="Order of the rows (highest first).")
    ] = "priority",
    top: TopOption = None,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """List the ranked files with filters and sorting."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    with cli_guard(session.err_console):
        query = make_query(band=band, debt_type=debt_type, path=path, sort=sort)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    limit = session.prefs.top if top is None else top
    with cli_guard(session.err_console):
        rows = select_files(loaded.result, query)
        shown = rows[:limit] if limit else rows
        if output_format == "json":
            document = {
                **json_header(loaded, FILES_SCHEMA),
                "query": {
                    "band": list(query.bands),
                    "type": list(query.debt_types),
                    "path": query.path_prefix,
                    "sort": query.sort,
                    "top": limit,
                },
                "total": len(rows),
                "files": [file_row(item) for item in shown],
            }
            sys.stdout.write(json.dumps(document, indent=2) + "\n")
            return
        console = session.console
        if not loaded.result.files:
            console.print(Text(NO_FILES, style="dim"))
            return
        if not rows:
            console.print(Text(NO_MATCH, style="dim"))
            return
        console.print(Text(query.describe(), style="bold"))
        console.print(ranked_table(shown, glyphs=session.glyphs))
        if len(shown) < len(rows):
            console.print(
                Text(f"Showing {len(shown)} of {len(rows)} files. Use --top 0 to see all.", style="dim")
            )


def file_cli(
    ctx: typer.Context,
    file: Annotated[
        str, typer.Argument(help="Path, unique ending or rank of a ranked file.", show_default=False)
    ],
    target: TargetArgument = ".",
    ref: RefOption = None,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Show the full detail of one ranked file."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        item = find_file(loaded.result, file)
        if output_format == "json":
            sys.stdout.write(json.dumps(file_json(loaded, item), indent=2) + "\n")
            return
        session.console.print(
            file_detail(loaded, item, glyphs=session.glyphs, view=view_glyphs(session.console))
        )


def register(registry: CommandRegistry) -> None:
    """Register ``/files``, ``/file``, ``tdp files`` and ``tdp file``."""
    registry.add(
        Command(
            name="files",
            group="Explore",
            help="List ranked files with filters, sorting and paging",
            usage="/files [--band B] [--type T] [--path PREFIX] [--sort priority|risk|cost] [--top N]",
            details=(
                "Shows the ranked files with their priority and band, risk of maintenance pain in the next "
                "12 months, estimated fix hours, debt types and top reason. --band and --type take comma "
                "lists (critical,high or test,complexity), --path keeps paths starting with a folder prefix, "
                "--sort orders by priority, risk or cost, and --top keeps that many rows. The list is paged "
                "by the top setting: type n for the next page, p for the previous one, a number from the # "
                "column to open that file, and Enter or q to go back."
            ),
            handler=files_handler,
            complete=complete_files,
            needs_analysis=True,
            cli=files_cli,
        )
    )
    registry.add(
        Command(
            name="file",
            group="Explore",
            help="Show the full detail of one file",
            usage="/file <path or rank>",
            details=(
                "Name the file by its path, a unique ending such as views.py, or its rank from /files. The "
                "screen shows the priority, band and risk range, the reasons and mitigations, the SHAP "
                "contribution of each feature family, the most complex functions, the files it imports and "
                "that import it, its commit activity over 24 months, its debt types with their usual remedy, "
                "and the breakdown of the fix estimate."
            ),
            handler=file_handler,
            complete=complete_file,
            needs_analysis=True,
            cli=file_cli,
        )
    )
