"""``/analyze`` and ``tdp analyze``: rank the files of a local repository or a GitHub URL."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.completion import complete_directories, options_completer
from tdp_cli.options import (
    DevModelOption,
    FreshOption,
    ModelDirOption,
    RefOption,
    TargetArgument,
    TopOption,
    cli_analysis,
    cli_session,
)
from tdp_cli.registry import Command, CommandRegistry, Suggestion

if TYPE_CHECKING:
    from tdp_cli.session import Session

_OPTIONS = options_completer("--ref", "--fresh")


def _complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    if args and args[-1] == "--ref":
        return
    yield from _OPTIONS(session, args, word)
    if not any(not a.startswith("-") and (i == 0 or args[i - 1] != "--ref") for i, a in enumerate(args)):
        yield from complete_directories(session, args, word)


def analyze_handler(session: Session, args: Sequence[str]) -> None:
    """``/analyze [PATH or URL] [--ref REF] [--fresh]``."""
    from tdp_cli.output import summary_card
    from tdp_cli.runner import run_analysis

    parsed = parse_args(args, values=("--ref",), flags=("--fresh",), max_positionals=1)
    target = parsed.positionals[0] if parsed.positionals else "."
    loaded, reused = run_analysis(session, target, ref=parsed.value("--ref"), fresh=parsed.flag("--fresh"))
    session.console.print(summary_card(loaded, top=5, glyphs=session.glyphs, reused=reused))
    if reused:
        session.console.print(Text("Saved analysis reused; add --fresh to run it again.", style="dim"))
    session.console.print(Text("Type /help to see what you can do next.", style="dim"))


def analyze_cli(
    ctx: typer.Context,
    target: TargetArgument = ".",
    ref: RefOption = None,
    top: TopOption = None,
    output_format: Annotated[
        Literal["table", "json", "md"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Rank the files of a repository by future maintenance pain."""
    from tdp_cli.output import analysis_json, analysis_markdown, files_table, summary_card

    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    rows = session.prefs.top if top is None else top
    if output_format == "json":
        sys.stdout.write(json.dumps(analysis_json(loaded, top=rows), indent=2) + "\n")
        return
    if output_format == "md":
        sys.stdout.write(analysis_markdown(loaded, top=rows))
        return
    console = session.console
    files = loaded.result.files
    console.print(summary_card(loaded, top=0, glyphs=session.glyphs))
    if files:
        console.print(files_table(files, loaded.result.extras, limit=rows, glyphs=session.glyphs))
        shown = min(rows, len(files)) if rows else len(files)
        if shown < len(files):
            console.print(
                Text(f"Showing {shown} of {len(files)} files. Use --top 0 to see all.", style="dim")
            )


def register(registry: CommandRegistry) -> None:
    """Register ``/analyze`` and ``tdp analyze``."""
    registry.add(
        Command(
            name="analyze",
            aliases=("analyse",),
            group="Analysis",
            help="Analyse a local repository or a GitHub URL",
            usage="/analyze [PATH or URL] [--ref REF] [--fresh]",
            details=(
                "Ranks the Python files of a repository by how much maintenance pain each is likely to cause "
                "in the next year. PATH is a folder inside a git repository (default: the current folder) or "
                "a bare repository; URL is https://github.com/<owner>/<name> (or gitlab.com), kept as a lean "
                "mirror under the Entropiq folder. --ref picks a branch, tag or commit (default HEAD). "
                "A saved "
                "analysis of the same commit and model is reopened instantly unless --fresh is given."
            ),
            handler=analyze_handler,
            complete=_complete,
            cli=analyze_cli,
        )
    )
