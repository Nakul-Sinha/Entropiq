"""``/whatif`` and ``tdp whatif``: simulate fixes on one file and compare its priority and risk.

The simulation is ``tdp_core.planning.whatif.what_if`` with the session's model bundle: the file's features
are moved to the targets of the chosen actions and the file is scored again; every other file keeps its
score. In the interactive shell on a real terminal, ``/whatif <path>`` without flags shows a checkbox picker.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group
from rich.table import Table
from rich.text import Text

from tdp_cli.args import ParsedArgs, parse_args
from tdp_cli.commands._file_arg import FileArgument, resolve_file
from tdp_cli.completion import complete_analysis_paths, options_completer
from tdp_cli.errors import CliError, UsageError
from tdp_cli.options import (
    DevModelOption,
    FreshOption,
    ModelDirOption,
    RefOption,
    TargetArgument,
    cli_analysis,
    cli_guard,
    cli_session,
)
from tdp_cli.registry import Command, CommandRegistry, Suggestion
from tdp_cli.theme import (
    UNICODE_GLYPHS,
    Glyphs,
    band_text,
    clean,
    format_risk,
    make_panel,
    make_table,
    notice,
)

if TYPE_CHECKING:
    from rich.panel import Panel

    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.planning.whatif import WhatIfResult

WHATIF_SCHEMA: Final = "entropiq.whatif.v1"

# Literal names keep tdp_core.contracts out of startup; a test compares them with WHATIF_ACTIONS.
ACTION_FLAGS: Final = MappingProxyType(
    {
        "--add-tests": "add_tests",
        "--split": "split_longest_function",
        "--dedupe": "remove_duplication",
        "--resolve-satd": "resolve_satd",
        "--break-cycle": "break_cycle",
    }
)
ACTION_LABELS: Final = MappingProxyType(
    {
        "add_tests": "Add tests",
        "split_longest_function": "Split the longest function",
        "remove_duplication": "Remove duplicated code",
        "resolve_satd": "Resolve TODO and FIXME comments",
        "break_cycle": "Break the import cycle",
    }
)
ACTION_HINTS: Final = MappingProxyType(
    {
        "add_tests": "a test file that imports it",
        "split_longest_function": "halve its most complex function",
        "remove_duplication": "no copied code left",
        "resolve_satd": "no self-admitted debt comments",
        "break_cycle": "no import cycle or hidden coupling",
    }
)
FEATURE_LABELS: Final = MappingProxyType(
    {
        "has_test_file": "has a matching test file (1 yes, 0 no)",
        "test_importers": "test files importing it",
        "test_cochange_rate": "test co-change rate",
        "cc_max": "highest cyclomatic complexity",
        "cog_max": "highest cognitive complexity",
        "nesting_max": "deepest nesting",
        "longest_function": "longest function, lines",
        "dup_ratio": "share of duplicated tokens",
        "satd_count": "self-admitted debt comments",
        "satd_density": "debt comments per 1,000 SLOC",
        "satd_age_days": "age of the oldest debt comment, days",
        "scc_size": "files in its import cycle",
        "hidden_coupling": "co-change partners without an import",
    }
)
ALL_FLAG: Final = "--all"
FLAG_LIST: Final = ", ".join([*ACTION_FLAGS, ALL_FLAG])
NO_ACTION: Final = "choose at least one action"
NO_ACTION_HINT: Final = f"Add one or more of {FLAG_LIST}."
NO_CHANGE: Final = "These actions change nothing for this file: it already meets their targets."
WHATIF_CAPTION: Final = "Other files keep their scores; cost, reasons and risk ranges are not recomputed."
USAGE: Final = "/whatif <path> [--add-tests] [--split] [--dedupe] [--resolve-satd] [--break-cycle] [--all]"

_OPTIONS = options_completer(*ACTION_FLAGS, ALL_FLAG)


def actions_from_flags(parsed: ParsedArgs) -> list[str]:
    """The actions named by the parsed flags, in canonical order (``--all`` gives all five)."""
    if parsed.flag(ALL_FLAG):
        return list(ACTION_FLAGS.values())
    return [action for flag, action in ACTION_FLAGS.items() if parsed.flag(flag)]


def selected_actions(
    *,
    add_tests: bool = False,
    split: bool = False,
    dedupe: bool = False,
    resolve_satd: bool = False,
    break_cycle: bool = False,
    all_actions: bool = False,
) -> list[str]:
    """The actions chosen with the command-line options, in canonical order."""
    chosen = (add_tests, split, dedupe, resolve_satd, break_cycle)
    return [action for action, on in zip(ACTION_FLAGS.values(), chosen, strict=True) if on or all_actions]


def format_feature_value(value: float) -> str:
    """A feature value for display: whole numbers without decimals, else 1 or 2 decimals by size.

    NaN and infinities are shown as Python prints them (``nan``, ``inf``).
    """
    if not math.isfinite(value):
        return str(value)
    if value == int(value):
        return str(int(value))
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def can_pick(session: Session) -> bool:
    """True when the checkbox picker can be shown: the interactive shell on a real terminal."""
    try:
        return bool(session.interactive and session.console.is_terminal and sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def simulate(session: Session, loaded: LoadedAnalysis, path: str, actions: Sequence[str]) -> WhatIfResult:
    """Run the what-if simulation for ``path`` with the session's bundle."""
    bundle = session.bundle()
    if loaded.meta.model_version != bundle.version:
        notice(
            session.err_console,
            f"This analysis used model {clean(loaded.meta.model_version)}; the simulation uses "
            f"{clean(bundle.version)}, so the before numbers can differ from the saved ranking.",
        )
    from tdp_core.planning.whatif import what_if

    try:
        return what_if(loaded.result.features, path, list(actions), bundle)
    except KeyError as exc:
        raise CliError(f"{clean(path)} has no feature row in this analysis", exit_code=2) from exc
    except ValueError as exc:
        raise CliError(f"Cannot simulate {clean(path)}: {clean(str(exc))}") from exc


def _priority_change(before: int, after: int, glyphs: Glyphs) -> Text:
    if after < before:
        return Text(f"{glyphs.down} {before - after}", style="green")
    if after > before:
        return Text(f"{glyphs.up} {after - before}", style="red")
    return Text("no change", style="dim")


def _risk_change(before: float, after: float, glyphs: Glyphs) -> Text:
    if after == before:
        return Text("no change", style="dim")
    points = round(100 * abs(after - before))
    amount = "<1 pt" if points == 0 else f"{points} pts" if points != 1 else "1 pt"
    if after < before:
        return Text(f"{glyphs.down} {amount}", style="green")
    return Text(f"{glyphs.up} {amount}", style="red")


def whatif_view(
    path: str, actions: Sequence[str], result: WhatIfResult, *, glyphs: Glyphs = UNICODE_GLYPHS
) -> Panel:
    """A panel with the actions, priority and risk before and after, and the features that changed."""
    before_priority, before_band, before_risk = result.before
    after_priority, after_band, after_risk = result.after
    parts: list[Any] = [
        Text.assemble(("Actions: ", "bold"), ", ".join(ACTION_LABELS.get(a, a) for a in actions))
    ]
    scores = Table.grid(padding=(0, 3))
    scores.add_column(style="bold")
    for _ in range(3):
        scores.add_column()
    scores.add_row(
        "", Text("Before", style="bold"), Text("After", style="bold"), Text("Change", style="bold")
    )
    scores.add_row(
        "Priority",
        band_text(before_band, before_priority, glyphs=glyphs),
        band_text(after_band, after_priority, glyphs=glyphs),
        _priority_change(before_priority, after_priority, glyphs),
    )
    scores.add_row(
        "Risk (12 months)",
        format_risk(before_risk),
        format_risk(after_risk),
        _risk_change(before_risk, after_risk, glyphs),
    )
    parts += [Text(""), scores, Text("")]
    if result.changed_features:
        table = make_table()
        table.add_column("Feature")
        table.add_column("Meaning", style="dim")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        for name, (old, new) in result.changed_features.items():
            table.add_row(
                name,
                FEATURE_LABELS.get(name, ""),
                format_feature_value(old),
                format_feature_value(new),
            )
        parts.append(table)
    else:
        parts.append(Text(NO_CHANGE, style="yellow"))
    parts.append(Text(WHATIF_CAPTION, style="dim"))
    return make_panel(Group(*parts), Text(f"What if: {clean(path)}"))


def whatif_json(
    loaded: LoadedAnalysis, path: str, actions: Sequence[str], result: WhatIfResult, model_version: str
) -> dict[str, Any]:
    """The simulation as a JSON-ready dict (schema ``entropiq.whatif.v1``)."""
    return {
        "schema": WHATIF_SCHEMA,
        "repo": loaded.meta.repo,
        "sha": loaded.meta.sha,
        "path": path,
        "actions": list(actions),
        "model_version": model_version,
        **result.to_json(),
    }


def _pick_actions(session: Session, path: str) -> list[str] | None:
    from tdp_cli.commands import _picker

    options = [(action, ACTION_LABELS[action], ACTION_HINTS[action]) for action in ACTION_FLAGS.values()]
    return _picker.pick_many(
        f"What if {clean(path)}? Pick the fixes to simulate", options, glyphs=session.glyphs
    )


def whatif_handler(session: Session, args: Sequence[str]) -> None:
    """``/whatif <path> [--add-tests] [--split] [--dedupe] [--resolve-satd] [--break-cycle] [--all]``."""
    parsed = parse_args(args, flags=(*ACTION_FLAGS, ALL_FLAG), max_positionals=1)
    if not parsed.positionals:
        raise UsageError("say which file to simulate, for example /whatif pkg/mod.py --add-tests")
    loaded = session.require_analysis()
    path = resolve_file(loaded, parsed.positionals[0]).path
    actions = actions_from_flags(parsed)
    if not actions:
        if not can_pick(session):
            raise UsageError(NO_ACTION, hint=NO_ACTION_HINT)
        picked = _pick_actions(session, path)
        if picked is None:
            notice(session.console, "Cancelled.")
            return
        if not picked:
            notice(session.console, "No action chosen, nothing to simulate.")
            return
        actions = picked
    result = simulate(session, loaded, path, actions)
    session.console.print(whatif_view(path, actions, result, glyphs=session.glyphs))


def _complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    if word.startswith("-"):
        yield from _OPTIONS(session, args, word)
        return
    if not any(not a.startswith("-") for a in args):
        yield from complete_analysis_paths(session, args, word)


def whatif_cli(
    ctx: typer.Context,
    path: FileArgument,
    target: TargetArgument = ".",
    add_tests: Annotated[bool, typer.Option("--add-tests", help="Add a test file that imports it.")] = False,
    split: Annotated[bool, typer.Option("--split", help="Halve its most complex function.")] = False,
    dedupe: Annotated[bool, typer.Option("--dedupe", help="Remove its duplicated code.")] = False,
    resolve_satd: Annotated[
        bool, typer.Option("--resolve-satd", help="Resolve its TODO and FIXME comments.")
    ] = False,
    break_cycle: Annotated[
        bool, typer.Option("--break-cycle", help="Break its import cycle and hidden coupling.")
    ] = False,
    all_actions: Annotated[bool, typer.Option("--all", help="Apply every action.")] = False,
    output_format: Annotated[
        Literal["table", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    ref: RefOption = None,
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Simulate fixes on one file and compare its priority and risk."""
    actions = selected_actions(
        add_tests=add_tests,
        split=split,
        dedupe=dedupe,
        resolve_satd=resolve_satd,
        break_cycle=break_cycle,
        all_actions=all_actions,
    )
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    with cli_guard(session.err_console):
        if not actions:
            raise UsageError(NO_ACTION, hint=NO_ACTION_HINT)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        name = resolve_file(loaded, path).path
        result = simulate(session, loaded, name, actions)
        if output_format == "json":
            version = session.model.version or ""
            sys.stdout.write(json.dumps(whatif_json(loaded, name, actions, result, version), indent=2) + "\n")
        else:
            session.console.print(whatif_view(name, actions, result, glyphs=session.glyphs))


def register(registry: CommandRegistry) -> None:
    """Register ``/whatif`` and ``tdp whatif``."""
    registry.add(
        Command(
            name="whatif",
            aliases=("what-if",),
            group="Explore",
            help="Simulate fixes on one file and compare its priority and risk",
            usage=USAGE,
            details=(
                "Moves the file's metrics to what each fix would give (--add-tests: a test file that imports "
                "it; --split: its most complex function halved; --dedupe: no duplicated code; "
                "--resolve-satd: no TODO or FIXME comments; --break-cycle: no import cycle; --all: every "
                "fix) and scores it again with the current model. Without flags the shell shows a picker. "
                "On the command line: tdp whatif FILE [PATH] --add-tests."
            ),
            handler=whatif_handler,
            complete=_complete,
            needs_analysis=True,
            cli=whatif_cli,
        )
    )
