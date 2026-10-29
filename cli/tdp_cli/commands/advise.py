"""``/advise`` and ``tdp advise``: grounded AI fix advice for one file through ``tdp_core.advisor``.

The advice comes from Anthropic's API (``DEFAULT_SETTINGS.advisor_model``) when ``ANTHROPIC_API_KEY`` is set.
The advisor checks every suggested step against the functions of the file and drops steps that name code that
is not there. Answers are cached under ``<Entropiq home>/advice/<key>.json`` with the platform's key
(``Advisor.cache_key(blob_sha, model)``), so the same file content and model never cost a second request
unless ``--refresh`` is given. Saved advice is shown even when no key is set.

Tests replace :data:`client_factory` so no real client is ever built.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group
from rich.text import Text

from tdp_cli.args import parse_args
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
from tdp_cli.theme import UNICODE_GLYPHS, Glyphs, activity, clean, format_hour_range, make_panel
from tdp_core.config import DEFAULT_SETTINGS

if TYPE_CHECKING:
    from rich.panel import Panel

    from tdp_cli.home import EntropiqHome
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.advisor.client import LLMClient
    from tdp_core.advisor.schema import Advice
    from tdp_core.contracts import FileExtra, FileScore, FileSyntax

logger = logging.getLogger(__name__)

ADVICE_SCHEMA: Final = "entropiq.advice.v1"
CACHE_SCHEMA: Final = 1
ENV_API_KEY: Final = "ANTHROPIC_API_KEY"
ADVICE_DIR: Final = "advice"
RETRY_HINT: Final = "The service may be busy or unreachable; try again in a minute."
INSTALL_HINT: Final = "Install the advisor extra: pip install 'tdp-core[advisor]'"
NO_PATH: Final = "say which file to get advice for, for example /advise pkg/mod.py"
NO_STEPS: Final = "No concrete steps were suggested."
ALL_DROPPED: Final = "Every suggested step named code that is not in this file, so none is shown."
USAGE: Final = "/advise <path> [--refresh]"

_OPTIONS = options_completer("--refresh")


@dataclass(frozen=True, slots=True)
class AdviceEntry:
    """Advice, the model that wrote it, when (UTC, whole seconds) and whether it came from the cache."""

    advice: Advice
    model: str
    created_at: datetime
    cached: bool = False


type ClientFactory = Callable[[str, str], LLMClient]
"""Builds the LLM client from ``(model, api_key)``."""


def anthropic_client(model: str, api_key: str) -> LLMClient:
    """The real Anthropic client (raises ``LLMError`` when the ``anthropic`` package is missing)."""
    from tdp_core.advisor.client import AnthropicClient

    return AnthropicClient(model, api_key=api_key)


client_factory: ClientFactory = anthropic_client
"""The factory the commands use, looked up at call time (tests replace it with a fake)."""


def api_key(environ: Mapping[str, str] | None = None) -> str | None:
    """The stripped ``ANTHROPIC_API_KEY`` of ``environ`` (default ``os.environ``), None when blank."""
    env = os.environ if environ is None else environ
    key = env.get(ENV_API_KEY, "").strip()
    return key or None


def advice_dir(home: EntropiqHome) -> Path:
    """The folder that holds saved advice: ``<home>/advice``."""
    return home.root / ADVICE_DIR


def cache_path(root: Path, blob_sha: str, model: str) -> Path:
    """The cache file of one file content and model: ``<root>/<Advisor.cache_key(blob_sha, model)>.json``."""
    from tdp_core.advisor.advisor import Advisor

    return root / f"{Advisor.cache_key(blob_sha, model)}.json"


def _utc_text(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def read_cached(path: Path) -> AdviceEntry | None:
    """The advice saved at ``path``; None when it is missing or unreadable (the next request replaces it)."""
    from pydantic import ValidationError

    from tdp_core.advisor.schema import Advice

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != CACHE_SCHEMA:
            logger.debug("ignoring advice cache %s: unknown schema", path)
            return None
        created = datetime.fromisoformat(str(data["created_at"]).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return AdviceEntry(
            Advice.model_validate(data["advice"]), str(data["model"]), created.astimezone(UTC), cached=True
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        logger.debug("ignoring advice cache %s: %s", path, exc)
        return None


def save_cached(path: Path, entry: AdviceEntry, *, file_path: str, blob_sha: str) -> bool:
    """Save ``entry`` at ``path`` atomically. Returns False (and logs a warning) when it cannot be written."""
    from tdp_core.advisor.advisor import PROMPT_VERSION

    document = {
        "schema": CACHE_SCHEMA,
        "model": entry.model,
        "prompt_version": PROMPT_VERSION,
        "path": file_path,
        "blob_sha": blob_sha,
        "created_at": _utc_text(entry.created_at),
        "advice": entry.advice.model_dump(mode="json"),
    }
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
        return True
    except OSError as exc:
        logger.warning("could not save the advice to %s: %s", path, exc)
        return False
    finally:
        with contextlib.suppress(OSError):  # gone after the rename, or its folder never existed
            temp.unlink(missing_ok=True)


def read_source(loaded: LoadedAnalysis, path: str, blob_sha: str) -> tuple[str, FileSyntax]:
    """The analysed content of ``path`` (decoded as UTF-8 with replacements) and its parsed syntax.

    Raises :class:`CliError` when the blob cannot be read from the repository or the file type has no parser.
    """
    from tdp_core.mining.git import GitError
    from tdp_core.mining.tree import read_blobs
    from tdp_core.parsing import get_plugin

    try:
        blob = read_blobs(Path(loaded.meta.repo_dir), [blob_sha])[blob_sha]
    except (GitError, OSError, ValueError, KeyError) as exc:
        raise CliError(
            f"Cannot read {clean(path)} from {clean(loaded.meta.repo_dir)}",
            hint="The repository may have moved; run /analyze on it again.",
        ) from exc
    plugin = get_plugin(path)
    if plugin is None:
        raise CliError(f"{clean(path)} is not a file Entropiq can parse")
    return blob.decode("utf-8", errors="replace"), plugin.parse(path, blob)


def request_advice(
    session: Session,
    *,
    path: str,
    source: str,
    syntax: FileSyntax,
    score: FileScore,
    extra: FileExtra,
    model: str,
    key: str,
) -> AdviceEntry:
    """Ask the model for advice on one file (with a spinner) and return it as a fresh entry.

    Raises :class:`CliError` when the client cannot be built, the request fails (with a retry hint when it is
    worth trying again) or no valid advice came back. Ctrl+C propagates.
    """
    from tdp_core.advisor.advisor import Advisor
    from tdp_core.advisor.client import LLMError
    from tdp_core.advisor.schema import AdviceError

    try:
        client = client_factory(model, key)
    except LLMError as exc:
        message = str(exc)
        hint = INSTALL_HINT if "not installed" in message else None
        raise CliError(f"AI advice is unavailable: {clean(message)}", hint=hint) from exc
    try:
        with activity(session.err_console, f"Asking {model} for fix advice on {clean(path)}"):
            advice = Advisor(client).advise(path=path, source=source, syntax=syntax, score=score, extra=extra)
    except LLMError as exc:
        raise CliError(
            f"The advice request failed: {clean(str(exc))}", hint=RETRY_HINT if exc.retryable else None
        ) from exc
    except AdviceError as exc:
        raise CliError(f"No usable advice for {clean(path)}: {clean(str(exc))}") from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return AdviceEntry(advice, model, datetime.now(UTC).replace(microsecond=0))


def get_advice(
    session: Session,
    loaded: LoadedAnalysis,
    score: FileScore,
    *,
    refresh: bool = False,
    environ: Mapping[str, str] | None = None,
) -> AdviceEntry | None:
    """Saved advice for ``score.path`` (unless ``refresh``), else fresh advice; None when no key is set.

    Saved advice is looked up before the key, so it shows without a key. Fresh advice is saved for next time.
    """
    extra = loaded.result.extras.get(score.path)
    if extra is None:
        raise CliError(f"{clean(score.path)} has no saved details; run /analyze again")
    model = DEFAULT_SETTINGS.advisor_model
    file = cache_path(advice_dir(session.home), extra.blob_sha, model)
    if not refresh:
        hit = read_cached(file)
        if hit is not None:
            return hit
    key = api_key(environ)
    if key is None:
        return None
    source, syntax = read_source(loaded, score.path, extra.blob_sha)
    entry = request_advice(
        session, path=score.path, source=source, syntax=syntax, score=score, extra=extra, model=model, key=key
    )
    save_cached(file, entry, file_path=score.path, blob_sha=extra.blob_sha)
    return entry


def _dropped_line(n: int) -> str:
    if n == 1:
        return "1 suggested step was dropped because it named code that is not in this file."
    return f"{n} suggested steps were dropped because they named code that is not in this file."


def _bullets(title: str, items: Sequence[str], glyphs: Glyphs) -> list[Text]:
    if not items:
        return []
    return [Text(""), Text(title, style="bold"), *(Text(f"{glyphs.bullet} {clean(item)}") for item in items)]


def advice_view(path: str, entry: AdviceEntry, *, glyphs: Glyphs = UNICODE_GLYPHS) -> Panel:
    """The advice as a panel: summary, numbered steps with targets and effort, tests, risks and confidence.

    Every string from the model goes through ``clean`` into ``Text`` (never markup). The footer labels the
    content as AI suggestions to review and says when saved advice is shown.
    """
    advice = entry.advice
    parts: list[Text] = [Text(clean(advice.summary)), Text(""), Text("Steps", style="bold")]
    if advice.steps:
        for step in advice.steps:
            parts.append(Text(f"{step.order}. {clean(step.action)}"))
            low, high = step.effort_hours
            targets = ", ".join(clean(t) for t in step.targets)
            parts.append(Text(f"   Targets: {targets}   Effort: {format_hour_range(low, high)}", style="dim"))
    else:
        parts.append(Text(ALL_DROPPED if advice.dropped_steps > 0 else NO_STEPS, style="yellow"))
    parts += _bullets("Tests to add", advice.tests_to_add, glyphs)
    parts += _bullets("Risks", advice.risks, glyphs)
    parts += [Text(""), Text(f"Confidence: {advice.confidence}")]
    if advice.dropped_steps > 0:
        parts.append(Text(_dropped_line(advice.dropped_steps), style="dim"))
    footer = f"AI suggestions from {clean(entry.model)}: review them before you change any code."
    if entry.cached:
        saved = entry.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
        footer += f" Saved {saved}; add --refresh to ask again."
    parts.append(Text(footer, style="dim italic"))
    return make_panel(Group(*parts), Text(f"AI suggestions for {clean(path)}"))


def disabled_view(model: str, *, command: str = "/advise") -> Panel:
    """The panel shown when no API key is set: what the feature does, how to turn it on, and what is sent."""
    lines = [
        f"Entropiq can ask Claude ({model}) for a short, step-by-step fix plan for one file.",
        f"To turn it on, set {ENV_API_KEY} to an Anthropic API key, then run {command} again:",
        f'  PowerShell   $env:{ENV_API_KEY} = "<your key>"',
        f"  bash or zsh  export {ENV_API_KEY}=<your key>",
        "The request sends the file's scores, reasons and up to "
        f"{DEFAULT_SETTINGS.advisor_max_excerpt_lines} lines of its code to Anthropic.",
        "Create a key at https://console.anthropic.com/settings/keys",
    ]
    return make_panel(Group(*(Text(line) for line in lines)), Text("AI fix advice is off"))


def advice_json(loaded: LoadedAnalysis, path: str, blob_sha: str, entry: AdviceEntry) -> dict[str, Any]:
    """The advice as a JSON-ready dict (schema ``entropiq.advice.v1``)."""
    return {
        "schema": ADVICE_SCHEMA,
        "repo": loaded.meta.repo,
        "sha": loaded.meta.sha,
        "path": path,
        "blob_sha": blob_sha,
        "model": entry.model,
        "cached": entry.cached,
        "created_at": _utc_text(entry.created_at),
        "advice": entry.advice.model_dump(mode="json"),
    }


def advise_handler(session: Session, args: Sequence[str]) -> None:
    """``/advise <path> [--refresh]``."""
    parsed = parse_args(args, flags=("--refresh",), max_positionals=1)
    if not parsed.positionals:
        raise UsageError(NO_PATH)
    loaded = session.require_analysis()
    score = resolve_file(loaded, parsed.positionals[0])
    entry = get_advice(session, loaded, score, refresh=parsed.flag("--refresh"))
    if entry is None:
        session.console.print(disabled_view(DEFAULT_SETTINGS.advisor_model))
        return
    session.console.print(advice_view(score.path, entry, glyphs=session.glyphs))


def _complete(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    if word.startswith("-"):
        yield from _OPTIONS(session, args, word)
        return
    if not any(not a.startswith("-") for a in args):
        yield from complete_analysis_paths(session, args, word)


def advise_cli(
    ctx: typer.Context,
    path: FileArgument,
    target: TargetArgument = ".",
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ask the model again instead of showing saved advice.")
    ] = False,
    output_format: Annotated[
        Literal["text", "json"], typer.Option("--format", "-f", help="Output format.")
    ] = "text",
    ref: RefOption = None,
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Get AI fix advice for one file (needs ANTHROPIC_API_KEY)."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        score = resolve_file(loaded, path)
        entry = get_advice(session, loaded, score, refresh=refresh)
    if entry is None:
        session.err_console.print(disabled_view(DEFAULT_SETTINGS.advisor_model, command="tdp advise"))
        raise typer.Exit(1)
    if output_format == "json":
        blob_sha = loaded.result.extras[score.path].blob_sha
        document = advice_json(loaded, score.path, blob_sha, entry)
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
    else:
        session.console.print(advice_view(score.path, entry, glyphs=session.glyphs))


def register(registry: CommandRegistry) -> None:
    """Register ``/advise`` and ``tdp advise``."""
    registry.add(
        Command(
            name="advise",
            aliases=("advice",),
            group="Plan",
            help="Get AI fix advice for one file (needs ANTHROPIC_API_KEY)",
            usage=USAGE,
            details=(
                "Name the file by its path, a unique ending of it or its rank. Asks Claude "
                f"({DEFAULT_SETTINGS.advisor_model}) for a short fix plan and keeps only the steps that name "
                "functions of that file. Answers are saved in the advice folder of the Entropiq home and "
                "shown again for the same file content; --refresh asks again. The request sends the file's "
                f"scores, reasons and up to {DEFAULT_SETTINGS.advisor_max_excerpt_lines} lines of its code "
                "to Anthropic. Without ANTHROPIC_API_KEY the command explains how to turn it on."
            ),
            handler=advise_handler,
            complete=_complete,
            needs_analysis=True,
            cli=advise_cli,
        )
    )
