"""``/diff`` (plus ``tdp diff``): the ranked files a branch changed since a base commit.

The range is ``BASE...HEAD`` like ``git diff``: the changes from the merge base of BASE and the analysed
commit to the analysed commit, with renames detected. The priorities shown are the analysed commit's, so
``tdp diff`` analyses ``--ref`` (default HEAD) first. All git calls go through ``run_git`` with external diff
drivers and text conversion disabled. ``tdp diff --format json`` and ``--format md`` serve the GitHub Action.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

import typer
from rich.console import Group, RenderableType
from rich.text import Text

from tdp_cli.args import parse_args
from tdp_cli.commands._views import (
    NO_FILES,
    RankedFile,
    file_row,
    json_header,
    limited,
    parse_count,
    ranked_by_path,
    ranked_table,
)
from tdp_cli.completion import options_completer
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
from tdp_cli.theme import BAND_ORDER, UNICODE_GLYPHS, Glyphs, clean, short_sha

if TYPE_CHECKING:
    from tdp_cli.session import Session
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import AnalysisResult

DIFF_SCHEMA: Final = "entropiq.diff.v1"
CHANGE_WORDS: Final[dict[str, str]] = {
    "A": "added",
    "C": "copied",
    "D": "deleted",
    "M": "modified",
    "R": "renamed",
    "T": "type changed",
}
"""Words for the ``git diff --name-status`` letters; any other letter counts as ``modified``."""
MAX_UNRANKED_SHOWN: Final = 10
DIFF_OPTIONS: Final = ("--base", "--top")
NO_BASE: Final = "say which commit to compare with: --base REV (a branch, tag or commit)"
_OPTIONS = options_completer(*DIFF_OPTIONS)


@dataclass(frozen=True, slots=True)
class ChangedFile:
    """One changed path: its path at the head (the removed path for a deletion), the change word, and the
    path before a rename or copy."""

    path: str
    change: str
    old_path: str | None = None


@dataclass(frozen=True, slots=True)
class DiffRange:
    """``base...head``: the base as typed, the full shas, and the changed files sorted by path."""

    base: str
    base_sha: str
    merge_base: str
    head_sha: str
    changes: tuple[ChangedFile, ...]


def parse_name_status(raw: bytes) -> list[ChangedFile]:
    """Parse ``git diff --name-status -z`` output, sorted by path.

    Records are ``<status>\\0<path>\\0`` or, for renames and copies, ``<R|C><score>\\0<old>\\0<new>\\0``.
    Paths are decoded as UTF-8 with replacement characters, like ``tdp_core.mining.tree.list_files``, so
    they match ranked paths. A truncated last record is ignored.
    """
    tokens = raw.decode("utf-8", errors="replace").split("\x00")
    out: list[ChangedFile] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        i += 1
        if not status:
            continue
        letter = status[0]
        if letter in ("R", "C"):
            if i + 1 >= len(tokens) or not tokens[i + 1]:
                break
            out.append(ChangedFile(tokens[i + 1], CHANGE_WORDS[letter], tokens[i]))
            i += 2
        else:
            if i >= len(tokens) or not tokens[i]:
                break
            out.append(ChangedFile(tokens[i], CHANGE_WORDS.get(letter, "modified")))
            i += 1
    return sorted(out, key=lambda c: (c.path, c.change))


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return clean(line.strip())[:300]
    return "no message"


def diff_range(repo_dir: Path, base: str, head: str, *, display: str) -> DiffRange:
    """The files changed in ``base...head`` of the repository at ``repo_dir`` (``display`` names it).

    Raises :class:`UsageError` for a base git could read as an option, and :class:`CliError` when the folder
    is gone (exit code 1), the base names no commit (exit code 2), the two commits share no history (exit
    code 2), or git fails or times out (exit code 1).
    """
    from tdp_core.mining.git import GitError, TransientGitError, check_rev, run_git

    if not repo_dir.is_dir():
        raise CliError(
            f"The repository folder {clean(str(repo_dir))} no longer exists.", hint="Run /analyze again."
        )
    try:
        check_rev(base)
    except ValueError:
        raise UsageError(f"{clean(base)!r} is not a valid revision") from None
    shown_base, shown_head = clean(base), short_sha(head)
    try:
        out = run_git(
            ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{base}^{{commit}}"],
            repo_dir,
            timeout=60,
        )
        base_sha = out.decode("ascii", "replace").strip()
        if not base_sha:
            raise ValueError("empty rev-parse output")
    except (GitError, ValueError) as exc:
        raise CliError(
            f"{shown_base!r} does not name a commit in {clean(display)}",
            exit_code=2,
            hint=(
                "Pass --base with a branch, tag or commit of the analysed repository "
                "(in CI fetch the full history)."
            ),
        ) from exc
    try:
        merge_base = (
            run_git(["merge-base", base_sha, head], repo_dir, timeout=60).decode("ascii", "replace").strip()
        )
        raw = run_git(
            [
                "diff",
                "--name-status",
                "-z",
                "-M",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                merge_base,
                head,
                "--",
            ],
            repo_dir,
            timeout=120,
        )
    except TransientGitError as exc:
        raise CliError("git timed out", hint=_first_line(str(exc))) from exc
    except GitError as exc:
        if exc.returncode == 1 and not exc.stderr.strip():
            raise CliError(f"{shown_base} and {shown_head} have no common history", exit_code=2) from exc
        raise CliError(f"git failed: {_first_line(exc.stderr)}") from exc
    return DiffRange(base, base_sha, merge_base, head, tuple(parse_name_status(raw)))


def split_changes(
    result: AnalysisResult, changes: Sequence[ChangedFile]
) -> tuple[list[RankedFile], list[ChangedFile]]:
    """The changes split into ranked files (rank order) and the rest (by path): deletions, tests, skipped
    and non-Python files."""
    by_path = ranked_by_path(result)
    ranked: list[RankedFile] = []
    unranked: list[ChangedFile] = []
    for change in changes:
        item = by_path.get(change.path) if change.change != "deleted" else None
        if item is None:
            unranked.append(change)
        else:
            ranked.append(item)
    ranked.sort(key=lambda r: r.rank)
    unranked.sort(key=lambda c: (c.path, c.change))
    return ranked, unranked


def _band_counts(rows: Sequence[RankedFile]) -> dict[str, int]:
    counts = {band: 0 for band in BAND_ORDER}
    for row in rows:
        counts[row.score.band] = counts.get(row.score.band, 0) + 1
    return counts


def _dim(text: str) -> Text:
    return Text(text, style="dim")


def _changed_noun(count: int) -> str:
    return f"{count} changed file" if count == 1 else f"{count} changed files"


def diff_view(
    loaded: LoadedAnalysis, diff: DiffRange, *, top: int = 0, glyphs: Glyphs = UNICODE_GLYPHS
) -> Group:
    """The ``/diff`` screen: the ranked changed files (``top`` rows, 0 = all), a band summary and the
    changed paths that are not ranked."""
    head = short_sha(diff.head_sha)
    title = Text(f"Files changed in {clean(diff.base)}...{head}", style="bold")
    title.append(f" (merge base {short_sha(diff.merge_base)})", style="dim")
    parts: list[RenderableType] = [title]
    if not diff.changes:
        parts.append(_dim(f"No files changed between {clean(diff.base)} and {head}."))
        return Group(*parts)
    ranked, unranked = split_changes(loaded.result, diff.changes)
    changes = {c.path: c.change for c in diff.changes if c.change != "deleted"}
    total = len(diff.changes)
    if ranked:
        shown = ranked[:top] if top else ranked
        parts.append(ranked_table(shown, glyphs=glyphs, changes=changes))
        if len(shown) < len(ranked):
            parts.append(_dim(f"Showing {len(shown)} of {len(ranked)} ranked files."))
        counts = _band_counts(ranked)
        listed = ", ".join(f"{counts[band]} {band}" for band in BAND_ORDER)
        verb = "is" if len(ranked) == 1 else "are"
        parts.append(_dim(f"{len(ranked)} of {_changed_noun(total)} {verb} ranked: {listed}."))
    elif total == 1:
        parts.append(_dim("The changed file is not ranked."))
    else:
        parts.append(_dim(f"None of the {total} changed files are ranked."))
    if unranked:
        names, more = limited([f"{clean(c.path)} ({c.change})" for c in unranked], MAX_UNRANKED_SHOWN)
        line = f"Not ranked (deleted, tests, skipped or not Python): {', '.join(names)}"
        parts.append(_dim(line + (f" and {more} more" if more else "")))
    return Group(*parts)


def diff_json(loaded: LoadedAnalysis, diff: DiffRange, *, top: int = 0) -> dict[str, Any]:
    """The ``tdp diff --format json`` document (schema ``entropiq.diff.v1``).

    ``summary`` counts every ranked change; ``files`` (rank order, ``top`` rows, 0 = all) are
    ``tdp analyze`` JSON rows plus ``change`` and ``old_path``; ``unranked`` lists the other changed paths.
    """
    ranked, unranked = split_changes(loaded.result, diff.changes)
    by_path = {c.path: c for c in diff.changes if c.change != "deleted"}
    counts = _band_counts(ranked)
    shown = ranked[:top] if top else ranked
    return {
        **json_header(loaded, DIFF_SCHEMA),
        "base": diff.base,
        "base_sha": diff.base_sha,
        "merge_base": diff.merge_base,
        "head_sha": diff.head_sha,
        "summary": {
            "changed": len(diff.changes),
            "ranked": len(ranked),
            **{band.lower(): counts[band] for band in BAND_ORDER},
        },
        "files": [
            {**file_row(item), "change": by_path[item.path].change, "old_path": by_path[item.path].old_path}
            for item in shown
        ],
        "unranked": [{"path": c.path, "change": c.change, "old_path": c.old_path} for c in unranked],
    }


def diff_markdown(loaded: LoadedAnalysis, diff: DiffRange, *, top: int = 0) -> str:
    """The ``tdp diff --format md`` text: the report of the changed ranked files plus a comparison line.

    Ends with exactly one newline; ``top`` caps the table rows (0 = all).
    """
    from tdp_core.report.markdown import code_span, to_markdown

    paths = [c.path for c in diff.changes if c.change != "deleted"]
    report = to_markdown(loaded.result, top=top or None, only_paths=paths)
    ranked, _ = split_changes(loaded.result, diff.changes)
    line = (
        f"Compared with {code_span(diff.base, in_table=False)} "
        f"(merge base {code_span(diff.merge_base[:12], in_table=False)}): "
        f"{_changed_noun(len(diff.changes))}, {len(ranked)} of them ranked."
    )
    return report.rstrip("\n") + "\n\n" + line + "\n"


def _base_of(parsed_base: str | None, positionals: Sequence[str]) -> str:
    if parsed_base is not None and positionals:
        raise UsageError("give the base once")
    base = parsed_base if parsed_base is not None else (positionals[0] if positionals else None)
    if base is None or not base.strip():
        raise UsageError(NO_BASE)
    return base.strip()


def diff_handler(session: Session, args: Sequence[str]) -> None:
    """``/diff --base REV [--top N]``: the ranked files changed in ``REV...`` the analysed commit."""
    parsed = parse_args(args, values=DIFF_OPTIONS, max_positionals=1)
    base = _base_of(parsed.value("--base"), parsed.positionals)
    raw_top = parsed.value("--top")
    top = 0 if raw_top is None else parse_count(raw_top, "--top")
    loaded = session.require_analysis()
    if not loaded.result.files:
        session.console.print(_dim(NO_FILES))
        return
    meta = loaded.meta
    diff = diff_range(Path(meta.repo_dir), base, loaded.result.sha, display=meta.repo)
    session.console.print(diff_view(loaded, diff, top=top, glyphs=session.glyphs))


def complete_diff(session: Session, args: Sequence[str], word: str) -> Iterator[Suggestion]:
    """Completion for ``/diff``: its option names."""
    if args and args[-1] in DIFF_OPTIONS:
        return
    yield from _OPTIONS(session, args, word)


def diff_cli(
    ctx: typer.Context,
    base: Annotated[
        str,
        typer.Option(
            "--base",
            help="Branch, tag or commit to compare with (files changed in BASE...HEAD).",
            show_default=False,
        ),
    ],
    target: TargetArgument = ".",
    ref: RefOption = None,
    top: Annotated[int, typer.Option("--top", min=0, help="Rows to show (0 = all).")] = 0,
    output_format: Annotated[
        Literal["table", "json", "md"], typer.Option("--format", "-f", help="Output format.")
    ] = "table",
    fresh: FreshOption = False,
    model_dir: ModelDirOption = None,
    dev_model: DevModelOption = False,
) -> None:
    """Rank the files changed since a base commit."""
    session = cli_session(ctx, model_dir=model_dir, dev_model=dev_model)
    with cli_guard(session.err_console):
        from tdp_core.mining.git import check_rev

        try:
            check_rev(base.strip())
        except ValueError:
            raise UsageError(f"{clean(base)!r} is not a valid revision") from None
    loaded = cli_analysis(session, target, ref=ref, fresh=fresh)
    with cli_guard(session.err_console):
        meta = loaded.meta
        diff = diff_range(Path(meta.repo_dir), base.strip(), loaded.result.sha, display=meta.repo)
        if output_format == "json":
            sys.stdout.write(json.dumps(diff_json(loaded, diff, top=top), indent=2) + "\n")
            return
        if output_format == "md":
            sys.stdout.write(diff_markdown(loaded, diff, top=top))
            return
        session.console.print(diff_view(loaded, diff, top=top, glyphs=session.glyphs))


def register(registry: CommandRegistry) -> None:
    """Register ``/diff`` and ``tdp diff``."""
    registry.add(
        Command(
            name="diff",
            group="Report",
            help="Rank the files changed since a base commit",
            usage="/diff --base REV [--top N]",
            details=(
                "Lists the ranked files changed in REV...HEAD (as git diff does: from the merge base of REV "
                "and the analysed commit to the analysed commit, renames detected) with their priority, "
                "band, risk, cost and change, then the changed paths that are not ranked (deleted files, "
                "tests, skipped and non-Python files). REV is a branch, tag or commit such as main or "
                "HEAD~10; it may also be given without --base. --top keeps that many rows (0 = all)."
            ),
            handler=diff_handler,
            complete=complete_diff,
            needs_analysis=True,
            cli=diff_cli,
        )
    )
