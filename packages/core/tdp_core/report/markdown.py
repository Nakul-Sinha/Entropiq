"""Markdown reports: the ranking table (``to_markdown``) and the budgeted fix plan (``plan_to_markdown``).

The output is GitHub Flavored Markdown meant for PR comments and job summaries. Repository paths are
untrusted: they go into code spans, free text is backslash-escaped, and characters that are not printable
(control, bidi, zero-width, unusual spaces, lone surrogates) are written as visible escapes such as ``\\n``
or ``\\u202e``, so a crafted file name can neither break the table nor hide text.

A full report is ``to_markdown(result) + "\\n" + plan_to_markdown(result.plan)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Final

from tdp_core.contracts import AnalysisResult, Plan
from tdp_core.report.common import format_percent, ranked_files, summarize

DEFAULT_TOP: Final = 20
_TEXT_SPECIALS: Final = frozenset("\\`*_[]<>|~&$")
_BACKTICK_RUN: Final = re.compile(r"`+")


def _visible(text: str) -> str:
    """``text`` with every non-printable character replaced by its Python escape (``\\t``, ``\\u202e``)."""
    return "".join(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii") for ch in text)


def code_span(text: str, *, in_table: bool = True) -> str:
    """``text`` as one inline code span whose rendered content is exactly the visible form of ``text``.

    Non-printable characters become visible escapes first (a newline shows as the two characters ``\\n``).
    The fence is one backtick longer than the longest backtick run inside, and one space is added on each
    side when the content starts or ends with a backtick, or starts and ends with a space without being all
    spaces (CommonMark strips exactly one such pair). With ``in_table=True`` (the default) every ``|`` is
    written as ``\\|``, because GFM splits table cells on bare pipes even inside code spans and removes the
    backslash again; pass ``in_table=False`` for a code span outside a table. Empty input gives ``""``.
    """
    shown = _visible(text)
    if not shown:
        return ""
    longest = max((len(run) for run in _BACKTICK_RUN.findall(shown)), default=0)
    fence = "`" * (longest + 1)
    inner = shown.replace("|", "\\|") if in_table else shown
    spaced = shown.startswith(" ") and shown.endswith(" ") and shown.strip(" ") != ""
    if shown.startswith("`") or shown.endswith("`") or spaced:
        inner = f" {inner} "
    return f"{fence}{inner}{fence}"


def escape_text(text: str) -> str:
    """``text`` as plain inline Markdown text: it renders as the visible form of ``text``, with no markup.

    Non-printable characters become visible escapes, then a backslash goes before each of
    ``\\ ` * _ [ ] < > | ~ & $``, which covers emphasis, code, links, HTML, entities, strikethrough, math and
    table pipes. Safe in a table cell and inside a line. It does not neutralise block syntax at the start of
    a line (``#``, ``>``, list markers), nor GitHub mentions, issue references or bare URL autolinks.
    """
    return "".join(f"\\{ch}" if ch in _TEXT_SPECIALS else ch for ch in _visible(text))


def _count(k: int, noun: str) -> str:
    return f"{k} {noun}" if k == 1 else f"{k} {noun}s"


def _hours(value: float) -> str:
    return f"{value:.1f}" if math.isfinite(value) else "n/a"


def _utc_minute(moment: datetime) -> str:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("analyzed_at must be timezone-aware")
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def to_markdown(
    result: AnalysisResult, top: int | None = DEFAULT_TOP, only_paths: Iterable[str] | None = None
) -> str:
    """The ranking report: a heading, the commit and model, a summary sentence and a table of files.

    Files are listed by priority descending, then path. ``top`` caps the table rows (``None`` lists every
    file); when rows are cut a footer says how many were shown. ``only_paths`` switches to selected mode, used
    for the files a pull request changes: only files whose path equals one of the given repo-relative POSIX
    paths are summarised and listed, and paths that were not ranked are ignored. The table has the columns
    File, Priority (with band), Risk (12 mo, the calibrated probability as a percentage) and Top reason
    (``-`` when a file has no reasons); there is no table when no file is listed. The text ends with exactly
    one newline and has no trailing spaces and no carriage returns.

    Raises ``ValueError`` when ``top`` is below 1 or ``analyzed_at`` is naive, and ``TypeError`` when
    ``only_paths`` is a single string.
    """
    if top is not None and top < 1:
        raise ValueError("top must be at least 1, or None for every file")
    if isinstance(only_paths, str):
        raise TypeError("only_paths must be a collection of paths, not a single string")
    ranked = ranked_files(result.files)
    total = len(ranked)
    if only_paths is None:
        listed = ranked
    else:
        wanted = frozenset(only_paths)
        listed = [f for f in ranked if f.path in wanted]
    summary = summarize(listed, result.extras)

    blocks = [
        "## Technical debt report",
        f"Commit {code_span(result.sha[:12], in_table=False)}, model "
        f"{code_span(result.model_version, in_table=False)}, analysed {_utc_minute(result.analyzed_at)} UTC.",
    ]
    counts = (
        f"{summary['critical']} Critical, {summary['high']} High, "
        f"{summary['medium']} Medium, {summary['low']} Low"
    )
    if not listed:
        sentence = "No files were ranked." if total == 0 else "None of the selected files were ranked."
    elif only_paths is None:
        sentence = f"{_count(total, 'file')} ranked: {counts}."
    else:
        sentence = f"This report covers {len(listed)} of {_count(total, 'ranked file')}: {counts}."
    if summary["critical"] > 0 and summary["critical_hours"] > 0:
        sentence += (
            f" Fixing {_count(summary['critical'], 'Critical file')} is estimated at "
            f"{summary['critical_hours']:.1f} hours."
        )
    blocks.append(sentence)

    if listed:
        shown = listed if top is None else listed[:top]
        lines = ["| File | Priority | Risk (12 mo) | Top reason |", "|---|---|---|---|"]
        for f in shown:
            reason = escape_text(f.reasons[0].text.strip()) if f.reasons else ""
            lines.append(
                f"| {code_span(f.path)} | {f.priority} {escape_text(f.band)} | "
                f"{format_percent(f.risk)} | {reason or '-'} |"
            )
        blocks.append("\n".join(lines))
        if len(shown) < len(listed):
            blocks.append(f"Showing the top {len(shown)} of {len(listed)} files.")
    return "\n\n".join(blocks) + "\n"


def plan_to_markdown(plan: Plan) -> str:
    """The budgeted fix plan: a heading, a sentence with the budget and total hours, and a table of steps.

    Steps are listed by ``order``, then path, with their estimated hours (1 decimal; ``n/a`` when not finite)
    and the scheduler's reason (``-`` when empty). An empty plan gives the heading and one sentence. The text
    ends with exactly one newline.
    """
    steps = sorted(plan.steps, key=lambda s: (s.order, s.path))
    blocks = ["### Fix plan"]
    if not steps:
        blocks.append(f"No files were chosen within the budget of {_hours(plan.budget_hours)} hours.")
    else:
        blocks.append(
            f"{_count(len(steps), 'file')} chosen within the budget of {_hours(plan.budget_hours)} hours, "
            f"{_hours(plan.total_hours)} hours in total."
        )
        lines = ["| Step | File | Hours | Reason |", "|---|---|---|---|"]
        for step in steps:
            reason = escape_text(step.reason.strip())
            lines.append(
                f"| {step.order} | {code_span(step.path)} | {_hours(step.cost.hours)} | {reason or '-'} |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
