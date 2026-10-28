from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.contracts import AnalysisResult, CostEstimate, FileScore, Plan, PlanStep, Reason
from tdp_core.report.markdown import (
    DEFAULT_TOP,
    _visible,
    code_span,
    escape_text,
    plan_to_markdown,
    to_markdown,
)

MakeScore = Callable[..., FileScore]
MakeResult = Callable[..., AnalysisResult]
Golden = Callable[[str, str], None]

NASTY_PATHS = (
    "a|b.py",
    "pipe\\|escaped.py",
    "back`tick.py",
    "``double``.py",
    "`leading.py",
    "trailing`",
    "tab\there.py",
    "new\nline.py",
    "carriage\r\nreturn.py",
    "<b>bold</b>.py",
    "*a*.py",
    "__x__.py",
    "[link](http://example.com).py",
    "rtl\u202eyp.exe",
    "zero\u200bwidth.py",
    "nbsp\xa0name.py",
    " lead and trail .py ",
    "  ",
    "\\",
    "ends with backslash\\",
    "&amp;.py",
    "~~strike~~.py",
    "$x$.py",
    "#hash.py",
)


def reason(text: str) -> Reason:
    return Reason("S", "sloc", 1.0, 50.0, 0.1, text)


def check_layout(text: str) -> None:
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert "\r" not in text
    assert "\u2013" not in text
    assert "\u2014" not in text
    for line in text.split("\n"):
        assert line == line.rstrip(), repr(line)


def table_rows(text: str) -> list[list[str]]:
    return [line.split(" | ") for line in text.split("\n") if line.startswith("| ")]


def test_golden_report(sample_result: AnalysisResult, golden: Golden) -> None:
    text = to_markdown(sample_result)
    golden("report.md", text)
    check_layout(text)


def test_golden_selected_report(sample_result: AnalysisResult, golden: Golden) -> None:
    text = to_markdown(sample_result, only_paths=["setup.py", "src/billing/invoice.py", "docs/README.md"])
    golden("report_selected.md", text)
    assert "This report covers 2 of 5 ranked files: 0 Critical, 1 High, 0 Medium, 1 Low." in text
    assert "Fixing" not in text
    check_layout(text)


def test_golden_plan(sample_result: AnalysisResult, golden: Golden) -> None:
    text = plan_to_markdown(sample_result.plan)
    golden("plan.md", text)
    check_layout(text)


def test_full_report_composition(sample_result: AnalysisResult) -> None:
    text = to_markdown(sample_result) + "\n" + plan_to_markdown(sample_result.plan)
    assert text.startswith("## Technical debt report\n\n")
    assert "\n\n### Fix plan\n\n" in text
    check_layout(text)


def test_top_cuts_rows(sample_result: AnalysisResult) -> None:
    text = to_markdown(sample_result, top=2)
    rows = table_rows(text)
    assert len(rows) == 3
    assert rows[1][0] == "| `src/payments/processor.py`"
    assert rows[2][0] == "| `src/billing/invoice.py`"
    assert text.endswith("\n\nShowing the top 2 of 5 files.\n")
    assert "5 files ranked" in text
    assert "Showing" not in to_markdown(sample_result, top=5)


def test_default_top_and_every_file(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [make_score(f"f{i:02d}.py", 100 - i) for i in range(25)]
    result = make_result(files)
    default = to_markdown(result)
    assert DEFAULT_TOP == 20
    assert len(table_rows(default)) == 21
    assert default.endswith("Showing the top 20 of 25 files.\n")
    every = to_markdown(result, top=None)
    assert len(table_rows(every)) == 26
    assert "Showing" not in every
    check_layout(default)
    check_layout(every)


@pytest.mark.parametrize("top", [0, -1])
def test_top_must_be_positive(sample_result: AnalysisResult, top: int) -> None:
    with pytest.raises(ValueError, match="top must be at least 1"):
        to_markdown(sample_result, top=top)


def test_only_paths_must_not_be_a_string(sample_result: AnalysisResult) -> None:
    with pytest.raises(TypeError, match="not a single string"):
        to_markdown(sample_result, only_paths="setup.py")


def test_only_paths_selection(sample_result: AnalysisResult) -> None:
    text = to_markdown(sample_result, only_paths=iter(["src/payments/processor.py", "SETUP.PY", "setup.py/"]))
    assert "This report covers 1 of 5 ranked files: 1 Critical, 0 High, 0 Medium, 0 Low." in text
    assert "Fixing 1 Critical file is estimated at 12.5 hours." in text
    assert len(table_rows(text)) == 2
    none = to_markdown(sample_result, only_paths=set())
    assert "None of the selected files were ranked." in none
    assert "| File |" not in none
    unknown = to_markdown(sample_result, only_paths=["docs/README.md"])
    assert "None of the selected files were ranked." in unknown
    check_layout(none)


def test_only_paths_with_top(sample_result: AnalysisResult) -> None:
    paths = ["setup.py", "src/pkg/__init__.py", "src/legacy/old utils.py"]
    text = to_markdown(sample_result, top=1, only_paths=paths)
    assert "This report covers 3 of 5 ranked files" in text
    assert text.endswith("Showing the top 1 of 3 files.\n")


def test_empty_result(make_result: MakeResult) -> None:
    result = make_result()
    for text in (to_markdown(result), to_markdown(result, only_paths=["a.py"])):
        assert "No files were ranked." in text
        assert "| File |" not in text
        check_layout(text)


def test_singular_counts(make_score: MakeScore, make_result: MakeResult) -> None:
    text = to_markdown(make_result([make_score("a.py", 40)]))
    assert "1 file ranked: 0 Critical, 0 High, 0 Medium, 1 Low." in text
    selected = to_markdown(make_result([make_score("a.py", 40)]), only_paths=["a.py"])
    assert "This report covers 1 of 1 ranked file: " in selected


def test_critical_hours_sentence(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [make_score("a.py", 95), make_score("b.py", 92)]
    assert "Fixing" not in to_markdown(make_result(files))


def test_reversed_input_is_sorted(sample_result: AnalysisResult) -> None:
    shuffled = dataclasses.replace(sample_result, files=tuple(reversed(sample_result.files)))
    assert to_markdown(shuffled) == to_markdown(sample_result)


def test_analyzed_at_is_converted_to_utc(sample_result: AnalysisResult) -> None:
    local = datetime(2026, 1, 15, 15, 0, 59, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    text = to_markdown(dataclasses.replace(sample_result, analyzed_at=local))
    assert "analysed 2026-01-15 09:30 UTC." in text


def test_naive_analyzed_at_is_rejected(sample_result: AnalysisResult) -> None:
    naive = dataclasses.replace(sample_result, analyzed_at=datetime(2026, 1, 15, 9, 30))
    with pytest.raises(ValueError, match="timezone-aware"):
        to_markdown(naive)


def test_top_reason_cell(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [
        make_score("a.py", 80, reasons=(reason("  padded *bold* reason  "), reason("second"))),
        make_score("b.py", 70, reasons=(reason("   "),)),
        make_score("c.py", 60, risk=float("nan")),
    ]
    rows = table_rows(to_markdown(make_result(files)))
    assert rows[1] == ["| `a.py`", "80 High", "50%", "padded \\*bold\\* reason |"]
    assert rows[2][3] == "- |"
    assert rows[3][2] == "n/a"


def test_meta_line_code_spans(sample_result: AnalysisResult) -> None:
    result = dataclasses.replace(sample_result, sha="ab|cd`ef\n", model_version="m|v")
    text = to_markdown(result)
    assert "Commit ``ab|cd`ef\\n``, model `m|v`, analysed" in text


def test_empty_plan() -> None:
    plan = Plan(budget_hours=4.0, steps=(), total_benefit=0.0, total_hours=0.0)
    text = plan_to_markdown(plan)
    assert text == "### Fix plan\n\nNo files were chosen within the budget of 4.0 hours.\n"


def test_plan_steps_are_sorted_and_hours_formatted() -> None:
    cost = CostEstimate(hours=1.25, low=1.0, high=1.5, breakdown={})
    bad = CostEstimate(hours=float("inf"), low=0.0, high=0.0, breakdown={})
    steps = (
        PlanStep(order=2, path="b.py", benefit=1.0, cost=bad, roi=0.0, group=2, reason=""),
        PlanStep(order=1, path="z|y.py", benefit=1.0, cost=cost, roi=0.8, group=1, reason="a <b> reason"),
    )
    text = plan_to_markdown(Plan(budget_hours=10.0, steps=steps, total_benefit=2.0, total_hours=float("nan")))
    assert "2 files chosen within the budget of 10.0 hours, n/a hours in total." in text
    rows = table_rows(text)
    assert rows[1] == ["| 1", "`z\\|y.py`", "1.2", "a \\<b\\> reason |"]
    assert rows[2] == ["| 2", "`b.py`", "n/a", "- |"]
    one = plan_to_markdown(Plan(budget_hours=10.0, steps=steps[1:], total_benefit=1.0, total_hours=1.25))
    assert "1 file chosen within the budget of 10.0 hours, 1.2 hours in total." in one
    check_layout(text)


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("a.py", "`a.py`"),
        ("a`b", "``a`b``"),
        ("`a", "`` `a ``"),
        ("a`", "`` a` ``"),
        ("a``b`c", "```a``b`c```"),
        ("a|b", "`a\\|b`"),
        ("a\\|b", "`a\\\\|b`"),
        ("  ", "`  `"),
        (" a ", "`  a  `"),
        (" a", "` a`"),
        ("a\nb", "`a\\nb`"),
        ("x\u202ey", "`x\\u202ey`"),
        ("", ""),
    ],
)
def test_code_span(text: str, span: str) -> None:
    assert code_span(text) == span


def test_code_span_outside_tables() -> None:
    assert code_span("a|b", in_table=False) == "`a|b`"
    assert code_span("`a|", in_table=False) == "`` `a| ``"


@pytest.mark.parametrize(
    ("text", "escaped"),
    [
        ("plain text (top 1% in repo)", "plain text (top 1% in repo)"),
        ("recency-weighted", "recency-weighted"),
        ("a*b_c", "a\\*b\\_c"),
        ("[x](y)", "\\[x\\](y)"),
        ("<b>&amp;", "\\<b\\>\\&amp;"),
        ("a|b", "a\\|b"),
        ("back\\slash", "back\\\\slash"),
        ("`code`", "\\`code\\`"),
        ("~~s~~ $m$", "\\~\\~s\\~\\~ \\$m\\$"),
        ("tab\tnew\n", "tab\\\\tnew\\\\n"),
        ("", ""),
    ],
)
def test_escape_text(text: str, escaped: str) -> None:
    assert escape_text(text) == escaped


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("a\tb", "a\\tb"),
        ("\n", "\\n"),
        ("\r", "\\r"),
        ("\x00", "\\x00"),
        ("\x7f", "\\x7f"),
        ("\xa0", "\\xa0"),
        ("\u200b", "\\u200b"),
        ("\u202e", "\\u202e"),
        ("\u2028", "\\u2028"),
        ("\udc80", "\\udc80"),
        ("caf\xe9 \u4e2d", "caf\xe9 \u4e2d"),
        ("\U0001f600", "\U0001f600"),
    ],
)
def test_visible(text: str, shown: str) -> None:
    assert _visible(text) == shown


def test_nasty_paths_keep_the_layout(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [make_score(path, 90 - i, reasons=(reason(path),)) for i, path in enumerate(NASTY_PATHS)]
    result = make_result(files)
    check_layout(to_markdown(result, top=None))
    steps = tuple(
        PlanStep(
            order=i + 1,
            path=path,
            benefit=1.0,
            cost=CostEstimate(hours=1.0, low=0.8, high=1.2, breakdown={}),
            roi=1.0,
            group=i,
            reason=path,
        )
        for i, path in enumerate(NASTY_PATHS)
    )
    check_layout(plan_to_markdown(Plan(budget_hours=40.0, steps=steps, total_benefit=1.0, total_hours=1.0)))


def parser() -> Any:
    """GFM oracle: markdown-it-py with the table and strikethrough rules splits cells like cmark-gfm."""
    markdown_it = pytest.importorskip("markdown_it")
    return markdown_it.MarkdownIt("commonmark").enable(["table", "strikethrough"])


def parsed_cells(text: str) -> list[list[Any]]:
    """Inline tokens of every body cell, row by row."""
    rows: list[list[Any]] = []
    in_cell = False
    for token in parser().parse(text):
        if token.type == "tr_open":
            rows.append([])
        elif token.type in ("td_open", "th_open"):
            in_cell = True
        elif token.type in ("td_close", "th_close"):
            in_cell = False
        elif token.type == "inline" and in_cell:
            rows[-1].append(token)
    return rows[1:]


def is_code(cell: Any, content: str) -> bool:
    children = cell.children or []
    return len(children) == 1 and children[0].type == "code_inline" and children[0].content == content


def plain_text(cell: Any) -> str | None:
    children = cell.children or []
    if any(child.type != "text" for child in children):
        return None
    return "".join(child.content for child in children)


def test_gfm_parses_nasty_paths_into_the_right_cells(make_score: MakeScore, make_result: MakeResult) -> None:
    files = [make_score(path, 90 - i, reasons=(reason(path),)) for i, path in enumerate(NASTY_PATHS)]
    ranked = sorted(files, key=lambda f: (-f.priority, f.path))
    rows = parsed_cells(to_markdown(make_result(files), top=None))
    assert len(rows) == len(NASTY_PATHS)
    for row, score in zip(rows, ranked, strict=True):
        assert len(row) == 4
        assert is_code(row[0], _visible(score.path)), (score.path, row[0].content)
        assert plain_text(row[1]) == f"{score.priority} {score.band}"
        assert plain_text(row[2]) == "50%"
        expected = _visible(score.path.strip()).strip(" ") or "-"
        assert plain_text(row[3]) == expected, (score.path, row[3].content)


def test_gfm_parses_the_plan_cells() -> None:
    steps = tuple(
        PlanStep(
            order=i + 1,
            path=path,
            benefit=1.0,
            cost=CostEstimate(hours=2.0, low=1.6, high=2.4, breakdown={}),
            roi=0.5,
            group=i,
            reason=f"fix before {path}",
        )
        for i, path in enumerate(NASTY_PATHS)
    )
    rows = parsed_cells(
        plan_to_markdown(Plan(budget_hours=40.0, steps=steps, total_benefit=1.0, total_hours=2.0))
    )
    assert len(rows) == len(NASTY_PATHS)
    for row, step in zip(rows, steps, strict=True):
        assert len(row) == 4
        assert plain_text(row[0]) == str(step.order)
        assert is_code(row[1], _visible(step.path))
        assert plain_text(row[2]) == "2.0"
        assert plain_text(row[3]) == _visible(step.reason.strip()).strip(" ")


def one_cell_table(cell: str) -> str:
    return f"| h |\n|---|\n| {cell} |\n"


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=1, max_size=40))
def test_gfm_code_span_property(text: str) -> None:
    rows = parsed_cells(one_cell_table(code_span(text)))
    assert len(rows) == 1
    assert len(rows[0]) == 1
    assert is_code(rows[0][0], _visible(text))


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=1, max_size=40))
def test_gfm_escape_text_property(text: str) -> None:
    escaped = escape_text(text)
    rows = parsed_cells(one_cell_table(escaped))
    assert len(rows) == 1
    assert len(rows[0]) == 1
    shown = plain_text(rows[0][0])
    assert shown == _visible(text).strip(" ")


def test_gfm_escape_text_outside_tables() -> None:
    for text in NASTY_PATHS:
        tokens = parser().parse(f"Before {escape_text(text)} after.\n")
        inline = next(token for token in tokens if token.type == "inline")
        assert plain_text(inline) == f"Before {_visible(text)} after.", text
