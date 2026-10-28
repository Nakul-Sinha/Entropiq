from __future__ import annotations

import math
from collections.abc import Callable

import pytest

import tdp_core
from tdp_core.contracts import BANDS, DEBT_TYPES, AnalysisResult, CostEstimate, FileExtra, FileScore
from tdp_core.report.common import (
    BAND_ORDER,
    HISTOGRAM_BUCKETS,
    TOOL_NAME,
    format_percent,
    ranked_files,
    summarize,
    tool_version,
)

MakeScore = Callable[..., FileScore]


def extra_with_hours(hours: float) -> FileExtra:
    return FileExtra(
        blob_sha="a" * 40,
        functions=(),
        dependents=(),
        dependencies=(),
        test_importers=(),
        commit_activity=(),
        cost=CostEstimate(hours=hours, low=0.8 * hours, high=1.2 * hours, breakdown={}),
    )


def test_constants() -> None:
    assert BAND_ORDER == tuple(name for name, _ in BANDS) == ("Critical", "High", "Medium", "Low")
    assert TOOL_NAME == "tdp"
    assert HISTOGRAM_BUCKETS == 10


def test_ranked_files_sorts_by_priority_then_path(make_score: MakeScore) -> None:
    files = [make_score("b.py", 50), make_score("c.py", 90), make_score("a.py", 50), make_score("z.py", 10)]
    ranked = ranked_files(files)
    assert [f.path for f in ranked] == ["c.py", "a.py", "b.py", "z.py"]
    assert [f.path for f in files] == ["b.py", "c.py", "a.py", "z.py"]
    assert ranked_files([]) == []


def test_ranked_files_keeps_the_sample_order(sample_result: AnalysisResult) -> None:
    assert ranked_files(reversed(sample_result.files)) == list(sample_result.files)


def test_summarize_sample(sample_result: AnalysisResult) -> None:
    summary = summarize(sample_result.files, sample_result.extras)
    assert summary == {
        "files": 5,
        "critical": 1,
        "high": 2,
        "medium": 1,
        "low": 1,
        "critical_hours": 12.5,
        "debt_types": {
            "complexity": 0,
            "duplication": 1,
            "test": 1,
            "architecture": 1,
            "knowledge": 1,
            "admitted": 1,
            "defect_hotspot": 1,
        },
        "priority_histogram": [0, 1, 0, 0, 0, 0, 1, 0, 2, 1],
    }
    assert list(summary) == [
        "files",
        "critical",
        "high",
        "medium",
        "low",
        "critical_hours",
        "debt_types",
        "priority_histogram",
    ]
    assert list(summary["debt_types"]) == list(DEBT_TYPES)


def test_summarize_histogram_edges(make_score: MakeScore) -> None:
    priorities = [0, 9, 10, 99, 100, -5, 120]
    files = [make_score(f"f{i}.py", p, band="Low") for i, p in enumerate(priorities)]
    histogram = summarize(files, {})["priority_histogram"]
    assert histogram == [3, 1, 0, 0, 0, 0, 0, 0, 0, 3]
    for p, bucket in zip(priorities, [0, 0, 1, 9, 9, 0, 9], strict=True):
        single = summarize([make_score("x.py", p, band="Low")], {})["priority_histogram"]
        assert single[bucket] == 1
        assert sum(single) == 1


def test_summarize_critical_hours(make_score: MakeScore) -> None:
    files = [
        make_score("a.py", 99),
        make_score("b.py", 95),
        make_score("c.py", 91),
        make_score("d.py", 92),
        make_score("high.py", 80),
    ]
    extras = {
        "a.py": extra_with_hours(0.1),
        "b.py": extra_with_hours(0.2),
        "d.py": extra_with_hours(math.nan),
        "high.py": extra_with_hours(100.0),
    }
    summary = summarize(files, extras)
    assert summary["critical"] == 4
    assert summary["high"] == 1
    assert summary["critical_hours"] == 0.3
    assert summarize([make_score("c.py", 91)], {})["critical_hours"] == 0.0


def test_summarize_unknown_values(make_score: MakeScore) -> None:
    files = [
        make_score("a.py", 60, debt_types=("zeta", "complexity")),
        make_score("b.py", 60, debt_types=("alpha", "zeta")),
        make_score("c.py", 60, band="Unknown"),
    ]
    summary = summarize(files, {})
    assert list(summary["debt_types"]) == [*DEBT_TYPES, "alpha", "zeta"]
    assert summary["debt_types"]["zeta"] == 2
    assert summary["debt_types"]["alpha"] == 1
    assert summary["debt_types"]["complexity"] == 1
    assert summary["files"] == 3
    assert summary["medium"] == 2
    assert summary["critical"] + summary["high"] + summary["low"] == 0


def test_summarize_counts_a_debt_type_once_per_file(make_score: MakeScore) -> None:
    files = [
        make_score("a.py", 60, debt_types=("test", "test")),
        make_score("b.py", 60, debt_types=("zeta", "zeta", "test")),
    ]
    debt = summarize(files, {})["debt_types"]
    assert debt["test"] == 2
    assert debt["zeta"] == 1


def test_summarize_empty() -> None:
    summary = summarize([], {})
    assert summary["files"] == 0
    assert summary["critical"] == summary["high"] == summary["medium"] == summary["low"] == 0
    assert summary["critical_hours"] == 0.0
    assert summary["debt_types"] == dict.fromkeys(DEBT_TYPES, 0)
    assert summary["priority_histogram"] == [0] * 10


def test_summarize_accepts_a_generator(sample_result: AnalysisResult) -> None:
    summary = summarize((f for f in sample_result.files), sample_result.extras)
    assert summary["files"] == 5


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.64, "64%"),
        (0.0, "0%"),
        (1.0, "100%"),
        (0.035, "4%"),
        (0.004, "0%"),
        (math.nan, "n/a"),
        (math.inf, "n/a"),
        (-math.inf, "n/a"),
    ],
)
def test_format_percent(value: float, text: str) -> None:
    assert format_percent(value) == text


def test_tool_version_follows_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tool_version() == tdp_core.__version__
    monkeypatch.setattr(tdp_core, "__version__", "9.9.9")
    assert tool_version() == "9.9.9"
