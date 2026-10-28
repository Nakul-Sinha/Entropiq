"""Shared fixtures for the report writer tests: a hand-built ``AnalysisResult`` and golden file helpers.

Set ``TDP_UPDATE_GOLDEN=1`` to rewrite the golden files under ``golden/`` instead of comparing with them,
then review the diff and run the tests again without the variable.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tdp_core
from tdp_core.contracts import (
    EXTRACTOR_VERSION,
    FAMILY_ORDER,
    AnalysisResult,
    CostEstimate,
    FileExtra,
    FileScore,
    FunctionInfo,
    GraphData,
    GraphEdge,
    GraphNode,
    Plan,
    PlanStep,
    Reason,
    band_for,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
PROCESSOR = "src/payments/processor.py"
INVOICE = "src/billing/invoice.py"
OLD_UTILS = "src/legacy/old utils.py"
PKG_INIT = "src/pkg/__init__.py"
SETUP = "setup.py"
COST_KINDS = ("complexity", "duplication", "tests", "cycle", "satd")
FEATURE_ORDER = [SETUP, INVOICE, OLD_UTILS, PROCESSOR, PKG_INIT]  # sorted paths, not the ranking order
SLOC = {SETUP: 20, INVOICE: 410, OLD_UTILS: 220, PROCESSOR: 640, PKG_INIT: 35}
DEPENDENTS = {PROCESSOR: 1, INVOICE: 2}


def _shap(**values: float) -> dict[str, float]:
    out = dict.fromkeys(FAMILY_ORDER, 0.0)
    out.update(values)
    return out


def _cost(hours: float, low: float, high: float) -> CostEstimate:
    minutes = hours * 60.0
    breakdown = dict.fromkeys(COST_KINDS, 0.0)
    breakdown["complexity"] = minutes / 2
    breakdown["tests"] = minutes / 2
    return CostEstimate(hours=hours, low=low, high=high, breakdown=breakdown)


def _function(name: str, start: int, end: int, cyclomatic: int, cognitive: int) -> FunctionInfo:
    return FunctionInfo(
        name=name,
        start_line=start,
        end_line=end,
        cyclomatic=cyclomatic,
        cognitive=cognitive,
        max_nesting=2,
        params=1,
    )


def _extra(
    cost: CostEstimate,
    functions: tuple[FunctionInfo, ...] = (),
    dependents: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    commit_activity: tuple[tuple[str, int], ...] = (),
) -> FileExtra:
    return FileExtra(
        blob_sha="b" * 40,
        functions=functions,
        dependents=dependents,
        dependencies=dependencies,
        test_importers=(),
        commit_activity=commit_activity,
        cost=cost,
    )


def build_sample_result() -> AnalysisResult:
    """Five files covering every band, a priority tie, a path with a space and a file without extras."""
    processor = FileScore(
        path=PROCESSOR,
        score=2.31,
        priority=95,
        band="Critical",
        risk=0.64,
        risk_low=0.55,
        risk_high=0.72,
        reasons=(
            Reason(
                "E",
                "churn_decayed",
                831.0,
                99.0,
                0.92,
                "Very high recent churn: 831 recency-weighted lines added or deleted (top 1% in repo)",
            ),
            Reason(
                "D",
                "prior_latent_bugs",
                4.0,
                99.0,
                0.61,
                "Many past bugs: 4 past bugs traced back to its code (top 1% in repo)",
            ),
            Reason("T", "has_test_file", 0.0, 0.0, 0.33, "No test file named after it"),
            Reason(
                "S",
                "cc_max",
                31.0,
                97.0,
                0.12,
                "Very complex function: its most complex function has a cyclomatic complexity of 31 "
                "(top 3% in repo)",
            ),
        ),
        mitigations=(Reason("O", "n_authors", 2.0, 40.0, -0.2, "Few authors: changed by 2 authors"),),
        debt_types=("defect_hotspot", "test"),
        shap_by_family=_shap(S=0.12, E=0.92, O=-0.2, D=0.61, T=0.33),
    )
    invoice = FileScore(
        path=INVOICE,
        score=1.42,
        priority=81,
        band="High",
        risk=0.41,
        risk_low=0.33,
        risk_high=0.5,
        reasons=(
            Reason(
                "I",
                "pagerank",
                0.031,
                91.0,
                0.4,
                "Central module in the import graph (top 9% in repo by PageRank)",
            ),
            Reason(
                "O",
                "n_authors",
                9.0,
                96.0,
                0.25,
                "Very many authors: changed by 9 authors (top 4% in repo)",
            ),
        ),
        mitigations=(),
        debt_types=("architecture", "knowledge"),
        shap_by_family=_shap(I=0.4, O=0.25),
    )
    old_utils = FileScore(
        path=OLD_UTILS,
        score=1.38,
        priority=81,
        band="High",
        risk=0.35,
        risk_low=0.27,
        risk_high=0.44,
        reasons=(
            Reason(
                "S",
                "dup_ratio",
                0.32,
                97.0,
                0.3,
                "Heavy duplication: 32% of its code is duplicated (top 3% in repo)",
            ),
        ),
        mitigations=(),
        debt_types=("duplication",),
        shap_by_family=_shap(S=0.3),
    )
    pkg_init = FileScore(
        path=PKG_INIT,
        score=0.52,
        priority=60,
        band="Medium",
        risk=0.18,
        risk_low=0.12,
        risk_high=0.25,
        reasons=(
            Reason(
                "Q",
                "satd_count",
                3.0,
                88.0,
                0.15,
                "Admitted debt: 3 TODO, FIXME or HACK style comments (top 12% in repo)",
            ),
        ),
        mitigations=(),
        debt_types=("admitted",),
        shap_by_family=_shap(Q=0.15),
    )
    setup = FileScore(
        path=SETUP,
        score=-0.8,
        priority=12,
        band="Low",
        risk=0.03,
        risk_low=0.01,
        risk_high=0.06,
        reasons=(),
        mitigations=(),
        debt_types=(),
        shap_by_family=_shap(),
    )
    features = pd.DataFrame(
        {
            "sloc": [SLOC[path] for path in FEATURE_ORDER],
            "cc_max": [1, 12, 9, 31, 2],
            "dup_ratio": [0.0, 0.1, 0.32, 0.05, 0.0],
            "pct_sloc": [0.2, 0.8, 0.6, 1.0, 0.4],
        },
        index=pd.Index(FEATURE_ORDER, name="path"),
        dtype=np.float32,
    )
    processor_cost = _cost(12.5, 10.0, 15.0)
    invoice_cost = _cost(7.5, 6.0, 9.0)
    old_utils_cost = _cost(7.5, 6.0, 9.0)
    extras = {
        PROCESSOR: _extra(
            processor_cost,
            functions=(
                _function("Processor.charge", 40, 95, 18, 25),
                _function("Processor.refund", 120, 160, 18, 30),
                _function("helper", 10, 20, 2, 1),
            ),
            dependents=("src/api/views.py",),
            dependencies=(INVOICE,),
            commit_activity=(("2025-12", 4), ("2026-01", 7)),
        ),
        OLD_UTILS: _extra(old_utils_cost),
        INVOICE: _extra(
            invoice_cost,
            functions=(_function("total", 30, 60, 7, 9), _function("Invoice.render", 12, 25, 7, 9)),
        ),
        PKG_INIT: _extra(_cost(2.0, 1.6, 2.4)),
    }
    scores = (processor, invoice, old_utils, pkg_init, setup)
    graph = GraphData(
        nodes=tuple(
            GraphNode(
                id=s.path,
                priority=s.priority,
                band=s.band,
                sloc=SLOC[s.path],
                dependents=DEPENDENTS.get(s.path, 0),
            )
            for s in scores
        ),
        edges=(
            GraphEdge(source=PROCESSOR, target=INVOICE, kind="import", weight=1.0),
            GraphEdge(source=INVOICE, target=OLD_UTILS, kind="hidden_coupling", weight=0.62),
        ),
    )
    plan = Plan(
        budget_hours=40.0,
        steps=(
            PlanStep(
                order=1,
                path=INVOICE,
                benefit=0.95,
                cost=invoice_cost,
                roi=0.127,
                group=1,
                reason=f"fix before {PROCESSOR}, which depends on it",
            ),
            PlanStep(
                order=2,
                path=PROCESSOR,
                benefit=1.9,
                cost=processor_cost,
                roi=0.152,
                group=2,
                reason="best return on effort among the remaining files",
            ),
            PlanStep(
                order=3,
                path=OLD_UTILS,
                benefit=0.7,
                cost=old_utils_cost,
                roi=0.093,
                group=3,
                reason=f"fix before {PKG_INIT}, which depends on it",
            ),
        ),
        total_benefit=3.55,
        total_hours=27.5,
    )
    return AnalysisResult(
        sha="0123456789abcdef0123456789abcdef01234567",
        analyzed_at=datetime(2026, 1, 15, 9, 30, 5, tzinfo=UTC),
        model_version="dev-20260115T093005Z",
        extractor_version=EXTRACTOR_VERSION,
        files=scores,
        features=features,
        extras=extras,
        graph=graph,
        plan=plan,
        stats={"n_files": 5, "skipped": 2, "timings_s": {"mine": 1.5, "parse": 0.75, "features": 2.25}},
    )


@pytest.fixture
def sample_result() -> AnalysisResult:
    """A fresh copy of the hand-built sample analysis (see ``build_sample_result``)."""
    return build_sample_result()


@pytest.fixture
def pinned_version(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin ``tdp_core.__version__`` to ``"0.0.0"`` so golden files do not change with releases."""
    monkeypatch.setattr(tdp_core, "__version__", "0.0.0")
    return "0.0.0"


@pytest.fixture
def golden() -> Callable[[str, str], None]:
    """``check(name, text)`` asserts that ``text`` equals ``golden/<name>`` (see the module docstring)."""

    def check(name: str, text: str) -> None:
        path = GOLDEN_DIR / name
        if os.environ.get("TDP_UPDATE_GOLDEN") == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            return
        assert text == path.read_text(encoding="utf-8")

    return check


@pytest.fixture
def make_score() -> Callable[..., FileScore]:
    """Factory for ad hoc ``FileScore`` objects: ``make_score(path, priority, **overrides)``."""

    def make(path: str, priority: int, **overrides: object) -> FileScore:
        values: dict[str, object] = {
            "path": path,
            "score": priority / 10.0,
            "priority": priority,
            "band": band_for(max(0, min(priority, 100))),
            "risk": 0.5,
            "risk_low": 0.4,
            "risk_high": 0.6,
            "reasons": (),
            "mitigations": (),
            "debt_types": (),
            "shap_by_family": _shap(),
        }
        values.update(overrides)
        return FileScore(**values)  # type: ignore[arg-type]

    return make


@pytest.fixture
def make_result() -> Callable[..., AnalysisResult]:
    """Factory for small results: ``make_result(files, **fields)`` with no extras, graph, plan or features."""

    def make(files: tuple[FileScore, ...] | list[FileScore] = (), **fields: object) -> AnalysisResult:
        base = AnalysisResult(
            sha="f" * 40,
            analyzed_at=datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
            model_version="dev-test",
            extractor_version=EXTRACTOR_VERSION,
            files=tuple(files),
            features=pd.DataFrame(),
            extras={},
            graph=GraphData(nodes=(), edges=()),
            plan=Plan(budget_hours=40.0, steps=(), total_benefit=0.0, total_hours=0.0),
            stats={},
        )
        return dataclasses.replace(base, **fields)  # type: ignore[arg-type]

    return make
