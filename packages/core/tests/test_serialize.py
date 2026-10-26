from datetime import UTC, datetime

import pytest

from tdp_core.contracts import CostEstimate, FileScore, Plan, PlanStep, Reason, Snapshot
from tdp_core.serialize import from_jsonable, to_jsonable


def _score() -> FileScore:
    reason = Reason(
        family="E", feature="commits_180d", value=47.0, percentile=96.0, shap=0.9, text="Very often"
    )
    return FileScore(
        path="src/pay.py",
        score=2.45,
        priority=92,
        band="Critical",
        risk=0.64,
        risk_low=0.55,
        risk_high=0.71,
        reasons=(reason,),
        mitigations=(),
        debt_types=("test",),
        shap_by_family={"E": 0.9, "S": -0.1},
    )


def test_filescore_round_trip() -> None:
    score = _score()
    data = to_jsonable(score)
    assert data["reasons"][0]["feature"] == "commits_180d"
    assert from_jsonable(FileScore, data) == score


def test_plan_round_trip() -> None:
    cost = CostEstimate(hours=6.0, low=4.8, high=7.2, breakdown={"tests": 240.0})
    plan = Plan(
        budget_hours=40,
        steps=(PlanStep(1, "a.py", 1.5, cost, 0.25, 0, "best return"),),
        total_benefit=1.5,
        total_hours=6.0,
    )
    assert from_jsonable(Plan, to_jsonable(plan)) == plan


def test_datetime_is_utc_z() -> None:
    snap = Snapshot(repo="o__r", index=0, time=datetime(2022, 1, 3, 12, tzinfo=UTC), sha="a" * 40)
    data = to_jsonable(snap)
    assert data["time"] == "2022-01-03T12:00:00Z"
    assert from_jsonable(Snapshot, data) == snap


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError):
        to_jsonable(datetime(2022, 1, 1))


def test_bad_literal_rejected() -> None:
    data = to_jsonable(_score())
    data["band"] = "Severe"
    with pytest.raises(ValueError):
        from_jsonable(FileScore, data)
