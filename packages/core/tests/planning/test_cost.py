from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tdp_core.contracts import CloneStats, CostEstimate, FileSyntax, FunctionInfo
from tdp_core.planning import cost as cost_module
from tdp_core.planning.cost import (
    COST_DEFAULTS_PATH,
    COST_KINDS,
    COST_MODEL_KEYS,
    estimate_cost,
    load_cost_model,
    public_function_names,
)
from tdp_core.serialize import from_jsonable, to_jsonable

PLAN_DEFAULTS = {
    "complexity_threshold": 10.0,
    "complexity_per_point_over": 10.0,
    "duplication_per_group": 30.0,
    "duplication_per_line": 1.0,
    "tests_per_public_function": 20.0,
    "cycle_per_file": 120.0,
    "satd_per_item": 30.0,
    "blast_radius_factor": 0.1,
    "min_hours": 0.5,
}
NO_CLONES = CloneStats(dup_ratio=0.0, dup_tokens=0, clone_groups=0, dup_lines=0)


def fn(name: str, cyclomatic: int = 1) -> FunctionInfo:
    return FunctionInfo(name, 1, 2, cyclomatic, 0, 0, 0)


def syntax(*functions: FunctionInfo) -> FileSyntax:
    return FileSyntax(
        path="pkg/mod.py",
        language="python",
        sloc=100,
        module_cc=7,
        functions=tuple(functions),
        classes=1,
        comments=(),
        imports=(),
        tokens=(),
        token_lines=(),
        parse_error=False,
    )


def row(
    test_importers: float = 0, scc_size: float = 1, satd_count: float = 0, fan_in: float = 0
) -> pd.Series:
    return pd.Series(
        {
            "test_importers": test_importers,
            "scc_size": scc_size,
            "satd_count": satd_count,
            "fan_in": fan_in,
            "sloc": 100.0,
        }
    )


HANDCRAFTED = syntax(
    fn("parse", 14),
    fn("Parser.run", 25),
    fn("_helper", 3),
    fn("Parser._step", 11),
    fn("outer", 2),
    fn("outer.inner", 1),
)
HANDCRAFTED_CLONES = CloneStats(dup_ratio=0.2, dup_tokens=120, clone_groups=2, dup_lines=15)


def test_defaults_match_the_plan() -> None:
    assert load_cost_model() == PLAN_DEFAULTS
    raw = yaml.safe_load(COST_DEFAULTS_PATH.read_text(encoding="utf-8"))
    assert tuple(raw) == COST_MODEL_KEYS
    assert all(type(v) is float for v in load_cost_model().values())


def test_load_returns_a_fresh_copy() -> None:
    first = load_cost_model()
    first["min_hours"] = 99.0
    assert load_cost_model()["min_hours"] == 0.5


def test_default_file_has_no_dashes() -> None:
    text = COST_DEFAULTS_PATH.read_text(encoding="utf-8")
    assert chr(0x2013) not in text
    assert chr(0x2014) not in text


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cost_model.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_overrides_one_key(tmp_path: Path) -> None:
    model = load_cost_model(write(tmp_path, "cycle_per_file: 60\n"))
    assert model == {**PLAN_DEFAULTS, "cycle_per_file": 60.0}


def test_load_warns_about_unknown_keys(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="tdp_core.planning.cost"):
        model = load_cost_model(write(tmp_path, "zeta: 1\nalpha: 2\nmin_hours: 1\n"))
    assert model == {**PLAN_DEFAULTS, "min_hours": 1.0}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "alpha, zeta" in warnings[0].getMessage()


def test_load_empty_file_gives_defaults(tmp_path: Path) -> None:
    assert load_cost_model(write(tmp_path, "")) == PLAN_DEFAULTS
    assert load_cost_model(write(tmp_path, "# only a comment\n")) == PLAN_DEFAULTS


@pytest.mark.parametrize(
    "text",
    [
        "min_hours: -1\n",
        "min_hours: .nan\n",
        "min_hours: .inf\n",
        "min_hours: lots\n",
        "min_hours: true\n",
        "min_hours: [1, 2]\n",
        "- 1\n- 2\n",
        "just text\n",
    ],
)
def test_load_rejects_bad_values(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        load_cost_model(write(tmp_path, text))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cost_model(tmp_path / "missing.yaml")


def test_load_bad_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(yaml.YAMLError):
        load_cost_model(write(tmp_path, "a: [1, 2\n"))


@pytest.mark.parametrize(
    ("text", "message"), [("- 1\n", "mapping"), ("min_hours: 0.5\n", "missing keys: complexity_threshold")]
)
def test_broken_defaults_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, message: str
) -> None:
    monkeypatch.setattr(cost_module, "COST_DEFAULTS_PATH", write(tmp_path, text))
    cost_module._default_params.cache_clear()
    try:
        with pytest.raises(ValueError, match=message):
            load_cost_model()
    finally:
        monkeypatch.undo()
        cost_module._default_params.cache_clear()
    assert load_cost_model() == PLAN_DEFAULTS


def test_public_function_names() -> None:
    functions = [
        fn("parse"),
        fn("_helper"),
        fn("Parser.run"),
        fn("Parser._step"),
        fn("Parser.__init__"),
        fn("Parser.prop"),
        fn("Parser.prop"),
        fn("outer"),
        fn("outer.inner"),
        fn("outer.Inner.method"),
        fn("Config.Meta.render"),
        fn("<unknown>"),
    ]
    assert public_function_names(functions) == (
        "Config.Meta.render",
        "Parser.prop",
        "Parser.run",
        "outer",
        "parse",
    )
    assert public_function_names([]) == ()
    nested = [fn("Class.<unknown>"), fn("Class.method"), fn("Class.method.helper"), fn("f.Impl.run"), fn("f")]
    assert public_function_names(nested) == ("Class.method", "f")


def test_handcrafted_breakdown() -> None:
    estimate = estimate_cost(
        row(test_importers=0, scc_size=3, satd_count=2, fan_in=3), HANDCRAFTED, HANDCRAFTED_CLONES, {}
    )
    assert estimate.breakdown == {
        "complexity": 200.0,
        "duplication": 75.0,
        "tests": 60.0,
        "cycle": 120.0,
        "satd": 60.0,
    }
    assert tuple(estimate.breakdown) == COST_KINDS
    assert math.fsum(estimate.breakdown.values()) == 515.0
    # multiplier 1 + 0.1 * log2(4) = 1.2, so 515 * 1.2 / 60 = 10.3 hours
    assert estimate.hours == pytest.approx(10.3)
    assert estimate.low == pytest.approx(8.24)
    assert estimate.high == pytest.approx(12.36)
    assert estimate.low == pytest.approx(0.8 * estimate.hours)
    assert estimate.high == pytest.approx(1.2 * estimate.hours)


def test_tested_files_cost_no_test_minutes() -> None:
    for importers in (1, 4):
        estimate = estimate_cost(row(test_importers=importers), HANDCRAFTED, NO_CLONES, {})
        assert estimate.breakdown["tests"] == 0.0


def test_cycle_minutes_only_inside_a_cycle() -> None:
    assert estimate_cost(row(scc_size=1), HANDCRAFTED, NO_CLONES, {}).breakdown["cycle"] == 0.0
    assert estimate_cost(row(scc_size=2), HANDCRAFTED, NO_CLONES, {}).breakdown["cycle"] == 120.0


@pytest.mark.parametrize(("fan_in", "multiplier"), [(0, 1.0), (1, 1.1), (3, 1.2), (7, 1.3)])
def test_blast_radius_multiplier(fan_in: int, multiplier: float) -> None:
    # 600 minutes of complexity (one function at cc 70) so the floor never applies
    estimate = estimate_cost(row(test_importers=1, fan_in=fan_in), syntax(fn("big", 70)), NO_CLONES, {})
    assert estimate.breakdown["complexity"] == 600.0
    assert estimate.hours == pytest.approx(10.0 * multiplier)


def test_clean_file_gets_the_floor() -> None:
    estimate = estimate_cost(row(test_importers=2), syntax(fn("_private", 4)), NO_CLONES, {})
    assert estimate.hours == 0.5
    assert estimate.low == pytest.approx(0.4)
    assert estimate.high == pytest.approx(0.6)
    assert estimate.breakdown == dict.fromkeys(COST_KINDS, 0.0)


def test_empty_file_with_no_functions() -> None:
    estimate = estimate_cost(row(), syntax(), NO_CLONES, {})
    assert estimate.hours == 0.5
    assert estimate.breakdown["tests"] == 0.0


def test_module_level_complexity_is_not_costed() -> None:
    busy_module = syntax()
    assert busy_module.module_cc == 7
    assert estimate_cost(row(test_importers=1), busy_module, NO_CLONES, {}).breakdown["complexity"] == 0.0


def test_cost_model_is_merged_over_defaults() -> None:
    sample = row(test_importers=0, scc_size=3, satd_count=2, fan_in=3)
    assert estimate_cost(sample, HANDCRAFTED, HANDCRAFTED_CLONES, {}) == estimate_cost(
        sample, HANDCRAFTED, HANDCRAFTED_CLONES, load_cost_model()
    )
    halved = estimate_cost(sample, HANDCRAFTED, HANDCRAFTED_CLONES, {"cycle_per_file": 60, "unknown": "x"})
    assert halved.breakdown["cycle"] == 60.0
    assert halved.breakdown["complexity"] == 200.0
    floor = estimate_cost(row(test_importers=1), syntax(), NO_CLONES, {"min_hours": 2})
    assert floor.hours == 2.0
    stricter = estimate_cost(row(test_importers=1), HANDCRAFTED, NO_CLONES, {"complexity_threshold": 20})
    assert stricter.breakdown["complexity"] == 50.0


@pytest.mark.parametrize("bad", ["x", -1, float("nan"), float("inf"), True, None, [1]])
def test_bad_cost_model_values_raise(bad: Any) -> None:
    with pytest.raises(ValueError):
        estimate_cost(row(), HANDCRAFTED, NO_CLONES, {"min_hours": bad})


def test_missing_row_column_raises_key_error() -> None:
    sample = row().drop("satd_count")
    with pytest.raises(KeyError):
        estimate_cost(sample, HANDCRAFTED, NO_CLONES, {})


@pytest.mark.parametrize("column", ["test_importers", "scc_size", "satd_count", "fan_in"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_row_values_raise(column: str, bad: float) -> None:
    sample = row()
    sample[column] = bad
    with pytest.raises(ValueError, match=column):
        estimate_cost(sample, HANDCRAFTED, NO_CLONES, {})


def test_float32_row_gives_builtin_floats_that_round_trip() -> None:
    sample = row(test_importers=0, scc_size=3, satd_count=2, fan_in=3).astype(np.float32)
    estimate = estimate_cost(
        sample, HANDCRAFTED, HANDCRAFTED_CLONES, {"blast_radius_factor": np.float32(0.1)}
    )
    values = [estimate.hours, estimate.low, estimate.high, *estimate.breakdown.values()]
    assert all(type(v) is float for v in values)
    assert estimate.hours == pytest.approx(10.3)
    assert from_jsonable(CostEstimate, to_jsonable(estimate)) == estimate


def test_row_from_a_feature_frame() -> None:
    frame = pd.DataFrame(
        {
            "test_importers": [0.0, 1.0],
            "scc_size": [1.0, 2.0],
            "satd_count": [0.0, 1.0],
            "fan_in": [0.0, 1.0],
        },
        index=pd.Index(["a.py", "b.py"], name="path"),
        dtype=np.float32,
    )
    for path, features in frame.iterrows():
        estimate = estimate_cost(features, HANDCRAFTED, NO_CLONES, {})
        assert estimate.hours >= 0.5, path
