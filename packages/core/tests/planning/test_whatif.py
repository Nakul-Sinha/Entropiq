from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tdp_core.contracts import (
    FEATURE_COLUMNS,
    FEATURE_INPUT_COLUMNS,
    PCT_COLUMNS,
    WHATIF_ACTIONS,
    band_for,
    family_of,
)
from tdp_core.model.bundle import DEV_MONOTONE_CONSTRAINTS, Bundle, load_bundle, make_dev_bundle
from tdp_core.model.priority import priority_from_scores
from tdp_core.model.synthetic import synthetic_dataset
from tdp_core.planning.cost import load_cost_model
from tdp_core.planning.whatif import (
    ACTION_COLUMNS,
    CC_MAX_FLOOR,
    RAISED_COLUMNS,
    WhatIfResult,
    apply_actions,
    what_if,
)
from tdp_core.serialize import from_jsonable, to_jsonable

ALL = WHATIF_ACTIONS
ACTION_SETS = [(action,) for action in WHATIF_ACTIONS] + [ALL]
SET_IDS = [*WHATIF_ACTIONS, "all"]


def traced() -> bool:
    """True when coverage or a debugger slows the interpreter down."""
    return sys.gettrace() is not None or sys.monitoring.get_tool(sys.monitoring.COVERAGE_ID) is not None


def f32(x: float) -> float:
    return float(np.float32(x))


def bits(values: Any) -> np.ndarray:
    """float32 values as raw bits, for exact comparisons (NaN-safe, sign of zero included)."""
    return np.asarray(values, dtype=np.float32).view(np.uint32)


# ---------- a tiny hand-made repository for exact rule checks (no model needed) ----------

TINY_PATHS = ("a.py", "b.py", "c.py", "d.py", "e.py")
TINY_VALUES: dict[str, tuple[float, ...]] = {
    "has_test_file": (0, 1, 1, 1, 0),
    "test_importers": (0, 2, 1, 3, 0),
    "test_cochange_rate": (0.1, 0.2, 0.6, 0.4, 0.05),
    "cc_max": (30, 14, 6, 10, 0),
    "cog_max": (9, 4, 2, 0, 0),
    "longest_function": (80, 41, 10, 3, 0),
    "nesting_max": (3, 1, 0, 2, 0),
    "dup_ratio": (0, 0, 0.25, 0.5, 0.75),
    "satd_count": (2, 0, 1, 0, 0),
    "satd_density": (10, 0, 3, 0, 0),
    "satd_age_days": (300, 0, 20, 0, 0),
    "scc_size": (3, 3, 3, 1, 0),
    "hidden_coupling": (2, 0, 1, 0, 0),
}


def tiny_frame() -> pd.DataFrame:
    """Five files with TINY_VALUES, other columns synthetic, percentiles ranked as in feature assembly."""
    frame = synthetic_dataset(1, 5, seed=1).set_index("path")[list(FEATURE_INPUT_COLUMNS)].copy()
    frame.index = pd.Index(TINY_PATHS, name="path")
    for column, values in TINY_VALUES.items():
        frame[column] = np.asarray(values, dtype=np.float32)
    for pct in PCT_COLUMNS:
        frame[pct] = frame[pct.removeprefix("pct_")].rank(pct=True, method="average").astype(np.float32)
    return frame


@pytest.fixture(scope="module")
def tiny() -> pd.DataFrame:
    return tiny_frame()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    return load_bundle(make_dev_bundle(tmp_path_factory.mktemp("whatif") / "bundle"))


@pytest.fixture(scope="module")
def dataset_query() -> pd.DataFrame:
    """Query 0 of the default synthetic dataset in the dataset layout (path column, RangeIndex)."""
    data = synthetic_dataset(seed=42)
    return data[data["qid"] == 0].reset_index(drop=True)


@pytest.fixture(scope="module")
def repo(dataset_query: pd.DataFrame) -> pd.DataFrame:
    """The same 200 files in the AnalysisResult layout: indexed by path, model input columns only."""
    return dataset_query.set_index("path")[list(FEATURE_INPUT_COLUMNS)]


@pytest.fixture(scope="module")
def top_paths(repo: pd.DataFrame, bundle: Bundle) -> list[str]:
    """The three files the dev bundle scores highest."""
    scores = bundle.ranker.score(repo)
    return [str(repo.index[i]) for i in np.argsort(-scores, kind="stable")[:3]]


TINY_CASES = [
    pytest.param(
        "a.py",
        "add_tests",
        {
            "has_test_file": (0.0, 1.0),
            "test_importers": (0.0, 1.0),
            "test_cochange_rate": (f32(0.1), f32(0.4)),
        },
        {"pct_test_importers": 0.5, "pct_test_cochange_rate": 0.7},
        id="add_tests-untested",
    ),
    pytest.param(
        "b.py",
        "add_tests",
        # median of the OTHER tested files (c: .6, d: .4) is .5; counting b itself would give .4
        {"test_cochange_rate": (f32(0.2), 0.5)},
        {"pct_test_cochange_rate": 0.8},
        id="add_tests-raises-the-rate-only",
    ),
    pytest.param("c.py", "add_tests", {}, {}, id="add_tests-already-tested"),
    pytest.param(
        "e.py",
        "add_tests",
        {
            "has_test_file": (0.0, 1.0),
            "test_importers": (0.0, 1.0),
            "test_cochange_rate": (f32(0.05), f32(0.4)),
        },
        {"pct_test_importers": 0.5, "pct_test_cochange_rate": 0.7},
        id="add_tests-lowest-rate",
    ),
    pytest.param(
        "a.py",
        "split_longest_function",
        {
            "cc_max": (30.0, 15.0),
            "cog_max": (9.0, 4.5),
            "nesting_max": (3.0, 2.0),
            "longest_function": (80.0, 40.0),
        },
        {"pct_cc_max": 1.0, "pct_cog_max": 1.0, "pct_nesting_max": 0.9, "pct_longest_function": 0.8},
        id="split-halves",
    ),
    pytest.param(
        "b.py",
        "split_longest_function",
        {
            "cc_max": (14.0, 10.0),
            "cog_max": (4.0, 2.0),
            "nesting_max": (1.0, 0.0),
            "longest_function": (41.0, 20.5),
        },
        {"pct_cc_max": 0.7, "pct_cog_max": 0.7, "pct_nesting_max": 0.4, "pct_longest_function": 0.8},
        id="split-stops-at-the-floor",
    ),
    pytest.param(
        "c.py",
        "split_longest_function",
        # cc_max 6 is below the floor of 10 and is kept, never raised
        {"cog_max": (2.0, 1.0), "longest_function": (10.0, 5.0)},
        {"pct_cog_max": 0.6, "pct_longest_function": 0.6},
        id="split-below-the-floor",
    ),
    pytest.param(
        "d.py",
        "split_longest_function",
        {"nesting_max": (2.0, 1.0), "longest_function": (3.0, 1.5)},
        {"pct_nesting_max": 0.7, "pct_longest_function": 0.4},
        id="split-at-the-floor",
    ),
    pytest.param("e.py", "split_longest_function", {}, {}, id="split-no-functions"),
    pytest.param("e.py", "remove_duplication", {"dup_ratio": (0.75, 0.0)}, {"pct_dup_ratio": 0.4}, id="dup"),
    pytest.param("a.py", "remove_duplication", {}, {}, id="dup-none"),
    pytest.param(
        "a.py",
        "resolve_satd",
        {"satd_count": (2.0, 0.0), "satd_density": (10.0, 0.0), "satd_age_days": (300.0, 0.0)},
        {"pct_satd_count": 0.5, "pct_satd_density": 0.5, "pct_satd_age_days": 0.5},
        id="satd",
    ),
    pytest.param("b.py", "resolve_satd", {}, {}, id="satd-none"),
    pytest.param(
        "a.py",
        "break_cycle",
        {"scc_size": (3.0, 1.0), "hidden_coupling": (2.0, 0.0)},
        {"pct_scc_size": 0.5, "pct_hidden_coupling": 0.5},
        id="cycle",
    ),
    pytest.param("b.py", "break_cycle", {"scc_size": (3.0, 1.0)}, {"pct_scc_size": 0.5}, id="cycle-only"),
    pytest.param("d.py", "break_cycle", {}, {}, id="cycle-none"),
    pytest.param("e.py", "break_cycle", {}, {}, id="cycle-out-of-range-zero-kept"),
]


@pytest.mark.parametrize(("path", "action", "changed", "pcts"), TINY_CASES)
def test_rules_on_a_tiny_repo(
    tiny: pd.DataFrame,
    path: str,
    action: str,
    changed: dict[str, tuple[float, float]],
    pcts: dict[str, float],
) -> None:
    row, got = apply_actions(tiny, path, [action])
    assert got == changed
    assert list(got) == [c for c in FEATURE_COLUMNS if c in got]
    assert all(type(old) is float and type(new) is float for old, new in got.values())
    assert set(got) <= set(ACTION_COLUMNS[action])
    assert row.name == path
    assert row.dtype == np.float32
    assert list(row.index) == list(FEATURE_INPUT_COLUMNS)
    want = tiny.loc[path].to_numpy(dtype=np.float32).copy()
    for column, (_, new) in got.items():
        want[FEATURE_INPUT_COLUMNS.index(column)] = new
    for pct, value in pcts.items():
        want[FEATURE_INPUT_COLUMNS.index(pct)] = value
    assert set(pcts) == {f"pct_{c}" for c in got if f"pct_{c}" in FEATURE_INPUT_COLUMNS}
    np.testing.assert_array_equal(bits(row.to_numpy()), bits(want))


def test_add_tests_keeps_the_rate_when_no_other_file_is_tested(tiny: pd.DataFrame) -> None:
    untested = tiny.copy()
    untested["test_importers"] = np.float32(0)
    untested["pct_test_importers"] = untested["test_importers"].rank(pct=True).astype(np.float32)
    row, changed = apply_actions(untested, "a.py", ["add_tests"])
    assert changed == {"has_test_file": (0.0, 1.0), "test_importers": (0.0, 1.0)}
    assert row["pct_test_importers"] == np.float32(1.0)
    assert row["pct_test_cochange_rate"] == untested.at["a.py", "pct_test_cochange_rate"]


def test_all_actions_combine_the_single_rules(tiny: pd.DataFrame) -> None:
    for path in TINY_PATHS:
        row, changed = apply_actions(tiny, path, ALL)
        merged: dict[str, tuple[float, float]] = {}
        for action in WHATIF_ACTIONS:
            single_row, single = apply_actions(tiny, path, [action])
            merged.update(single)
            for column in ACTION_COLUMNS[action]:
                for name in (column, f"pct_{column}"):
                    if name in row.index:
                        assert bits(row[name]) == bits(single_row[name])
        assert changed == {c: merged[c] for c in FEATURE_COLUMNS if c in merged}


# ---------- contract and model checks with the dev bundle ----------


def test_action_columns_match_the_contract() -> None:
    assert tuple(ACTION_COLUMNS) == WHATIF_ACTIONS
    columns = [c for cols in ACTION_COLUMNS.values() for c in cols]
    assert len(columns) == len(set(columns))
    assert set(RAISED_COLUMNS) <= set(columns)
    for cols in ACTION_COLUMNS.values():
        assert list(cols) == [c for c in FEATURE_COLUMNS if c in cols]
    for column in columns:
        assert family_of(column) not in {"E", "O", "D"}
        sign = -1 if column in RAISED_COLUMNS else 1
        assert DEV_MONOTONE_CONSTRAINTS[column] == sign
        if f"pct_{column}" in FEATURE_INPUT_COLUMNS:
            assert DEV_MONOTONE_CONSTRAINTS[f"pct_{column}"] == sign
    assert "pkg_test_ratio" not in columns
    assert load_cost_model()["complexity_threshold"] == CC_MAX_FLOOR


def test_stored_percentiles_follow_the_feature_assembly_rule(repo: pd.DataFrame) -> None:
    # the simulator's +1 formula is only right if stored pct_ columns are rank(pct=True, method="average")
    for pct in PCT_COLUMNS:
        ranked = repo[pct.removeprefix("pct_")].rank(pct=True, method="average").astype(np.float32)
        np.testing.assert_array_equal(bits(ranked.to_numpy()), bits(repo[pct].to_numpy()))


def test_before_matches_full_scoring(repo: pd.DataFrame, bundle: Bundle) -> None:
    scores = bundle.ranker.score(repo)
    priority = priority_from_scores(scores)
    risk = bundle.calibrator.predict(scores)[0]
    for i in range(0, len(repo), 10):
        result = what_if(repo, str(repo.index[i]), ALL, bundle)
        p = int(priority[i])
        assert result.before == (p, band_for(p), float(risk[i]))
        assert result.path == repo.index[i]
        for state in (result.before, result.after):
            assert type(state[0]) is int
            assert type(state[1]) is str
            assert type(state[2]) is float
            assert 0 <= state[0] <= 100
            assert 0.0 <= state[2] <= 1.0
            assert state[1] == band_for(state[0])


def test_after_matches_rescoring_the_modified_frame(repo: pd.DataFrame, bundle: Bundle) -> None:
    for i in range(0, len(repo), 20):
        path = str(repo.index[i])
        row, changed = apply_actions(repo, path, ALL)
        values = repo.to_numpy(dtype=np.float32, copy=True)
        values[i] = row.to_numpy()
        modified = pd.DataFrame(values, index=repo.index, columns=repo.columns)
        scores = bundle.ranker.score(modified)
        p = int(priority_from_scores(scores)[i])
        result = what_if(repo, path, ALL, bundle)
        assert result.after == (p, band_for(p), float(bundle.calibrator.predict(scores)[0][i]))
        assert result.changed_features == changed


@pytest.mark.parametrize("actions", ACTION_SETS, ids=SET_IDS)
def test_never_worse_with_the_dev_bundle(
    repo: pd.DataFrame, bundle: Bundle, actions: tuple[str, ...]
) -> None:
    allowed = {c for action in actions for c in ACTION_COLUMNS[action]}
    files_changed = priority_drops = 0
    for path in repo.index:
        result = what_if(repo, str(path), actions, bundle)
        assert result.after[0] <= result.before[0]
        assert result.after[2] <= result.before[2]
        assert set(result.changed_features) <= allowed
        for column, (old, new) in result.changed_features.items():
            assert (new - old) * DEV_MONOTONE_CONSTRAINTS[column] < 0, column
        files_changed += bool(result.changed_features)
        priority_drops += result.after[0] < result.before[0]
    assert files_changed > 0
    if actions != ("break_cycle",):
        # the dev ranker barely uses the cycle columns, so break_cycle may move no priority at all
        assert priority_drops > 0


def test_score_never_rises(repo: pd.DataFrame, bundle: Bundle) -> None:
    rows = [apply_actions(repo, str(path), ALL)[0] for path in repo.index]
    after = bundle.ranker.score(pd.DataFrame(rows))
    before = bundle.ranker.score(repo)
    assert np.all(after <= before)
    assert np.count_nonzero(after < before) > len(repo) // 2


def test_add_tests_on_untested_files(repo: pd.DataFrame, bundle: Bundle) -> None:
    # the plan's acceptance check, made strict: adding tests never raises an untested file's priority
    untested = repo.index[(repo["has_test_file"] == 0) & (repo["test_importers"] == 0)]
    assert len(untested) >= 20
    drops = 0
    for path in untested:
        result = what_if(repo, str(path), ["add_tests"], bundle)
        assert result.after[0] <= result.before[0]
        assert result.after[2] <= result.before[2]
        assert result.changed_features["has_test_file"] == (0.0, 1.0)
        assert result.changed_features["test_importers"] == (0.0, 1.0)
        drops += result.after[0] < result.before[0]
    assert drops >= 0.75 * len(untested)


def test_more_actions_never_worse(repo: pd.DataFrame, bundle: Bundle, top_paths: list[str]) -> None:
    subsets = [frozenset(combo) for k in range(len(ALL) + 1) for combo in itertools.combinations(ALL, k)]
    checks = 0
    for path in top_paths:
        results = {s: what_if(repo, path, sorted(s), bundle) for s in subsets}
        for subset in subsets:
            for action in set(ALL) - subset:
                bigger = results[subset | {action}].after
                smaller = results[subset].after
                assert bigger[0] <= smaller[0]
                assert bigger[2] <= smaller[2]
                checks += 1
    assert checks == 3 * 80


def test_actions_are_a_set(repo: pd.DataFrame, bundle: Bundle, top_paths: list[str]) -> None:
    path = top_paths[0]
    canonical = what_if(repo, path, ALL, bundle)
    shuffled = [*reversed(ALL), "add_tests", "break_cycle"]
    assert what_if(repo, path, shuffled, bundle) == canonical
    assert what_if(repo, path, tuple(ALL), bundle) == canonical
    row, changed = apply_actions(repo, path, ["split_longest_function"])
    row2, changed2 = apply_actions(repo, path, ["split_longest_function", "split_longest_function"])
    assert changed2 == changed
    np.testing.assert_array_equal(bits(row2.to_numpy()), bits(row.to_numpy()))


def test_no_actions_and_no_ops(repo: pd.DataFrame, bundle: Bundle, top_paths: list[str]) -> None:
    path = top_paths[0]
    no_actions: tuple[Sequence[str], ...] = ([], ())
    for actions in no_actions:
        result = what_if(repo, path, actions, bundle)
        assert result.after == result.before
        assert result.changed_features == {}
        row, changed = apply_actions(repo, path, actions)
        assert changed == {}
        np.testing.assert_array_equal(bits(row.to_numpy()), bits(repo.loc[path].to_numpy()))
    no_satd = str(repo.index[repo["satd_count"] == 0][0])
    assert repo.at[no_satd, "satd_density"] == 0 and repo.at[no_satd, "satd_age_days"] == 0
    result = what_if(repo, no_satd, ["resolve_satd"], bundle)
    assert result.after == result.before
    assert result.changed_features == {}


def test_percentiles_match_pandas_rank(repo: pd.DataFrame) -> None:
    for i in range(0, len(repo), 10):
        path = str(repo.index[i])
        row, changed = apply_actions(repo, path, ALL)
        original = repo.iloc[i]
        for column in FEATURE_COLUMNS:
            if column in changed:
                assert row[column] == np.float32(changed[column][1])
            else:
                assert bits(row[column]) == bits(original[column])
            pct = f"pct_{column}"
            if pct not in FEATURE_INPUT_COLUMNS:
                continue
            if column in changed:
                values = repo[column].to_numpy(copy=True)
                values[i] = row[column]
                ranked = pd.Series(values).rank(pct=True, method="average").astype(np.float32)
                assert bits(row[pct]) == bits(ranked.iloc[i]), (path, column)
            else:
                assert bits(row[pct]) == bits(original[pct]), (path, column)


def test_single_file_repo(repo: pd.DataFrame, bundle: Bundle, top_paths: list[str]) -> None:
    one = repo.loc[[top_paths[0]]]
    result = what_if(one, top_paths[0], ALL, bundle)
    assert result.before[0] == result.after[0] == 100
    assert result.before[1] == result.after[1] == "Critical"
    assert result.after[2] <= result.before[2]
    assert result.changed_features
    assert "test_cochange_rate" not in result.changed_features  # no other tested file to compare with
    row, changed = apply_actions(one, top_paths[0], ALL)
    assert changed == result.changed_features
    for column in changed:
        if f"pct_{column}" in row.index:
            assert row[f"pct_{column}"] == np.float32(1.0)


def test_accepts_the_dataset_shape(dataset_query: pd.DataFrame, repo: pd.DataFrame, bundle: Bundle) -> None:
    assert isinstance(dataset_query.index, pd.RangeIndex)
    assert {"repo", "snapshot_idx", "snapshot_time", "grade", "qid"} <= set(dataset_query.columns)
    for path in (str(repo.index[5]), str(repo.index[150])):
        assert what_if(dataset_query, path, ALL, bundle) == what_if(repo, path, ALL, bundle)
        row, changed = apply_actions(dataset_query, path, ALL)
        row2, changed2 = apply_actions(repo, path, ALL)
        assert changed == changed2
        np.testing.assert_array_equal(bits(row.to_numpy()), bits(row2.to_numpy()))


def test_frame_rebuilt_from_json(repo: pd.DataFrame, bundle: Bundle) -> None:
    # the API path: file_results.features stored as JSON numbers, rebuilt with from_dict(orient="index")
    stored = {str(path): {c: float(v) for c, v in values.items()} for path, values in repo.iterrows()}
    loaded = json.loads(json.dumps(stored))
    rebuilt = pd.DataFrame.from_dict(loaded, orient="index")
    rebuilt = rebuilt[list(reversed(rebuilt.columns))]
    rebuilt["id"] = np.arange(len(rebuilt))
    assert rebuilt["sloc"].dtype == np.float64
    for path in (str(repo.index[0]), str(repo.index[99]), str(repo.index[199])):
        assert what_if(rebuilt, path, ALL, bundle) == what_if(repo, path, ALL, bundle)


@pytest.mark.parametrize("shape", ["indexed", "dataset"])
def test_inputs_are_not_modified(
    repo: pd.DataFrame, dataset_query: pd.DataFrame, bundle: Bundle, shape: str
) -> None:
    frame = repo if shape == "indexed" else dataset_query
    snapshot = copy.deepcopy(frame)
    actions = [*ALL]
    path = str(repo.index[3])
    apply_actions(frame, path, actions)
    what_if(frame, path, actions, bundle)
    pd.testing.assert_frame_equal(frame, snapshot, check_exact=True)
    assert actions == [*ALL]


Call = Callable[[pd.DataFrame, str, Sequence[str], Bundle], object]
CALLS: dict[str, Call] = {
    "apply_actions": lambda frame, path, actions, bundle: apply_actions(frame, path, actions),
    "what_if": what_if,
}


@pytest.mark.parametrize("name", list(CALLS))
def test_errors(repo: pd.DataFrame, bundle: Bundle, name: str) -> None:
    call = CALLS[name]
    path = str(repo.index[0])
    with pytest.raises(ValueError, match=r"'refactor'.*'nope'.*add_tests, split_longest_function"):
        call(repo, path, ["add_tests", "refactor", "nope", "refactor"], bundle)
    with pytest.raises(ValueError, match="unknown what-if action"):
        call(pd.DataFrame(), "a.py", ["nope"], bundle)  # actions are checked before the frame
    with pytest.raises(TypeError, match="not a string"):
        call(repo, path, "add_tests", bundle)
    with pytest.raises(KeyError, match=r"no feature row for 'missing\.py'"):
        call(repo, "missing.py", ALL, bundle)
    with pytest.raises(KeyError):
        call(repo.iloc[:0], path, ALL, bundle)
    with pytest.raises(ValueError, match="repeated paths"):
        call(pd.concat([repo, repo.iloc[[0]]]), path, ALL, bundle)
    with pytest.raises(ValueError, match=r"lacks 1 model input columns \(pct_sloc\)"):
        call(repo.drop(columns=["pct_sloc"]), path, ALL, bundle)
    with pytest.raises(ValueError, match="lacks 89 model input columns"):
        call(pd.DataFrame(index=pd.Index([path])), path, ALL, bundle)
    with pytest.raises(ValueError, match="repeats model input columns"):
        call(pd.concat([repo, repo[["sloc"]]], axis=1), path, ALL, bundle)
    for bad in (np.nan, np.inf, -np.inf):
        broken = repo.copy()
        broken.at[broken.index[7], "churn_total"] = bad
        with pytest.raises(ValueError, match=r"NaN or infinite values in 1 columns \(churn_total\)"):
            call(broken, path, ALL, bundle)
    huge = repo.astype(np.float64)
    huge.at[huge.index[7], "sloc"] = 1e300  # overflows float32
    with pytest.raises(ValueError, match="NaN or infinite"):
        call(huge, path, ALL, bundle)
    text = repo.astype(object)
    text.at[text.index[7], "sloc"] = "many"
    with pytest.raises(ValueError, match="non-numeric"):
        call(text, path, ALL, bundle)


def test_serialisation(repo: pd.DataFrame, bundle: Bundle, top_paths: list[str]) -> None:
    handmade = WhatIfResult("a.py", (91, "Critical", 0.5), (40, "Low", 0.25), {"cc_max": (30.0, 15.0)})
    assert handmade.to_json() == {
        "before": {"priority": 91, "band": "Critical", "risk": 0.5},
        "after": {"priority": 40, "band": "Low", "risk": 0.25},
        "changed_features": {"cc_max": [30.0, 15.0]},
    }
    result = what_if(repo, top_paths[0], ALL, bundle)
    body = result.to_json()
    assert set(body) == {"before", "after", "changed_features"}
    assert body["before"] == dict(zip(("priority", "band", "risk"), result.before, strict=True))
    assert body["after"] == dict(zip(("priority", "band", "risk"), result.after, strict=True))
    assert list(body["changed_features"]) == list(result.changed_features)
    assert json.loads(json.dumps(body, allow_nan=False)) == body
    assert from_jsonable(WhatIfResult, to_jsonable(result)) == result
    assert from_jsonable(WhatIfResult, json.loads(json.dumps(to_jsonable(result)))) == result


def test_concurrent_calls_agree(repo: pd.DataFrame, bundle: Bundle) -> None:
    paths = [str(p) for p in repo.index[::25]]
    sequential = [what_if(repo, p, ALL, bundle) for p in paths]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda p: what_if(repo, p, ALL, bundle), paths))
    assert parallel == sequential


def test_import_does_not_load_xgboost() -> None:
    code = "import sys, tdp_core.planning.whatif; print('xgboost' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "False"


def test_large_repo_is_fast(bundle: Bundle) -> None:
    big = synthetic_dataset(1, 5000, seed=7).set_index("path")[list(FEATURE_INPUT_COLUMNS)]
    what_if(big, str(big.index[0]), ALL, bundle)  # warm up
    for path in (str(big.index[1]), str(big.index[2500]), str(big.index[4999])):
        start = time.perf_counter()
        result = what_if(big, path, ALL, bundle)
        elapsed = time.perf_counter() - start
        assert result.after[0] <= result.before[0]
        if not traced():
            assert elapsed < 1.0, f"{elapsed:.3f} s"
