import json
from typing import Any

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from tdp_core.model.calibration import JSON_KEYS, IsotonicCalibrator


def make_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, n)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(2.0 * scores - 1.0)))).astype(np.float64)
    return scores, y


def is_non_decreasing(a: np.ndarray) -> bool:
    return bool(np.all(np.diff(a) >= 0))


def test_main_curve_matches_sklearn() -> None:
    s, y = make_data(1500)
    cal = IsotonicCalibrator.fit(s, y, n_boot=0)
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    np.testing.assert_array_equal(cal.x, ref.X_thresholds_)
    grid = np.linspace(-5.0, 5.0, 501)
    risk, low, high = cal.predict(grid)
    np.testing.assert_allclose(risk, ref.predict(grid), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(low, risk)
    np.testing.assert_array_equal(high, risk)


def test_bands_bracket_risk_and_are_monotone() -> None:
    s, y = make_data(2000)
    cal = IsotonicCalibrator.fit(s, y, n_boot=200)
    for curve in (cal.y, cal.y_low, cal.y_high):
        assert is_non_decreasing(curve)
    grid = np.concatenate([[-100.0], np.linspace(-4.0, 4.0, 400), [100.0]])
    risk, low, high = cal.predict(grid)
    assert np.all((low >= 0) & (high <= 1))
    assert np.all(low <= risk)
    assert np.all(risk <= high)
    for curve in (risk, low, high):
        assert is_non_decreasing(curve)
    assert float(np.mean(high - low)) > 0


def test_bands_narrow_with_more_data() -> None:
    def width(n: int) -> float:
        s, y = make_data(n, seed=1)
        cal = IsotonicCalibrator.fit(s, y, n_boot=100, seed=3)
        grid = np.linspace(-1.5, 1.5, 200)
        _, low, high = cal.predict(grid)
        return float(np.mean(high - low))

    assert width(200) > width(5000)


def test_scores_outside_range_clip_to_end_values() -> None:
    s, y = make_data(500)
    cal = IsotonicCalibrator.fit(s, y, n_boot=20)
    risk, low, high = cal.predict(np.array([cal.x[0] - 50.0, cal.x[-1] + 50.0]))
    assert risk.tolist() == [cal.y[0], cal.y[-1]]
    assert low.tolist() == [cal.y_low[0], cal.y_low[-1]]
    assert high.tolist() == [cal.y_high[0], cal.y_high[-1]]


def test_json_round_trip_is_exact() -> None:
    s, y = make_data(800)
    cal = IsotonicCalibrator.fit(s, y, n_boot=50)
    payload = cal.to_json()
    assert tuple(payload) == JSON_KEYS
    assert all(isinstance(v, float) for values in payload.values() for v in values)
    again = IsotonicCalibrator.from_json(json.loads(json.dumps(payload)))
    assert again.to_json() == payload
    for key in JSON_KEYS:
        np.testing.assert_array_equal(getattr(again, key), getattr(cal, key))


def test_from_json_ignores_extra_keys() -> None:
    cal = IsotonicCalibrator.from_json(
        {"x": [0.0], "y": [0.5], "y_low": [0.4], "y_high": [0.6], "note": [1.0]}
    )
    assert cal.predict([-1.0, 3.0])[0].tolist() == [0.5, 0.5]


def test_same_seed_is_deterministic_and_seed_changes_only_bands() -> None:
    s, y = make_data(600)
    a = IsotonicCalibrator.fit(s, y, n_boot=40, seed=7)
    b = IsotonicCalibrator.fit(s, y, n_boot=40, seed=7)
    c = IsotonicCalibrator.fit(s, y, n_boot=40, seed=8)
    assert a.to_json() == b.to_json()
    assert a.to_json()["y"] == c.to_json()["y"]
    assert a.to_json()["x"] == c.to_json()["x"]
    assert (a.to_json()["y_low"], a.to_json()["y_high"]) != (c.to_json()["y_low"], c.to_json()["y_high"])


def test_single_sample() -> None:
    cal = IsotonicCalibrator.fit([0.3], [1.0], n_boot=10)
    assert cal.x.tolist() == [0.3]
    risk, low, high = cal.predict([-2.0, 0.3, 9.0])
    assert risk.tolist() == low.tolist() == high.tolist() == [1.0, 1.0, 1.0]


def test_constant_scores_give_one_threshold() -> None:
    cal = IsotonicCalibrator.fit(np.full(10, 2.0), [0, 1, 0, 0, 1, 0, 0, 0, 1, 0], n_boot=30)
    assert cal.x.size == 1
    risk, low, high = cal.predict([-1.0, 2.0, 5.0])
    np.testing.assert_allclose(risk, 0.3)
    assert np.all(low <= risk)
    assert np.all(risk <= high)


def test_all_zero_labels_give_zero_risk() -> None:
    s, _ = make_data(100)
    cal = IsotonicCalibrator.fit(s, np.zeros(100), n_boot=10)
    risk, low, high = cal.predict(np.linspace(-3, 3, 7))
    assert risk.tolist() == low.tolist() == high.tolist() == [0.0] * 7


def test_bool_labels_are_accepted() -> None:
    s, y = make_data(300)
    a = IsotonicCalibrator.fit(s, y.astype(bool), n_boot=5)
    b = IsotonicCalibrator.fit(s, y, n_boot=5)
    assert a.to_json() == b.to_json()


@pytest.mark.parametrize(
    ("scores", "labels", "kwargs"),
    [
        ([], [], {}),
        ([0.1, 0.2], [1.0], {}),
        ([0.1, float("nan")], [1.0, 0.0], {}),
        ([0.1, float("inf")], [1.0, 0.0], {}),
        ([0.1, 0.2], [2.0, 0.0], {}),
        ([0.1, 0.2], [-0.1, 0.0], {}),
        ([0.1, 0.2], [0.0, float("nan")], {}),
        ([[0.1, 0.2]], [[1.0, 0.0]], {}),
        ([0.1, 0.2], [1.0, 0.0], {"n_boot": -1}),
    ],
    ids=["empty", "length", "nan", "inf", "label2", "negative", "label_nan", "2d", "n_boot"],
)
def test_fit_rejects_bad_input(scores: Any, labels: Any, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        IsotonicCalibrator.fit(scores, labels, **kwargs)


GOOD: dict[str, list[float]] = {
    "x": [0.0, 1.0, 2.0],
    "y": [0.1, 0.4, 0.8],
    "y_low": [0.05, 0.3, 0.7],
    "y_high": [0.2, 0.5, 0.9],
}


@pytest.mark.parametrize(
    "change",
    [
        {"y_high": None},
        {"y": [0.1, 0.4]},
        {"x": [], "y": [], "y_low": [], "y_high": []},
        {"x": [0.0, 1.0, 1.0]},
        {"x": [0.0, 2.0, 1.0]},
        {"y": [0.1, 0.4, 0.3], "y_low": [0.05, 0.3, 0.2]},
        {"y": [0.1, 0.4, 1.2], "y_high": [0.2, 0.5, 1.3]},
        {"y_low": [0.05, 0.5, 0.7]},
        {"y_high": [0.2, 0.3, 0.9]},
        {"y": [0.1, float("nan"), 0.8]},
        {"x": [[0.0, 1.0, 2.0]]},
        {"y": ["a", "b", "c"]},
    ],
    ids=[
        "missing_key",
        "unequal",
        "empty",
        "x_tie",
        "x_decreasing",
        "y_decreasing",
        "y_above_one",
        "low_above_y",
        "high_below_y",
        "nan",
        "2d",
        "strings",
    ],
)
def test_from_json_rejects_bad_payload(change: dict[str, Any]) -> None:
    payload: dict[str, Any] = {**GOOD, **change}
    payload = {k: v for k, v in payload.items() if v is not None}
    with pytest.raises(ValueError):
        IsotonicCalibrator.from_json(payload)


def test_from_json_rejects_non_mapping() -> None:
    with pytest.raises(ValueError):
        IsotonicCalibrator.from_json([1.0, 2.0])  # type: ignore[arg-type]


def test_tiny_tolerance_is_clipped_into_range() -> None:
    cal = IsotonicCalibrator([0.0, 1.0], [0.0, 1.0 + 1e-12], [-1e-12, 1.0], [0.0, 1.0])
    assert cal.y.tolist() == [0.0, 1.0]
    assert cal.y_low.tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "bad",
    [[0.1, float("nan")], [[0.1, 0.2]], [float("-inf")]],
    ids=["nan", "2d", "inf"],
)
def test_predict_rejects_bad_scores(bad: Any) -> None:
    cal = IsotonicCalibrator.from_json(GOOD)
    with pytest.raises(ValueError):
        cal.predict(bad)


def test_predict_empty_and_interpolates() -> None:
    cal = IsotonicCalibrator.from_json(GOOD)
    risk, low, high = cal.predict([])
    assert risk.shape == low.shape == high.shape == (0,)
    risk, low, high = cal.predict([0.5, 1.5])
    np.testing.assert_allclose(risk, [0.25, 0.6])
    np.testing.assert_allclose(low, [0.175, 0.5])
    np.testing.assert_allclose(high, [0.35, 0.7])
    assert risk.dtype == np.float64


def test_arrays_are_read_only_copies() -> None:
    x = np.array(GOOD["x"])
    cal = IsotonicCalibrator(x, GOOD["y"], GOOD["y_low"], GOOD["y_high"])
    x[0] = -9.0
    assert cal.x[0] == 0.0
    for key in JSON_KEYS:
        with pytest.raises(ValueError):
            getattr(cal, key)[0] = 0.5
    assert repr(cal) == "IsotonicCalibrator(thresholds=3)"
