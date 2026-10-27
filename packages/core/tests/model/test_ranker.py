import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tdp_core.contracts import FEATURE_INPUT_COLUMNS
from tdp_core.model.bundle import Bundle
from tdp_core.model.ranker import DEFAULT_NTHREAD, Ranker

COLS = list(FEATURE_INPUT_COLUMNS)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[COLS].copy()


def numpy_booster(n_features: int, rounds: int = 3, seed: int = 0) -> xgb.Booster:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(64, n_features)).astype(np.float32)
    y = X[:, 0] + rng.normal(scale=0.1, size=64)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train({"max_depth": 2, "nthread": 1, "seed": seed}, dtrain, num_boost_round=rounds)


def test_score_matches_dmatrix_prediction(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    X = features(synthetic_frame)
    scores = dev_bundle.ranker.score(X)
    ref = dev_bundle.ranker.booster.predict(xgb.DMatrix(X), output_margin=True)
    assert scores.dtype == np.float64
    assert scores.shape == (len(X),)
    np.testing.assert_array_equal(scores, ref.astype(np.float64))
    assert np.unique(scores).size > 100


@pytest.mark.parametrize(
    "mutate",
    [
        lambda X: X.drop(columns=["sloc"]),
        lambda X: X.assign(extra=1.0),
        lambda X: X[COLS[::-1]],
        lambda X: X.rename(columns={"sloc": "lines"}),
        lambda X: X[[*COLS[:-1], COLS[0]]],
    ],
    ids=["dropped", "extra", "reversed", "renamed", "repeated"],
)
def test_score_rejects_wrong_columns(
    dev_bundle: Bundle, synthetic_frame: pd.DataFrame, mutate: object
) -> None:
    X = mutate(features(synthetic_frame.head(5)))  # type: ignore[operator]
    with pytest.raises(ValueError, match="columns"):
        dev_bundle.ranker.score(X)


def test_score_empty_frame(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    out = dev_bundle.ranker.score(features(synthetic_frame).iloc[:0])
    assert out.shape == (0,)
    assert out.dtype == np.float64


def test_score_ignores_index_and_handles_nan(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    X = features(synthetic_frame.head(50))
    base = dev_bundle.ranker.score(X)
    relabelled = X.set_index(pd.Index([f"file_{i}.py" for i in range(len(X))]))
    np.testing.assert_array_equal(dev_bundle.ranker.score(relabelled), base)
    with_nan = X.copy()
    with_nan["sloc"] = np.nan
    assert np.isfinite(dev_bundle.ranker.score(with_nan)).all()


def test_rejects_booster_with_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="89 features"):
        Ranker(numpy_booster(3))


def test_rejects_booster_with_other_feature_names() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(40, len(COLS))).astype(np.float32), columns=COLS[::-1])
    booster = xgb.train({"nthread": 1}, xgb.DMatrix(X, label=rng.normal(size=40)), num_boost_round=2)
    with pytest.raises(ValueError, match="feature names"):
        Ranker(booster)


def test_accepts_unnamed_booster_with_89_features(synthetic_frame: pd.DataFrame) -> None:
    booster = numpy_booster(len(COLS))
    ranker = Ranker(booster)
    X = features(synthetic_frame.head(30))
    ref = booster.predict(xgb.DMatrix(X.to_numpy(dtype=np.float32)), output_margin=True)
    np.testing.assert_array_equal(ranker.score(X), ref.astype(np.float64))
    assert repr(ranker) == "Ranker(features=89, rounds=3)"


def test_warns_when_best_iteration_is_before_last_round(caplog: pytest.LogCaptureFixture) -> None:
    booster = numpy_booster(len(COLS), rounds=5)
    booster.set_attr(best_iteration="1")
    with caplog.at_level(logging.WARNING, logger="tdp_core.model.ranker"):
        Ranker(booster)
    assert "scoring uses every round" in caplog.text


def test_from_json_round_trip(tmp_path: Path, dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    target = tmp_path / "model ünïcode" / "ranker.json"
    target.parent.mkdir()
    dev_bundle.ranker.booster.save_model(str(target))
    loaded = Ranker.from_json(target)
    X = features(synthetic_frame.head(200))
    np.testing.assert_array_equal(loaded.score(X), dev_bundle.ranker.score(X))
    config = json.loads(loaded.booster.save_config())
    assert config["learner"]["generic_param"]["nthread"] == str(DEFAULT_NTHREAD)
    assert isinstance(loaded.booster, xgb.Booster)


def test_from_json_custom_thread_count(tmp_path: Path, dev_bundle: Bundle) -> None:
    target = tmp_path / "ranker.json"
    dev_bundle.ranker.booster.save_model(str(target))
    loaded = Ranker.from_json(target, nthread=2)
    assert json.loads(loaded.booster.save_config())["learner"]["generic_param"]["nthread"] == "2"
    with pytest.raises(ValueError):
        Ranker.from_json(target, nthread=0)


def test_from_json_missing_empty_and_garbage(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Ranker.from_json(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        Ranker.from_json(tmp_path)
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        Ranker.from_json(empty)
    garbage = tmp_path / "garbage.json"
    garbage.write_text('{"not": "a model"}', encoding="utf-8")
    with pytest.raises(ValueError):
        Ranker.from_json(garbage)
    noise = tmp_path / "noise.json"
    noise.write_bytes(b"\x00\x01binary garbage\xff")
    with pytest.raises(ValueError):
        Ranker.from_json(noise)
