import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml

from tdp_core.contracts import BINARY_FEATURES, EXTRACTOR_VERSION, FEATURE_INPUT_COLUMNS
from tdp_core.model import bundle as bundle_mod
from tdp_core.model.bundle import (
    CALIBRATOR_FILE,
    COST_MODEL_FILE,
    DEV_MONOTONE_CONSTRAINTS,
    DEV_RANKER_PARAMS,
    FEATURE_SCHEMA_FILE,
    METADATA_FILE,
    MODEL_CARD_FILE,
    RANKER_FILE,
    REASONS_FILE,
    REQUIRED_FILES,
    Bundle,
    BundleSchemaError,
    load_bundle,
    make_dev_bundle,
    save_bundle,
)

COLS = list(FEATURE_INPUT_COLUMNS)
DASHES = re.compile("[\u2013\u2014]")


def ndcg_at_k(grades: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    gains = 2.0 ** grades[order] - 1.0
    discounts = 1.0 / np.log2(np.arange(2, order.size + 2))
    ideal = 2.0 ** np.sort(grades)[::-1][:k] - 1.0
    best = float(np.sum(ideal * discounts[: ideal.size]))
    return float(np.sum(gains * discounts)) / best if best > 0 else 1.0


def mean_ndcg(frame: pd.DataFrame, scores: np.ndarray) -> float:
    values = []
    for _, idx in frame.groupby("qid").indices.items():
        values.append(ndcg_at_k(frame["grade"].to_numpy()[idx].astype(float), scores[idx]))
    return float(np.mean(values))


def copy_bundle(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    return dst


def small_booster(n_features: int = 3) -> xgb.Booster:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features)).astype(np.float32)
    return xgb.train({"nthread": 1}, xgb.DMatrix(X, label=X[:, 0]), num_boost_round=2)


def test_dev_bundle_files_and_metadata(dev_bundle_dir: Path, dev_bundle: Bundle) -> None:
    names = {p.name for p in dev_bundle_dir.iterdir()}
    assert set(REQUIRED_FILES) | {MODEL_CARD_FILE} <= names
    schema = json.loads((dev_bundle_dir / FEATURE_SCHEMA_FILE).read_text(encoding="utf-8"))
    assert schema == {"extractor_version": EXTRACTOR_VERSION, "columns": COLS}
    meta = dev_bundle.metadata
    for key in (
        "version",
        "created_at",
        "manifest_sha256",
        "code_git_sha",
        "params",
        "metrics",
        "training_rows",
        "synthetic",
        "seed",
    ):
        assert key in meta, key
    assert re.fullmatch(r"dev-\d{8}T\d{6}Z", dev_bundle.version)
    assert meta["version"] == dev_bundle.version
    assert meta["created_at"].endswith("Z")
    assert meta["training_rows"] == 6000
    assert meta["synthetic"] is True
    assert meta["seed"] == 42
    assert meta["params"]["monotone_constraints"] == dict(DEV_MONOTONE_CONSTRAINTS)
    assert meta["params"]["objective"] == "rank:ndcg"
    assert dev_bundle.path == dev_bundle_dir
    assert isinstance(dev_bundle.reasons, dict)
    assert isinstance(dev_bundle.cost_model, dict)
    assert dev_bundle.ranker.booster.num_boosted_rounds() == DEV_RANKER_PARAMS["n_estimators"]
    assert "reasons" not in repr(dev_bundle)
    card = (dev_bundle_dir / MODEL_CARD_FILE).read_text(encoding="utf-8")
    for heading in ("## Intended use", "## Data", "## Metrics", "## Limitations"):
        assert heading in card
    assert "synthetic" in card.lower()
    assert "seed 42" in card
    assert "never be promoted" in card
    assert not DASHES.search(card)


def test_dev_bundle_is_fast_and_reproducible(tmp_path: Path, dev_bundle_dir: Path) -> None:
    start = time.perf_counter()
    out = make_dev_bundle(tmp_path / "b", version="dev-fixed")
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0
    assert out == tmp_path / "b"
    assert load_bundle(out).version == "dev-fixed"
    for name in (RANKER_FILE, CALIBRATOR_FILE, FEATURE_SCHEMA_FILE):
        assert (out / name).read_bytes() == (dev_bundle_dir / name).read_bytes(), name


def test_dev_bundle_can_be_rebuilt_in_place(tmp_path: Path) -> None:
    target = tmp_path / "dev"
    make_dev_bundle(target, version="dev-a")
    make_dev_bundle(target, version="dev-b")
    assert load_bundle(target).version == "dev-b"


def test_dev_bundle_refuses_real_bundle_and_foreign_files(tmp_path: Path, dev_bundle_dir: Path) -> None:
    real = copy_bundle(dev_bundle_dir, tmp_path)
    meta = json.loads((real / METADATA_FILE).read_text(encoding="utf-8"))
    meta.pop("synthetic")
    (real / METADATA_FILE).write_text(json.dumps(meta), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in real.iterdir()}
    with pytest.raises(FileExistsError):
        make_dev_bundle(real)
    assert {p.name: p.read_bytes() for p in real.iterdir()} == before

    (real / METADATA_FILE).write_text("not json", encoding="utf-8")
    with pytest.raises(FileExistsError):
        make_dev_bundle(real)

    other = tmp_path / "models"
    other.mkdir()
    (other / "production.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        make_dev_bundle(other)
    assert [p.name for p in other.iterdir()] == ["production.json"]

    a_file = tmp_path / "file"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        make_dev_bundle(a_file)
    with pytest.raises(ValueError):
        make_dev_bundle(tmp_path / "bad-version", version="bad version")


def test_dev_constraints_cover_the_what_if_features() -> None:
    assert len(DEV_MONOTONE_CONSTRAINTS) == 33
    assert list(DEV_MONOTONE_CONSTRAINTS) == [c for c in COLS if c in DEV_MONOTONE_CONSTRAINTS]
    negative = {c for c, sign in DEV_MONOTONE_CONSTRAINTS.items() if sign == -1}
    assert negative == {
        "has_test_file",
        "test_importers",
        "test_cochange_rate",
        "pct_test_importers",
        "pct_test_cochange_rate",
    }
    for base in (
        "churn_decayed",
        "prior_latent_bugs",
        "transitive_dependents",
        "longest_function",
        "dup_ratio",
    ):
        assert DEV_MONOTONE_CONSTRAINTS[base] == DEV_MONOTONE_CONSTRAINTS[f"pct_{base}"] == 1
    with pytest.raises(TypeError):
        DEV_MONOTONE_CONSTRAINTS["sloc"] = 1  # type: ignore[index]


def test_dev_ranker_beats_churn_alone(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    scores = dev_bundle.ranker.score(synthetic_frame[COLS])
    model = mean_ndcg(synthetic_frame, scores)
    churn = mean_ndcg(synthetic_frame, synthetic_frame["churn_decayed"].to_numpy(dtype=np.float64))
    assert model > 0.6
    assert model > churn + 0.1


def test_dev_ranker_is_monotone_in_constrained_features(
    dev_bundle: Bundle, synthetic_frame: pd.DataFrame
) -> None:
    X = synthetic_frame[COLS].sample(200, random_state=0).reset_index(drop=True)
    before = dev_bundle.ranker.score(X)
    violations = []
    for column, sign in DEV_MONOTONE_CONSTRAINTS.items():
        values = synthetic_frame[column].to_numpy(dtype=np.float64)
        span = float(values.max() - values.min())
        bumps: list[float | None]
        if column in BINARY_FEATURES:
            bumps = [None]
        elif column.startswith("pct_"):
            bumps = [0.2]
        else:
            bumps = [1.0, 0.25 * span]
        for bump in bumps:
            changed = X.copy()
            current = changed[column].to_numpy(dtype=np.float32)
            if bump is None:
                changed[column] = np.ones_like(current)
            elif column.startswith("pct_"):
                changed[column] = np.minimum(current + np.float32(bump), np.float32(1.0))
            else:
                changed[column] = current + np.float32(bump)
            assert changed[column].dtype == np.float32
            after = dev_bundle.ranker.score(changed)
            if np.any(sign * (after - before) < 0):
                violations.append((column, bump))
    assert violations == []


def test_calibrated_risk_separates_outcomes(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    scores = dev_bundle.ranker.score(synthetic_frame[COLS])
    risk, low, high = dev_bundle.calibrator.predict(scores)
    positive = synthetic_frame["has_latent_defect"].to_numpy()
    assert risk[positive].mean() - risk[~positive].mean() > 0.1
    assert np.all(low <= risk)
    assert np.all(risk <= high)
    assert np.all((low >= 0) & (high <= 1))


def test_save_and_load_round_trip(tmp_path: Path, dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    reasons = {
        "sloc": {"template": "Large file: {value:.0f} lines", "adjectives": [[95, "Very"], [0, "Fairly"]]}
    }
    cost_model = {"minutes_per_function": 12.5, "kinds": ["complexity", "duplication"]}
    created = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
    metadata: dict[str, Any] = {
        "created_at": created,
        "params": {"max_depth": 6},
        "metrics": {"ndcg@10": np.float32(0.5), "folds": (1, 2)},
        "training_rows": 1234,
    }
    target = tmp_path / "ranker-2026.10.1"
    out = save_bundle(
        target,
        version="ranker-2026.10.1",
        booster=dev_bundle.ranker.booster,
        calibrator=dev_bundle.calibrator,
        metadata=metadata,
        reasons=reasons,
        cost_model=cost_model,
    )
    assert out == target
    assert metadata == {
        "created_at": created,
        "params": {"max_depth": 6},
        "metrics": {"ndcg@10": np.float32(0.5), "folds": (1, 2)},
        "training_rows": 1234,
    }
    loaded = load_bundle(target)
    assert loaded.version == "ranker-2026.10.1"
    assert loaded.path == target
    X = synthetic_frame[COLS].head(300)
    np.testing.assert_array_equal(loaded.ranker.score(X), dev_bundle.ranker.score(X))
    assert loaded.calibrator.to_json() == dev_bundle.calibrator.to_json()
    assert loaded.reasons == reasons
    assert loaded.cost_model == cost_model
    meta = loaded.metadata
    assert meta["created_at"] == "2026-10-01T09:30:00Z"
    assert meta["params"] == {"max_depth": 6}
    assert meta["metrics"] == {"ndcg@10": 0.5, "folds": [1, 2]}
    assert meta["training_rows"] == 1234
    assert meta["manifest_sha256"] is None
    assert meta["code_git_sha"] is None
    assert MODEL_CARD_FILE not in {p.name for p in target.iterdir()}
    for name in (CALIBRATOR_FILE, FEATURE_SCHEMA_FILE, METADATA_FILE, REASONS_FILE, COST_MODEL_FILE):
        raw = (target / name).read_bytes()
        assert b"\r\n" not in raw, name
        assert raw.endswith(b"\n"), name


def test_save_fills_defaults(tmp_path: Path, dev_bundle: Bundle) -> None:
    target = save_bundle(
        tmp_path / "b",
        version="ranker-2026.10.2",
        booster=dev_bundle.ranker.booster,
        calibrator=dev_bundle.calibrator,
        metadata={"version": "ranker-2026.10.2"},
    )
    meta = json.loads((target / METADATA_FILE).read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", meta["created_at"])
    assert meta["params"] == {}
    assert meta["metrics"] == {}
    assert meta["training_rows"] is None


@pytest.mark.parametrize(
    ("version", "metadata"),
    [
        ("", {}),
        ("../x", {}),
        ("a b", {}),
        ("-x", {}),
        ("x.", {}),
        ("x" * 129, {}),
        ("ranker-2026.10.1", {"version": "ranker-2026.10.2"}),
        ("ranker-2026.10.1", {"metrics": {"ndcg": float("nan")}}),
        ("ranker-2026.10.1", {"metrics": {"ndcg": float("inf")}}),
        ("ranker-2026.10.1", {"params": {"weights": np.zeros(3)}}),
        ("ranker-2026.10.1", {"params": object()}),
    ],
    ids=[
        "empty",
        "dots",
        "space",
        "leading",
        "trailing",
        "long",
        "conflict",
        "nan",
        "inf",
        "array",
        "object",
    ],
)
def test_save_rejects_bad_version_and_metadata(
    tmp_path: Path, dev_bundle: Bundle, version: str, metadata: dict[str, Any]
) -> None:
    target = tmp_path / "b"
    with pytest.raises(ValueError):
        save_bundle(
            target,
            version=version,
            booster=dev_bundle.ranker.booster,
            calibrator=dev_bundle.calibrator,
            metadata=metadata,
        )
    assert not target.exists()


def test_save_rejects_booster_with_wrong_features(tmp_path: Path, dev_bundle: Bundle) -> None:
    with pytest.raises(BundleSchemaError):
        save_bundle(
            tmp_path / "b",
            version="v1",
            booster=small_booster(),
            calibrator=dev_bundle.calibrator,
            metadata={},
        )
    assert not (tmp_path / "b").exists()


def test_save_trims_early_stopped_booster(
    tmp_path: Path, synthetic_frame: pd.DataFrame, dev_bundle: Bundle
) -> None:
    X = synthetic_frame[COLS]
    model = xgb.XGBRanker(objective="rank:ndcg", n_estimators=10, max_depth=3, n_jobs=1, random_state=0)
    model.fit(X, synthetic_frame["grade"].astype(np.int32), qid=synthetic_frame["qid"])
    original = model.get_booster().copy()
    original.set_attr(best_iteration="3")
    target = save_bundle(
        tmp_path / "b",
        version="v1",
        booster=original,
        calibrator=dev_bundle.calibrator,
        metadata={},
    )
    assert original.num_boosted_rounds() == 10
    loaded = load_bundle(target)
    assert loaded.ranker.booster.num_boosted_rounds() == 4
    ref = original[:4].predict(xgb.DMatrix(X), output_margin=True)
    np.testing.assert_array_equal(loaded.ranker.score(X), ref.astype(np.float64))


@pytest.mark.parametrize(
    "schema",
    [
        {"extractor_version": EXTRACTOR_VERSION, "columns": COLS[:-1]},
        {"extractor_version": EXTRACTOR_VERSION, "columns": [*COLS, "extra"]},
        {"extractor_version": EXTRACTOR_VERSION, "columns": COLS[1:] + COLS[:1]},
        {"extractor_version": EXTRACTOR_VERSION, "columns": "sloc"},
        {"extractor_version": "0", "columns": COLS},
        {"columns": COLS},
        COLS,
    ],
    ids=["dropped", "extra", "reordered", "not_list", "old_extractor", "no_extractor", "json_list"],
)
def test_load_rejects_feature_schema_mismatch(tmp_path: Path, dev_bundle_dir: Path, schema: Any) -> None:
    target = copy_bundle(dev_bundle_dir, tmp_path)
    (target / FEATURE_SCHEMA_FILE).write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(BundleSchemaError, match=FEATURE_SCHEMA_FILE):
        load_bundle(target)


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_load_rejects_missing_required_file(tmp_path: Path, dev_bundle_dir: Path, name: str) -> None:
    target = copy_bundle(dev_bundle_dir, tmp_path)
    (target / name).unlink()
    with pytest.raises(BundleSchemaError, match=re.escape(name)):
        load_bundle(target)


def test_load_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "absent")
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "file")


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (METADATA_FILE, b"{not json"),
        (METADATA_FILE, b'{"created_at": "2026-01-01T00:00:00Z"}'),
        (METADATA_FILE, b'{"version": "bad version"}'),
        (METADATA_FILE, b'{"version": 3}'),
        (METADATA_FILE, b'["dev-1"]'),
        (METADATA_FILE, b"\xff\xfe\x00"),
        (CALIBRATOR_FILE, b'{"x": [1.0, 0.0], "y": [0.1, 0.2], "y_low": [0.1, 0.2], "y_high": [0.1, 0.2]}'),
        (CALIBRATOR_FILE, b'{"x": [0.0]}'),
        (CALIBRATOR_FILE, b"[]"),
        (RANKER_FILE, b'{"learner": "garbage"}'),
        (RANKER_FILE, b""),
        (REASONS_FILE, b"- a\n- b\n"),
        (REASONS_FILE, b"key: [unclosed\n"),
        (COST_MODEL_FILE, b"just a string\n"),
    ],
    ids=[
        "meta_not_json",
        "meta_no_version",
        "meta_bad_version",
        "meta_version_not_str",
        "meta_list",
        "meta_not_utf8",
        "cal_bad_x",
        "cal_missing_keys",
        "cal_list",
        "ranker_garbage",
        "ranker_empty",
        "reasons_list",
        "reasons_bad_yaml",
        "cost_string",
    ],
)
def test_load_rejects_corrupt_files(tmp_path: Path, dev_bundle_dir: Path, name: str, content: bytes) -> None:
    target = copy_bundle(dev_bundle_dir, tmp_path)
    (target / name).write_bytes(content)
    with pytest.raises(BundleSchemaError, match=re.escape(name)):
        load_bundle(target)


def test_empty_yaml_gives_empty_mapping(tmp_path: Path, dev_bundle_dir: Path) -> None:
    target = copy_bundle(dev_bundle_dir, tmp_path)
    (target / REASONS_FILE).write_text("# nothing yet\n", encoding="utf-8")
    assert load_bundle(target).reasons == {}


def test_package_defaults_are_copied_and_used(
    tmp_path: Path, dev_bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    reasons_default = defaults / "reasons.yaml"
    reasons_default.write_bytes(b"# reason templates\nsloc:\n  template: 'Large file'  # keep\n")
    cost_default = defaults / "cost_defaults.yaml"
    cost_default.write_bytes("# cost constants\nminutes: 30\nlabel: caf\u00e9\n".encode())
    mapping = {REASONS_FILE: reasons_default, COST_MODEL_FILE: cost_default}
    monkeypatch.setattr(bundle_mod, "_package_default", lambda name: mapping.get(name))

    target = save_bundle(
        tmp_path / "b",
        version="v1",
        booster=dev_bundle.ranker.booster,
        calibrator=dev_bundle.calibrator,
        metadata={},
    )
    assert (target / REASONS_FILE).read_bytes() == reasons_default.read_bytes()
    assert (target / COST_MODEL_FILE).read_bytes() == cost_default.read_bytes()
    loaded = load_bundle(target)
    assert loaded.reasons == {"sloc": {"template": "Large file"}}
    assert loaded.cost_model == {"minutes": 30, "label": "caf\u00e9"}

    (target / REASONS_FILE).unlink()
    (target / COST_MODEL_FILE).unlink()
    fallback = load_bundle(target)
    assert fallback.reasons == {"sloc": {"template": "Large file"}}
    assert fallback.cost_model == {"minutes": 30, "label": "caf\u00e9"}

    monkeypatch.setattr(bundle_mod, "_package_default", lambda name: None)
    bare = load_bundle(target)
    assert bare.reasons == {}
    assert bare.cost_model == {}


def test_resave_without_default_removes_stale_yaml(
    tmp_path: Path, dev_bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    common: dict[str, Any] = {
        "booster": dev_bundle.ranker.booster,
        "calibrator": dev_bundle.calibrator,
        "metadata": {},
    }
    target = save_bundle(tmp_path / "b", version="v1", reasons={"a": 1}, cost_model={"b": 2}, **common)
    assert yaml.safe_load((target / REASONS_FILE).read_text(encoding="utf-8")) == {"a": 1}
    monkeypatch.setattr(bundle_mod, "_package_default", lambda name: None)
    save_bundle(target, version="v2", **common)
    assert not (target / REASONS_FILE).exists()
    assert not (target / COST_MODEL_FILE).exists()
    loaded = load_bundle(target)
    assert loaded.version == "v2"
    assert loaded.reasons == {}


def test_package_default_lookup() -> None:
    for name in (REASONS_FILE, COST_MODEL_FILE):
        found = bundle_mod._package_default(name)
        assert found is None or found.is_file()
    assert bundle_mod._package_default(METADATA_FILE) is None
