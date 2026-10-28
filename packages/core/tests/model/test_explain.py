"""Tests for the SHAP explainer, the reason templates, reasons, mitigations and debt types."""

import copy
import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xgboost as xgb
import yaml

import tdp_core.model.explain as explain_mod
from tdp_core.contracts import (
    BINARY_FEATURES,
    DEBT_TYPES,
    FAMILY_ORDER,
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_INPUT_COLUMNS,
    FileScore,
    Reason,
    band_for,
    family_of,
)
from tdp_core.model.bundle import REASONS_FILE, Bundle, load_bundle, save_bundle
from tdp_core.model.explain import (
    DEBT_TYPE_FAMILIES,
    DOMINANT_DEBT_TYPE,
    MAX_DEBT_TYPES,
    MAX_REASONS,
    REASON_MIN_SHARE,
    TEMPLATE_FIELDS,
    TEMPLATE_KEYS,
    Explainer,
    Explanation,
    assign_debt_types,
    family_totals,
    feature_totals,
    load_templates,
    render_reasons,
    render_text,
    validate_templates,
)
from tdp_core.model.priority import priority_from_scores
from tdp_core.model.ranker import Ranker
from tdp_core.serialize import from_jsonable, to_jsonable

COLS = list(FEATURE_INPUT_COLUMNS)
PACKAGED = Path(explain_mod.__file__).with_name("reasons.yaml")
RENDER_VALUES = (0.0, 0.004, 0.5, 1.0, 1.4, 2.0, 37.25, 1234.5, -3.2)
RENDER_PERCENTILES = (0.0, 19.9, 20.0, 50.0, 80.0, 95.0, 99.5, 100.0)
ZERO_TOTALS = dict.fromkeys(FAMILY_ORDER, 0.0)


# ---------------------------------------------------------------- helpers and fixtures


def make_row(**values: float) -> pd.Series:
    """A float32 feature row on which no debt rule triggers, with the given overrides."""
    data = {c: (0.5 if c.startswith("pct_") else 0.0) for c in COLS}
    data.update(scc_size=1.0, owner_active=1.0, has_test_file=1.0, test_importers=1.0)
    for name, value in values.items():
        assert name in data, f"not a feature column: {name}"
        data[name] = value
    return pd.Series(data, dtype=np.float32)


def make_phi(**contrib: float) -> npt.NDArray[np.float64]:
    phi = np.zeros(len(COLS), dtype=np.float64)
    for name, value in contrib.items():
        phi[COLS.index(name)] = value
    return phi


def totals_with(**values: float) -> dict[str, float]:
    return {**ZERO_TOTALS, **values}


def cleaned(value: float) -> float:
    return float(str(np.float32(value)))


def assert_clean_text(text: str) -> None:
    assert text
    assert text == text.strip()
    assert re.fullmatch(r"[ -~]+", text), text
    assert not set("{}|<>") & set(text), text
    assert not re.search(r"\b(nan|inf)\b", text, flags=re.IGNORECASE), text
    assert len(text) <= 140, text


@pytest.fixture(scope="module")
def templates() -> dict[str, Any]:
    return load_templates()


@pytest.fixture(scope="module")
def explainer(dev_bundle: Bundle) -> Explainer:
    return Explainer(dev_bundle)


@pytest.fixture(scope="module")
def query(synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    return synthetic_frame[synthetic_frame["qid"] == 0]


# ---------------------------------------------------------------- A. SHAP values


def test_shap_values_add_up_to_scores(
    explainer: Explainer, dev_bundle: Bundle, synthetic_frame: pd.DataFrame
) -> None:
    X = synthetic_frame[COLS]
    phi = explainer.shap_values(X)
    assert phi.shape == (6000, 89)
    assert phi.dtype == np.float64
    assert np.isfinite(phi).all()
    error = np.abs(phi.sum(axis=1) + explainer.expected_value - dev_bundle.ranker.score(X)).max()
    assert error < 1e-4


def test_shap_values_match_xgboost_contributions(
    explainer: Explainer, dev_bundle: Bundle, synthetic_frame: pd.DataFrame
) -> None:
    X = synthetic_frame[COLS].iloc[:300]
    contribs = dev_bundle.ranker.booster.predict(
        xgb.DMatrix(X.to_numpy(dtype=np.float32)), pred_contribs=True, validate_features=False
    )
    np.testing.assert_allclose(explainer.shap_values(X), contribs[:, :-1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(contribs[:, -1], contribs[0, -1], rtol=0, atol=1e-6)
    assert isinstance(explainer.expected_value, float)
    assert explainer.expected_value == pytest.approx(float(contribs[0, -1]), abs=1e-6)
    assert dev_bundle.version in repr(explainer)
    assert repr(explainer).startswith("Explainer(version=")


def test_every_boosting_round_is_used(dev_bundle: Bundle, synthetic_frame: pd.DataFrame) -> None:
    booster = dev_bundle.ranker.booster.copy()
    booster.set_attr(best_iteration="9")
    ranker = Ranker(booster)
    exp = Explainer(dataclasses.replace(dev_bundle, ranker=ranker))
    X = synthetic_frame[COLS].iloc[:500]
    error = np.abs(exp.shap_values(X).sum(axis=1) + exp.expected_value - ranker.score(X)).max()
    assert error < 1e-4


def test_shap_values_empty_frame(explainer: Explainer, query: pd.DataFrame) -> None:
    phi = explainer.shap_values(query[COLS].iloc[:0])
    assert phi.shape == (0, 89)
    assert phi.dtype == np.float64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda X: X.drop(columns=["sloc"]), "1 missing"),
        (lambda X: X.assign(extra=1.0), "1 extra"),
        (lambda X: X[COLS[::-1]], "wrong order"),
    ],
    ids=["missing", "extra", "order"],
)
def test_wrong_columns_are_rejected(
    explainer: Explainer,
    query: pd.DataFrame,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    X = mutate(query[COLS].head(5))
    with pytest.raises(ValueError, match=message):
        explainer.shap_values(X)
    with pytest.raises(ValueError, match=message):
        explainer.explain(X)


def test_shap_values_treat_nan_as_missing(
    explainer: Explainer, dev_bundle: Bundle, synthetic_frame: pd.DataFrame
) -> None:
    X = synthetic_frame[COLS].iloc[:200].copy()
    X.iloc[0, 0] = np.nan
    X.iloc[5, 10] = np.nan
    X.iloc[7, 50] = np.nan
    phi = explainer.shap_values(X)
    assert np.isfinite(phi).all()
    error = np.abs(phi.sum(axis=1) + explainer.expected_value - dev_bundle.ranker.score(X)).max()
    assert error < 1e-4


# ---------------------------------------------------------------- B. totals


def test_family_totals_sum_columns_per_family() -> None:
    phi = np.random.default_rng(0).normal(size=89)
    totals = family_totals(phi)
    assert list(totals) == list(FAMILY_ORDER)
    for fam in FAMILY_ORDER:
        expected = sum(phi[j] for j, c in enumerate(COLS) if family_of(c) == fam)
        assert totals[fam] == pytest.approx(expected, abs=1e-12)
        assert type(totals[fam]) is float
    assert sum(totals.values()) == pytest.approx(phi.sum(), abs=1e-12)
    assert family_totals(phi.tolist()) == totals


def test_feature_totals_add_raw_and_pct() -> None:
    phi = np.random.default_rng(1).normal(size=89)
    totals = feature_totals(phi)
    assert list(totals) == list(FEATURE_COLUMNS)
    for name in FEATURE_COLUMNS:
        raw = phi[COLS.index(name)]
        if name in BINARY_FEATURES:
            assert totals[name] == raw
        else:
            assert totals[name] == pytest.approx(raw + phi[COLS.index(f"pct_{name}")], abs=1e-12)
        assert type(totals[name]) is float
    assert sum(totals.values()) == pytest.approx(phi.sum(), abs=1e-12)


@pytest.mark.parametrize("func", [family_totals, feature_totals])
@pytest.mark.parametrize(
    "phi",
    [np.zeros(88), np.zeros((2, 89)), make_phi(sloc=np.nan), make_phi(fan_in=np.inf)],
    ids=["short", "two_rows", "nan", "inf"],
)
def test_totals_reject_bad_rows(func: Callable[[Any], dict[str, float]], phi: Any) -> None:
    with pytest.raises(ValueError, match="phi_row"):
        func(phi)


# ---------------------------------------------------------------- C. templates


def test_every_feature_has_a_template(templates: dict[str, Any]) -> None:
    assert list(templates) == list(FEATURE_COLUMNS)
    validate_templates(templates)
    for entry in templates.values():
        assert set(entry) <= TEMPLATE_KEYS


def test_every_template_renders_cleanly(templates: dict[str, Any]) -> None:
    kinds: tuple[Any, ...] = ("template", "mitigation")
    for feature in FEATURE_COLUMNS:
        for kind in kinds:
            for value in RENDER_VALUES:
                for percentile in RENDER_PERCENTILES:
                    assert_clean_text(render_text(templates, feature, kind, value, percentile))


def test_packaged_file_is_plain_ascii() -> None:
    raw = PACKAGED.read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


@pytest.mark.parametrize(
    ("feature", "kind", "value", "percentile", "expected"),
    [
        ("commits_180d", "template", 0.0, 99.0, "Not changed in the last 6 months"),
        ("commits_180d", "mitigation", 1.0, 40.0, "Rarely changed: 1 commit in the last 6 months"),
        (
            "commits_180d",
            "template",
            7.0,
            97.0,
            "Very often changed: 7 commits in the last 6 months (top 3% in repo)",
        ),
        ("scc_size", "template", 1.0, 50.0, "Not part of an import cycle"),
        ("scc_size", "mitigation", 1.0, 0.0, "Not part of an import cycle"),
        (
            "scc_size",
            "template",
            3.0,
            96.0,
            "Import cycle: one of 3 files that import each other in a loop (top 4% in repo)",
        ),
        ("has_test_file", "template", 0.0, 0.0, "No test file named after it"),
        ("has_test_file", "mitigation", 1.0, 100.0, "Has a matching test file"),
        ("sloc", "template", 1200.0, 95.0, "Very large file: 1,200 lines of code (top 5% in repo)"),
        ("sloc", "template", 1200.0, 94.99, "Large file: 1,200 lines of code (top 5% in repo)"),
        ("sloc", "template", 900.0, 99.5, "Very large file: 900 lines of code (top 1% in repo)"),
        ("sloc", "template", 900.0, 100.0, "Very large file: 900 lines of code (top 1% in repo)"),
        ("sloc", "mitigation", 1.0, 0.0, "Very small file: 1 line of code"),
        ("sloc", "mitigation", 0.0, 0.0, "Very small file: 0 lines of code"),
        ("sloc", "mitigation", 2.0, 0.0, "Very small file: 2 lines of code"),
        ("sloc", "mitigation", 1.6, 0.0, "Very small file: 2 lines of code"),
        ("sloc", "mitigation", 1.4, 0.0, "Very small file: 1 line of code"),
        (
            "dup_ratio",
            "template",
            0.1,
            97.5,
            "Heavy duplication: 10% of its code is duplicated (top 2% in repo)",
        ),
        (
            "cc_trend",
            "mitigation",
            -3.25,
            5.0,
            "Complexity falling: total complexity changes by -3.2 per year",
        ),
    ],
)
def test_render_text_golden(
    templates: dict[str, Any], feature: str, kind: Any, value: float, percentile: float, expected: str
) -> None:
    assert render_text(templates, feature, kind, value, percentile) == expected


def test_render_text_errors(templates: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="'nope'"):
        render_text(templates, "nope", "template", 1.0, 50.0)
    with pytest.raises(ValueError, match="kind"):
        render_text(templates, "sloc", cast(Any, "reason"), 1.0, 50.0)
    with pytest.raises(ValueError, match="finite"):
        render_text(templates, "sloc", "template", float("nan"), 50.0)
    with pytest.raises(ValueError, match="finite"):
        render_text(templates, "sloc", "template", 1.0, float("inf"))
    with pytest.raises(ValueError, match="mitigation of feature 'sloc'"):
        render_text({"sloc": {"template": "x"}}, "sloc", "mitigation", 1.0, 50.0)
    with pytest.raises(ValueError, match="exact sentence"):
        render_text({"sloc": {"template": "x", "exact": [[0, 5]]}}, "sloc", "template", 0.0, 50.0)
    with pytest.raises(ValueError, match="template of feature 'sloc'"):
        render_text({"sloc": {"template": "{foo}"}}, "sloc", "template", 1.0, 50.0)


def _set(feature: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(t: dict[str, Any]) -> None:
        t[feature][key] = value

    return mutate


def _drop(feature: str, key: str | None = None) -> Callable[[dict[str, Any]], None]:
    def mutate(t: dict[str, Any]) -> None:
        if key is None:
            del t[feature]
        else:
            del t[feature][key]

    return mutate


def _replace_entry(t: dict[str, Any]) -> None:
    t["sloc"] = "Large file"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop("sloc"), "miss 1 of 46 features: sloc"),
        (_replace_entry, "'sloc': must be a mapping"),
        (_drop("sloc", "template"), "'sloc': template must be a non-empty string"),
        (_set("sloc", "mitigation", ""), "'sloc': mitigation must be a non-empty string"),
        (_set("sloc", "mitigation", 3), "'sloc': mitigation must be a non-empty string"),
        (_set("sloc", "mitigaton", "x"), "unknown key"),
        (_set("sloc", "template", "{foo} lines"), "unknown field"),
        (_set("sloc", "template", "{value.real} lines"), "unknown field"),
        (_set("sloc", "template", "{0} lines"), "unknown field"),
        (_set("sloc", "template", "{} lines"), "unknown field"),
        (_set("sloc", "template", "{value!r} lines"), "conversion"),
        (_set("sloc", "template", "{value:{top}} lines"), "nest"),
        (_set("sloc", "template", "{value lines"), "format string"),
        (_set("sloc", "template", "{value:.0q} lines"), "does not render"),
        (_set("sloc", "template", "{adj:.1f} lines"), "does not render"),
        (_set("sloc", "template", "Large\nfile"), "printable line"),
        (_set("sloc", "template", "Large file | big"), "must not contain"),
        (_set("sloc", "mitigation", "<b>Large</b> file"), "must not contain"),
        (_set("sloc", "template", "{{value}} lines"), "must not contain"),
        (_set("sloc", "adjectives", [[80, "a"], [95, "b"], [0, "c"]]), "strictly decreasing"),
        (_set("sloc", "adjectives", [[95, "a"], [10, "b"]]), "last adjective threshold must be 0"),
        (_set("sloc", "adjectives", [[101, "a"], [0, "b"]]), "from 0 to 100"),
        (_set("sloc", "adjectives", [[True, "a"], [0, "b"]]), "from 0 to 100"),
        (_set("sloc", "adjectives", [[95, "a", "x"], [0, "b"]]), r"adjectives\[0\] must be"),
        (_set("sloc", "adjectives", []), "non-empty list"),
        (_set("sloc", "adjectives", [[95, 7], [0, "b"]]), "text must be a string"),
        (_set("sloc", "adjectives", [[95, "a\tb"], [0, "b"]]), "printable line"),
        (_set("sloc", "adjectives", [[95, " "], [0, "b"]]), "empty sentence"),
        (_drop("sloc", "adjectives"), "no adjectives"),
        (_set("cc_mean", "exact", {0: "No functions"}), "exact must be a list"),
        (_set("cc_mean", "exact", [["0", "No functions"]]), "finite number"),
        (_set("cc_mean", "exact", [[0, ""]]), "non-empty string"),
        (_set("cc_mean", "exact", [[0]]), "pair"),
        (_set("cc_mean", "exact", [[0, "None {foo}"]]), "unknown field"),
    ],
)
def test_validate_templates_rejects(
    templates: dict[str, Any], mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    broken = copy.deepcopy(templates)
    mutate(broken)
    with pytest.raises(ValueError, match=message):
        validate_templates(broken)


def test_validate_templates_rejects_non_mapping_and_ignores_extra_keys(templates: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_templates(cast(Any, ["sloc"]))
    extended = copy.deepcopy(templates)
    extended["not_a_feature"] = "ignored"
    validate_templates(extended)
    assert {"adj", "value", "percentile", "top", "s"} == TEMPLATE_FIELDS


def test_load_templates(tmp_path: Path, templates: dict[str, Any]) -> None:
    assert load_templates() == yaml.safe_load(PACKAGED.read_text(encoding="utf-8"))

    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(templates, sort_keys=False), encoding="utf-8")
    assert load_templates(custom) == templates

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_templates(bad_yaml)

    a_list = tmp_path / "list.yaml"
    a_list.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_templates(a_list)

    with pytest.raises(FileNotFoundError):
        load_templates(tmp_path / "missing.yaml")

    broken = copy.deepcopy(templates)
    broken["fan_in"]["template"] = "{foo}"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="'fan_in'"):
        load_templates(invalid)


def test_bundles_carry_the_packaged_templates(dev_bundle: Bundle, dev_bundle_dir: Path) -> None:
    assert dev_bundle.reasons == load_templates()
    assert (dev_bundle_dir / REASONS_FILE).read_bytes() == PACKAGED.read_bytes()


def test_templates_round_trip_through_save_bundle(tmp_path: Path, dev_bundle: Bundle) -> None:
    target = save_bundle(
        tmp_path / "b",
        version="rt-1",
        booster=dev_bundle.ranker.booster,
        calibrator=dev_bundle.calibrator,
        metadata={},
        reasons=load_templates(),
    )
    loaded = load_bundle(target)
    assert loaded.reasons == load_templates()
    validate_templates(loaded.reasons)
    Explainer(loaded)


# ---------------------------------------------------------------- D. render_reasons


def golden_case() -> tuple[pd.Series, npt.NDArray[np.float64]]:
    row = make_row(
        churn_decayed=1459,
        pct_churn_decayed=0.99,
        prior_latent_bugs=4,
        pct_prior_latent_bugs=0.97,
        fan_in=12,
        pct_fan_in=0.85,
        test_importers=3,
        pct_test_importers=0.9,
        sloc=120,
        pct_sloc=0.4,
    )
    phi = make_phi(
        churn_decayed=2.0,
        pct_churn_decayed=1.0,
        prior_latent_bugs=1.5,
        sloc=0.1,
        fan_in=0.6,
        owner_share=-0.1,
        test_importers=-2.0,
    )
    return row, phi


def test_render_reasons_golden(templates: dict[str, Any]) -> None:
    row, phi = golden_case()
    reasons, mitigations = render_reasons(row, phi, templates)
    assert reasons == (
        Reason(
            "E",
            "churn_decayed",
            1459.0,
            99.0,
            3.0,
            "Very high recent churn: 1,459 recency-weighted lines added or deleted (top 1% in repo)",
        ),
        Reason(
            "D",
            "prior_latent_bugs",
            4.0,
            97.0,
            1.5,
            "Many past bugs: 4 past bugs traced back to its code (top 3% in repo)",
        ),
        Reason("I", "fan_in", 12.0, 85.0, 0.6, "Widely used: imported by 12 files (top 15% in repo)"),
    )
    assert mitigations == (
        Reason("T", "test_importers", 3.0, 90.0, -2.0, "Well tested: imported by 3 test files"),
    )
    assert family_totals(phi) == {"S": 0.1, "E": 3.0, "O": -0.1, "D": 1.5, "I": 0.6, "T": -2.0, "Q": 0.0}


def test_reason_share_rule_is_strict(templates: dict[str, Any]) -> None:
    row = make_row()
    assert REASON_MIN_SHARE * 20.0 == 1.0
    at_floor, _ = render_reasons(row, make_phi(churn_decayed=19.0, sloc=1.0), templates)
    assert [r.family for r in at_floor] == ["E"]
    above, _ = render_reasons(row, make_phi(churn_decayed=19.0, sloc=1.01), templates)
    assert [r.family for r in above] == ["E", "S"]


def test_max_reasons(templates: dict[str, Any]) -> None:
    row = make_row()
    phi = make_phi(
        sloc=6.0,
        churn_decayed=5.0,
        n_authors=4.0,
        prior_latent_bugs=3.0,
        fan_in=2.0,
        test_importers=1.5,
        satd_count=-8.0,
    )
    reasons, mitigations = render_reasons(row, phi, templates)
    assert MAX_REASONS == 4
    assert [r.family for r in reasons] == ["S", "E", "O", "D"]
    assert [m.family for m in mitigations] == ["Q"]
    two, mitigations_two = render_reasons(row, phi, templates, max_reasons=2)
    assert [r.family for r in two] == ["S", "E"]
    assert mitigations_two == mitigations
    none, mitigations_none = render_reasons(row, phi, templates, max_reasons=0)
    assert none == ()
    assert mitigations_none == mitigations
    with pytest.raises(ValueError, match="max_reasons"):
        render_reasons(row, phi, templates, max_reasons=-1)


def test_feature_choice_adds_raw_and_pct(templates: dict[str, Any]) -> None:
    row = make_row(sloc=500, pct_sloc=0.7, cc_max=12, pct_cc_max=0.8)
    phi = make_phi(sloc=0.3, pct_sloc=0.3, cc_max=0.5, pct_cc_max=-0.2)
    reasons, _ = render_reasons(row, phi, templates)
    assert len(reasons) == 1
    assert reasons[0].feature == "sloc"
    assert reasons[0].shap == family_totals(phi)["S"]
    assert reasons[0].shap == pytest.approx(0.9)
    assert reasons[0].percentile == 70.0


def test_binary_feature_percentiles(templates: dict[str, Any]) -> None:
    reasons, _ = render_reasons(make_row(has_test_file=0), make_phi(has_test_file=1.0), templates)
    assert reasons[0].feature == "has_test_file"
    assert reasons[0].percentile == 0.0
    assert reasons[0].value == 0.0
    assert reasons[0].text == "No test file named after it"

    _, mitigations = render_reasons(make_row(), make_phi(churn_decayed=1.0, owner_active=-1.0), templates)
    assert mitigations[0].feature == "owner_active"
    assert mitigations[0].percentile == 100.0
    assert mitigations[0].text == "Main author still active: committed in the last 90 days"


def test_mitigation_rule(templates: dict[str, Any]) -> None:
    row = make_row()
    _, at_quarter = render_reasons(row, make_phi(churn_decayed=4.0, test_importers=-1.0), templates)
    assert at_quarter == ()
    _, above = render_reasons(row, make_phi(churn_decayed=4.0, test_importers=-1.01), templates)
    assert [(m.family, m.feature) for m in above] == [("T", "test_importers")]

    phi = make_phi(churn_decayed=4.0, test_importers=-1.5, test_cochange_rate=-0.5, n_authors=-1.9)
    _, most_negative = render_reasons(row, phi, templates)
    assert [(m.family, m.feature, m.shap) for m in most_negative] == [("T", "test_importers", -2.0)]

    reasons, lone = render_reasons(row, make_phi(n_authors=-0.1), templates)
    assert reasons == ()
    assert [(m.family, m.feature) for m in lone] == [("O", "n_authors")]

    assert render_reasons(row, np.zeros(89), templates) == ((), ())


def test_ties_and_determinism(templates: dict[str, Any]) -> None:
    row = make_row()
    reasons, _ = render_reasons(row, make_phi(n_authors=1.0, churn_decayed=1.0, sloc=1.0), templates)
    assert [r.family for r in reasons] == ["S", "E", "O"]
    reasons, _ = render_reasons(row, make_phi(prior_latent_bugs=1.0, sloc=1.0), templates)
    assert [r.family for r in reasons] == ["S", "D"]

    reasons, _ = render_reasons(row, make_phi(sloc=0.5, cc_sum=0.5), templates)
    assert reasons[0].feature == "sloc"
    reasons, _ = render_reasons(row, make_phi(test_importers=0.5, has_test_file=0.5), templates)
    assert reasons[0].feature == "has_test_file"
    _, mitigations = render_reasons(row, make_phi(cc_sum=-0.5, sloc=-0.5), templates)
    assert mitigations[0].feature == "sloc"

    row, phi = golden_case()
    first = render_reasons(row, phi, templates)
    assert render_reasons(row, phi, templates) == first
    assert render_reasons(row.iloc[::-1], phi, templates) == first
    with_labels = pd.concat([pd.Series({"path": "a.py", "repo": "r"}), row.astype(object)])
    assert render_reasons(with_labels, phi, templates) == first


def test_values_are_cleaned_float32(templates: dict[str, Any]) -> None:
    row = make_row(dup_ratio=0.1, pct_dup_ratio=0.975)
    assert float(row["dup_ratio"]) != 0.1
    reasons, _ = render_reasons(row, make_phi(dup_ratio=1.0), templates)
    assert reasons[0].value == 0.1
    assert reasons[0].percentile == 97.5
    assert reasons[0].text == "Heavy duplication: 10% of its code is duplicated (top 2% in repo)"


def test_render_reasons_bad_input(templates: dict[str, Any]) -> None:
    row, phi = golden_case()
    with pytest.raises(KeyError):
        render_reasons(row.drop("sloc"), phi, templates)
    with pytest.raises(ValueError, match="sloc is not finite"):
        render_reasons(make_row(sloc=np.nan), phi, templates)
    with pytest.raises(ValueError, match="repeated"):
        render_reasons(pd.concat([row, row[["sloc"]]]), phi, templates)
    with pytest.raises(ValueError, match="shape"):
        render_reasons(row, phi[:-1], templates)
    with pytest.raises(ValueError, match="finite"):
        render_reasons(row, make_phi(sloc=np.nan), templates)
    partial = {k: v for k, v in templates.items() if k != "churn_decayed"}
    with pytest.raises(ValueError, match="churn_decayed"):
        render_reasons(row, phi, partial)


# ---------------------------------------------------------------- E. assign_debt_types


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"cc_max": 16}, "complexity"),
        ({"nesting_max": 5}, "complexity"),
        ({"dup_ratio": 0.11}, "duplication"),
        ({"test_importers": 0, "has_test_file": 0}, "test"),
        ({"scc_size": 2}, "architecture"),
        ({"hidden_coupling": 1, "pct_hidden_coupling": 0.95}, "architecture"),
        ({"betweenness": 0.2, "pct_betweenness": 0.96}, "architecture"),
        ({"owner_active": 0}, "knowledge"),
        ({"author_entropy": 1.5, "pct_author_entropy": 0.95}, "knowledge"),
        ({"satd_count": 2, "pct_satd_density": 0.9}, "admitted"),
    ],
)
def test_each_debt_rule(values: dict[str, float], expected: str) -> None:
    assert assign_debt_types(make_row(**values), ZERO_TOTALS) == (expected,)


@pytest.mark.parametrize(
    "values",
    [
        {"cc_max": 15},
        {"nesting_max": 4},
        {"dup_ratio": 0.10},
        {"scc_size": 1},
        {"pct_hidden_coupling": 0.9, "hidden_coupling": 1},
        {"pct_betweenness": 0.95, "betweenness": 0.1},
        {"pct_author_entropy": 0.9, "author_entropy": 1},
        {"satd_count": 1, "pct_satd_density": 0.89},
        {"satd_count": 0, "pct_satd_density": 0.99},
        {"test_importers": 1, "has_test_file": 0},
        {"test_importers": 0, "has_test_file": 1},
        {"pct_hidden_coupling": 1.0, "hidden_coupling": 0},
        {"pct_betweenness": 1.0, "betweenness": 0},
        {"pct_author_entropy": 1.0, "author_entropy": 0},
    ],
)
def test_debt_rule_boundaries(values: dict[str, float]) -> None:
    assert assign_debt_types(make_row(**values), ZERO_TOTALS) == ()


def test_defect_hotspot_rule() -> None:
    quiet = make_row()
    assert assign_debt_types(quiet, totals_with(D=2.0, E=1.0)) == ("defect_hotspot",)
    complex_row = make_row(cc_max=16)
    hotspot_first = assign_debt_types(complex_row, totals_with(D=2.0, E=1.0, S=0.5))
    assert hotspot_first == ("defect_hotspot", "complexity")
    assert assign_debt_types(complex_row, totals_with(E=2.0, S=1.5, D=1.0)) == ("complexity",)
    assert assign_debt_types(complex_row, totals_with(D=1.0, S=-1.0)) == ("complexity",)


def test_debt_type_order_and_cap() -> None:
    row = make_row(cc_max=16, test_importers=0, has_test_file=0, owner_active=0)
    assert MAX_DEBT_TYPES == 2
    assert assign_debt_types(row, totals_with(S=0.5, T=1.0, O=0.2)) == ("test", "complexity")
    both = make_row(cc_max=16, dup_ratio=0.2)
    assert assign_debt_types(both, totals_with(S=1.0)) == ("complexity", "duplication")
    assert assign_debt_types(make_row(cc_max=16), totals_with(D=1.0, E=3.0, S=0.5)) == (
        "defect_hotspot",
        "complexity",
    )
    negative = make_row(cc_max=16, owner_active=0)
    assert assign_debt_types(negative, totals_with(S=-1.0, O=-0.5)) == ("knowledge", "complexity")
    assert set(DEBT_TYPE_FAMILIES) == set(DEBT_TYPES)


@pytest.mark.parametrize("family", FAMILY_ORDER)
def test_dominant_family_fallback(family: str) -> None:
    totals = {f: (1.0 if f == family else -0.1) for f in FAMILY_ORDER}
    assert assign_debt_types(make_row(), totals) == (DOMINANT_DEBT_TYPE[family],)


def test_no_debt_type_without_a_positive_family() -> None:
    assert assign_debt_types(make_row(), ZERO_TOTALS) == ()
    assert assign_debt_types(make_row(), dict.fromkeys(FAMILY_ORDER, -0.5)) == ()


def test_assign_debt_types_bad_input() -> None:
    with pytest.raises(KeyError):
        assign_debt_types(make_row(), {f: 0.0 for f in FAMILY_ORDER if f != "Q"})
    with pytest.raises(ValueError, match="not finite"):
        assign_debt_types(make_row(), totals_with(S=float("nan")))
    with pytest.raises(ValueError, match="not finite"):
        assign_debt_types(make_row(cc_max=np.nan), ZERO_TOTALS)


# ---------------------------------------------------------------- F. Explainer.explain


def test_explain_equals_building_blocks(
    explainer: Explainer, query: pd.DataFrame, templates: dict[str, Any]
) -> None:
    X = query[COLS]
    phi = explainer.shap_values(X)
    centred = phi - phi.mean(axis=0)
    explanations = explainer.explain(X)
    assert len(explanations) == len(X)
    for i, explanation in enumerate(explanations):
        row = X.iloc[i]
        totals = family_totals(centred[i])
        reasons, mitigations = render_reasons(row, centred[i], templates)
        expected = Explanation(
            reasons, mitigations, assign_debt_types(row, totals), totals, feature_totals(centred[i])
        )
        assert explanation == expected


def test_explain_baselines_add_up(explainer: Explainer, dev_bundle: Bundle, query: pd.DataFrame) -> None:
    X = query[COLS]
    scores = dev_bundle.ranker.score(X)
    for baseline, reference in (("repo", scores.mean()), ("model", explainer.expected_value)):
        explanations = explainer.explain(X, baseline=cast(Any, baseline))
        by_family = np.array([sum(e.shap_by_family.values()) for e in explanations])
        by_feature = np.array([sum(e.shap_by_feature.values()) for e in explanations])
        assert np.abs(by_family - (scores - reference)).max() < 1e-4
        assert np.abs(by_feature - (scores - reference)).max() < 1e-4


def test_explain_single_row_and_empty(explainer: Explainer, query: pd.DataFrame) -> None:
    X = query[COLS]
    one = explainer.explain(X.iloc[:1])
    assert len(one) == 1
    assert one == explainer.explain(X.iloc[:1], baseline="model")
    assert explainer.explain(X.iloc[:0]) == []


def test_explain_rejections(explainer: Explainer, query: pd.DataFrame) -> None:
    X = query[COLS].head(10)
    with pytest.raises(ValueError, match="baseline"):
        explainer.explain(X, baseline=cast(Any, "global"))
    with pytest.raises(ValueError, match="max_reasons"):
        explainer.explain(X, max_reasons=-1)
    bad = X.copy()
    bad.iloc[3, COLS.index("fan_in")] = np.nan
    with pytest.raises(ValueError, match=rf"row {bad.index[3]}: feature fan_in is not finite"):
        explainer.explain(bad)
    bad.index = pd.Index(np.arange(10, 20, dtype=np.int32))
    with pytest.raises(ValueError, match="row 13: feature fan_in is not finite"):
        explainer.explain(bad)
    bad.index = pd.Index([f"pkg/m{k}.py" for k in range(10)])
    with pytest.raises(ValueError, match=re.escape("row 'pkg/m3.py': feature fan_in")):
        explainer.explain(bad)
    with pytest.raises(ValueError, match="columns"):
        explainer.explain(X.drop(columns=["fan_in"]))
    assert all(len(e.reasons) <= 1 for e in explainer.explain(X, max_reasons=1))


def test_explanations_cover_every_query(
    explainer: Explainer, dev_bundle: Bundle, synthetic_frame: pd.DataFrame
) -> None:
    high_files = 0
    for qid in range(30):
        Xq = synthetic_frame.loc[synthetic_frame["qid"] == qid, COLS]
        priorities = priority_from_scores(dev_bundle.ranker.score(Xq))
        explanations = explainer.explain(Xq)
        for (_, row), priority, e in zip(Xq.iterrows(), priorities, explanations, strict=True):
            if priority >= 75:
                high_files += 1
                assert e.reasons, f"query {qid}: a file with priority {priority} has no reason"
            assert len(e.reasons) <= MAX_REASONS
            shaps = [r.shap for r in e.reasons]
            assert all(s > 0 for s in shaps)
            assert shaps == sorted(shaps, reverse=True)
            assert len({r.family for r in e.reasons}) == len(e.reasons)
            assert len(e.mitigations) <= 1
            assert all(m.shap < 0 for m in e.mitigations)
            for r in (*e.reasons, *e.mitigations):
                assert r.feature in FEATURE_FAMILIES[r.family]
                assert r.shap == e.shap_by_family[r.family]
                assert 0.0 <= r.percentile <= 100.0
                assert r.value == cleaned(float(row[r.feature]))
                assert_clean_text(r.text)
            assert set(e.debt_types) <= set(DEBT_TYPES)
            assert len(e.debt_types) <= MAX_DEBT_TYPES
            assert len(set(e.debt_types)) == len(e.debt_types)
            assert list(e.shap_by_family) == list(FAMILY_ORDER)
            assert list(e.shap_by_feature) == list(FEATURE_COLUMNS)
    assert high_files > 1000


def test_explain_ignores_row_order(explainer: Explainer, query: pd.DataFrame) -> None:
    X = query[COLS]
    forward = explainer.explain(X)
    backward = explainer.explain(X.iloc[::-1])[::-1]
    for a, b in zip(forward, backward, strict=True):
        assert [(r.feature, r.text) for r in a.reasons] == [(r.feature, r.text) for r in b.reasons]
        assert [(m.feature, m.text) for m in a.mitigations] == [(m.feature, m.text) for m in b.mitigations]
        assert a.debt_types == b.debt_types
        for fam in FAMILY_ORDER:
            assert a.shap_by_family[fam] == pytest.approx(b.shap_by_family[fam], abs=1e-9)


def test_explainer_rejects_bad_templates(dev_bundle: Bundle) -> None:
    with pytest.raises(ValueError, match="miss 46 of 46"):
        Explainer(dataclasses.replace(dev_bundle, reasons={}))
    broken = copy.deepcopy(dev_bundle.reasons)
    broken["fan_in"]["template"] = "{value!r}"
    with pytest.raises(ValueError, match="'fan_in'"):
        Explainer(dataclasses.replace(dev_bundle, reasons=broken))


def test_file_score_from_explanation_serialises(
    explainer: Explainer, dev_bundle: Bundle, query: pd.DataFrame
) -> None:
    X = query[COLS]
    scores = dev_bundle.ranker.score(X)
    priorities = priority_from_scores(scores)
    risk, low, high = dev_bundle.calibrator.predict(scores)
    explanations = explainer.explain(X)
    i = int(np.argmax(scores))
    e = explanations[i]
    assert e.reasons
    fs = FileScore(
        path=str(query["path"].iloc[i]),
        score=float(scores[i]),
        priority=int(priorities[i]),
        band=band_for(int(priorities[i])),
        risk=float(risk[i]),
        risk_low=float(low[i]),
        risk_high=float(high[i]),
        reasons=e.reasons,
        mitigations=e.mitigations,
        debt_types=e.debt_types,
        shap_by_family=e.shap_by_family,
    )
    text = json.dumps(to_jsonable(fs), allow_nan=False)
    assert from_jsonable(FileScore, json.loads(text)) == fs
