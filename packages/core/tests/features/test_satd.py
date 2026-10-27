from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from tdp_core.contracts import FEATURE_FAMILIES, CommentInfo, FileSyntax
from tdp_core.features.satd import (
    SATD_COLUMNS,
    SATD_PATTERN,
    SatdClassifier,
    satd_candidates,
    satd_features,
)
from tdp_core.features.satd_model import load_default_classifier
from tdp_core.parsing.python import parse_python


def c(line: int, text: str) -> CommentInfo:
    return CommentInfo(line, text)


def syn(sloc: int, comments: Sequence[CommentInfo]) -> FileSyntax:
    """A FileSyntax with only SLOC and comments filled in."""
    return FileSyntax("pkg/mod.py", "python", sloc, 1, (), 0, tuple(comments), (), (), (), False)


class FakeClassifier:
    """Returns a fixed probability per text and records the texts of every call."""

    def __init__(self, probs: Mapping[str, float], default: float = 0.0) -> None:
        self.probs = dict(probs)
        self.default = default
        self.calls: list[list[str]] = []

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        self.calls.append(list(texts))
        return np.array([self.probs.get(t, self.default) for t in texts], dtype=np.float64)


class FixedOutput:
    """Returns the same array whatever it is asked."""

    def __init__(self, output: npt.NDArray[np.float64]) -> None:
        self.output = output

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        return self.output


FOUR = (c(3, "TODO: x"), c(5, "plain"), c(9, "HACK"), c(12, "temporary file"))
PROBS = {"TODO: x": 0.9, "HACK": 0.5, "temporary file": 0.2}


def test_columns_are_family_q() -> None:
    assert FEATURE_FAMILIES["Q"] == SATD_COLUMNS
    assert SATD_COLUMNS == ("satd_count", "satd_density", "satd_age_days")


# The positive and negative lists are exact on purpose: they pin the word boundaries and the keyword list.
@pytest.mark.parametrize(
    "text",
    [
        "TODO: remove",
        "todo later",
        "Todo(ana): x",
        "FIXME",
        "fixme: off by one",
        "HACK",
        "this is a hack",
        "XXX: check this",
        "xxx",
        "KLUDGE",
        "kludge for py2",
        "workaround for bpo-123",
        "Workaround",
        "work around the bug",
        "temporary until 2.0",
        "temp fix",
        "quick fix for #12",
        "ugly but works",
        "#TODO nested hash",
        "@todo",
        "see TODO.",
        "(HACK)",
    ],
)
def test_pattern_positives(text: str) -> None:
    assert satd_candidates([c(1, text)]) == [c(1, text)]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "normal comment",
        "mastodon",
        "todolist",
        "todo_list",
        "TODOs",
        "fixmes",
        "hacks",
        "hackathon",
        "shack",
        "kludges",
        "workarounds",
        "work-around",
        "temporarily",
        "temp-fix",
        "quick-fix",
        "uglify",
        "XXXL",
        "type: ignore",
        "noqa: E501",
        "pragma: no cover",
    ],
)
def test_pattern_negatives(text: str) -> None:
    assert SATD_PATTERN.search(text) is None
    assert satd_candidates([c(1, text)]) == []


def test_candidates_keep_order_and_identity() -> None:
    result = satd_candidates(FOUR)
    assert [x.line for x in result] == [3, 9, 12]
    assert all(x is FOUR[i] for x, i in zip(result, (0, 2, 3), strict=True))
    assert satd_candidates([]) == []


def test_without_classifier_counts_every_candidate() -> None:
    out = satd_features(syn(40, FOUR), None, None)
    assert out == {"satd_count": 3.0, "satd_density": 75.0, "satd_age_days": 0.0}
    ages = {3: 10.0, 5: 999.0, 9: 400.0, 12: 30.0}
    # Line 5 is not a candidate, so its age is ignored.
    assert satd_features(syn(40, FOUR), None, ages)["satd_age_days"] == 400.0


def test_zero_sloc_and_no_candidates() -> None:
    out = satd_features(syn(0, [c(1, "TODO"), c(2, "FIXME")]), None, None)
    assert out["satd_density"] == 2000.0
    clf = FakeClassifier({}, default=1.0)
    empty = satd_features(syn(10, [c(1, "plain"), c(2, "also plain")]), clf, {1: 50.0})
    assert empty == {"satd_count": 0.0, "satd_density": 0.0, "satd_age_days": 0.0}
    assert clf.calls == []


def test_ages() -> None:
    s = syn(40, FOUR)
    assert satd_features(s, None, None)["satd_age_days"] == 0.0
    assert satd_features(s, None, {1: 100.0, 5: 200.0})["satd_age_days"] == 0.0
    assert satd_features(s, None, {3: 7.5, 12: 2.0})["satd_age_days"] == 7.5
    assert satd_features(s, None, {3: -4.0, 9: -1.0})["satd_age_days"] == 0.0
    assert satd_features(s, None, {3: math.nan, 9: 5.0})["satd_age_days"] == 5.0
    assert satd_features(s, None, {3: math.inf, 9: 5.0, 12: -math.inf})["satd_age_days"] == 5.0


def test_with_classifier_counts_candidates_at_or_above_threshold() -> None:
    s = syn(40, FOUR)
    clf = FakeClassifier(PROBS)
    out = satd_features(s, clf, {3: 10.0, 9: 20.0, 12: 900.0})
    assert out == {"satd_count": 2.0, "satd_density": 50.0, "satd_age_days": 20.0}
    assert clf.calls == [["TODO: x", "HACK", "temporary file"]]
    assert satd_features(s, FakeClassifier(PROBS), None, threshold=0.95)["satd_count"] == 0.0
    assert satd_features(s, FakeClassifier(PROBS), None, threshold=0.1)["satd_count"] == 3.0


@pytest.mark.parametrize(
    "output",
    [
        np.array([0.9, 0.5], dtype=np.float64),
        np.full((3, 2), 0.5, dtype=np.float64),
        np.array([0.9, math.nan, 0.2], dtype=np.float64),
    ],
)
def test_bad_classifier_output_raises(output: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        satd_features(syn(40, FOUR), FixedOutput(output), None)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan])
def test_bad_threshold_raises_even_without_candidates(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        satd_features(syn(10, FOUR), None, None, threshold=threshold)
    with pytest.raises(ValueError, match="threshold"):
        satd_features(syn(10, []), None, None, threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_bounds_are_accepted(threshold: float) -> None:
    out = satd_features(syn(40, FOUR), FakeClassifier(PROBS), None, threshold=threshold)
    assert out["satd_count"] == (3.0 if threshold == 0.0 else 0.0)


def test_output_keys_and_types() -> None:
    for clf in (None, FakeClassifier(PROBS)):
        out = satd_features(syn(40, FOUR), clf, {3: 1.0})
        assert list(out) == list(SATD_COLUMNS)
        assert all(type(v) is float for v in out.values())


def test_fake_classifier_satisfies_the_protocol() -> None:
    clf: SatdClassifier = FakeClassifier({})
    assert satd_features(syn(1, []), clf, None)["satd_count"] == 0.0


def test_with_real_parser_docstrings_are_not_comments() -> None:
    source = (
        b"import os  # TODO: drop this\n"
        b"\n"
        b"def f():\n"
        b'    """TODO: not a comment, a docstring."""\n'
        b"    x = 1  # hack around the parser\n"
        b"    # FIXME later\n"
        b"    return x\n"
    )
    syntax = parse_python("pkg/mod.py", source)
    assert syntax.sloc == 5
    assert [x.line for x in syntax.comments] == [1, 5, 6]
    out = satd_features(syntax, None, None)
    assert out["satd_count"] == 3.0
    assert out["satd_density"] == 600.0


def test_shipped_classifier_filters_candidates() -> None:
    clf: SatdClassifier = load_default_classifier()
    comments = (
        c(2, "TODO: remove this hack"),
        c(4, "Create a temporary file"),
        c(7, "FIXME: this is ugly"),
        c(8, "plain comment"),
    )
    out = satd_features(syn(20, comments), clf, {2: 30.0, 4: 900.0, 7: 10.0, 8: 5000.0})
    # The keyword-only count is 3; the classifier rejects the temporary file comment and its age.
    assert satd_features(syn(20, comments), None, None)["satd_count"] == 3.0
    assert out == {"satd_count": 2.0, "satd_density": 100.0, "satd_age_days": 30.0}
