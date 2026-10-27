import copy
import hashlib
import json
import math
import pickle
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tdp_core.features import satd_model
from tdp_core.features.satd_model import (
    DEFAULT_MODEL_PATH,
    MODEL_FORMAT,
    TOKEN_PATTERN,
    JsonSatdClassifier,
    SatdModelError,
    load_default_classifier,
)

PINNED_DATASET_URL = (
    "https://raw.githubusercontent.com/maldonado/tse.satd.data/"
    "1c6842f27be1739cd54eff38fdfbc10ea98e091e/dataset/technical_debt_dataset.zip"
)

DEBT_WORDS = ("todo", "fixme", "hack", "workaround", "temporary", "ugly", "should", "not sure")
PLAIN_WORDS = (
    "return", "the", "value", "of", "list", "user", "name", "file", "path", "count", "rows", "table",
    "data", "parse", "input", "config", "default", "cache", "item", "index", "x1", "foo_bar", "v2",
    "a", "b", "is", "empty", "when", "called", "with", "none", "2024", "http_client",
)  # fmt: skip
SEPARATORS = (" ", "  ", ", ", ". ", ": ", "-", "/", "\t", "(", ") ", "\n", "_", "!", "'")

EDGE_TEXTS = [
    "",
    "   ",
    "!!!",
    "a b c",
    "todo todo todo",
    "TODO TODO todo ToDo",
    "caf\u00e9 na\u00efve r\u00e9sum\u00e9",
    "\u65e5\u672c\u8a9e todo",
    "\u00c9COLE \u00e9cole \u0130stanbul",
    "x_1 __init__ foo123 12 3",
    "fix-me later... maybe?",
    " ".join(f"w{i % 37} todo" if i % 11 == 0 else f"w{i % 37}" for i in range(500)),
]


def make_text(rng: random.Random, debt: bool) -> str:
    words = [rng.choice(PLAIN_WORDS) for _ in range(rng.randint(1, 14))]
    if debt:
        for _ in range(rng.randint(1, 2)):
            words.insert(rng.randrange(len(words) + 1), rng.choice(DEBT_WORDS))
    if rng.random() < 0.2:
        words.append(rng.choice(words))
    styled = []
    for word in words:
        roll = rng.random()
        styled.append(word.upper() if roll < 0.1 else word.capitalize() if roll < 0.25 else word)
    return "".join(word + rng.choice(SEPARATORS) for word in styled).strip()


def make_corpus(seed: int, n: int) -> tuple[list[str], np.ndarray]:
    """Synthetic comments; about 30% debt, with 8% label noise so the classes overlap."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[bool] = []
    for _ in range(n):
        debt = rng.random() < 0.3
        texts.append(make_text(rng, debt))
        labels.append(debt if rng.random() >= 0.08 else not debt)
    return texts, np.array(labels, dtype=bool)


def fit_sklearn(
    texts: Sequence[str], labels: np.ndarray, **vectorizer_kw: Any
) -> tuple[TfidfVectorizer, LogisticRegression]:
    params: dict[str, Any] = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True, "lowercase": True}
    params.update(vectorizer_kw)
    vectorizer = TfidfVectorizer(**params)
    x = vectorizer.fit_transform(list(texts))
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(x, labels)
    return vectorizer, model


def sklearn_proba(vectorizer: TfidfVectorizer, model: LogisticRegression, texts: Sequence[str]) -> np.ndarray:
    return np.asarray(model.predict_proba(vectorizer.transform(list(texts)))[:, 1], dtype=np.float64)


def eval_texts() -> list[str]:
    texts, _ = make_corpus(seed=7, n=200 - len(EDGE_TEXTS))
    return EDGE_TEXTS + texts


@pytest.fixture(scope="module")
def fitted() -> tuple[TfidfVectorizer, LogisticRegression]:
    texts, labels = make_corpus(seed=42, n=400)
    return fit_sklearn(texts, labels)


def tiny(**kw: Any) -> JsonSatdClassifier:
    return JsonSatdClassifier(["todo", "hack", "todo hack"], [1.5, 2.0, 3.0], [1.2, 0.7, -0.4], -0.3, **kw)


# ---------------------------------------------------------------------------------------------- tokenising
@pytest.mark.parametrize(
    ("ngram_range", "lowercase"),
    [
        ((1, 1), True),
        ((1, 2), True),
        ((1, 2), False),
        ((1, 3), True),
        ((2, 2), True),
        ((2, 3), False),
        ((3, 3), True),
    ],
)
def test_analyze_matches_sklearn_analyzer(ngram_range: tuple[int, int], lowercase: bool) -> None:
    analyzer = TfidfVectorizer(ngram_range=ngram_range, lowercase=lowercase).build_analyzer()
    clf = tiny(ngram_range=ngram_range, lowercase=lowercase)
    for text in eval_texts():
        assert clf.analyze(text) == analyzer(text), text


# ---------------------------------------------------------------------------------------------- parity
def test_parity_with_sklearn_on_200_comments(
    fitted: tuple[TfidfVectorizer, LogisticRegression], tmp_path: Path
) -> None:
    vectorizer, model = fitted
    texts = eval_texts()
    assert len(texts) == 200
    expected = sklearn_proba(vectorizer, model, texts)
    assert 0.05 < float(np.mean(expected >= 0.5)) < 0.95  # the model is not trivial

    clf = JsonSatdClassifier.from_sklearn(vectorizer, model)
    assert clf.n_features == len(vectorizer.vocabulary_)
    assert float(np.max(np.abs(clf.predict_proba(texts) - expected))) < 1e-10
    margins = model.decision_function(vectorizer.transform(texts))
    assert float(np.max(np.abs(clf.decision_function(texts) - margins))) < 1e-9

    path = tmp_path / "model.json"
    clf.to_json(path)
    loaded = JsonSatdClassifier.from_json(path)
    assert float(np.max(np.abs(loaded.predict_proba(texts) - expected))) < 1e-6


@pytest.mark.parametrize(
    "vectorizer_kw",
    [
        {"ngram_range": (1, 1), "lowercase": True, "sublinear_tf": True},
        {"ngram_range": (1, 2), "lowercase": False, "sublinear_tf": True, "min_df": 2},
        {"ngram_range": (1, 3), "lowercase": True, "sublinear_tf": False, "max_features": 50},
        {"ngram_range": (2, 2), "lowercase": True, "sublinear_tf": True},
    ],
)
def test_parity_across_vectorizer_settings(vectorizer_kw: dict[str, Any], tmp_path: Path) -> None:
    texts, labels = make_corpus(seed=3, n=300)
    vectorizer, model = fit_sklearn(texts, labels, **vectorizer_kw)
    clf = JsonSatdClassifier.from_sklearn(vectorizer, model)
    assert clf.ngram_range == tuple(vectorizer_kw["ngram_range"])
    assert clf.lowercase is vectorizer_kw["lowercase"]
    assert clf.sublinear_tf is vectorizer_kw["sublinear_tf"]
    clf.to_json(tmp_path / "m.json")
    loaded = JsonSatdClassifier.from_json(tmp_path / "m.json")
    sample = eval_texts()
    assert (
        float(np.max(np.abs(loaded.predict_proba(sample) - sklearn_proba(vectorizer, model, sample)))) < 1e-6
    )


def test_unknown_and_empty_texts_score_the_intercept() -> None:
    clf = tiny()
    texts = ["", "   ", "!!!", "a b c", "zzz unknown words only", "\u65e5\u672c"]
    margins = clf.decision_function(texts)
    assert np.all(margins == clf.intercept)
    assert np.allclose(clf.predict_proba(texts), 1.0 / (1.0 + math.exp(-clf.intercept)), rtol=0, atol=1e-15)


def test_hand_computed_score() -> None:
    clf = tiny()
    # "TODO hack todo": todo x2, hack x1, "todo hack" x1, "hack todo" unknown.
    tf = np.array([1.0 + math.log(2.0), 1.0, 1.0])
    w = tf * np.array([1.5, 2.0, 3.0])
    z = -0.3 + float(w @ np.array([1.2, 0.7, -0.4])) / math.sqrt(float(w @ w))
    assert clf.decision_function(["TODO hack todo"])[0] == pytest.approx(z, abs=1e-12)
    assert clf.predict_proba(["TODO hack todo"])[0] == pytest.approx(1.0 / (1.0 + math.exp(-z)), abs=1e-12)


def test_predict_proba_shapes() -> None:
    clf = tiny()
    out = clf.predict_proba(["todo", "nothing here", "hack todo hack"])
    assert out.dtype == np.float64
    assert out.shape == (3,)
    assert np.all((out >= 0.0) & (out <= 1.0))
    empty = clf.predict_proba([])
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    assert clf.predict_proba(("todo",)).shape == (1,)
    assert clf.decision_function([]).shape == (0,)


def test_extreme_margins_do_not_overflow() -> None:
    clf = JsonSatdClassifier(["todo"], [1.0], [5000.0], 0.0)
    with np.errstate(all="raise"):
        p = clf.predict_proba(["todo", "x"])
    assert p[0] == pytest.approx(1.0)
    assert p[1] == 0.5


@pytest.mark.parametrize("texts", ["todo", b"x", ["ok", None], [b"x"], [1]])
def test_rejects_single_string_and_non_strings(texts: Any) -> None:
    clf = tiny()
    with pytest.raises(TypeError):
        clf.predict_proba(texts)
    with pytest.raises(TypeError):
        clf.decision_function(texts)


# ---------------------------------------------------------------------------------------------- JSON format
def test_json_round_trip_and_size_guard(
    fitted: tuple[TfidfVectorizer, LogisticRegression], tmp_path: Path
) -> None:
    vectorizer, model = fitted
    clf = JsonSatdClassifier.from_sklearn(vectorizer, model, metadata={"note": "synthetic", "n": [1, 2]})
    texts = eval_texts()

    path = tmp_path / "sub" / "model.json"
    written = clf.to_json(path)
    raw = path.read_bytes()
    assert written == len(raw)
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b"\r" not in raw
    raw.decode("ascii")
    data = json.loads(raw)
    assert list(data) == [
        "format",
        "format_version",
        "vectorizer",
        "intercept",
        "metadata",
        "terms",
        "idf",
        "coef",
    ]
    assert list(data["vectorizer"]) == ["lowercase", "token_pattern", "ngram_range", "sublinear_tf", "norm"]
    assert data["format"] == MODEL_FORMAT
    assert data["vectorizer"]["token_pattern"] == TOKEN_PATTERN
    assert data["metadata"] == {"note": "synthetic", "n": [1, 2]}
    assert data["terms"] == list(clf.terms)
    assert all(len(repr(v).split(".")[-1]) <= 9 for v in data["coef"])
    assert not list(tmp_path.glob("**/*.tmp"))

    nine = JsonSatdClassifier.from_json(path)
    assert nine.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert float(np.max(np.abs(nine.predict_proba(texts) - clf.predict_proba(texts)))) < 1e-9

    full_path = tmp_path / "full.json"
    clf.to_json(full_path, decimals=None)
    full = JsonSatdClassifier.from_json(full_path)
    assert np.array_equal(full.idf, clf.idf)
    assert np.array_equal(full.coef, clf.coef)
    assert full.intercept == clf.intercept
    assert np.array_equal(full.predict_proba(texts), clf.predict_proba(texts))
    assert dict(full.metadata) == dict(clf.metadata)

    small = tmp_path / "small.json"
    with pytest.raises(ValueError, match="bytes"):
        clf.to_json(small, max_bytes=100)
    assert not small.exists()
    assert not (tmp_path / "small.json.tmp").exists()
    # An existing file is left untouched when the new model is too large.
    path_before = path.read_bytes()
    with pytest.raises(ValueError):
        clf.to_json(path, max_bytes=100)
    assert path.read_bytes() == path_before


def test_in_memory_model_has_no_source_hash() -> None:
    clf = tiny()
    assert clf.source_sha256 is None
    assert JsonSatdClassifier.from_dict(clf.to_dict()).source_sha256 is None
    assert "n_features=3" in repr(clf)
    assert "ngram_range=(1, 2)" in repr(clf)


def test_weights_are_read_only() -> None:
    clf = tiny(metadata={"a": 1})
    with pytest.raises(ValueError):
        clf.idf[0] = 9.0
    with pytest.raises(ValueError):
        clf.coef[0] = 9.0
    with pytest.raises(TypeError):
        clf.metadata["a"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        clf.extra = 1  # type: ignore[attr-defined]
    source = np.array([1.0, 2.0, 3.0])
    clf2 = JsonSatdClassifier(["a1", "b2", "c3"], source, [0.0, 0.0, 0.0], 0.0)
    source[0] = 100.0
    assert clf2.idf[0] == 1.0  # the constructor copies its inputs


Mutation = Callable[[dict[str, Any]], None]


def _set(*keys: str, value: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target = data
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return mutate


def _delete(*keys: str) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        target = data
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return mutate


def _append_term(term: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        data["terms"].append(term)
        data["idf"].append(1.0)
        data["coef"].append(0.0)

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _set("format", value="other-format"),
        _set("format_version", value=2),
        _set("format_version", value="1"),
        _set("vectorizer", "token_pattern", value=r"\b\w+\b"),
        _set("vectorizer", "norm", value="l1"),
        _set("vectorizer", "ngram_range", value=[0, 1]),
        _set("vectorizer", "ngram_range", value=[2, 1]),
        _set("vectorizer", "ngram_range", value=[1, 4]),
        _set("vectorizer", "ngram_range", value="x"),
        _set("vectorizer", "ngram_range", value=[1.0, 2.0]),
        _set("vectorizer", "lowercase", value="yes"),
        _set("vectorizer", "sublinear_tf", value=1),
        _set("vectorizer", value=[]),
        _set("idf", value=[1.0, 2.0]),
        _set("coef", value=[1.0]),
        _set("idf", value="1,2,3"),
        _set("coef", value=[[1.0, 2.0, 3.0]]),
        _append_term("todo"),
        _append_term(7),
        _append_term(""),
        _set("terms", value=[]),
        _set("terms", value="todo"),
        _set("idf", value=[1.0, float("nan"), 3.0]),
        _set("coef", value=[1.0, float("inf"), 3.0]),
        _set("coef", value=[1.0, "x", 3.0]),
        _set("intercept", value=float("nan")),
        _set("intercept", value=float("inf")),
        _set("intercept", value=True),
        _set("intercept", value="0.5"),
        _set("idf", value=[1.0, 0.0, 3.0]),
        _set("idf", value=[1.0, -2.0, 3.0]),
        _delete("terms"),
        _delete("format"),
        _delete("intercept"),
        _delete("vectorizer", "norm"),
        _delete("vectorizer", "lowercase"),
        _set("metadata", value=[1, 2]),
    ],
)
def test_from_dict_rejects_malformed(mutate: Mutation) -> None:
    data = tiny().to_dict()
    mutate(data)
    with pytest.raises(SatdModelError):
        JsonSatdClassifier.from_dict(data)


def test_from_dict_accepts_missing_metadata() -> None:
    data = tiny().to_dict()
    del data["metadata"]
    assert dict(JsonSatdClassifier.from_dict(data).metadata) == {}


def test_from_json_rejects_non_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe not json")
    with pytest.raises(SatdModelError):
        JsonSatdClassifier.from_json(bad)
    bad.write_bytes(b"[1, 2]")
    with pytest.raises(SatdModelError):
        JsonSatdClassifier.from_json(bad)
    with pytest.raises(FileNotFoundError):
        JsonSatdClassifier.from_json(tmp_path / "missing.json")


def test_constructor_rejects_non_serialisable_metadata() -> None:
    with pytest.raises(SatdModelError):
        tiny(metadata={"x": object()})
    with pytest.raises(SatdModelError):
        tiny(metadata={"x": float("nan")})


@pytest.mark.parametrize(
    "vectorizer_kw",
    [
        {"token_pattern": r"(?u)\b\w+\b"},
        {"analyzer": "char"},
        {"stop_words": "english"},
        {"binary": True},
        {"use_idf": False},
        {"norm": "l1"},
        {"norm": None},
        {"strip_accents": "unicode"},
        {"dtype": np.float32},
        {"preprocessor": str.upper},
    ],
)
def test_from_sklearn_rejects_unsupported_vectorizers(vectorizer_kw: dict[str, Any]) -> None:
    texts, labels = make_corpus(seed=5, n=120)
    vectorizer, model = fit_sklearn(texts, labels, **vectorizer_kw)
    with pytest.raises(ValueError):
        JsonSatdClassifier.from_sklearn(vectorizer, model)


def test_from_sklearn_rejects_unsupported_models() -> None:
    texts, labels = make_corpus(seed=5, n=150)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x = vectorizer.fit_transform(texts)

    string_labels = LogisticRegression(max_iter=1000).fit(x, np.where(labels, "debt", "clean"))
    with pytest.raises(ValueError, match="classes"):
        JsonSatdClassifier.from_sklearn(vectorizer, string_labels)

    three = LogisticRegression(max_iter=1000).fit(x, np.arange(len(texts)) % 3)
    with pytest.raises(ValueError):
        JsonSatdClassifier.from_sklearn(vectorizer, three)

    with pytest.raises(ValueError, match="fitted"):
        JsonSatdClassifier.from_sklearn(TfidfVectorizer(), string_labels)
    with pytest.raises(ValueError, match="fitted"):
        JsonSatdClassifier.from_sklearn(vectorizer, LogisticRegression())

    int_labels = LogisticRegression(max_iter=1000).fit(x, labels.astype(int))
    clf = JsonSatdClassifier.from_sklearn(vectorizer, int_labels)
    sample = eval_texts()
    assert (
        float(np.max(np.abs(clf.predict_proba(sample) - sklearn_proba(vectorizer, int_labels, sample))))
        < 1e-10
    )


def test_classifier_survives_deepcopy_and_pickle(fitted: tuple[TfidfVectorizer, LogisticRegression]) -> None:
    clf = JsonSatdClassifier.from_sklearn(*fitted, metadata={"k": {"nested": [1]}})
    texts = eval_texts()
    expected = clf.predict_proba(texts)
    for clone in (copy.deepcopy(clf), pickle.loads(pickle.dumps(clf)), copy.copy(clf)):
        assert np.array_equal(clone.predict_proba(texts), expected)
        assert clone.terms == clf.terms
        assert dict(clone.metadata) == dict(clf.metadata)
        assert not clone.idf.flags.writeable
        assert not clone.coef.flags.writeable


# ---------------------------------------------------------------------------------------------- shipped model
def test_default_model_file_is_small_and_loads() -> None:
    assert DEFAULT_MODEL_PATH.is_file()
    size = DEFAULT_MODEL_PATH.stat().st_size
    assert size < 2_000_000
    clf = JsonSatdClassifier.from_json(DEFAULT_MODEL_PATH)
    assert clf.n_features >= 1000
    assert clf.ngram_range == (1, 2)
    assert clf.lowercase and clf.sublinear_tf
    meta = clf.metadata
    assert meta["dataset"]["url"] == PINNED_DATASET_URL
    assert meta["dataset"]["rows"] > 60_000
    assert 0.0 < meta["lopo"]["macro_f1"] <= 1.0
    assert meta["parity"]["max_abs_diff"] < 1e-6
    assert meta["vocabulary_size"] == clf.n_features
    assert clf.source_sha256 == hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()


def python_like_comments(n: int = 200) -> list[str]:
    rng = random.Random(2024)
    verbs = ("Return", "Compute", "Parse", "Load", "Close", "Check", "Build", "Skip", "Retry", "Cache")
    nouns = ("the config", "user rows", "this file", "the parser state", "HTTP headers", "a list of paths")
    tails = (
        "", ".", " if it is empty.", " before the loop", " (see issue 12)", " for Python 3.12",
        " TODO: handle None", " FIXME this is slow", " hack: avoid the lock", " temporary workaround",
        " noqa: E501", " type: ignore[attr-defined]", " -> int", " x == y and not z",
    )  # fmt: skip
    out = [f"{rng.choice(verbs)} {rng.choice(nouns)}{rng.choice(tails)}" for _ in range(n - len(EDGE_TEXTS))]
    return EDGE_TEXTS + out


def test_default_model_matches_sklearn_on_200_comments() -> None:
    clf = load_default_classifier()
    vectorizer = TfidfVectorizer(
        vocabulary=list(clf.terms),
        ngram_range=clf.ngram_range,
        lowercase=clf.lowercase,
        sublinear_tf=clf.sublinear_tf,
    )
    vectorizer.idf_ = np.array(clf.idf)
    model = LogisticRegression()
    model.coef_ = np.array(clf.coef).reshape(1, -1)
    model.intercept_ = np.array([clf.intercept])
    model.classes_ = np.array([False, True])
    texts = python_like_comments(200)
    assert len(texts) == 200
    expected = sklearn_proba(vectorizer, model, texts)
    assert float(np.max(np.abs(clf.predict_proba(texts) - expected))) < 1e-6


DEBT_EXAMPLES = (
    "TODO: this is a temporary hack, clean it up later",
    "FIXME: this breaks when the input list is empty",
    "XXX ugly workaround for a bug in the parser",
    "HACK: remove this once the upstream fix is released",
)
CLEAN_EXAMPLES = (
    "Return the number of rows in the table.",
    "Initialise the connection pool with default settings.",
    "Iterate over all files in the directory.",
    "The user name is stored in lower case.",
)


def test_default_model_sanity() -> None:
    clf = load_default_classifier()
    p_debt = clf.predict_proba(DEBT_EXAMPLES)
    p_clean = clf.predict_proba(CLEAN_EXAMPLES)
    for text, p in zip(DEBT_EXAMPLES, p_debt, strict=True):
        assert p >= 0.5, (text, p)
    for text, p in zip(CLEAN_EXAMPLES, p_clean, strict=True):
        assert p < 0.5, (text, p)
    assert float(np.mean(p_debt) - np.mean(p_clean)) > 0.3


def test_load_default_classifier_is_cached() -> None:
    assert load_default_classifier() is load_default_classifier()


def test_load_default_classifier_reports_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(satd_model, "DEFAULT_MODEL_PATH", tmp_path / "missing.json")
    load_default_classifier.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="tdp-ml satd train"):
            load_default_classifier()
    finally:
        load_default_classifier.cache_clear()


class _SatdClassifierLike(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


def _count_debt(classifier: _SatdClassifierLike, texts: Sequence[str], threshold: float = 0.5) -> int:
    return int((classifier.predict_proba(texts) >= threshold).sum())


def test_default_classifier_fits_the_satd_protocol() -> None:
    assert _count_debt(load_default_classifier(), list(DEBT_EXAMPLES) + list(CLEAN_EXAMPLES)) == len(
        DEBT_EXAMPLES
    )


def test_runtime_does_not_import_sklearn() -> None:
    code = (
        "import sys\n"
        "from tdp_core.features.satd_model import load_default_classifier\n"
        "assert load_default_classifier().predict_proba(['todo: hack']).shape == (1,)\n"
        "assert 'sklearn' not in sys.modules, 'sklearn was imported'\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
