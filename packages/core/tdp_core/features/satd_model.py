"""Numpy runtime of the self-admitted technical debt (SATD) comment classifier.

The classifier is a TF-IDF vectoriser followed by a binary logistic regression, trained offline with
scikit-learn by ``tdp-ml satd train`` (module ``tdp_ml.satd.train``) on the labelled comment dataset of
Maldonado, Shihab and Tsantalis (IEEE TSE 2017). The trained weights ship next to this module as
``satd_model.json``, and :class:`JsonSatdClassifier` scores comments with numpy and the standard library only,
reproducing scikit-learn's ``TfidfVectorizer`` and ``LogisticRegression.predict_proba`` exactly (up to
floating point rounding), so analysis never has to import scikit-learn or unpickle anything.

Model file format (``MODEL_FORMAT`` version ``MODEL_FORMAT_VERSION``), one compact JSON object with keys in
this order:

- ``format``, ``format_version``: identify the layout.
- ``vectorizer``: ``{"lowercase", "token_pattern", "ngram_range": [min_n, max_n], "sublinear_tf",
  "norm": "l2"}``.
- ``intercept``: the logistic regression intercept (full precision).
- ``metadata``: free-form provenance (dataset, parameters, evaluation results); never read by scoring.
- ``terms``: the vocabulary in column order, so a term's position in the list is its column index.
- ``idf``, ``coef``: one float per term (inverse document frequency, logistic regression weight).

:class:`JsonSatdClassifier` satisfies the ``SatdClassifier`` protocol of :mod:`tdp_core.features.satd`:
``predict_proba(texts)`` returns P(debt) per text as a 1-D float64 array.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

TOKEN_PATTERN: Final = r"(?u)\b\w\w+\b"
"""scikit-learn's default ``token_pattern``: runs of two or more word characters."""

MODEL_FORMAT: Final = "tdp-satd-tfidf-logreg"
MODEL_FORMAT_VERSION: Final = 1
WEIGHT_DECIMALS: Final = 9
"""Decimal places kept for ``idf`` and ``coef`` in the JSON file (changes scores by about 1e-10)."""

DEFAULT_MODEL_PATH: Final = Path(__file__).with_name("satd_model.json")

MAX_NGRAM: Final = 3
_Z_CLIP: Final = 500.0


class SatdModelError(ValueError):
    """A SATD model is malformed, inconsistent or uses settings the numpy runtime does not support."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _weights(name: str, values: npt.ArrayLike, n: int) -> FloatArray:
    try:
        arr = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise SatdModelError(f"{name} must be a list of numbers: {exc}") from exc
    if arr.ndim != 1:
        raise SatdModelError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise SatdModelError(f"{name} has {arr.shape[0]} values but there are {n} terms")
    if not np.all(np.isfinite(arr)):
        raise SatdModelError(f"{name} contains NaN or infinite values")
    arr.setflags(write=False)
    return arr


def _plain_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise SatdModelError(f"metadata must be a mapping, got {type(metadata).__name__}")
    try:
        copied = json.loads(json.dumps(dict(metadata), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SatdModelError(f"metadata must be JSON-serialisable: {exc}") from exc
    if not isinstance(copied, dict):  # pragma: no cover - json round trip of a dict is a dict
        raise SatdModelError("metadata must be a JSON object")
    return copied


def _rounded(values: FloatArray, decimals: int | None) -> list[float]:
    if decimals is None:
        return [float(v) for v in values.tolist()]
    return [round(float(v), decimals) for v in values.tolist()]


class JsonSatdClassifier:
    """TF-IDF plus logistic regression SATD classifier evaluated with numpy.

    Scoring one text (exactly scikit-learn's ``TfidfVectorizer(analyzer="word", norm="l2", use_idf=True)``
    followed by ``LogisticRegression.predict_proba(X)[:, 1]``):

    1. ``doc = text.lower()`` when ``lowercase``; ``tokens = re.findall(TOKEN_PATTERN, doc)``.
    2. Word n-grams for every n in ``ngram_range`` (tokens joined by one space), in scikit-learn's order.
    3. ``count[j]`` = occurrences of vocabulary term ``j``; n-grams outside the vocabulary are ignored.
    4. ``tf[j] = 1 + ln(count[j])`` when ``sublinear_tf``, else ``count[j]``; ``w[j] = tf[j] * idf[j]``.
    5. ``z = intercept + sum_j(w[j] * coef[j]) / sqrt(sum_j(w[j] ** 2))``; a text with no vocabulary term
       (empty, punctuation only, single letters, unknown words) scores ``z = intercept``.
    6. ``P(debt) = 1 / (1 + exp(-z))`` with ``z`` clipped to [-500, 500].

    Instances are immutable after construction, safe to share between threads, and picklable (for process
    pools). Weights are float64; ``terms`` must be unique non-empty strings.
    """

    __slots__ = (
        "_coef",
        "_idf",
        "_index",
        "_intercept",
        "_lowercase",
        "_metadata",
        "_ngram_range",
        "_pattern",
        "_source_sha256",
        "_sublinear_tf",
        "_terms",
    )

    def __init__(
        self,
        terms: Sequence[str],
        idf: npt.ArrayLike,
        coef: npt.ArrayLike,
        intercept: float,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
        sublinear_tf: bool = True,
        metadata: Mapping[str, Any] | None = None,
        source_sha256: str | None = None,
    ) -> None:
        """Validate and freeze the model parameters.

        Args:
            terms: vocabulary in column order (term ``terms[j]`` uses ``idf[j]`` and ``coef[j]``).
            idf: inverse document frequency per term, each finite and > 0.
            coef: logistic regression weight per term (positive pushes towards debt), each finite.
            intercept: logistic regression intercept, a finite number.
            ngram_range: ``(min_n, max_n)`` word n-gram lengths, ``1 <= min_n <= max_n <= 3``.
            lowercase: lowercase texts before tokenising.
            sublinear_tf: use ``1 + ln(count)`` instead of raw counts.
            metadata: JSON-serialisable provenance kept with the model (deep-copied, read-only).
            source_sha256: SHA-256 of the file the model was read from, if any.

        Raises:
            SatdModelError: any parameter is missing, malformed or inconsistent.
        """
        if isinstance(terms, str | bytes) or not isinstance(terms, Iterable):
            raise SatdModelError("terms must be a list of strings")
        term_tuple = tuple(terms)
        if not term_tuple:
            raise SatdModelError("terms must not be empty")
        for term in term_tuple:
            if not isinstance(term, str) or not term:
                raise SatdModelError(f"every term must be a non-empty string, got {term!r}")
        index = {term: i for i, term in enumerate(term_tuple)}
        if len(index) != len(term_tuple):
            raise SatdModelError("terms must be unique")
        n = len(term_tuple)
        idf_arr = _weights("idf", idf, n)
        if not np.all(idf_arr > 0):
            raise SatdModelError("idf values must be > 0")
        coef_arr = _weights("coef", coef, n)
        if not _is_number(intercept) or not math.isfinite(float(intercept)):
            raise SatdModelError(f"intercept must be a finite number, got {intercept!r}")
        if (
            not isinstance(ngram_range, tuple | list)
            or len(ngram_range) != 2
            or not all(_is_int(v) for v in ngram_range)
            or not 1 <= ngram_range[0] <= ngram_range[1] <= MAX_NGRAM
        ):
            raise SatdModelError(
                f"ngram_range must be two integers with 1 <= min_n <= max_n <= {MAX_NGRAM}, "
                f"got {ngram_range!r}"
            )
        if not isinstance(lowercase, bool) or not isinstance(sublinear_tf, bool):
            raise SatdModelError("lowercase and sublinear_tf must be booleans")
        if source_sha256 is not None and not isinstance(source_sha256, str):
            raise SatdModelError("source_sha256 must be a string or None")

        self._terms: tuple[str, ...] = term_tuple
        self._index: dict[str, int] = index
        self._idf: FloatArray = idf_arr
        self._coef: FloatArray = coef_arr
        self._intercept: float = float(intercept)
        self._ngram_range: tuple[int, int] = (int(ngram_range[0]), int(ngram_range[1]))
        self._lowercase: bool = lowercase
        self._sublinear_tf: bool = sublinear_tf
        self._metadata: MappingProxyType[str, Any] = MappingProxyType(_plain_metadata(metadata))
        self._source_sha256: str | None = source_sha256
        self._pattern: re.Pattern[str] = re.compile(TOKEN_PATTERN)

    # ------------------------------------------------------------------ construction from files and sklearn
    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> JsonSatdClassifier:
        """Load a model written by :meth:`to_json`.

        ``source_sha256`` of the result is the SHA-256 of the file bytes, so callers can record exactly
        which weights produced their features.

        Raises:
            FileNotFoundError: the file does not exist.
            SatdModelError: the file is not valid UTF-8 JSON or not a supported model.
        """
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SatdModelError(f"{path} is not a valid JSON model file: {exc}") from exc
        if not isinstance(data, dict):
            raise SatdModelError(f"{path} must hold a JSON object")
        return cls.from_dict(data, source_sha256=digest)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, source_sha256: str | None = None) -> JsonSatdClassifier:
        """Build a classifier from the dictionary layout produced by :meth:`to_dict`.

        Raises:
            SatdModelError: a key is missing or ill-typed, the format or version is unknown, or the
                vectoriser settings (token pattern, norm) differ from what the runtime implements.
        """
        if not isinstance(data, Mapping):
            raise SatdModelError("model data must be a mapping")
        try:
            if data["format"] != MODEL_FORMAT:
                raise SatdModelError(f"unknown model format {data['format']!r}, expected {MODEL_FORMAT!r}")
            version = data["format_version"]
            if not _is_int(version) or version != MODEL_FORMAT_VERSION:
                raise SatdModelError(
                    f"unsupported format_version {version!r}, expected {MODEL_FORMAT_VERSION}"
                )
            vec = data["vectorizer"]
            if not isinstance(vec, Mapping):
                raise SatdModelError("vectorizer must be an object")
            if vec["token_pattern"] != TOKEN_PATTERN:
                raise SatdModelError(f"unsupported token_pattern {vec['token_pattern']!r}")
            if vec["norm"] != "l2":
                raise SatdModelError(f"unsupported norm {vec['norm']!r}, only 'l2' is implemented")
            ngram = vec["ngram_range"]
            if not isinstance(ngram, list) or len(ngram) != 2 or not all(_is_int(v) for v in ngram):
                raise SatdModelError(f"ngram_range must be a list of two integers, got {ngram!r}")
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise SatdModelError("metadata must be an object")
            terms = data["terms"]
            if not isinstance(terms, list):
                raise SatdModelError("terms must be a list")
            idf = data["idf"]
            coef = data["coef"]
            if not isinstance(idf, list) or not isinstance(coef, list):
                raise SatdModelError("idf and coef must be lists")
            return cls(
                terms,
                idf,
                coef,
                data["intercept"],
                ngram_range=(ngram[0], ngram[1]),
                lowercase=vec["lowercase"],
                sublinear_tf=vec["sublinear_tf"],
                metadata=metadata,
                source_sha256=source_sha256,
            )
        except KeyError as exc:
            raise SatdModelError(f"model data is missing key {exc}") from exc
        except TypeError as exc:
            raise SatdModelError(f"model data is ill-typed: {exc}") from exc

    @classmethod
    def from_sklearn(
        cls, vectorizer: Any, model: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> JsonSatdClassifier:
        """Copy the weights of a fitted ``TfidfVectorizer`` and binary ``LogisticRegression``.

        The objects are duck-typed (this module never imports scikit-learn). Only settings the runtime
        reproduces exactly are accepted: word analyser on string input with :data:`TOKEN_PATTERN`, no custom
        tokenizer, preprocessor, stop words or accent stripping, ``binary=False``, ``use_idf=True``,
        ``norm="l2"`` and float64 output. The model must have two classes that are ``False``/``True`` (or
        ``0``/``1``), so column 1 of ``predict_proba`` is "debt"; string labels are rejected because their
        sorted order could silently invert the model.

        Raises:
            SatdModelError: unsupported settings, an unfitted object, or inconsistent shapes.
        """
        try:
            params = vectorizer.get_params()
            expected: dict[str, Any] = {
                "analyzer": "word",
                "input": "content",
                "token_pattern": TOKEN_PATTERN,
                "tokenizer": None,
                "preprocessor": None,
                "stop_words": None,
                "binary": False,
                "use_idf": True,
                "norm": "l2",
            }
            for key, value in expected.items():
                if params[key] != value:
                    raise SatdModelError(f"unsupported vectorizer setting {key}={params[key]!r}")
            if params["strip_accents"]:
                accents = params["strip_accents"]
                raise SatdModelError(f"unsupported vectorizer setting strip_accents={accents!r}")
            if np.dtype(params["dtype"]) != np.float64:
                raise SatdModelError(f"unsupported vectorizer dtype {params['dtype']!r}, expected float64")
            vocabulary: Mapping[str, Any] = vectorizer.vocabulary_
            n = len(vocabulary)
            by_index = sorted((int(i), str(t)) for t, i in vocabulary.items())
            if [i for i, _ in by_index] != list(range(n)):
                raise SatdModelError("vocabulary_ indices must be exactly 0..n-1")
            terms = [t for _, t in by_index]
            idf = np.asarray(vectorizer.idf_, dtype=np.float64)
            coef = np.asarray(model.coef_, dtype=np.float64)
            intercept = np.asarray(model.intercept_, dtype=np.float64)
            classes = np.asarray(model.classes_)
            ngram = params["ngram_range"]
        except AttributeError as exc:
            raise SatdModelError(f"vectorizer and model must be fitted scikit-learn objects: {exc}") from exc
        except (KeyError, TypeError) as exc:
            raise SatdModelError(f"unsupported vectorizer or model: {exc}") from exc
        if classes.shape != (2,) or classes.dtype.kind not in "biu" or classes.tolist() != [0, 1]:
            raise SatdModelError(
                f"model classes must be [False, True] (or [0, 1]), got {classes.tolist()!r}; "
                "train on boolean labels so column 1 is debt"
            )
        if coef.shape != (1, n):
            raise SatdModelError(f"model coef_ must have shape (1, {n}), got {coef.shape}")
        if intercept.shape != (1,):
            raise SatdModelError(f"model intercept_ must have shape (1,), got {intercept.shape}")
        return cls(
            terms,
            idf,
            coef[0],
            float(intercept[0]),
            ngram_range=(ngram[0], ngram[1]),
            lowercase=bool(params["lowercase"]),
            sublinear_tf=bool(params["sublinear_tf"]),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ serialisation
    def to_dict(self, *, decimals: int | None = WEIGHT_DECIMALS) -> dict[str, Any]:
        """The JSON layout of this model (see the module docstring).

        Args:
            decimals: decimal places kept for ``idf`` and ``coef`` (Python ``round``, shortest repr);
                ``None`` keeps full float64 precision. The intercept always keeps full precision.
        """
        return {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION,
            "vectorizer": {
                "lowercase": self._lowercase,
                "token_pattern": TOKEN_PATTERN,
                "ngram_range": [self._ngram_range[0], self._ngram_range[1]],
                "sublinear_tf": self._sublinear_tf,
                "norm": "l2",
            },
            "intercept": self._intercept,
            "metadata": json.loads(json.dumps(dict(self._metadata))),
            "terms": list(self._terms),
            "idf": _rounded(self._idf, decimals),
            "coef": _rounded(self._coef, decimals),
        }

    def to_json(
        self,
        path: str | os.PathLike[str],
        *,
        decimals: int | None = WEIGHT_DECIMALS,
        max_bytes: int | None = None,
    ) -> int:
        """Write the model as one line of ASCII JSON plus a newline; returns the number of bytes written.

        The file is written to ``<name>.tmp`` in the same folder and then renamed over ``path``, so readers
        never see a partial file. Parent folders are created.

        Args:
            path: destination file.
            decimals: see :meth:`to_dict`.
            max_bytes: refuse to write a file larger than this many bytes.

        Raises:
            ValueError: the encoded model is larger than ``max_bytes`` (nothing is written).
        """
        text = json.dumps(
            self.to_dict(decimals=decimals), separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
        payload = (text + "\n").encode("ascii")
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(
                f"the SATD model needs {len(payload):,} bytes, more than the limit of {max_bytes:,} bytes"
            )
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        logger.info("wrote SATD model with %d terms to %s (%d bytes)", self.n_features, dest, len(payload))
        return len(payload)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _restore,
            (
                self._terms,
                self._idf,
                self._coef,
                self._intercept,
                self._ngram_range,
                self._lowercase,
                self._sublinear_tf,
                dict(self._metadata),
                self._source_sha256,
            ),
        )

    # ------------------------------------------------------------------ scoring
    def analyze(self, text: str) -> list[str]:
        """The word n-grams of ``text`` in scikit-learn's order (equal to ``build_analyzer()(text)``).

        Unigrams come first (when ``min_n == 1``), then all bigrams, then trigrams. Duplicates are kept.
        """
        doc = text.lower() if self._lowercase else text
        tokens: list[str] = self._pattern.findall(doc)
        min_n, max_n = self._ngram_range
        if max_n == 1:
            return tokens
        if min_n == 1:
            grams = list(tokens)
            min_n = 2
        else:
            grams = []
        n_tokens = len(tokens)
        for n in range(min_n, min(max_n + 1, n_tokens + 1)):
            for i in range(n_tokens - n + 1):
                grams.append(" ".join(tokens[i : i + n]))
        return grams

    def decision_function(self, texts: Sequence[str]) -> FloatArray:
        """Logistic regression margin ``z`` per text (log-odds of debt), float64 of shape ``(len(texts),)``.

        Raises:
            TypeError: ``texts`` is a single ``str``/``bytes`` or contains a non-string.
        """
        items = _check_texts(texts)
        n_texts = len(items)
        rows: list[int] = []
        cols: list[int] = []
        counts: list[int] = []
        index = self._index
        for row, text in enumerate(items):
            hits: dict[int, int] = {}
            for gram in self.analyze(text):
                col = index.get(gram)
                if col is not None:
                    hits[col] = hits.get(col, 0) + 1
            for col, count in hits.items():
                rows.append(row)
                cols.append(col)
                counts.append(count)
        z = np.full(n_texts, self._intercept, dtype=np.float64)
        if not cols:
            return z
        row_arr = np.asarray(rows, dtype=np.intp)
        col_arr = np.asarray(cols, dtype=np.intp)
        tf = np.asarray(counts, dtype=np.float64)
        if self._sublinear_tf:
            tf = np.log(tf) + 1.0
        w = tf * self._idf[col_arr]
        sq = np.bincount(row_arr, weights=w * w, minlength=n_texts)
        dot = np.bincount(row_arr, weights=w * self._coef[col_arr], minlength=n_texts)
        has_terms = sq > 0.0
        z[has_terms] += dot[has_terms] / np.sqrt(sq[has_terms])
        return z

    def predict_proba(self, texts: Sequence[str]) -> FloatArray:
        """P(debt) per text in [0, 1], float64 of shape ``(len(texts),)``; ``[]`` gives shape ``(0,)``.

        This is column 1 of scikit-learn's ``predict_proba`` (the ``SatdClassifier`` protocol). The model
        was trained with balanced class weights, so 0.5 is the balanced decision threshold.

        Raises:
            TypeError: ``texts`` is a single ``str``/``bytes`` or contains a non-string.
        """
        z = np.clip(self.decision_function(texts), -_Z_CLIP, _Z_CLIP)
        proba: FloatArray = 1.0 / (1.0 + np.exp(-z))
        return proba

    # ------------------------------------------------------------------ read-only views
    @property
    def terms(self) -> tuple[str, ...]:
        """Vocabulary in column order."""
        return self._terms

    @property
    def idf(self) -> FloatArray:
        """Inverse document frequency per term (read-only float64 array)."""
        return self._idf

    @property
    def coef(self) -> FloatArray:
        """Logistic regression weight per term (read-only float64 array)."""
        return self._coef

    @property
    def intercept(self) -> float:
        """Logistic regression intercept (the margin of a text with no vocabulary term)."""
        return self._intercept

    @property
    def ngram_range(self) -> tuple[int, int]:
        """``(min_n, max_n)`` word n-gram lengths."""
        return self._ngram_range

    @property
    def lowercase(self) -> bool:
        """Whether texts are lowercased before tokenising."""
        return self._lowercase

    @property
    def sublinear_tf(self) -> bool:
        """Whether term frequency is ``1 + ln(count)``."""
        return self._sublinear_tf

    @property
    def n_features(self) -> int:
        """Vocabulary size."""
        return len(self._terms)

    @property
    def metadata(self) -> MappingProxyType[str, Any]:
        """Provenance stored with the model (read-only view)."""
        return self._metadata

    @property
    def source_sha256(self) -> str | None:
        """SHA-256 of the file the model was loaded from, or ``None`` for an in-memory model."""
        return self._source_sha256

    def __repr__(self) -> str:
        return f"JsonSatdClassifier(n_features={self.n_features}, ngram_range={self._ngram_range})"


def _check_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str | bytes):
        raise TypeError("texts must be a sequence of strings, not a single string")
    items = list(texts)
    for text in items:
        if not isinstance(text, str):
            raise TypeError(f"every text must be a str, got {type(text).__name__}")
    return items


def _restore(
    terms: tuple[str, ...],
    idf: FloatArray,
    coef: FloatArray,
    intercept: float,
    ngram_range: tuple[int, int],
    lowercase: bool,
    sublinear_tf: bool,
    metadata: dict[str, Any],
    source_sha256: str | None,
) -> JsonSatdClassifier:
    return JsonSatdClassifier(
        terms,
        idf,
        coef,
        intercept,
        ngram_range=ngram_range,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,
        metadata=metadata,
        source_sha256=source_sha256,
    )


@functools.cache
def load_default_classifier() -> JsonSatdClassifier:
    """The trained classifier shipped with ``tdp_core`` (``DEFAULT_MODEL_PATH``), loaded once per process.

    Raises:
        FileNotFoundError: the weights file is missing (train it with ``uv run tdp-ml satd train``).
        SatdModelError: the file is corrupt.
    """
    if not DEFAULT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SATD model weights not found at {DEFAULT_MODEL_PATH}; train it with `uv run tdp-ml satd train`"
        )
    return JsonSatdClassifier.from_json(DEFAULT_MODEL_PATH)
