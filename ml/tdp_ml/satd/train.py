"""Download the SATD comment dataset, evaluate the classifier by project, and export the JSON weights.

Dataset: E. da S. Maldonado, E. Shihab and N. Tsantalis, "Using Natural Language Processing to Automatically
Detect Self-Admitted Technical Debt", IEEE Transactions on Software Engineering 43(11), 2017
(doi:10.1109/TSE.2017.2654244). About 62k source code comments from 10 Java projects, each labelled with a
debt type (DESIGN, DEFECT, DOCUMENTATION, IMPLEMENTATION, TEST) or WITHOUT_CLASSIFICATION. The download is
pinned to one commit of the authors' repository and verified by hash. The dataset carries no license file:
it is used here for training only and is never redistributed (it stays under ``$TDP_DATA_DIR/satd/``); only
the derived weights are committed.

Model: ``TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)`` (capped at the
``max_features`` most frequent n-grams) followed by ``LogisticRegression(class_weight="balanced")`` with C
chosen from ``C_GRID`` by leave-one-project-out (LOPO) macro F1. The final model is refitted on every row and
exported with :meth:`tdp_core.features.satd_model.JsonSatdClassifier.to_json`, after a parity check against
scikit-learn.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import platform
import re
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.features.satd_model import JsonSatdClassifier
from tdp_core.paths import data_dir, results_dir

logger = logging.getLogger(__name__)

Bools = npt.NDArray[np.bool_]
Floats = npt.NDArray[np.float64]

DATASET_NAME = "Maldonado, Shihab and Tsantalis SATD comment dataset (IEEE TSE 2017)"
DATASET_REPO_URL = "https://github.com/maldonado/tse.satd.data"
DATASET_COMMIT = "1c6842f27be1739cd54eff38fdfbc10ea98e091e"
DATASET_URL = (
    "https://raw.githubusercontent.com/maldonado/tse.satd.data/"
    f"{DATASET_COMMIT}/dataset/technical_debt_dataset.zip"
)
DATASET_BLOB_SHA1 = "7ea5be9fd95e143c38905e37dda77e418a9631ca"
DATASET_ZIP_BYTES = 1_180_637
DATASET_ZIP_SHA256 = "d67fa276b618f78797a0b124e0345e1978c975637f8cc84dba1fad4a5ff08472"
DATASET_CSV_SHA256 = "bd01a13566634ecba72261714bb9f1b3e40f7249cb9b27d2babc495c986b7492"
DATASET_CITATION = (
    'E. da S. Maldonado, E. Shihab, N. Tsantalis, "Using Natural Language Processing to Automatically Detect '
    'Self-Admitted Technical Debt", IEEE Transactions on Software Engineering 43(11), 2017, '
    "doi:10.1109/TSE.2017.2654244"
)
DATASET_LICENSE = (
    "The dataset is published without a license file as research data accompanying the paper; it is used "
    "for training only and is not redistributed, and only the derived weights are shipped."
)
ZIP_NAME = "technical_debt_dataset.zip"
CSV_NAME = "technical_debt_dataset.csv"
NEGATIVE_CLASS = "WITHOUT_CLASSIFICATION"

C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
DEFAULT_MAX_FEATURES = 40_000
MAX_MODEL_BYTES = 2_000_000
MAX_DOWNLOAD_BYTES = 50_000_000
PARITY_SAMPLE = 200
PARITY_TOLERANCE = 1e-6
PARITY_EXTRA_TEXTS = ("", "   ", "!!!", "TODO", "todo: remove this hack", "Return the number of rows.")
THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_THRESHOLD = 0.5
MIN_DF = 2
NGRAM_RANGE = (1, 2)
MAX_ITER = 3000
TOP_TERMS = 20

# Diagnostic copy of the SATD candidate rule of tdp_core.features.satd (kept separate on purpose: the
# report measures how the classifier behaves on the comments that rule would hand it).
CANDIDATE_RE = re.compile(
    r"\b(?:TODO|FIXME|HACK|XXX|KLUDGE|workaround|work around|temporary|temp fix|quick fix|ugly)\b",
    re.IGNORECASE,
)
CANDIDATE_TERMS = (
    "todo",
    "fixme",
    "hack",
    "xxx",
    "kludge",
    "workaround",
    "work around",
    "temporary",
    "temp fix",
    "quick fix",
    "ugly",
)

_REQUIRED_COLUMNS = ("projectname", "classification", "commenttext")


# ---------------------------------------------------------------------------------------------- paths
def default_dataset_dir() -> Path:
    """``$TDP_DATA_DIR/satd``: where the dataset zip and CSV are stored."""
    return data_dir() / "satd"


def default_report_path() -> Path:
    """``$TDP_RESULTS_DIR/satd/lopo_report.json``: the full training and evaluation report."""
    return results_dir() / "satd" / "lopo_report.json"


# ---------------------------------------------------------------------------------------------- download
def git_blob_sha1(data: bytes) -> str:
    """Git's object id of ``data`` stored as a blob: ``sha1(b"blob <len>\\0" + data)``, lowercase hex."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _fetch(url: str, *, retries: int, wait_s: float, timeout_s: float, max_bytes: int) -> bytes:
    attempts = max(retries, 0) + 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                data: bytes = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"{url} is larger than {max_bytes:,} bytes")
            return data
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
            problem: Exception = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            if attempt == attempts:
                raise
            problem = exc
        logger.warning(
            "download attempt %d of %d failed (%s); retrying in %.0f s", attempt, attempts, problem, wait_s
        )
        time.sleep(wait_s)
    raise AssertionError("unreachable")  # pragma: no cover


def _hash_problem(data: bytes, expected_blob_sha1: str | None) -> str | None:
    if expected_blob_sha1 is None:
        return None
    actual = git_blob_sha1(data)
    if actual != expected_blob_sha1.lower():
        return f"git blob SHA-1 is {actual}, expected {expected_blob_sha1}"
    return None


def download_dataset(
    dest_dir: Path,
    *,
    url: str = DATASET_URL,
    expected_blob_sha1: str | None = DATASET_BLOB_SHA1,
    retries: int = 5,
    wait_s: float = 30.0,
    timeout_s: float = 120.0,
    force: bool = False,
) -> Path:
    """Fetch the dataset zip into ``dest_dir``, verify it, and extract the CSV; returns the CSV path.

    A zip already in ``dest_dir`` whose git blob SHA-1 matches is reused without a download (unless
    ``force``). Downloads are retried ``retries`` times (so ``retries + 1`` attempts), ``wait_s`` seconds
    apart, on connection errors, timeouts, truncated reads and HTTP 5xx; HTTP 4xx fails at once.

    Args:
        dest_dir: target folder (created); receives ``technical_debt_dataset.zip`` and ``.csv``.
        url: ``https://`` or ``file:`` URL of the zip (other schemes are refused).
        expected_blob_sha1: git blob SHA-1 the zip must have; ``None`` skips verification.
        retries: extra attempts after the first failure.
        wait_s: pause between attempts in seconds.
        timeout_s: socket timeout per attempt in seconds.
        force: download even when a verified zip exists.

    Raises:
        ValueError: unsupported URL scheme, hash mismatch (nothing is kept), or a bad zip.
        urllib.error.URLError: the download kept failing.
    """
    if not (url.startswith("https://") or url.startswith("file:")):
        raise ValueError(f"only https:// and file: URLs are allowed, got {url!r}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / ZIP_NAME
    if not force and zip_path.is_file() and _hash_problem(zip_path.read_bytes(), expected_blob_sha1) is None:
        logger.info("reusing verified dataset zip %s", zip_path)
    else:
        started = time.perf_counter()
        data = _fetch(url, retries=retries, wait_s=wait_s, timeout_s=timeout_s, max_bytes=MAX_DOWNLOAD_BYTES)
        problem = _hash_problem(data, expected_blob_sha1)
        if problem is not None:
            raise ValueError(f"downloaded dataset from {url} failed verification: {problem}")
        part = zip_path.with_name(zip_path.name + ".part")
        try:
            part.write_bytes(data)
            os.replace(part, zip_path)
        finally:
            part.unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        logger.info("downloaded %s (%d bytes) in %.1f s to %s", url, len(data), elapsed, zip_path)
    return extract_csv(zip_path, dest_dir)


def extract_csv(zip_path: Path, dest_dir: Path, *, max_bytes: int = 200_000_000) -> Path:
    """Extract only the dataset CSV member of ``zip_path`` to ``dest_dir/technical_debt_dataset.csv``.

    The member named exactly ``technical_debt_dataset.csv`` is used, or else the single member with that
    base name outside ``__MACOSX/``. Member paths are never used for output and no other member is read.

    Raises:
        ValueError: not a zip file, no (or an ambiguous) CSV member, or the member is larger than
            ``max_bytes`` bytes uncompressed.
    """
    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            exact = [i for i in infos if i.filename == CSV_NAME]
            if not exact:
                exact = [
                    i
                    for i in infos
                    if PurePosixPath(i.filename).name == CSV_NAME
                    and "__MACOSX" not in PurePosixPath(i.filename).parts
                ]
            if len(exact) != 1:
                names = [i.filename for i in infos]
                raise ValueError(f"{zip_path} must contain exactly one {CSV_NAME}, found members {names}")
            info = exact[0]
            if info.file_size > max_bytes:
                raise ValueError(f"{info.filename} is {info.file_size:,} bytes, more than {max_bytes:,}")
            with archive.open(info) as member:
                data = member.read(max_bytes + 1)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{zip_path} is not a valid zip file: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError(f"{info.filename} is larger than {max_bytes:,} bytes")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / CSV_NAME
    part = out.with_name(out.name + ".part")
    try:
        part.write_bytes(data)
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------------------------- dataset
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Read the labelled comment CSV into ``project, classification, text, is_debt`` columns.

    Bytes that are not valid UTF-8 become U+FFFD; no value is treated as missing (a comment ``NA`` stays
    text). Column names are matched after stripping, removing a byte order mark and lowercasing. Project
    names are stripped, classifications stripped and uppercased; ``is_debt`` is ``classification !=
    WITHOUT_CLASSIFICATION``. Rows whose text is blank are dropped; the original row order is kept and the
    index is reset. ``df.attrs`` records ``rows_read``, ``dropped_empty`` and ``replacement_chars``.

    Raises:
        ValueError: a required column is missing, or a row has an empty project or classification.
    """
    raw = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    raw.columns = [str(c).strip().lstrip("\ufeff").strip().lower() for c in raw.columns]
    missing = [c for c in _REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks columns {missing}; found {list(raw.columns)}")
    project = raw["projectname"].astype(str).str.strip()
    classification = raw["classification"].astype(str).str.strip().str.upper()
    text = raw["commenttext"].astype(str)
    if (project == "").any():
        raise ValueError(f"{csv_path}: {int((project == '').sum())} rows have an empty projectname")
    if (classification == "").any():
        raise ValueError(f"{csv_path}: {int((classification == '').sum())} rows have an empty classification")
    keep = (text.str.strip() != "").to_numpy(dtype=bool)
    df = pd.DataFrame(
        {
            "project": project.to_numpy(dtype=object)[keep],
            "classification": classification.to_numpy(dtype=object)[keep],
            "text": text.to_numpy(dtype=object)[keep],
        }
    )
    df["is_debt"] = (df["classification"] != NEGATIVE_CLASS).to_numpy(dtype=bool)
    replacement = int(sum(t.count("\ufffd") for t in df["text"]))
    df.attrs.update(
        {"rows_read": len(raw), "dropped_empty": int((~keep).sum()), "replacement_chars": replacement}
    )
    logger.info(
        "read %d rows from %s: kept %d, dropped %d blank, %d replacement characters",
        len(raw),
        csv_path,
        len(df),
        int((~keep).sum()),
        replacement,
    )
    return df


# ---------------------------------------------------------------------------------------------- model
def make_vectorizer(max_features: int | None = DEFAULT_MAX_FEATURES) -> TfidfVectorizer:
    """The TF-IDF vectoriser of the plan: word 1-2 grams, ``min_df=2``, sublinear tf, lowercase, l2 norm.

    ``max_features`` keeps the most frequent n-grams (by total count in the training texts); ``None`` keeps
    all n-grams that pass ``min_df``.
    """
    return TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,
        lowercase=True,
        max_features=max_features,
    )


def make_classifier(c: float, seed: int = DEFAULT_SETTINGS.seed) -> LogisticRegression:
    """Balanced L2 logistic regression (lbfgs, ``MAX_ITER`` iterations) with inverse regularisation ``c``."""
    return LogisticRegression(
        C=c, class_weight="balanced", solver="lbfgs", max_iter=MAX_ITER, random_state=seed
    )


def _fit(model: LogisticRegression, x: Any, y: Bools, label: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            logger.warning("%s: %s", label, warning.message)
        else:  # pragma: no cover - re-emit anything unrelated
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    if [bool(c) for c in model.classes_] != [False, True]:
        raise RuntimeError(f"{label}: expected classes [False, True], got {list(model.classes_)}")


# ---------------------------------------------------------------------------------------------- metrics
@dataclass(frozen=True, slots=True)
class BinaryCounts:
    """Confusion counts of the debt class. Ratios with a zero denominator are 0.0."""

    n: int
    positives: int
    tp: int
    fp: int
    fn: int

    @property
    def predicted(self) -> int:
        """Rows predicted as debt."""
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        """``tp / (tp + fp)``."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """``tp / (tp + fn)``."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """``2 tp / (2 tp + fp + fn)``."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    def __add__(self, other: BinaryCounts) -> BinaryCounts:
        return BinaryCounts(
            self.n + other.n,
            self.positives + other.positives,
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )

    def to_dict(self) -> dict[str, int | float]:
        """Counts plus precision, recall and F1 (rounded to 6 decimals)."""
        return {
            "n": self.n,
            "positives": self.positives,
            "predicted": self.predicted,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


ZERO_COUNTS = BinaryCounts(0, 0, 0, 0, 0)


def binary_counts(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> BinaryCounts:
    """Confusion counts of two equal-length boolean vectors (``True`` = debt).

    Raises:
        ValueError: the vectors differ in length.
    """
    truth = np.asarray(y_true, dtype=bool).ravel()
    pred = np.asarray(y_pred, dtype=bool).ravel()
    if truth.shape != pred.shape:
        raise ValueError(f"y_true has {truth.size} values but y_pred has {pred.size}")
    return BinaryCounts(
        n=int(truth.size),
        positives=int(truth.sum()),
        tp=int((truth & pred).sum()),
        fp=int((~truth & pred).sum()),
        fn=int((truth & ~pred).sum()),
    )


def candidate_mask(texts: Sequence[str]) -> Bools:
    """``True`` for each text that matches the SATD candidate keywords (``CANDIDATE_RE``)."""
    return np.fromiter((CANDIDATE_RE.search(t) is not None for t in texts), dtype=bool, count=len(texts))


def _macro(counts: Mapping[str, BinaryCounts], attr: str) -> float:
    values = [float(getattr(counts[p], attr)) for p in sorted(counts)]
    return float(np.mean(values)) if values else 0.0


def _pooled(counts: Mapping[str, BinaryCounts]) -> BinaryCounts:
    total = ZERO_COUNTS
    for p in sorted(counts):
        total = total + counts[p]
    return total


def _per_project_counts(projects: npt.NDArray[Any], y: Bools, pred: Bools) -> dict[str, BinaryCounts]:
    names = sorted({str(p) for p in projects.tolist()})
    return {p: binary_counts(y[projects == p], pred[projects == p]) for p in names}


@dataclass(frozen=True)
class LopoResult:
    """Leave-one-project-out evaluation for every C of the grid.

    ``proba[c]`` holds the held-out P(debt) of every dataset row (each row is scored by the model trained
    without its project), ``per_project[c][p]`` the counts of project ``p`` at ``proba >= threshold`` and
    ``n_iter[c]`` the lbfgs iterations per fold in project order.
    """

    projects: tuple[str, ...]
    c_grid: tuple[float, ...]
    threshold: float
    proba: dict[float, Floats]
    per_project: dict[float, dict[str, BinaryCounts]]
    n_iter: dict[float, list[int]]

    def macro_f1(self, c: float) -> float:
        """Mean over projects of the debt-class F1 (the headline LOPO F1)."""
        return _macro(self.per_project[c], "f1")

    def macro_precision(self, c: float) -> float:
        """Mean over projects of the debt-class precision."""
        return _macro(self.per_project[c], "precision")

    def macro_recall(self, c: float) -> float:
        """Mean over projects of the debt-class recall."""
        return _macro(self.per_project[c], "recall")

    def pooled(self, c: float) -> BinaryCounts:
        """Counts summed over all held-out projects (pooled, or micro, F1 comes from these)."""
        return _pooled(self.per_project[c])

    def selected_c(self) -> float:
        """The C with the highest macro F1; ties go to the smaller C (stronger regularisation)."""
        return max(sorted(self.c_grid), key=self.macro_f1)


def _check_frame(df: pd.DataFrame) -> None:
    missing = [c for c in ("project", "text", "is_debt") if c not in df.columns]
    if missing:
        raise ValueError(f"dataset frame lacks columns {missing}")


def lopo_evaluate(
    df: pd.DataFrame,
    c_grid: Sequence[float] = C_GRID,
    *,
    max_features: int | None = DEFAULT_MAX_FEATURES,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = DEFAULT_SETTINGS.seed,
) -> LopoResult:
    """Leave-one-project-out evaluation: for each project, train on all others and score it.

    The vectoriser is refitted on every training fold, so the held-out project contributes neither
    vocabulary nor idf. Within a fold every C of ``c_grid`` is fitted on the same matrix.

    Args:
        df: frame from :func:`load_dataset` (columns ``project``, ``text``, ``is_debt``).
        c_grid: inverse regularisation strengths (positive; duplicates are merged, order is sorted).
        max_features: vocabulary cap per fold (see :func:`make_vectorizer`).
        threshold: P(debt) at or above which a comment counts as debt.
        seed: passed to the classifier.

    Raises:
        ValueError: fewer than two projects, an empty or non-positive grid, or a training fold that has only
            one class.
    """
    _check_frame(df)
    grid = tuple(sorted({float(c) for c in c_grid}))
    if not grid or any(c <= 0 for c in grid):
        raise ValueError(f"c_grid must hold positive values, got {list(c_grid)}")
    projects_col = df["project"].to_numpy(dtype=object)
    projects = tuple(sorted({str(p) for p in projects_col}))
    if len(projects) < 2:
        raise ValueError(f"leave-one-project-out needs at least two projects, got {list(projects)}")
    texts = df["text"].to_numpy(dtype=object)
    y = df["is_debt"].to_numpy(dtype=bool)
    proba = {c: np.full(len(df), np.nan, dtype=np.float64) for c in grid}
    per_project: dict[float, dict[str, BinaryCounts]] = {c: {} for c in grid}
    n_iter: dict[float, list[int]] = {c: [] for c in grid}
    for project in projects:
        test = projects_col == project
        train = ~test
        if y[train].all() or not y[train].any():
            raise ValueError(f"the training fold without {project!r} has only one class")
        started = time.perf_counter()
        vectorizer = make_vectorizer(max_features)
        x_train = vectorizer.fit_transform(texts[train].tolist())
        x_test = vectorizer.transform(texts[test].tolist())
        for c in grid:
            model = make_classifier(c, seed)
            _fit(model, x_train, y[train], f"fold {project} C={c}")
            held_out = model.predict_proba(x_test)[:, 1]
            proba[c][test] = held_out
            per_project[c][project] = binary_counts(y[test], held_out >= threshold)
            n_iter[c].append(int(np.max(model.n_iter_)))
        logger.info(
            "fold %s: %d train rows, %d held out, %d terms, F1 %s, %.1f s",
            project,
            int(train.sum()),
            int(test.sum()),
            len(vectorizer.vocabulary_),
            ", ".join(f"C={c}: {per_project[c][project].f1:.3f}" for c in grid),
            time.perf_counter() - started,
        )
    for c in grid:
        if np.isnan(proba[c]).any():  # pragma: no cover - every row belongs to exactly one fold
            raise RuntimeError(f"rows without a held-out score for C={c}")
    return LopoResult(projects, grid, float(threshold), proba, per_project, n_iter)


def keyword_baseline(df: pd.DataFrame) -> dict[str, BinaryCounts]:
    """Per-project counts when every comment matching ``CANDIDATE_RE`` is called debt (no classifier)."""
    _check_frame(df)
    texts = df["text"].tolist()
    return _per_project_counts(
        df["project"].to_numpy(dtype=object), df["is_debt"].to_numpy(dtype=bool), candidate_mask(texts)
    )


def fit_final(
    df: pd.DataFrame,
    c: float,
    *,
    max_features: int | None = DEFAULT_MAX_FEATURES,
    seed: int = DEFAULT_SETTINGS.seed,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit the vectoriser and classifier with inverse regularisation ``c`` on every row of ``df``."""
    _check_frame(df)
    vectorizer = make_vectorizer(max_features)
    x = vectorizer.fit_transform(df["text"].tolist())
    model = make_classifier(c, seed)
    _fit(model, x, df["is_debt"].to_numpy(dtype=bool), f"final fit C={c}")
    return vectorizer, model


def parity_texts(df: pd.DataFrame, n: int = PARITY_SAMPLE, seed: int = DEFAULT_SETTINGS.seed) -> list[str]:
    """A deterministic sample of ``n`` texts, half debt and half not (fewer when a class is short).

    Rows are drawn with ``numpy.random.default_rng(seed)`` and returned in dataset order.
    """
    _check_frame(df)
    y = df["is_debt"].to_numpy(dtype=bool)
    rng = np.random.default_rng(seed)
    debt = np.flatnonzero(y)
    clean = np.flatnonzero(~y)
    k_debt = min(n // 2, debt.size)
    k_clean = min(n - k_debt, clean.size)
    chosen = np.sort(
        np.concatenate(
            [rng.choice(debt, size=k_debt, replace=False), rng.choice(clean, size=k_clean, replace=False)]
        )
    )
    texts = df["text"].tolist()
    return [texts[int(i)] for i in chosen]


# ---------------------------------------------------------------------------------------------- training run
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _f(value: float) -> float:
    return round(float(value), 6)


def _counts_by_project(counts: Mapping[str, BinaryCounts]) -> dict[str, dict[str, int | float]]:
    return {p: counts[p].to_dict() for p in sorted(counts)}


def _grid_key(c: float) -> str:
    return repr(float(c))


def _dataset_source(csv_path: Path, pinned: bool) -> dict[str, Any]:
    if pinned:
        return {
            "name": DATASET_NAME,
            "repo": DATASET_REPO_URL,
            "url": DATASET_URL,
            "commit": DATASET_COMMIT,
            "blob_sha1": DATASET_BLOB_SHA1,
            "zip_sha256": DATASET_ZIP_SHA256,
        }
    return {"name": f"custom CSV {csv_path.name}", "repo": None, "url": None, "commit": None}


def run_training(
    csv_path: Path,
    out_path: Path,
    report_path: Path,
    *,
    c_grid: Sequence[float] = C_GRID,
    max_features: int | None = DEFAULT_MAX_FEATURES,
    max_bytes: int = MAX_MODEL_BYTES,
    jobs: int = 4,
    seed: int = DEFAULT_SETTINGS.seed,
) -> dict[str, Any]:
    """Evaluate by project, pick C, refit on everything, export the JSON weights and write a report.

    Steps: load the CSV; leave-one-project-out over ``c_grid``; pick the C with the best macro F1;
    diagnostics at that C (keyword baseline, the keyword-candidate subset, thresholds, recall per debt
    type); refit on all rows; export with 9-decimal weights after checking that the numpy runtime matches
    scikit-learn within ``PARITY_TOLERANCE`` on ``PARITY_SAMPLE`` dataset comments plus a few edge texts;
    reload the written file and check it scores identically.

    Args:
        csv_path: the dataset CSV (see :func:`load_dataset`).
        out_path: where the model JSON goes (``tdp_core/features/satd_model.json`` for the shipped model).
        report_path: where the JSON report goes.
        c_grid: candidate C values.
        max_features: vocabulary cap (``None`` for no cap).
        max_bytes: largest model file allowed; a bigger model raises before anything is written.
        jobs: thread cap for numeric libraries.
        seed: classifier seed and parity sample seed.

    Returns:
        The report (plain JSON types), also written to ``report_path``.

    Raises:
        ValueError: bad dataset, or the model would exceed ``max_bytes`` (the existing file is kept).
        RuntimeError: the exported model does not match scikit-learn.
    """
    started = time.perf_counter()
    with threadpool_limits(limits=max(int(jobs), 1)):
        df = load_dataset(csv_path)
        texts: list[str] = df["text"].tolist()
        y = df["is_debt"].to_numpy(dtype=bool)
        projects_col = df["project"].to_numpy(dtype=object)
        project_rows = {str(p): int(n) for p, n in sorted(df["project"].value_counts().items())}
        project_debt = {str(p): int(df.loc[df["project"] == p, "is_debt"].sum()) for p in project_rows}
        for p in project_rows:
            logger.info("project %s: %d comments, %d debt", p, project_rows[p], project_debt[p])
        uncapped = len(CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF).fit(texts).vocabulary_)
        logger.info("vocabulary without a cap: %d n-grams (min_df=%d)", uncapped, MIN_DF)

        lopo_started = time.perf_counter()
        lopo = lopo_evaluate(df, c_grid, max_features=max_features, seed=seed)
        lopo_s = time.perf_counter() - lopo_started
        best_c = lopo.selected_c()
        logger.info("selected C=%s with LOPO macro F1 %.4f", best_c, lopo.macro_f1(best_c))

        held_out = lopo.proba[best_c]
        predicted = held_out >= lopo.threshold
        baseline = keyword_baseline(df)
        candidates = candidate_mask(texts)
        cand_counts = _per_project_counts(projects_col[candidates], y[candidates], predicted[candidates])
        thresholds: dict[str, Any] = {}
        for t in THRESHOLDS:
            at_t = _per_project_counts(projects_col, y, held_out >= t)
            thresholds[repr(t)] = {"macro_f1": _f(_macro(at_t, "f1")), "pooled": _pooled(at_t).to_dict()}
        recall_by_type: dict[str, dict[str, int | float]] = {}
        classes = df["classification"].to_numpy(dtype=object)
        for kind in sorted(set(classes.tolist()) - {NEGATIVE_CLASS}):
            rows = classes == kind
            recall_by_type[str(kind)] = {"n": int(rows.sum()), "recall": _f(predicted[rows].mean())}

        fit_started = time.perf_counter()
        vectorizer, model = fit_final(df, best_c, max_features=max_features, seed=seed)
        fit_s = time.perf_counter() - fit_started
        vocabulary: dict[str, int] = {str(t): int(i) for t, i in vectorizer.vocabulary_.items()}
        terms_by_index = [t for t, _ in sorted(vocabulary.items(), key=lambda kv: kv[1])]
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        order = np.argsort(-coef, kind="stable")
        top_debt = [[terms_by_index[i], _f(coef[i])] for i in order[:TOP_TERMS]]
        top_clean = [[terms_by_index[i], _f(coef[i])] for i in order[::-1][:TOP_TERMS]]
        keyword_coef = {t: (_f(coef[vocabulary[t]]) if t in vocabulary else None) for t in CANDIDATE_TERMS}

        sample = list(PARITY_EXTRA_TEXTS) + parity_texts(df, PARITY_SAMPLE, seed)
        sklearn_p = model.predict_proba(vectorizer.transform(sample))[:, 1]
        in_memory = JsonSatdClassifier.from_sklearn(vectorizer, model)
        rounded = JsonSatdClassifier.from_dict(in_memory.to_dict())
        memory_diff = float(np.max(np.abs(in_memory.predict_proba(sample) - sklearn_p)))
        rounded_p = rounded.predict_proba(sample)
        parity_diff = float(np.max(np.abs(rounded_p - sklearn_p)))
        logger.info("parity with scikit-learn: %.3g in memory, %.3g after rounding", memory_diff, parity_diff)
        if parity_diff >= PARITY_TOLERANCE:
            raise RuntimeError(
                f"the numpy runtime differs from scikit-learn by {parity_diff:.3g} (limit {PARITY_TOLERANCE})"
            )

        csv_sha256 = _sha256_file(csv_path)
        pinned = csv_sha256 == DATASET_CSV_SHA256
        metadata: dict[str, Any] = {
            "created_at": _now(),
            "dataset": {
                **_dataset_source(csv_path, pinned),
                "csv_sha256": csv_sha256,
                "rows": len(df),
                "debt_rows": int(y.sum()),
                "projects": project_rows,
            },
            "license": f"{DATASET_LICENSE} Cite: {DATASET_CITATION}." if pinned else None,
            "params": {
                "vectorizer": {
                    "ngram_range": list(NGRAM_RANGE),
                    "min_df": MIN_DF,
                    "max_features": max_features,
                    "sublinear_tf": True,
                    "lowercase": True,
                    "norm": "l2",
                    "smooth_idf": True,
                },
                "classifier": {
                    "C": best_c,
                    "class_weight": "balanced",
                    "solver": "lbfgs",
                    "max_iter": MAX_ITER,
                },
            },
            "lopo": {
                "threshold": lopo.threshold,
                "selected_c": best_c,
                "macro_f1": _f(lopo.macro_f1(best_c)),
                "pooled_f1": _f(lopo.pooled(best_c).f1),
                "per_project_f1": {p: _f(lopo.per_project[best_c][p].f1) for p in lopo.projects},
                "macro_f1_by_c": {_grid_key(c): _f(lopo.macro_f1(c)) for c in lopo.c_grid},
                "keyword_baseline_macro_f1": _f(_macro(baseline, "f1")),
            },
            "vocabulary_size": len(vocabulary),
            "vocabulary_size_uncapped": uncapped,
            "parity": {"texts": len(sample), "max_abs_diff": parity_diff},
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
        }
        final = JsonSatdClassifier.from_sklearn(vectorizer, model, metadata=metadata)
        try:
            n_bytes = final.to_json(out_path, max_bytes=max_bytes)
        except ValueError as exc:
            raise ValueError(f"{exc}; retrain with a smaller --max-features") from exc
        loaded = JsonSatdClassifier.from_json(out_path)
        if not np.array_equal(loaded.predict_proba(sample), rounded_p):
            raise RuntimeError(f"{out_path} does not score like the model that was written")

        by_c: dict[str, Any] = {}
        for c in lopo.c_grid:
            by_c[_grid_key(c)] = {
                "macro_f1": _f(lopo.macro_f1(c)),
                "macro_precision": _f(lopo.macro_precision(c)),
                "macro_recall": _f(lopo.macro_recall(c)),
                "pooled": lopo.pooled(c).to_dict(),
                "per_project": _counts_by_project(lopo.per_project[c]),
                "n_iter_max": max(lopo.n_iter[c]),
            }
        all_debt = binary_counts(y[candidates], np.ones(int(candidates.sum()), dtype=bool))
        report: dict[str, Any] = {
            "created_at": metadata["created_at"],
            "runtime_s": {
                "total": _f(time.perf_counter() - started),
                "lopo": _f(lopo_s),
                "final_fit": _f(fit_s),
            },
            "dataset": {
                "csv": str(csv_path),
                "csv_sha256": csv_sha256,
                "pinned_dataset": pinned,
                "rows_read": int(df.attrs.get("rows_read", len(df))),
                "rows": len(df),
                "debt_rows": int(y.sum()),
                "dropped_empty": int(df.attrs.get("dropped_empty", 0)),
                "replacement_chars": int(df.attrs.get("replacement_chars", 0)),
                "projects": [
                    {"name": p, "rows": project_rows[p], "debt_rows": project_debt[p]} for p in project_rows
                ],
                "classification_counts": {
                    str(k): int(v) for k, v in sorted(df["classification"].value_counts().items())
                },
            },
            "config": {
                "c_grid": list(lopo.c_grid),
                "max_features": max_features,
                "max_bytes": max_bytes,
                "jobs": jobs,
                "seed": seed,
                "threshold": lopo.threshold,
                "ngram_range": list(NGRAM_RANGE),
                "min_df": MIN_DF,
                "max_iter": MAX_ITER,
            },
            "vocabulary_size_uncapped": uncapped,
            "lopo": {
                "by_c": by_c,
                "selected_c": best_c,
                "macro_f1": _f(lopo.macro_f1(best_c)),
                "pooled_f1": _f(lopo.pooled(best_c).f1),
                "thresholds": thresholds,
                "recall_by_type": recall_by_type,
            },
            "keyword_baseline": {
                "macro_f1": _f(_macro(baseline, "f1")),
                "pooled": _pooled(baseline).to_dict(),
                "per_project": _counts_by_project(baseline),
            },
            "candidates": {
                "n": int(candidates.sum()),
                "debt": int(y[candidates].sum()),
                "coverage": _f(y[candidates].sum() / max(int(y.sum()), 1)),
                "classifier": _pooled(cand_counts).to_dict(),
                "classifier_macro_f1": _f(_macro(cand_counts, "f1")),
                "all_candidates_debt": all_debt.to_dict(),
                "pipeline": binary_counts(y, candidates & predicted).to_dict(),
            },
            "final": {
                "c": best_c,
                "vocabulary_size": len(vocabulary),
                "n_iter": int(np.max(model.n_iter_)),
                "top_debt_terms": top_debt,
                "top_clean_terms": top_clean,
                "keyword_coefficients": keyword_coef,
            },
            "export": {
                "path": str(out_path),
                "bytes": n_bytes,
                "sha256": loaded.source_sha256,
                "parity_texts": len(sample),
                "parity_max_abs_diff": parity_diff,
                "parity_in_memory_max_abs_diff": memory_diff,
            },
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s in %.1f s", out_path, report_path, time.perf_counter() - started)
    return report
