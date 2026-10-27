"""Self-admitted technical debt features (family Q): ``satd_count``, ``satd_density``, ``satd_age_days``.

Comments are the ``FileSyntax.comments`` of a file: one ``CommentInfo`` per ``#`` comment line, with the text
after the first ``#``, stripped. Docstrings and other string literals are not comments, so a ``TODO`` inside a
docstring is never counted.

A comment is a **candidate** when ``SATD_PATTERN`` (a case-insensitive keyword list with word boundaries)
finds a match in its text. Without a classifier every candidate counts. With a ``SatdClassifier``, the
candidates are scored in one call and those with probability ``>= threshold`` count. The classifier exists
because some keywords (notably ``temporary``, ``ugly`` and ``XXX``) often appear in comments that admit no
debt at all.

Units: ``satd_count`` is a number of comment lines, ``satd_density`` is comments per 1,000 SLOC and
``satd_age_days`` is the age in days at the snapshot time T of the oldest counted comment line (from git
blame, computed by feature assembly).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from tdp_core.contracts import FEATURE_FAMILIES, CommentInfo, FileSyntax

__all__ = ["SATD_COLUMNS", "SATD_PATTERN", "SatdClassifier", "satd_candidates", "satd_features"]

SATD_COLUMNS: tuple[str, ...] = FEATURE_FAMILIES["Q"]

SATD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:TODO|FIXME|HACK|XXX|KLUDGE|workaround|work around|temporary|temp fix|quick fix|ugly)\b",
    re.IGNORECASE,
)


class SatdClassifier(Protocol):
    """Scores comment texts by how likely each one admits technical debt."""

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.floating[Any]]:
        """Probability that each text admits technical debt.

        Returns a 1-D float array with ``len(texts)`` values in 0..1, in input order. Each text is one
        ``CommentInfo.text`` (no leading ``#``, stripped).
        """
        ...


def satd_candidates(comments: Sequence[CommentInfo]) -> list[CommentInfo]:
    """The comments whose text matches ``SATD_PATTERN``, in input order (the same objects, not copies)."""
    return [c for c in comments if SATD_PATTERN.search(c.text)]


def satd_features(
    syntax: FileSyntax,
    classifier: SatdClassifier | None,
    line_ages_days: Mapping[int, float] | None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """``satd_count``, ``satd_density`` and ``satd_age_days`` of one parsed file.

    - ``classifier``: ``None`` counts every candidate. Otherwise ``predict_proba`` is called exactly once with
      the candidate texts (never with an empty list) and candidates with probability ``>= threshold`` count.
    - ``line_ages_days`` maps a 1-based line number to that line's age in days at T (git blame). ``None``, or
      no finite age for any counted line, gives ``satd_age_days = 0.0``. Ages of other lines are ignored,
      non-finite ages are skipped and the result is clipped at 0.
    - ``satd_count`` is the number of counted comment lines (a comment spanning several lines can count more
      than once), ``satd_density = 1000 * satd_count / max(sloc, 1)`` in comments per 1,000 SLOC, and
      ``satd_age_days`` the largest known age of a counted line, in days.

    Returns a dict with keys in ``SATD_COLUMNS`` order and float values. Raises ``ValueError`` when
    ``threshold`` is not in 0..1 (NaN included), or when the classifier output is not a 1-D array with one
    value per candidate or contains NaN.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in 0..1, got {threshold!r}")

    candidates = satd_candidates(syntax.comments)
    counted = candidates
    if classifier is not None and candidates:
        probs = np.asarray(classifier.predict_proba([c.text for c in candidates]), dtype=np.float64)
        if probs.shape != (len(candidates),):
            raise ValueError(
                f"classifier returned shape {probs.shape} for {len(candidates)} texts, expected a 1-D array"
            )
        if np.isnan(probs).any():
            raise ValueError("classifier returned NaN probabilities")
        counted = [c for c, p in zip(candidates, probs, strict=True) if p >= threshold]

    count = len(counted)
    age = 0.0
    if line_ages_days is not None:
        known = [a for c in counted if (a := line_ages_days.get(c.line)) is not None and math.isfinite(a)]
        if known:
            age = max(0.0, float(max(known)))

    return {
        "satd_count": float(count),
        "satd_density": 1000.0 * count / max(syntax.sloc, 1),
        "satd_age_days": age,
    }
