"""Priority percentiles from raw ranker scores.

The priority of a file is the share of files in the same repository whose score is strictly lower, written
as an integer percent from 0 to 100. Ties share a priority, so two files with the same score always get the
same number.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def priority_from_scores(scores: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return the priority (integer percent, 0 to 100) of every score within ``scores``.

    ``scores`` are raw ranker margins (unitless, any finite float, higher means more pain) for the files
    of one repository. For file ``i`` with ``lower_i`` files scoring strictly lower and ``n`` files in total,
    ``priority_i = round(100 * lower_i / n)`` with halves rounded up, computed in exact integer arithmetic.

    Edge cases:

    - An empty input gives an empty ``int64`` array; a single file gets ``[100]``.
    - Ties get equal priority. With two or more files the lowest score gets 0, so if every score is equal
      every file gets 0.
    - With ``n`` distinct scores the top file gets ``round(100 * (n - 1) / n)``: 100 for ``n >= 200``,
      99 for ``n = 100``.

    Raises ``ValueError`` if ``scores`` is not one-dimensional or holds NaN or infinite values.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite (no NaN or infinity)")
    n = int(s.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.full(1, 100, dtype=np.int64)
    lower = np.searchsorted(np.sort(s), s, side="left").astype(np.int64)
    priority = (200 * lower + n) // (2 * n)
    return np.clip(priority, 0, 100).astype(np.int64)
