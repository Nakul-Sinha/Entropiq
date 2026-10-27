"""Clone detection by winnowing token k-gram fingerprints (IMPLEMENTATION_PLAN.md step P1.3).

Every file's normalised token stream (``FileSyntax.tokens``) is turned into Rabin-Karp hashes of its k-grams,
winnowed into a sparse set of fingerprints, and every fingerprint found in two or more non-overlapping places
marks its k tokens as duplicated. The work is vectorised with numpy over the concatenated token stream of all
files, so a repository of a million tokens takes well under a second.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Mapping, Sequence
from itertools import chain

import numpy as np
import numpy.typing as npt

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.contracts import CloneStats, FileSyntax

logger = logging.getLogger(__name__)

BASE = 1_000_003
MOD = (1 << 61) - 1

_MASK32 = (1 << 32) - 1
_MASK29 = (1 << 29) - 1

U64 = npt.NDArray[np.uint64]
I64 = npt.NDArray[np.int64]
Bools = npt.NDArray[np.bool_]

_NO_CLONES = CloneStats(dup_ratio=0.0, dup_tokens=0, clone_groups=0, dup_lines=0)


def clone_stats(
    syntax_by_path: Mapping[str, FileSyntax], settings: Settings = DEFAULT_SETTINGS
) -> dict[str, CloneStats]:
    """Duplicated-code statistics for every file, from winnowed token k-gram fingerprints.

    Algorithm (k = ``settings.clone_k`` tokens, w = ``settings.clone_window`` k-grams):

    1. Each token string becomes ``zlib.crc32(token.encode())``.
    2. Every k-gram of a file with at least k tokens gets the Rabin-Karp hash
       ``sum(id[i + j] * 1_000_003 ** (k - 1 - j)) mod 2**61 - 1``.
    3. Winnowing: each window of w consecutive k-grams of one file selects its minimum hash (rightmost on
       ties); each selected position is one fingerprint. A file with fewer than w k-grams is a single window.
    4. A fingerprint hash is a clone if it occurs in two or more locations that are in different files or are
       at least k tokens apart in the same file (so a short repeated pattern cannot match itself).
    5. Every occurrence of a clone hash marks tokens ``[pos, pos + k)`` of its file as duplicated.

    Args:
        syntax_by_path: parsed files keyed by repo-relative path; only ``tokens`` and ``token_lines`` are
            read. Tokens of all files are compared with each other regardless of ``language``.
        settings: ``clone_k`` and ``clone_window`` are used; both must be at least 1.

    Returns:
        One ``CloneStats`` per input path, keyed in sorted path order. ``dup_ratio`` is duplicated tokens
        over total tokens (0..1), ``dup_tokens`` a token count, ``clone_groups`` the number of distinct clone
        hashes present in the file, ``dup_lines`` the number of distinct ``token_lines`` values of duplicated
        tokens. Files with fewer than k tokens (including empty files) get all zeros.

    Raises:
        ValueError: if ``clone_k`` or ``clone_window`` is below 1, or a file's ``tokens`` and ``token_lines``
            differ in length.
    """
    k, w = settings.clone_k, settings.clone_window
    if k < 1 or w < 1:
        raise ValueError(f"clone_k and clone_window must be >= 1, got {k} and {w}")
    paths = sorted(syntax_by_path)
    for path in paths:
        fs = syntax_by_path[path]
        if len(fs.tokens) != len(fs.token_lines):
            raise ValueError(f"{path}: {len(fs.tokens)} tokens but {len(fs.token_lines)} token_lines")

    result = dict.fromkeys(paths, _NO_CLONES)
    eligible = [path for path in paths if len(syntax_by_path[path].tokens) >= k]
    if not eligible:
        return result

    # All eligible files are laid end to end in one array; k-grams and windows that cross a file
    # boundary are computed but never selected.
    n_files = len(eligible)
    lengths = np.array([len(syntax_by_path[path].tokens) for path in eligible], dtype=np.int64)
    starts = np.zeros(n_files, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    n_total = int(lengths.sum())
    file_of = np.repeat(np.arange(n_files, dtype=np.int64), lengths)

    ids = _token_ids([syntax_by_path[path].tokens for path in eligible], n_total)
    hashes = _kgram_hashes(ids, k)
    fp_pos = _winnow(hashes, lengths, starts, file_of, k, w)
    clone_pos, groups = _clone_occurrences(hashes[fp_pos], fp_pos, file_of, k, n_files)
    covered = _coverage(clone_pos, n_total, k)

    dup_tokens = np.add.reduceat(covered.astype(np.int64), starts)
    lines = np.fromiter(
        chain.from_iterable(syntax_by_path[path].token_lines for path in eligible),
        dtype=np.int64,
        count=n_total,
    )
    dup_lines = _distinct_lines(covered, file_of, lines, n_files)

    for i, path in enumerate(eligible):
        dup = int(dup_tokens[i])
        result[path] = CloneStats(
            dup_ratio=dup / int(lengths[i]),
            dup_tokens=dup,
            clone_groups=int(groups[i]),
            dup_lines=int(dup_lines[i]),
        )
    logger.debug(
        "clone_stats: %d files, %d tokens, %d fingerprints, %d clone occurrences",
        len(paths),
        n_total,
        fp_pos.shape[0],
        clone_pos.shape[0],
    )
    return result


def _token_ids(token_streams: Sequence[Sequence[str]], n_total: int) -> U64:
    """crc32 of every token, concatenated over the streams."""
    flat = list(chain.from_iterable(token_streams))
    crc = {token: zlib.crc32(token.encode()) for token in set(flat)}
    return np.fromiter(map(crc.__getitem__, flat), dtype=np.uint64, count=n_total)


def _kgram_hashes(ids: U64, k: int) -> U64:
    """Rabin-Karp hash of every k-gram: ``h[i] = sum(ids[i + j] * BASE**(k - 1 - j)) mod MOD``.

    Returns ``len(ids) - k + 1`` hashes (requires ``len(ids) >= k`` and every id below 2**32). The value
    equals the rolling-hash recurrence; it is computed by Horner's rule over all positions at once, in k
    vector passes.
    """
    n = ids.shape[0] - k + 1
    h = np.zeros(n, dtype=np.uint64)
    lo = np.empty_like(h)
    hi = np.empty_like(h)
    for j in range(k):
        # h * BASE mod MOD in 64-bit lanes: split h (< 2**61) at bit 32, so h * BASE = lo * BASE +
        # (hi * BASE) * 2**32, and fold the high part with 2**61 = 1 (mod MOD).
        np.bitwise_and(h, _MASK32, out=lo)
        lo *= BASE  # < 2**52
        np.right_shift(h, 32, out=hi)
        hi *= BASE  # < 2**49
        np.right_shift(hi, 29, out=h)  # hi * 2**32 = (hi >> 29) * 2**61 + (hi & _MASK29) * 2**32
        lo += h
        hi &= _MASK29
        hi <<= 32
        lo += hi
        lo += ids[j : j + n]  # < 2**62 + 2**32
        np.bitwise_and(lo, MOD, out=h)
        lo >>= 61
        h += lo
        np.subtract(h, MOD, out=h, where=h >= MOD)
    return h


def _winnow(hashes: U64, lengths: I64, starts: I64, file_of: I64, k: int, w: int) -> I64:
    """Sorted global k-gram positions picked by winnowing, never crossing a file boundary.

    Each window of w consecutive k-grams inside one file picks its minimum hash, rightmost on ties. A file
    with fewer than w k-grams is treated as one window over all of them.
    """
    picked: list[I64] = []
    n_windows = hashes.shape[0] - w + 1
    if n_windows > 0:
        win_min = hashes[:n_windows].copy()
        for j in range(1, w):
            np.minimum(win_min, hashes[j : j + n_windows], out=win_min)
        offset = np.zeros(n_windows, dtype=np.int64)
        for j in range(1, w):  # a later j overwrites an earlier one, so ties go to the rightmost
            np.copyto(offset, j, where=hashes[j : j + n_windows] == win_min)
        # The window starting at i spans tokens [i, i + w + k - 2]; keep it if both ends are in one file.
        last = w + k - 2
        inside = file_of[:n_windows] == file_of[last : last + n_windows]
        picked.append((np.arange(n_windows, dtype=np.int64) + offset)[inside])
    n_grams = lengths - k + 1
    for f in np.flatnonzero(n_grams < w):
        s, m = int(starts[f]), int(n_grams[f])
        rightmost_min = m - 1 - int(np.argmin(hashes[s : s + m][::-1]))
        picked.append(np.array([s + rightmost_min], dtype=np.int64))
    return np.unique(np.concatenate(picked))


def _run_starts(*keys: npt.NDArray[np.generic]) -> Bools:
    """True where any key differs from the previous element (and at index 0)."""
    n = keys[0].shape[0]
    new = np.zeros(n, dtype=np.bool_)
    if n:
        new[0] = True
        for key in keys:
            new[1:] |= key[1:] != key[:-1]
    return new


def _clone_occurrences(fp_hash: U64, fp_pos: I64, file_of: I64, k: int, n_files: int) -> tuple[I64, I64]:
    """Global positions of every fingerprint whose hash is a clone, and distinct clone hashes per file."""
    order = np.lexsort((fp_pos, fp_hash))  # by hash, then position (so also by file)
    h, g = fp_hash[order], fp_pos[order]
    f = file_of[g]
    first = np.flatnonzero(_run_starts(h))
    last = np.append(first[1:], h.shape[0]) - 1
    is_clone = (last > first) & ((f[first] != f[last]) | (g[last] - g[first] >= k))
    keep = np.repeat(is_clone, last - first + 1)
    h, g, f = h[keep], g[keep], f[keep]
    groups = np.bincount(f[_run_starts(h, f)], minlength=n_files).astype(np.int64)
    return g, groups


def _coverage(clone_pos: I64, n_total: int, k: int) -> Bools:
    """True for every token inside at least one span ``[pos, pos + k)``."""
    delta = np.bincount(clone_pos, minlength=n_total + 1) - np.bincount(clone_pos + k, minlength=n_total + 1)
    covered: Bools = np.cumsum(delta[:n_total]) > 0
    return covered


def _distinct_lines(covered: Bools, file_of: I64, lines: I64, n_files: int) -> I64:
    """Per file, the number of distinct line numbers among covered tokens."""
    idx = np.flatnonzero(covered)
    if idx.shape[0] == 0:
        return np.zeros(n_files, dtype=np.int64)
    ln = lines[idx]
    low = int(ln.min())
    span = int(ln.max()) - low + 1
    keys = np.unique(file_of[idx] * span + (ln - low))
    return np.bincount(keys // span, minlength=n_files).astype(np.int64)
