"""Draw a stratified random sample of commits for hand-labelling the bug-fix detector.

The population is every commit the detector could call a bug fix (non-bulk, changes a source file; see
``tdp_core.labeling.bugfix.source_commits``); other commits are non-fixes by construction. Strata are the
detector's sources (``issue``, ``keyword``, ``none``), so both precision (the first two) and recall (misses
hide in ``none``) can be estimated. The labeller sees a blind CSV without the prediction; the stratum, its
size and the prediction go to a separate key file that ``score_bugfix`` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from tdp_core.labeling.bugfix import default_is_source, source_commits

STRATA = ("issue", "keyword", "none")
SAMPLE_COLUMNS = (
    "row",
    "sha",
    "date",
    "subject",
    "message",
    "source_files",
    "other_files",
    "url",
    "label",
    "note",
)
KEY_COLUMNS = ("sha", "stratum", "stratum_size", "sample_size", "predicted")
MAX_MESSAGE_CHARS = 4000
MAX_FILES_LISTED = 10


@dataclass(frozen=True)
class BugfixSample:
    rows: pd.DataFrame  # SAMPLE_COLUMNS, shuffled, label/note empty: what the labeller sees (no prediction)
    key: pd.DataFrame  # KEY_COLUMNS, sorted by sha


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Split ``n`` sample slots over strata as equally as their sizes allow.

    Equal shares over the strata that still have unsampled commits (in the given order; the remainder goes to
    the first ones), capped at each stratum's size; slots a small stratum cannot use are shared out again.
    The result sums to ``min(n, sum(sizes))``. Raises ``ValueError`` for ``n < 1`` or a negative size.
    """
    if n < 1:
        raise ValueError(f"sample size must be >= 1, got {n}")
    if any(size < 0 for size in sizes.values()):
        raise ValueError(f"stratum sizes must be >= 0, got {dict(sizes)}")
    alloc = dict.fromkeys(sizes, 0)
    remaining = n
    while remaining > 0:
        open_strata = [s for s in sizes if sizes[s] > alloc[s]]
        if not open_strata:
            break
        share, extra = divmod(remaining, len(open_strata))
        for i, stratum in enumerate(open_strata):
            take = min(share + (1 if i < extra else 0), sizes[stratum] - alloc[stratum])
            alloc[stratum] += take
            remaining -= take
    return alloc


def _file_entry(path: str, old_path: str | None, added: int, deleted: int, binary: bool) -> str:
    name = f"{old_path} => {path}" if old_path else path
    return f"{name} (binary)" if binary else f"{name} (+{added}/-{deleted})"


def _join(entries: list[str]) -> str:
    shown = entries[:MAX_FILES_LISTED]
    more = len(entries) - len(shown)
    return "; ".join(shown) + (f"; ... {more} more" if more else "")


def sample_bugfix(
    commits: pa.Table,
    file_changes: pa.Table,
    bugfix: pa.Table,
    *,
    n: int = 200,
    seed: int = 42,
    url_prefix: str | None = None,
    is_source: Callable[[str], bool] = default_is_source,
) -> BugfixSample:
    """Sample ``n`` eligible commits stratified by the detector's ``source`` (see ``allocate``).

    ``bugfix`` is ``classify_bugfix`` output for ``commits``. Within a stratum, commits are drawn uniformly
    without replacement with ``numpy.random.default_rng(seed)``; then all picked rows are shuffled and
    numbered ``row`` 1..m, so labelling rows 1..K in order is still a random subsample of every stratum. The
    same inputs and seed give the same sample. Row fields: ``date`` (author date, UTC, YYYY-MM-DD), the
    subject, the message (cut at ``MAX_MESSAGE_CHARS``), source and other changed files as
    ``path (+added/-deleted)`` (at most ``MAX_FILES_LISTED`` each), ``url`` (``<url_prefix>/commit/<sha>`` or
    empty) and empty ``label`` / ``note`` columns for the labeller.

    Raises ``ValueError`` when an eligible commit is missing from ``bugfix`` or has an unknown source, when no
    commit is eligible, or when ``n`` is smaller than the number of non-empty strata (every stratum must be
    sampled for the weighted estimates to be unbiased).
    """
    eligible = source_commits(commits, file_changes, is_source)
    source_by_sha = dict(
        zip(bugfix.column("sha").to_pylist(), bugfix.column("source").to_pylist(), strict=True)
    )
    members: dict[str, list[str]] = {s: [] for s in STRATA}
    for sha in sorted(eligible):
        source = source_by_sha.get(sha)
        if source is None:
            raise ValueError(f"commit {sha} is eligible but missing from the bug-fix table")
        if source not in members:
            raise ValueError(f"commit {sha} has unknown bug-fix source {source!r}")
        members[source].append(sha)
    sizes = {s: len(members[s]) for s in STRATA}
    non_empty = sum(1 for size in sizes.values() if size)
    if non_empty == 0:
        raise ValueError("no commit changes a source file; nothing to sample")
    if n < non_empty:
        raise ValueError(f"sample size {n} is smaller than the {non_empty} non-empty strata")
    alloc = allocate(sizes, n)

    rng = np.random.default_rng(seed)
    picked: list[tuple[str, str]] = []
    for stratum in STRATA:
        k = alloc[stratum]
        if k:
            idx = rng.choice(len(members[stratum]), size=k, replace=False)
            picked.extend((members[stratum][int(i)], stratum) for i in idx)
    order = rng.permutation(len(picked))
    picked = [picked[int(i)] for i in order]

    wanted = {sha for sha, _ in picked}
    info: dict[str, tuple[str, str, str]] = {}
    for sha, when, subject, message in zip(
        commits.column("sha").to_pylist(),
        commits.column("author_time").to_pylist(),
        commits.column("subject").to_pylist(),
        commits.column("message").to_pylist(),
        strict=True,
    ):
        if sha in wanted:
            info[sha] = (when.strftime("%Y-%m-%d"), subject or "", (message or "")[:MAX_MESSAGE_CHARS])
    files: dict[str, tuple[list[str], list[str]]] = {sha: ([], []) for sha in wanted}
    for sha, path, old_path, added, deleted, binary in zip(
        file_changes.column("sha").to_pylist(),
        file_changes.column("path").to_pylist(),
        file_changes.column("old_path").to_pylist(),
        file_changes.column("added").to_pylist(),
        file_changes.column("deleted").to_pylist(),
        file_changes.column("is_binary").to_pylist(),
        strict=True,
    ):
        if sha in files:
            src = is_source(path) or (old_path is not None and is_source(old_path))
            entry = _file_entry(path, old_path, added or 0, deleted or 0, bool(binary))
            files[sha][0 if src else 1].append(entry)

    rows = pd.DataFrame(
        [
            {
                "row": i + 1,
                "sha": sha,
                "date": info[sha][0],
                "subject": info[sha][1],
                "message": info[sha][2],
                "source_files": _join(files[sha][0]),
                "other_files": _join(files[sha][1]),
                "url": f"{url_prefix.rstrip('/')}/commit/{sha}" if url_prefix else "",
                "label": "",
                "note": "",
            }
            for i, (sha, _) in enumerate(picked)
        ],
        columns=list(SAMPLE_COLUMNS),
    )
    key = pd.DataFrame(
        [
            {
                "sha": sha,
                "stratum": stratum,
                "stratum_size": sizes[stratum],
                "sample_size": alloc[stratum],
                "predicted": int(stratum != "none"),
            }
            for sha, stratum in sorted(picked)
        ],
        columns=list(KEY_COLUMNS),
    )
    return BugfixSample(rows=rows, key=key)


def key_path(sample_csv: Path) -> Path:
    """The key file next to a sample: ``<dir>/<stem>.key.csv``."""
    return sample_csv.with_name(f"{sample_csv.stem}.key.csv")


def write_sample(sample: BugfixSample, out_csv: Path, *, force: bool = False) -> tuple[Path, Path]:
    """Write the sample CSV and its key (UTF-8, ``\\n`` line ends). Returns (sample path, key path).

    Refuses with ``FileExistsError`` when ``out_csv`` exists and ``force`` is false, so hand labels are never
    overwritten by accident. Parent folders are created.
    """
    key_csv = key_path(out_csv)
    if out_csv.exists() and not force:
        raise FileExistsError(f"{out_csv} exists (it may hold hand labels); pass force to overwrite it")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sample.rows.to_csv(out_csv, index=False, encoding="utf-8", lineterminator="\n")
    sample.key.to_csv(key_csv, index=False, encoding="utf-8", lineterminator="\n")
    return out_csv, key_csv
