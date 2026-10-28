"""Paths of every pipeline output under the data folder, and atomic writers for them.

All writers create the parent folder, write to a temporary file in the same folder and move it into place
with ``os.replace``, so a reader never sees a partial file and an interrupted stage leaves no output that
looks complete. Data folders are never removed recursively: several of them are symbolic links.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tdp_core.paths import data_dir


class PipelineError(RuntimeError):
    """A stage cannot run: an input is missing, a repository is unknown or not selected, or outputs disagree.

    The message names the command that produces the missing input.
    """


def _check_slug(slug: str) -> str:
    if not slug or "/" in slug or "\\" in slug or ".." in slug:
        raise ValueError(f"invalid repository slug: {slug!r}")
    return slug


@dataclass(frozen=True, slots=True)
class DataLayout:
    """The data folder layout (plan section 3.4 plus the pipeline's own files).

    Every method that takes a ``slug`` raises ``ValueError`` for an empty slug or one holding ``/``, ``\\``
    or ``..``, so a path can never leave ``root``. Nothing is created on disk by these methods.
    """

    root: Path

    @classmethod
    def from_env(cls) -> DataLayout:
        """The layout rooted at ``tdp_core.paths.data_dir()`` (``$TDP_DATA_DIR`` or ``./data``)."""
        return cls(data_dir())

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def mirror(self, slug: str) -> Path:
        return self.root / "mirrors" / f"{_check_slug(slug)}.git"

    def commits(self, slug: str) -> Path:
        return self.root / "mined" / _check_slug(slug) / "commits.parquet"

    def file_changes(self, slug: str) -> Path:
        return self.root / "mined" / _check_slug(slug) / "file_changes.parquet"

    def issues(self, slug: str) -> Path:
        return self.root / "issues" / f"{_check_slug(slug)}.parquet"

    def bugfix(self, slug: str, *, keyword: bool = False) -> Path:
        """Issue-aware bug fixes (labels), or with ``keyword`` the keyword-only table (features)."""
        suffix = ".keyword.parquet" if keyword else ".parquet"
        return self.root / "bugfix" / f"{_check_slug(slug)}{suffix}"

    def snapshots(self, slug: str) -> Path:
        return self.root / "snapshots" / f"{_check_slug(slug)}.json"

    def snapshot_plan(self, slug: str) -> Path:
        return self.root / "snapshots" / f"{_check_slug(slug)}.plan.json"

    def szz(self, slug: str, *, keyword: bool = False) -> Path:
        """SZZ rows for the label fixes, or with ``keyword`` for the keyword fixes (features)."""
        suffix = ".keyword.parquet" if keyword else ".parquet"
        return self.root / "szz" / f"{_check_slug(slug)}{suffix}"

    def features(self, slug: str, index: int) -> Path:
        return self.root / "features" / _check_slug(slug) / f"{int(index)}.parquet"

    def features_meta(self, slug: str, index: int) -> Path:
        return self.root / "features" / _check_slug(slug) / f"{int(index)}.json"

    def lineage(self, slug: str, index: int) -> Path:
        """The ``path_lineage`` of a snapshot (columns ``path``, ``name``), written by ``features``."""
        return self.root / "features" / _check_slug(slug) / f"{int(index)}.lineage.parquet"

    def labels(self, slug: str, index: int) -> Path:
        return self.root / "labels" / _check_slug(slug) / f"{int(index)}.parquet"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset.parquet"

    @property
    def dataset_summary(self) -> Path:
        return self.root / "dataset.json"

    @property
    def stats(self) -> Path:
        return self.root / "stats.json"

    @property
    def static_cache(self) -> Path:
        return self.root / "cache" / "static"

    def szz_cache(self, slug: str) -> Path:
        return self.root / "cache" / "szz" / _check_slug(slug)

    @property
    def tmp(self) -> Path:
        """Scratch space for temporary clones (inside ``cache``, so it shares the cache's disk)."""
        return self.root / "cache" / "tmp"

    def run_record(self, stage: str, slug: str | None = None) -> Path:
        """``runs/<slug>/<stage>.json`` for a repository stage, ``runs/<stage>.json`` for a global one."""
        if not stage or "/" in stage or "\\" in stage or ".." in stage:
            raise ValueError(f"invalid stage name: {stage!r}")
        if slug is None:
            return self.root / "runs" / f"{stage}.json"
        return self.root / "runs" / _check_slug(slug) / f"{stage}.json"


def _atomic(path: Path, write: Callable[[Path], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_json(path: Path, data: Any) -> Path:
    """Write ``data`` as UTF-8 JSON (indent 2, non-ASCII kept, trailing newline) atomically."""
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return _atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8", newline="\n"))


def read_json(path: Path) -> Any:
    """The JSON content of ``path``; :class:`PipelineError` when it is missing or not valid JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"{path} does not exist") from exc
    except (OSError, ValueError) as exc:
        raise PipelineError(f"{path} cannot be read as JSON: {exc}") from exc


def write_table(path: Path, table: pa.Table) -> Path:
    """Write an Arrow table as Parquet atomically."""
    return _atomic(path, lambda tmp: pq.write_table(table, tmp))


def write_frame(path: Path, frame: pd.DataFrame) -> Path:
    """Write a DataFrame as Parquet (``index=False``) atomically; dtypes round-trip through Arrow."""
    return _atomic(path, lambda tmp: frame.to_parquet(tmp, index=False))


def dir_bytes(path: Path) -> int:
    """Total size in bytes of the files below ``path`` (symbolic links followed); 0 when it is missing."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, _, files in os.walk(path, followlinks=True):
        for name in files:
            try:
                total += (Path(folder) / name).stat().st_size
            except OSError:
                continue
    return total
