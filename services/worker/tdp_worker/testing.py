"""In-memory stand-ins for running the worker tasks without Postgres (tests and local experiments).

:class:`MemoryStore` implements :class:`tdp_worker.persist.Store` with the state machine of
``tdp_platform.db.repos``; :class:`MemoryStaticCache` is a dict-backed ``StaticCache`` that counts writes.
Install a store with ``tdp_worker.persist.set_store(MemoryStore())``.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final

from tdp_core.contracts import FileExtra, FileSyntax
from tdp_core.features.cache import StaticCache
from tdp_core.serialize import from_jsonable
from tdp_platform.db.rows import FileResultRow
from tdp_worker.persist import ACTIVE_STATUSES, AnalysisInfo, FileInfo, ResultPayload, score_from_row

DEFAULT_REPOSITORY_ID: Final = uuid.UUID("00000000-0000-4000-8000-000000000001")


class MemoryStaticCache:
    """A ``StaticCache`` in a dict; ``puts`` counts every ``put`` call (hits are not counted)."""

    def __init__(self) -> None:
        self.entries: dict[str, FileSyntax] = {}
        self.puts = 0

    def get(self, blob_sha: str) -> FileSyntax | None:
        return self.entries.get(blob_sha)

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        self.puts += 1
        self.entries.setdefault(blob_sha, syntax)


@dataclass
class StoredFile:
    """A stored file row with its id and analysis."""

    id: int
    analysis_id: uuid.UUID
    row: FileResultRow


@dataclass
class StoredAdviceRow:
    """A stored advice row."""

    cache_key: str
    file_result_id: int
    model: str
    status: str = "queued"
    body: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class MemoryStore:
    """A :class:`~tdp_worker.persist.Store` held in memory, with the transitions of P5.1's repositories.

    ``analyses`` maps ids to :class:`AnalysisInfo` (plus ``errors``, ``summaries`` and ``stats``), files get
    sequential ids in the order stored, ``artifacts`` is keyed by ``(analysis_id, kind)``, ``plans`` by
    ``(analysis_id, budget_hours)``, ``advice`` by cache key. ``progress_log`` records every accepted
    ``set_progress`` call. ``cache`` is returned by :meth:`static_cache`.
    """

    analyses: dict[uuid.UUID, AnalysisInfo] = field(default_factory=dict)
    errors: dict[uuid.UUID, str | None] = field(default_factory=dict)
    summaries: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)
    stats: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)
    files: dict[int, StoredFile] = field(default_factory=dict)
    artifacts: dict[tuple[uuid.UUID, str], dict[str, Any]] = field(default_factory=dict)
    plans: dict[tuple[uuid.UUID, float], dict[str, Any]] = field(default_factory=dict)
    advice: dict[str, StoredAdviceRow] = field(default_factory=dict)
    progress_log: list[tuple[uuid.UUID, str, int]] = field(default_factory=list)
    cache: StaticCache | None = None
    closed: bool = False
    _next_file_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------- seeding ----------

    def add_analysis(
        self,
        *,
        commit_sha: str,
        model_version: str = "dev-worker",
        budget_hours: float = 40.0,
        url: str = "https://github.com/acme/demo",
        host: str = "github.com",
        owner: str = "acme",
        name: str = "demo",
        status: str = "queued",
        repository_id: uuid.UUID = DEFAULT_REPOSITORY_ID,
        progress: int = 0,
    ) -> uuid.UUID:
        """Add an analysis row and return its new id."""
        aid = uuid.uuid4()
        self.analyses[aid] = AnalysisInfo(
            id=aid,
            repository_id=repository_id,
            url=url,
            host=host,
            owner=owner,
            name=name,
            commit_sha=commit_sha,
            model_version=model_version,
            budget_hours=budget_hours,
            status=status,
            stage=None,
            progress=progress,
        )
        self.errors[aid] = None
        return aid

    def add_file(self, analysis_id: uuid.UUID, row: FileResultRow) -> int:
        """Store one file row for an analysis and return its id."""
        with self._lock:
            fid = self._next_file_id
            self._next_file_id += 1
            self.files[fid] = StoredFile(fid, analysis_id, row)
            return fid

    def rows_of(self, analysis_id: uuid.UUID) -> list[FileResultRow]:
        """The stored rows of an analysis in id order."""
        return [f.row for fid, f in sorted(self.files.items()) if f.analysis_id == analysis_id]

    # ---------- Store ----------

    def _transition(self, analysis_id: uuid.UUID, allowed: tuple[str, ...], **values: Any) -> bool:
        with self._lock:
            info = self.analyses.get(analysis_id)
            if info is None or info.status not in allowed:
                return False
            self.analyses[analysis_id] = replace(info, **values)
            return True

    def load_analysis(self, analysis_id: uuid.UUID) -> AnalysisInfo | None:
        return self.analyses.get(analysis_id)

    def set_running(self, analysis_id: uuid.UUID) -> bool:
        ok = self._transition(analysis_id, ACTIVE_STATUSES, status="running")
        if ok:
            self.errors[analysis_id] = None
        return ok

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        ok = self._transition(analysis_id, ACTIVE_STATUSES, stage=stage, progress=max(0, min(100, pct)))
        if ok:
            self.progress_log.append((analysis_id, stage, pct))
        return ok

    def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        ok = self._transition(analysis_id, ACTIVE_STATUSES, status="failed")
        if ok:
            self.errors[analysis_id] = error
        return ok

    def save_results(self, analysis_id: uuid.UUID, payload: ResultPayload) -> bool:
        if not self._transition(analysis_id, ("running",), status="succeeded", stage="done", progress=100):
            return False
        for fid in [fid for fid, f in self.files.items() if f.analysis_id == analysis_id]:
            del self.files[fid]
        for row in payload.rows:
            self.add_file(analysis_id, row)
        self.artifacts[(analysis_id, "graph")] = payload.graph
        self.artifacts[(analysis_id, "treemap")] = payload.treemap
        self.plans[(analysis_id, payload.budget_hours)] = payload.plan
        self.summaries[analysis_id] = payload.summary
        self.stats[analysis_id] = payload.stats
        self.errors[analysis_id] = None
        return True

    def load_file(self, file_result_id: int) -> FileInfo | None:
        stored = self.files.get(file_result_id)
        if stored is None:
            return None
        info = self.analyses[stored.analysis_id]
        row = stored.row
        return FileInfo(
            id=stored.id,
            analysis=info,
            path=row.path,
            blob_sha=row.blob_sha,
            score=score_from_row(row),
            extra=from_jsonable(FileExtra, dict(row.extra)),
        )

    def begin_advice(self, cache_key: str, *, file_result_id: int, model: str) -> str:
        with self._lock:
            row = self.advice.setdefault(cache_key, StoredAdviceRow(cache_key, file_result_id, model))
            return row.status

    def _advice_transition(self, cache_key: str, **values: Any) -> bool:
        with self._lock:
            row = self.advice.get(cache_key)
            if row is None or row.status != "queued":
                return False
            for name, value in values.items():
                setattr(row, name, value)
            return True

    def advice_ready(self, cache_key: str, body: Mapping[str, Any]) -> bool:
        return self._advice_transition(cache_key, status="ready", body=dict(body), error=None)

    def advice_failed(self, cache_key: str, error: str) -> bool:
        return self._advice_transition(cache_key, status="failed", body=None, error=error)

    def static_cache(self) -> StaticCache | None:
        return self.cache

    def close(self) -> None:
        self.closed = True
