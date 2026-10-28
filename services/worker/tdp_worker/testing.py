"""Test helpers for code that runs the worker tasks without Postgres or a Redis server.

:class:`MemoryStore` is an in-memory :class:`tdp_worker.persist.Store` with the state rules of the platform
repositories; install it with ``tdp_worker.persist.set_store(MemoryStore())``. :class:`RecordingCache` is a
dict-backed ``StaticCache`` that counts writes and hits. :class:`EventLog` collects the progress events of
one analysis from a (fake) Redis client, and :class:`SourceRepo` describes a scripted local repository.
Nothing here is used by the tasks themselves.
"""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import redis

from tdp_core.contracts import FileExtra, FileSyntax
from tdp_core.features.cache import StaticCache
from tdp_core.serialize import from_jsonable
from tdp_platform.db.rows import FileResultRow
from tdp_worker.persist import ACTIVE_STATUSES, AnalysisInfo, FileInfo, ResultPayload, score_from_row
from tdp_worker.progress import ProgressEvent, channel_for

DEFAULT_REPOSITORY_ID: Final = uuid.UUID("00000000-0000-4000-8000-000000000001")
DEFAULT_MODEL_VERSION: Final = "dev-worker"


@dataclass(frozen=True)
class SourceRepo:
    """A local git repository used as a clone source: its folder and two of its commits."""

    path: Path
    first: str
    head: str

    def blob(self, rel: str, rev: str | None = None) -> str:
        """The blob sha of ``rel`` at ``rev`` (default the head commit)."""
        out = subprocess.run(
            ["git", "rev-parse", f"{rev or self.head}:{rel}"],
            cwd=self.path,
            check=True,
            capture_output=True,
        )
        return out.stdout.decode().strip()


class RecordingCache:
    """A dict-backed ``StaticCache`` that counts ``put`` calls and cache hits."""

    def __init__(self) -> None:
        self.data: dict[str, FileSyntax] = {}
        self.puts = 0
        self.hits = 0

    def get(self, blob_sha: str) -> FileSyntax | None:
        found = self.data.get(blob_sha)
        if found is not None:
            self.hits += 1
        return found

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        self.puts += 1
        self.data[blob_sha] = syntax


class EventLog:
    """The progress events of one analysis, subscribed before the run and read with :meth:`drain`."""

    def __init__(self, client: redis.Redis, analysis_id: uuid.UUID) -> None:
        self._pubsub = client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(channel_for(analysis_id))
        self._pubsub.get_message(timeout=0.1)  # the subscribe confirmation

    def drain(self, timeout_s: float = 0.1) -> list[ProgressEvent]:
        """Every event published since the last call, waiting ``timeout_s`` seconds for each one."""
        events: list[ProgressEvent] = []
        while True:
            message = self._pubsub.get_message(timeout=timeout_s)
            if message is None:
                return events
            if message.get("type") == "message":
                events.append(ProgressEvent.from_json(message["data"]))

    def close(self) -> None:
        """Unsubscribe and release the connection."""
        self._pubsub.close()


@dataclass
class AdviceRow:
    """A stored advice row of :class:`MemoryStore`."""

    file_result_id: int
    model: str
    status: str
    body: dict[str, Any] | None = None
    error: str | None = None


class MemoryStore:
    """An in-memory ``Store`` with the transitions of the platform repositories.

    ``analyses`` maps ids to plain dicts (the columns plus ``error``, ``summary`` and ``stats``), ``files``
    holds each analysis's stored rows in order and ``file_rows`` every row by its sequential id.
    ``artifacts`` is keyed by ``(analysis_id, kind)``, ``plans`` by ``(analysis_id, budget_hours)`` and
    ``advice`` by cache key. ``progress_log`` records every accepted ``set_progress`` call and ``cache`` is
    what :meth:`static_cache` returns.
    """

    def __init__(self) -> None:
        self.analyses: dict[uuid.UUID, dict[str, Any]] = {}
        self.files: dict[uuid.UUID, list[FileResultRow]] = {}
        self.file_rows: dict[int, tuple[uuid.UUID, FileResultRow]] = {}
        self.artifacts: dict[tuple[uuid.UUID, str], Any] = {}
        self.plans: dict[tuple[uuid.UUID, float], Any] = {}
        self.advice: dict[str, AdviceRow] = {}
        self.progress_log: list[tuple[uuid.UUID, str, int]] = []
        self.cache: StaticCache | None = None
        self.closed = False
        self._next_file_id = 1

    # ---------- seeding ----------

    def add_analysis(
        self,
        *,
        commit_sha: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        budget_hours: float = 40.0,
        url: str = "https://github.com/acme/demo",
        host: str = "github.com",
        owner: str = "acme",
        name: str = "demo",
        status: str = "queued",
        repository_id: uuid.UUID = DEFAULT_REPOSITORY_ID,
    ) -> uuid.UUID:
        """Add an analysis row and return its new id."""
        aid = uuid.uuid4()
        self.analyses[aid] = {
            "repository_id": repository_id,
            "url": url,
            "host": host,
            "owner": owner,
            "name": name,
            "commit_sha": commit_sha,
            "model_version": model_version,
            "budget_hours": budget_hours,
            "status": status,
            "stage": None,
            "progress": 0,
            "error": None,
            "summary": None,
            "stats": None,
        }
        return aid

    def add_file(self, analysis_id: uuid.UUID, row: FileResultRow) -> int:
        """Store one file row for an analysis and return its id."""
        fid = self._next_file_id
        self._next_file_id += 1
        self.file_rows[fid] = (analysis_id, row)
        self.files.setdefault(analysis_id, []).append(row)
        return fid

    def row(self, analysis_id: uuid.UUID) -> dict[str, Any]:
        """The stored columns of an analysis (a live dict)."""
        return self.analyses[analysis_id]

    def _info(self, analysis_id: uuid.UUID) -> AnalysisInfo:
        a = self.analyses[analysis_id]
        return AnalysisInfo(
            id=analysis_id,
            repository_id=a["repository_id"],
            url=a["url"],
            host=a["host"],
            owner=a["owner"],
            name=a["name"],
            commit_sha=a["commit_sha"],
            model_version=a["model_version"],
            budget_hours=a["budget_hours"],
            status=a["status"],
            stage=a["stage"],
            progress=a["progress"],
        )

    def _transition(self, analysis_id: uuid.UUID, allowed: tuple[str, ...], **values: Any) -> bool:
        a = self.analyses.get(analysis_id)
        if a is None or a["status"] not in allowed:
            return False
        a.update(values)
        return True

    # ---------- Store ----------

    def load_analysis(self, analysis_id: uuid.UUID) -> AnalysisInfo | None:
        return self._info(analysis_id) if analysis_id in self.analyses else None

    def set_running(self, analysis_id: uuid.UUID) -> bool:
        return self._transition(analysis_id, ACTIVE_STATUSES, status="running", error=None)

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        ok = self._transition(analysis_id, ACTIVE_STATUSES, stage=stage, progress=pct)
        if ok:
            self.progress_log.append((analysis_id, stage, pct))
        return ok

    def set_failed(self, analysis_id: uuid.UUID, error: str) -> bool:
        return self._transition(analysis_id, ACTIVE_STATUSES, status="failed", error=error)

    def save_results(self, analysis_id: uuid.UUID, payload: ResultPayload) -> bool:
        if self.analyses.get(analysis_id, {}).get("status") != "running":
            return False
        self.files[analysis_id] = []
        for row in payload.rows:
            self.add_file(analysis_id, row)
        self.artifacts[(analysis_id, "graph")] = payload.graph
        self.artifacts[(analysis_id, "treemap")] = payload.treemap
        self.plans[(analysis_id, payload.budget_hours)] = payload.plan
        return self._transition(
            analysis_id,
            ("running",),
            status="succeeded",
            stage="done",
            progress=100,
            summary=payload.summary,
            stats=payload.stats,
        )

    def load_file(self, file_result_id: int) -> FileInfo | None:
        found = self.file_rows.get(file_result_id)
        if found is None:
            return None
        aid, row = found
        return FileInfo(
            id=file_result_id,
            analysis=self._info(aid),
            path=row.path,
            blob_sha=row.blob_sha,
            score=score_from_row(row),
            extra=from_jsonable(FileExtra, dict(row.extra)),
        )

    def begin_advice(self, cache_key: str, *, file_result_id: int, model: str) -> str:
        row = self.advice.get(cache_key)
        if row is None:
            row = AdviceRow(file_result_id=file_result_id, model=model, status="queued")
            self.advice[cache_key] = row
        return row.status

    def _advice_transition(self, cache_key: str, **values: Any) -> bool:
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
