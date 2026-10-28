"""Progress events of analyses: Redis pub/sub messages on ``analysis:<id>`` and the progress callback.

Every event is a compact JSON object ``{"stage": str, "pct": int, "message": str}``; the last event of an
analysis also carries ``"status"`` (``"succeeded"`` with stage ``done`` and pct 100, or ``"failed"`` with
stage ``failed``, the user-safe error as message and the last reached percentage). The API relays these to
the browser as server-sent events (plan section 3.7).

Publishing is best effort: Redis being down never fails an analysis, because the database row holds the
state too. :class:`ProgressReporter` is the ``progress`` callback for ``tdp_core.analyze.analyze``: it
records the stage in the database, then publishes the event, and stops the analysis with
:class:`AnalysisCancelled` once the row is no longer queued or running.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

import redis

from tdp_platform.settings import get_settings

logger = logging.getLogger(__name__)

CHANNEL_PREFIX: Final = "analysis:"
DONE_STAGE: Final = "done"  # equals tdp_platform.db.repos.DONE_STAGE
FAILED_STAGE: Final = "failed"
EventStatus = Literal["succeeded", "failed"]
EVENT_STATUSES: Final[tuple[str, ...]] = get_args(EventStatus)
REDIS_TIMEOUT_S: Final = 10.0

RecordFn = Callable[[uuid.UUID, str, int], bool]
"""Stores ``(analysis_id, stage, pct)``; returns False when the analysis is no longer queued or running."""

_client: redis.Redis | None = None
_client_pid: int | None = None
_client_lock = threading.Lock()


def channel_for(analysis_id: uuid.UUID | str) -> str:
    """The pub/sub channel of an analysis, ``analysis:<canonical uuid>``; ``ValueError`` for a non-UUID."""
    return f"{CHANNEL_PREFIX}{uuid.UUID(str(analysis_id))}"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One progress message. ``pct`` is the percentage of the whole analysis (0..100).

    ``status`` is set only on the terminal event of an analysis.
    """

    stage: str
    pct: int
    message: str
    status: EventStatus | None = None

    @property
    def terminal(self) -> bool:
        """True for the last event of an analysis (it carries a status)."""
        return self.status is not None

    def to_json(self) -> str:
        """Compact JSON with ``stage``, ``pct`` and ``message``, plus ``status`` when it is set."""
        data: dict[str, Any] = {"stage": self.stage, "pct": self.pct, "message": self.message}
        if self.status is not None:
            data["status"] = self.status
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: bytes | str) -> ProgressEvent:
        """Parse a published event.

        Raises ``ValueError`` for text that is not a JSON object, an empty or non-string stage, a pct that
        is not an integer 0..100 (booleans are rejected), a non-string message or an unknown status.
        """
        try:
            obj: Any = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"progress event is not JSON: {exc}") from None
        if not isinstance(obj, dict):
            raise ValueError("progress event must be a JSON object")
        stage, pct, message, status = obj.get("stage"), obj.get("pct"), obj.get("message"), obj.get("status")
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"progress event stage must be a non-empty string, got {stage!r}")
        if isinstance(pct, bool) or not isinstance(pct, int) or not 0 <= pct <= 100:
            raise ValueError(f"progress event pct must be an integer 0..100, got {pct!r}")
        if not isinstance(message, str):
            raise ValueError(f"progress event message must be a string, got {message!r}")
        if status is not None and status not in EVENT_STATUSES:
            raise ValueError(f"progress event status must be one of {EVENT_STATUSES}, got {status!r}")
        return cls(stage=stage, pct=pct, message=message, status=status)


def get_redis() -> redis.Redis:
    """The Redis client of this process for ``REDIS_URL`` (rebuilt after a fork).

    Connections are made lazily on first use, with ``REDIS_TIMEOUT_S`` second socket and connect timeouts.
    """
    global _client, _client_pid
    with _client_lock:
        pid = os.getpid()
        if _client is None or _client_pid != pid:
            _client = redis.Redis.from_url(
                get_settings().redis_url,
                socket_timeout=REDIS_TIMEOUT_S,
                socket_connect_timeout=REDIS_TIMEOUT_S,
            )
            _client_pid = pid
        return _client


def set_redis(client: redis.Redis | None) -> None:
    """Use ``client`` as this process's Redis client (tests), or None to build a new one on next use.

    The previous client is dropped, not closed: after a fork its connections belong to the parent.
    """
    global _client, _client_pid
    with _client_lock:
        _client = client
        _client_pid = os.getpid() if client is not None else None


def publish(analysis_id: uuid.UUID, event: ProgressEvent) -> bool:
    """Publish ``event`` on the analysis's channel. Best effort: False (and a warning) when Redis fails."""
    try:
        get_redis().publish(channel_for(analysis_id), event.to_json())
    except (redis.RedisError, OSError) as exc:
        logger.warning("could not publish progress of analysis %s: %s", analysis_id, exc)
        return False
    return True


class AnalysisCancelled(Exception):
    """The analysis row is no longer queued or running (failed or finished elsewhere); stop working on it."""


class ProgressReporter:
    """Progress callback ``(stage, pct, message)`` that records the stage in the database and publishes it.

    ``record`` is called first. If it raises, the error is logged and the event is still published
    (database progress is best effort). If it returns False the analysis was moved out of queued or running
    by another actor, so :class:`AnalysisCancelled` is raised and nothing is published; ``analyze`` passes
    callback exceptions through unchanged, which stops the analysis.
    """

    def __init__(self, analysis_id: uuid.UUID, record: RecordFn) -> None:
        self._analysis_id = analysis_id
        self._record = record
        self._last_pct = 0

    @property
    def last_pct(self) -> int:
        """The percentage of the last reported event (0 before the first one)."""
        return self._last_pct

    def __call__(self, stage: str, pct: int, message: str) -> None:
        """Report that the analysis reached ``stage`` at ``pct`` percent (clamped to 0..100)."""
        pct = max(0, min(100, int(pct)))
        try:
            ok = self._record(self._analysis_id, stage, pct)
        except Exception:
            logger.warning("could not record progress of analysis %s", self._analysis_id, exc_info=True)
            ok = True
        if ok is False:
            raise AnalysisCancelled(f"analysis {self._analysis_id} is no longer queued or running")
        self._last_pct = pct
        publish(self._analysis_id, ProgressEvent(stage=stage, pct=pct, message=message))
