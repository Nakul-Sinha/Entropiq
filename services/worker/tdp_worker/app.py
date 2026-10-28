"""The Celery application of the workers (``celery -A tdp_worker.app:app worker``).

Broker and result backend are ``REDIS_URL``. Two queues: ``analysis`` (fetching repositories, running
analyses and the failure handler) and ``llm`` (advice), so slow LLM calls never hold up analyses. Messages
are JSON only, each worker process takes one task at a time (``worker_prefetch_multiplier=1``) and long tasks
acknowledge late, so a crashed worker's task is delivered again. Redis redelivers unacknowledged tasks after
:data:`VISIBILITY_TIMEOUT_S`, which must stay above every task's hard time limit (960 s).

Creating the app never connects to anything. Forked worker processes drop the per-process store, Redis
client and bundle cache they inherited (:func:`reset_process_state`), so no connection is shared with the
parent.
"""

from __future__ import annotations

from typing import Any, Final

from celery import Celery
from celery.signals import worker_process_init

from tdp_platform.settings import PlatformSettings, get_settings
from tdp_worker import bundle_loader, persist, progress

ANALYSIS_QUEUE: Final = "analysis"
LLM_QUEUE: Final = "llm"
TASK_ROUTES: Final[dict[str, dict[str, str]]] = {
    "tdp_worker.fetch_repo": {"queue": ANALYSIS_QUEUE},
    "tdp_worker.run_analysis": {"queue": ANALYSIS_QUEUE},
    "tdp_worker.mark_failed": {"queue": ANALYSIS_QUEUE},
    "tdp_worker.generate_advice": {"queue": LLM_QUEUE},
}
VISIBILITY_TIMEOUT_S: Final = 3600  # seconds; above every task's hard time limit
RESULT_EXPIRES_S: Final = 86_400  # seconds that task results stay in Redis


def celery_config(settings: PlatformSettings) -> dict[str, Any]:
    """The Celery configuration for ``settings`` (broker and backend ``settings.redis_url``)."""
    return {
        "broker_url": settings.redis_url,
        "result_backend": settings.redis_url,
        "task_serializer": "json",
        "result_serializer": "json",
        "accept_content": ["json"],
        "task_default_queue": ANALYSIS_QUEUE,
        "task_routes": dict(TASK_ROUTES),
        "worker_prefetch_multiplier": 1,
        "task_track_started": True,
        "result_expires": RESULT_EXPIRES_S,
        "broker_transport_options": {"visibility_timeout": VISIBILITY_TIMEOUT_S},
        "result_backend_transport_options": {"visibility_timeout": VISIBILITY_TIMEOUT_S},
        "broker_connection_retry_on_startup": True,
        "enable_utc": True,
        "timezone": "UTC",
    }


def create_app(settings: PlatformSettings | None = None) -> Celery:
    """A Celery app named ``tdp_worker`` that loads ``tdp_worker.tasks`` (settings default to the env)."""
    celery_app = Celery("tdp_worker", include=["tdp_worker.tasks"])
    celery_app.conf.update(celery_config(settings or get_settings()))
    return celery_app


app = create_app()


def reset_process_state() -> None:
    """Drop this process's store, Redis client and loaded bundles; they are rebuilt on next use.

    Inherited connections are only dropped, never closed, because after a fork they belong to the parent.
    """
    persist.set_store(None)
    progress.set_redis(None)
    bundle_loader.clear_cache()


@worker_process_init.connect(weak=False)
def _on_worker_process_init(**_: Any) -> None:
    reset_process_state()
