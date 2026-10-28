"""The Celery app: task names, routes, limits, configuration and per-process state."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import fakeredis
import pytest

from tdp_platform.settings import PlatformSettings
from tdp_worker import bundle_loader, persist, progress, tasks
from tdp_worker.app import (
    ANALYSIS_QUEUE,
    LLM_QUEUE,
    TASK_ROUTES,
    VISIBILITY_TIMEOUT_S,
    app,
    celery_config,
    create_app,
    reset_process_state,
)
from tdp_worker.testing import MemoryStore

TASK_NAMES = ("tdp_worker.fetch_repo", "tdp_worker.run_analysis", "tdp_worker.mark_failed", "tdp_worker.generate_advice")


def _settings(**values: Any) -> PlatformSettings:
    kwargs: dict[str, Any] = {"_env_file": None, **values}
    return PlatformSettings(**kwargs)


def test_tasks_are_registered() -> None:
    assert set(TASK_NAMES) <= set(app.tasks)
    assert tasks.fetch_repo.name == "tdp_worker.fetch_repo"
    assert tasks.generate_advice.name == "tdp_worker.generate_advice"


@pytest.mark.parametrize("name", TASK_NAMES)
def test_routes(name: str) -> None:
    expected = LLM_QUEUE if name == "tdp_worker.generate_advice" else ANALYSIS_QUEUE
    assert app.amqp.router.route({}, name)["queue"].name == expected
    assert TASK_ROUTES[name] == {"queue": expected}


def test_task_limits() -> None:
    run = tasks.run_analysis
    assert (run.acks_late, run.soft_time_limit, run.time_limit) == (True, 900, 960)
    fetch = tasks.fetch_repo
    assert (fetch.acks_late, fetch.max_retries, fetch.soft_time_limit, fetch.time_limit) == (True, 3, 900, 960)
    advice = tasks.generate_advice
    assert (advice.acks_late, advice.max_retries, advice.time_limit) == (True, 3, 960)
    assert tasks.mark_failed.acks_late is True


def test_celery_config() -> None:
    config = celery_config(_settings(redis_url="redis://example:6379/3"))
    assert config["broker_url"] == config["result_backend"] == "redis://example:6379/3"
    assert config["task_serializer"] == config["result_serializer"] == "json"
    assert config["accept_content"] == ["json"]
    assert config["task_default_queue"] == "analysis"
    assert config["worker_prefetch_multiplier"] == 1
    assert config["broker_transport_options"]["visibility_timeout"] == VISIBILITY_TIMEOUT_S
    assert VISIBILITY_TIMEOUT_S > tasks.ANALYSIS_TIME_LIMIT_S
    assert config["enable_utc"] is True and config["timezone"] == "UTC"


def test_create_app_does_not_connect() -> None:
    created = create_app(_settings(redis_url="redis://127.0.0.1:1/9"))
    assert created.conf.broker_url == "redis://127.0.0.1:1/9"
    assert created.main == "tdp_worker"


def test_importing_tasks_loads_no_heavy_libraries() -> None:
    code = (
        "import sys, tdp_worker.app, tdp_worker.tasks\n"
        "heavy = [m for m in ('xgboost', 'shap', 'anthropic', 'sklearn') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, timeout=120)
    assert out.stdout.decode().strip() == ""


def test_reset_process_state(worker_env: dict[str, Path], models_dir: Path) -> None:
    memory = MemoryStore()
    persist.set_store(memory)
    client = fakeredis.FakeRedis()
    progress.set_redis(client)
    first = bundle_loader.get_bundle("dev-worker", model_dir=models_dir)

    reset_process_state()

    assert persist.get_store() is not memory
    assert progress.get_redis() is not client
    assert bundle_loader.get_bundle("dev-worker", model_dir=models_dir) is not first
