"""Progress events: channel names, the JSON shape, publishing and the progress callback."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import fakeredis
import pytest
import redis

from tdp_worker import progress
from tdp_worker.progress import (
    AnalysisCancelled,
    ProgressEvent,
    ProgressReporter,
    channel_for,
    get_redis,
    publish,
    set_redis,
)
from tdp_worker.testing import EventLog


def test_channel_for() -> None:
    aid = uuid.uuid4()
    assert channel_for(aid) == f"analysis:{aid}"
    assert channel_for(str(aid).upper()) == f"analysis:{aid}"
    with pytest.raises(ValueError):
        channel_for("not-a-uuid")


def test_event_round_trip() -> None:
    event = ProgressEvent("parse", 50, "Parsed the source files")
    assert event.to_json() == '{"stage":"parse","pct":50,"message":"Parsed the source files"}'
    assert not event.terminal
    assert ProgressEvent.from_json(event.to_json()) == event
    done = ProgressEvent("done", 100, "Analysis complete", "succeeded")
    assert done.terminal
    assert json.loads(done.to_json())["status"] == "succeeded"
    assert ProgressEvent.from_json(done.to_json().encode()) == done


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        b"\xff",
        "[1, 2]",
        '{"stage": "", "pct": 1, "message": "m"}',
        '{"stage": 3, "pct": 1, "message": "m"}',
        '{"stage": "s", "pct": true, "message": "m"}',
        '{"stage": "s", "pct": 1.5, "message": "m"}',
        '{"stage": "s", "pct": 101, "message": "m"}',
        '{"stage": "s", "pct": -1, "message": "m"}',
        '{"stage": "s", "pct": 1, "message": null}',
        '{"stage": "s", "pct": 1, "message": "m", "status": "running"}',
    ],
)
def test_from_json_rejects(data: str | bytes) -> None:
    with pytest.raises(ValueError):
        ProgressEvent.from_json(data)


def test_publish_reaches_a_subscriber(fake_redis: fakeredis.FakeRedis) -> None:
    aid = uuid.uuid4()
    log = EventLog(fake_redis, aid)
    assert publish(aid, ProgressEvent("mine", 25, "Mined 3 commits"))
    assert log.drain() == [ProgressEvent("mine", 25, "Mined 3 commits")]
    log.close()


class BrokenRedis:
    def publish(self, channel: str, message: str) -> int:
        raise redis.ConnectionError("refused")


def test_publish_is_best_effort(worker_env: dict[str, Path], caplog: pytest.LogCaptureFixture) -> None:
    broken: Any = BrokenRedis()
    set_redis(broken)
    assert publish(uuid.uuid4(), ProgressEvent("mine", 25, "x")) is False
    assert "could not publish" in caplog.text


def test_get_redis_is_per_process(worker_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_redis()
    assert get_redis() is first
    assert first.connection_pool.connection_kwargs["socket_timeout"] == progress.REDIS_TIMEOUT_S
    monkeypatch.setattr(progress.os, "getpid", lambda: -1)
    assert get_redis() is not first
    set_redis(None)
    assert progress._client is None


class Recorder:
    def __init__(self, answers: list[bool | Exception]) -> None:
        self.answers = answers
        self.calls: list[tuple[uuid.UUID, str, int]] = []

    def __call__(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        self.calls.append((analysis_id, stage, pct))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_reporter_records_then_publishes(fake_redis: fakeredis.FakeRedis) -> None:
    aid = uuid.uuid4()
    log = EventLog(fake_redis, aid)
    record = Recorder([True, True, True])
    reporter = ProgressReporter(aid, record)
    assert reporter.last_pct == 0

    reporter("mine", 25, "Mined")
    reporter("over", 150, "Clamped high")
    reporter("under", -3, "Clamped low")

    assert record.calls == [(aid, "mine", 25), (aid, "over", 100), (aid, "under", 0)]
    assert [(e.stage, e.pct) for e in log.drain()] == [("mine", 25), ("over", 100), ("under", 0)]
    assert reporter.last_pct == 0


def test_reporter_stops_a_cancelled_analysis(fake_redis: fakeredis.FakeRedis) -> None:
    aid = uuid.uuid4()
    log = EventLog(fake_redis, aid)
    reporter = ProgressReporter(aid, Recorder([True, False]))
    reporter("mine", 25, "Mined")
    with pytest.raises(AnalysisCancelled):
        reporter("parse", 50, "Parsed")
    assert [e.stage for e in log.drain()] == ["mine"]
    assert reporter.last_pct == 25


def test_reporter_survives_a_failing_record(
    fake_redis: fakeredis.FakeRedis, caplog: pytest.LogCaptureFixture
) -> None:
    aid = uuid.uuid4()
    log = EventLog(fake_redis, aid)
    reporter = ProgressReporter(aid, Recorder([RuntimeError("db down")]))
    reporter("graphs", 60, "Built the graphs")
    assert [e.stage for e in log.drain()] == ["graphs"]
    assert reporter.last_pct == 60
    assert "could not record progress" in caplog.text
