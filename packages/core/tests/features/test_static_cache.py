import dataclasses
import gzip
import hashlib
import json
import logging
import math
import multiprocessing
import operator
import threading
from pathlib import Path
from typing import Any

import pytest

from tdp_core.contracts import EXTRACTOR_VERSION, FileSyntax
from tdp_core.features import cache as cache_mod
from tdp_core.features.cache import (
    CACHE_FORMAT,
    PARSE_FAILED,
    PARSE_TIMEOUT,
    DiskStaticCache,
    ParseTask,
    parse_blobs,
    syntax_from_json,
    syntax_to_json,
)
from tdp_core.parsing import PythonPlugin
from tdp_core.parsing.python import parse_python
from tdp_core.serialize import from_jsonable

SOURCE = b"""#!/usr/bin/env python
'''Module docstring.'''
import os
from . import sibling
from ..pkg.mod import a, b as c


class Shape:
    def area(self, scale, *, exact=False):
        # TODO: cache this
        if exact and scale > 1:
            return scale * 2
        for _ in range(3):
            pass
        return 0


def helper(x):
    return [y for y in x if y]
"""


def _blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _source(i: int) -> bytes:
    return SOURCE + f"\n\ndef extra_{i}(v):\n    if v:\n        return {i}\n    return -{i}\n".encode()


def _task(i: int, path: str | None = None) -> ParseTask:
    source = _source(i)
    return ParseTask(_blob_sha(source), path or f"pkg/m{i}.py", source)


def _big_source(functions: int = 400) -> bytes:
    body = "".join(
        f"def f{i}(a, b):\n    if a > b:\n        return [x for x in range(a) if x % {i + 2}]\n"
        "    return b\n\n\n"
        for i in range(functions)
    )
    return body.encode()


class RecordingCache:
    """An in-memory StaticCache that records every call."""

    def __init__(self, stored: dict[str, FileSyntax] | None = None) -> None:
        self.stored = dict(stored or {})
        self.gets: list[str] = []
        self.puts: list[str] = []

    def get(self, blob_sha: str) -> FileSyntax | None:
        self.gets.append(blob_sha)
        return self.stored.get(blob_sha)

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        self.puts.append(blob_sha)
        self.stored[blob_sha] = syntax


class BrokenCache(RecordingCache):
    """A RecordingCache whose get or put records the call and then raises."""

    def __init__(
        self,
        stored: dict[str, FileSyntax] | None = None,
        *,
        fail_get: bool = False,
        fail_put: bool = False,
    ) -> None:
        super().__init__(stored)
        self.fail_get = fail_get
        self.fail_put = fail_put

    def get(self, blob_sha: str) -> FileSyntax | None:
        if self.fail_get:
            self.gets.append(blob_sha)
            raise RuntimeError("database is down")
        return super().get(blob_sha)

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        if self.fail_put:
            self.puts.append(blob_sha)
            raise OSError("disk full")
        super().put(blob_sha, syntax)


class Poison(bytes):
    """Bytes that cannot be unpickled: a worker raises while reading the task and dies holding it."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (operator.truediv, (1, 0))


@pytest.fixture
def syntax() -> FileSyntax:
    result = parse_python("pkg/shape.py", SOURCE)
    assert result.functions and result.classes and result.comments and len(result.imports) == 3
    assert any(i.level == 2 and i.names for i in result.imports)
    return result


@pytest.fixture
def no_children() -> Any:
    yield
    assert multiprocessing.active_children() == []


# ------------------------------------------------------------------------------------------- JSON codec
def test_json_round_trip(syntax: FileSyntax) -> None:
    data = syntax_to_json(syntax)
    assert json.loads(json.dumps(data)) == data
    assert syntax_from_json(data) == syntax
    assert from_jsonable(FileSyntax, data) == syntax
    assert syntax_from_json(json.loads(json.dumps(data))) == syntax
    empty = parse_python("e.py", b"")
    assert syntax_from_json(syntax_to_json(empty)) == empty


def _mutated(data: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(data))
    target: Any = out
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return out


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("sloc",), _DELETE),
        (("extra",), 1),
        (("sloc",), True),
        (("sloc",), 3.0),
        (("classes",), "1"),
        (("parse_error",), 0),
        (("path",), None),
        (("language",), 1),
        (("tokens", 0), 1),
        (("token_lines", 0), 1.0),
        (("token_lines", 0), False),
        (("tokens",), ["ID"]),
        (("tokens",), "ID"),
        (("functions",), {}),
        (("functions", 0, "cyclomatic"), _DELETE),
        (("functions", 0, "extra"), 1),
        (("functions", 0, "name"), 1),
        (("functions", 0, "params"), True),
        (("comments", 0, "line"), _DELETE),
        (("comments", 0, "text"), None),
        (("imports", 0, "names"), "x"),
        (("imports", 1, "names", 0), 1),
        (("imports", 0, "level"), 0.5),
        (("imports", 0, "extra"), 1),
    ],
)
def test_syntax_from_json_rejects_invalid_data(syntax: FileSyntax, path: tuple[Any, ...], value: Any) -> None:
    data = _mutated(syntax_to_json(syntax), path, value)
    with pytest.raises(ValueError):
        syntax_from_json(data)


@pytest.mark.parametrize("data", [None, [], "x", 1])
def test_syntax_from_json_rejects_non_objects(data: object) -> None:
    with pytest.raises(ValueError):
        syntax_from_json(data)


# ------------------------------------------------------------------------------------------- disk cache
def test_disk_cache_round_trip_layout_and_determinism(tmp_path: Path, syntax: FileSyntax) -> None:
    root = tmp_path / "static"
    cache = DiskStaticCache(root)
    sha = _blob_sha(SOURCE)
    assert cache.root == root
    assert cache.get(sha) is None
    assert not root.exists()

    cache.put(sha, syntax)
    path = cache.path_for(sha)
    assert path == root / sha[:2] / f"{sha}.{EXTRACTOR_VERSION}.json.gz"
    assert path.is_file()
    first = path.read_bytes()
    cache.put(sha, syntax)
    assert path.read_bytes() == first
    assert cache.get(sha) == syntax
    assert list(root.rglob("*.tmp")) == []
    assert [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()] == [
        f"{sha[:2]}/{sha}.1.json.gz"
    ]

    wrapper = json.loads(gzip.decompress(first))
    assert wrapper["format"] == CACHE_FORMAT
    assert wrapper["extractor_version"] == EXTRACTOR_VERSION
    assert wrapper["blob_sha"] == sha
    assert syntax_from_json(wrapper["syntax"]) == syntax

    sha256 = hashlib.sha256(SOURCE).hexdigest()
    cache.put(sha256, syntax)
    assert cache.get(sha256) == syntax


def _write(cache: DiskStaticCache, sha: str, payload: bytes, *, compress: bool = True) -> None:
    path = cache.path_for(sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload) if compress else payload)


def _wrapper(parsed: FileSyntax, sha: str, **override: Any) -> bytes:
    data: dict[str, Any] = {
        "format": CACHE_FORMAT,
        "extractor_version": EXTRACTOR_VERSION,
        "blob_sha": sha,
        "syntax": syntax_to_json(parsed),
    }
    data.update(override)
    return json.dumps(data).encode()


def test_disk_cache_get_treats_invalid_files_as_misses(tmp_path: Path, syntax: FileSyntax) -> None:
    cache = DiskStaticCache(tmp_path)
    sha = _blob_sha(SOURCE)
    other = _blob_sha(b"other")
    cases = [
        (b"not gzip at all", False),
        (gzip.compress(b"{}")[:-6], False),  # truncated gzip
        (b"{not json", True),
        (b"\xff\xfe\x00", True),
        (b"[1, 2]", True),
        (_wrapper(syntax, sha, format=2), True),
        (_wrapper(syntax, sha, format=True), True),
        (_wrapper(syntax, sha, extractor_version="0"), True),
        (_wrapper(syntax, other), True),
        (_wrapper(syntax, sha, syntax={"path": "x"}), True),
        (_wrapper(syntax, sha, extra=1), True),
    ]
    for payload, compress in cases:
        _write(cache, sha, payload, compress=compress)
        assert cache.get(sha) is None, payload[:40]
    _write(cache, sha, _wrapper(syntax, sha))
    assert cache.get(sha) == syntax
    cache.path_for(sha).unlink()
    cache.path_for(sha).mkdir()  # a folder where the file belongs cannot be read
    assert cache.get(sha) is None


@pytest.mark.parametrize("sha", ["A" * 40, "a" * 39, "a" * 41, "../x", "", "g" * 40, "a" * 63])
def test_disk_cache_rejects_invalid_shas(tmp_path: Path, syntax: FileSyntax, sha: str) -> None:
    cache = DiskStaticCache(tmp_path)
    with pytest.raises(ValueError):
        cache.path_for(sha)
    with pytest.raises(ValueError):
        cache.get(sha)
    with pytest.raises(ValueError):
        cache.put(sha, syntax)


def test_disk_cache_put_never_raises_for_io_errors(tmp_path: Path, syntax: FileSyntax) -> None:
    root = tmp_path / "file"
    root.write_text("not a folder")
    cache = DiskStaticCache(root)
    sha = _blob_sha(SOURCE)
    cache.put(sha, syntax)
    assert cache.get(sha) is None
    assert root.read_text() == "not a folder"


# ------------------------------------------------------------------------------------------- parse_blobs
def test_parse_blobs_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = cache_mod._parse_one

    def spy(path: str, source: bytes) -> FileSyntax | str:
        calls.append(path)
        return real(path, source)

    monkeypatch.setattr(cache_mod, "_parse_one", spy)
    tasks = [_task(0), _task(1), _task(0, "pkg/copy.py")]
    store = RecordingCache()
    out = parse_blobs(tasks, store, timeout_s=5.0, jobs=4)
    assert list(out) == [tasks[0].key, tasks[1].key]
    assert calls == ["pkg/m0.py", "pkg/m1.py"]
    for task in tasks[:2]:
        assert out[task.key] == parse_python(task.path, task.source)
    assert store.puts == [tasks[0].key, tasks[1].key]

    calls.clear()
    store.puts.clear()
    again = parse_blobs([_task(1, "other/name.py"), _task(2)], store, timeout_s=5.0)
    assert calls == ["pkg/m2.py"]
    assert store.puts == [_task(2).key]
    hit = again[_task(1).key]
    assert isinstance(hit, FileSyntax) and hit.path == "pkg/m1.py"  # a hit keeps the stored path

    out_none = parse_blobs([_task(3)], None, timeout_s=math.inf)
    assert out_none[_task(3).key] == parse_python("pkg/m3.py", _source(3))


def test_parse_blobs_treats_another_language_as_a_miss() -> None:
    task = _task(0)
    foreign = dataclasses.replace(parse_python(task.path, task.source), language="cobol", sloc=999)
    store = RecordingCache({task.key: foreign})
    out = parse_blobs([task], store, timeout_s=5.0)
    assert out[task.key] == parse_python(task.path, task.source)
    assert store.puts == [task.key]


def test_parse_blobs_survives_a_failing_get(caplog: pytest.LogCaptureFixture) -> None:
    tasks = [_task(0), _task(1), _task(2)]
    store = BrokenCache(fail_get=True)
    with caplog.at_level(logging.WARNING, logger="tdp_core.features.cache"):
        out = parse_blobs(tasks, store, timeout_s=5.0)
    assert out == {t.key: parse_python(t.path, t.source) for t in tasks}
    assert store.gets == [tasks[0].key]  # dropped after the first failure, puts included
    assert store.puts == []
    assert sum("static cache get failed" in r.message for r in caplog.records) == 1


def test_parse_blobs_survives_a_failing_put(caplog: pytest.LogCaptureFixture) -> None:
    tasks = [_task(0), _task(1), _task(2)]
    old = parse_python("old/name.py", tasks[0].source)
    store = BrokenCache({tasks[0].key: old}, fail_put=True)
    with caplog.at_level(logging.WARNING, logger="tdp_core.features.cache"):
        out = parse_blobs(tasks, store, timeout_s=5.0)
    assert out[tasks[0].key] is old  # a hit read before the failure is kept
    for task in tasks[1:]:
        assert out[task.key] == parse_python(task.path, task.source)
    assert store.gets == [t.key for t in tasks]
    assert store.puts == [tasks[1].key]
    assert sum("static cache put failed" in r.message for r in caplog.records) == 1


def test_parse_blobs_validates_arguments() -> None:
    store = RecordingCache()
    assert parse_blobs([], store, timeout_s=1.0) == {}
    assert store.gets == [] and store.puts == []
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="timeout_s"):
            parse_blobs([_task(0)], store, timeout_s=bad)
    with pytest.raises(ValueError, match="jobs"):
        parse_blobs([_task(0)], store, timeout_s=1.0, jobs=0)
    with pytest.raises(ValueError, match="plugin"):
        parse_blobs([_task(0), ParseTask("a" * 40, "README.md", b"")], store, timeout_s=1.0)
    assert store.gets == []


def test_parse_blobs_soft_limit_in_process(caplog: pytest.LogCaptureFixture) -> None:
    store = RecordingCache()
    tasks = [_task(0), _task(1)]
    with caplog.at_level(logging.WARNING, logger="tdp_core.features.cache"):
        out = parse_blobs(tasks, store, timeout_s=1e-9)
    assert out == {t.key: PARSE_TIMEOUT for t in tasks}
    assert store.puts == []
    assert sum("took over" in r.message for r in caplog.records) == 2


def test_parse_blobs_isolates_parser_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = PythonPlugin.parse

    def flaky(self: PythonPlugin, path: str, source: bytes) -> FileSyntax:
        if path == "pkg/m1.py":
            raise RecursionError("too deep")
        return real(self, path, source)

    monkeypatch.setattr(PythonPlugin, "parse", flaky)
    store = RecordingCache()
    tasks = [_task(0), _task(1), _task(2)]
    with caplog.at_level(logging.WARNING, logger="tdp_core.features.cache"):
        out = parse_blobs(tasks, store, timeout_s=5.0)
    assert out[tasks[1].key] == PARSE_FAILED
    assert isinstance(out[tasks[0].key], FileSyntax) and isinstance(out[tasks[2].key], FileSyntax)
    assert store.puts == [tasks[0].key, tasks[2].key]
    assert any("pkg/m1.py" in r.message and "failed" in r.message for r in caplog.records)
    assert cache_mod._parse_one("README.md", b"") == PARSE_FAILED


def test_parse_blobs_falls_back_in_daemonic_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_mod, "_MIN_TASKS_PER_WORKER", 1)
    monkeypatch.setattr(cache_mod, "_can_spawn", lambda: False)

    def no_pool(*args: object) -> None:
        raise AssertionError("the pool must not run")

    monkeypatch.setattr(cache_mod, "_parse_in_pool", no_pool)
    tasks = [_task(i) for i in range(4)]
    out = parse_blobs(tasks, None, timeout_s=5.0, jobs=4)
    assert out == {t.key: parse_python(t.path, t.source) for t in tasks}


def test_can_spawn_outside_daemons() -> None:
    assert cache_mod._can_spawn() is True


# ------------------------------------------------------------------------------------------- worker pool
def test_worker_main_answers_requests_until_told_to_stop() -> None:
    parent, child = multiprocessing.Pipe(duplex=True)
    worker = threading.Thread(target=cache_mod._worker_main, args=(child,))
    worker.start()
    assert parent.recv() == ("ready", "", None)
    task = _task(0)
    parent.send((task.key, task.path, task.source))
    assert parent.recv() == ("done", task.key, parse_python(task.path, task.source))
    parent.send(("k", "README.md", b""))
    assert parent.recv() == ("done", "k", PARSE_FAILED)
    parent.send(None)
    worker.join(10)
    assert not worker.is_alive()
    parent.close()

    parent, child = multiprocessing.Pipe(duplex=True)
    worker = threading.Thread(target=cache_mod._worker_main, args=(child,))
    worker.start()
    assert parent.recv()[0] == "ready"
    parent.close()  # the parent goes away: the worker returns
    worker.join(10)
    assert not worker.is_alive()


def test_pool_matches_in_process(monkeypatch: pytest.MonkeyPatch, no_children: None) -> None:
    monkeypatch.setattr(cache_mod, "_MIN_TASKS_PER_WORKER", 1)
    tasks = [_task(i) for i in range(6)]
    store = RecordingCache()
    out = parse_blobs(tasks, store, timeout_s=60.0, jobs=2)
    assert out == {t.key: parse_python(t.path, t.source) for t in tasks}
    assert sorted(store.puts) == sorted(t.key for t in tasks)


def test_pool_hard_limit_kills_workers(monkeypatch: pytest.MonkeyPatch, no_children: None) -> None:
    monkeypatch.setattr(cache_mod, "_MIN_TASKS_PER_WORKER", 1)
    big = _big_source()
    tasks = [ParseTask(_blob_sha(big + bytes([i])), f"pkg/big{i}.py", big + b"#" * i) for i in range(3)]
    store = RecordingCache()
    out = parse_blobs(tasks, store, timeout_s=1e-9, jobs=2)
    assert out == {t.key: PARSE_TIMEOUT for t in tasks}
    assert store.puts == []


def test_pool_survives_a_dying_worker(monkeypatch: pytest.MonkeyPatch, no_children: None) -> None:
    # Unpickling the source raises inside the worker, which then exits while it holds the task.
    monkeypatch.setattr(cache_mod, "_MIN_TASKS_PER_WORKER", 1)
    tasks = [ParseTask("a" * 40, "pkg/bad.py", Poison(b"x = 1\n"))] + [_task(i) for i in range(4)]
    out = parse_blobs(tasks, None, timeout_s=60.0, jobs=2)
    assert out["a" * 40] == PARSE_FAILED
    for task in tasks[1:]:
        assert out[task.key] == parse_python(task.path, task.source)


def test_pool_start_up_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, no_children: None
) -> None:
    monkeypatch.setattr(cache_mod, "_MIN_TASKS_PER_WORKER", 1)
    monkeypatch.setattr(cache_mod, "_STARTUP_TIMEOUT_S", 1e-3)
    tasks = [_task(i) for i in range(4)]
    with caplog.at_level(logging.WARNING, logger="tdp_core.features.cache"):
        out = parse_blobs(tasks, None, timeout_s=60.0, jobs=2)
    assert out == {t.key: parse_python(t.path, t.source) for t in tasks}
    assert any("parse workers failed" in r.message for r in caplog.records)


def test_pool_process_start_errors_fall_back(monkeypatch: pytest.MonkeyPatch, no_children: None) -> None:
    def refuse(self: object) -> None:
        raise AssertionError("daemonic processes are not allowed to have children")

    monkeypatch.setattr(multiprocessing.get_context("spawn").Process, "start", refuse)
    tasks = [_task(i) for i in range(3)]
    out = cache_mod._parse_in_pool(tasks, 2, 60.0)
    assert out == {t.key: parse_python(t.path, t.source) for t in tasks}


def test_pool_rejects_unexpected_messages() -> None:
    pool = cache_mod._Pool(1, 1.0, 1)
    parent, child = multiprocessing.Pipe()
    worker = cache_mod._Worker(multiprocessing.current_process(), parent, 0.0)
    results: dict[str, FileSyntax | str] = {}
    pool._receive(worker, ("ready", "", None), results)
    assert worker.ready
    for message in (("ready", "", None), ("done", "k", "x"), "garbage", ("done", "k")):
        with pytest.raises(cache_mod._PoolFailure):
            pool._receive(worker, message, results)
    worker.task = _task(0)
    pool._receive(worker, ("done", worker.task.key, PARSE_FAILED), results)
    assert results == {_task(0).key: PARSE_FAILED} and worker.task is None
    parent.close()
    child.close()
