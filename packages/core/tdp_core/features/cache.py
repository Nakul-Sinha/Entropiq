"""Static-analysis cache and cached, time-limited parsing of source blobs.

A :class:`~tdp_core.contracts.FileSyntax` depends only on the bytes of a blob and on
:data:`~tdp_core.contracts.EXTRACTOR_VERSION` (the path only lands in ``FileSyntax.path``), so parses are
cached by blob sha. A cache hit may therefore carry the path the blob was first parsed under; callers
re-path it with ``dataclasses.replace``.

:func:`parse_blobs` looks every blob up in a :class:`StaticCache`, parses the misses and stores the new
parses. Small batches are parsed in this process, where the time limit is soft: a parse cannot be
interrupted, so it runs to the end and its result is discarded when it took longer than the limit. Large
batches (a cold first snapshot of a big repository) are parsed by spawned worker processes, where the limit
is hard: a worker that overruns it is killed and replaced. Both modes skip the same files (those that take
longer than the limit), and a parser crash or exception only loses the one file. A cache that raises is
logged once and dropped for the rest of the call, so a storage outage only makes parsing slower.

:class:`DiskStaticCache` implements the cache as one gzip JSON file per blob in the
``data/cache/static/<sha[:2]>/<sha>.<EXTRACTOR_VERSION>.json.gz`` layout of plan section 3.4.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gzip
import json
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import re
import time
import uuid
import zlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from tdp_core.contracts import EXTRACTOR_VERSION, CommentInfo, FileSyntax, FunctionInfo, RawImport
from tdp_core.parsing import get_plugin
from tdp_core.serialize import to_jsonable

if TYPE_CHECKING:
    from multiprocessing.connection import _ConnectionBase
    from multiprocessing.process import BaseProcess

__all__ = [
    "CACHE_FORMAT",
    "PARSE_FAILED",
    "PARSE_TIMEOUT",
    "DiskStaticCache",
    "ParseTask",
    "StaticCache",
    "parse_blobs",
    "syntax_from_json",
    "syntax_to_json",
]

logger = logging.getLogger(__name__)

PARSE_TIMEOUT: Final = "timeout"
"""Result (and skip reason) for a file whose parse took longer than the time limit."""
PARSE_FAILED: Final = "parse_failed"
"""Result (and skip reason) for a file whose parser raised or whose worker process died."""
CACHE_FORMAT: Final = 1
"""Version of the :class:`DiskStaticCache` file wrapper (independent of ``EXTRACTOR_VERSION``)."""

# A worker pool is used only when every worker gets at least this many parses: starting one spawned worker
# (it imports pandas, pyarrow and tree-sitter) costs about as much as parsing a few thousand files here.
_MIN_TASKS_PER_WORKER = 500
_STARTUP_TIMEOUT_S = 120.0  # a worker that has not reported ready by then counts as a start-up failure
_STOP_TIMEOUT_S = 5.0  # wait for a worker to exit before killing it

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_WRAPPER_KEYS = frozenset({"format", "extractor_version", "blob_sha", "syntax"})
_SYNTAX_KEYS = frozenset(f.name for f in dataclasses.fields(FileSyntax))
_FUNCTION_KEYS = frozenset(f.name for f in dataclasses.fields(FunctionInfo))
_COMMENT_KEYS = frozenset(f.name for f in dataclasses.fields(CommentInfo))
_IMPORT_KEYS = frozenset(f.name for f in dataclasses.fields(RawImport))
_STR_ONLY = frozenset({str})
_INT_ONLY = frozenset({int})

# Messages between the parent and a worker.
_READY = "ready"
_DONE = "done"


class StaticCache(Protocol):
    """Storage for parsed files keyed by blob sha (P5.1's ``PostgresStaticCache`` implements it too).

    Implementations embed :data:`~tdp_core.contracts.EXTRACTOR_VERSION` in their key, so a version bump
    never returns stale parses, and should handle their own storage errors (``get`` returns None, ``put``
    returns quietly). :func:`parse_blobs` does not rely on it: the first exception from ``get`` or ``put`` is
    logged as a warning and the cache is not used again during that call.
    """

    def get(self, blob_sha: str) -> FileSyntax | None:
        """The ``FileSyntax`` stored for ``blob_sha`` under the current extractor version, or None.

        Its ``path`` may be the path the blob was first parsed under; callers re-path it.
        """
        ...

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        """Store ``syntax`` as the parse of ``blob_sha`` (best effort)."""
        ...


# ------------------------------------------------------------------------------------------- JSON codec
def syntax_to_json(syntax: FileSyntax) -> dict[str, Any]:
    """``syntax`` as plain JSON values (``tdp_core.serialize.to_jsonable``: dicts, lists, str, int, bool)."""
    data: dict[str, Any] = to_jsonable(syntax)
    return data


def _record(data: object, keys: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(data, dict) or data.keys() != keys:
        raise ValueError(f"{what} must be an object with exactly the keys {sorted(keys)}")
    return data


def _list(data: object, what: str) -> list[Any]:
    if not isinstance(data, list):
        raise ValueError(f"{what} must be a list")
    return data


def _str(data: object, what: str) -> str:
    if type(data) is str:
        return data
    raise ValueError(f"{what} must be a string")


def _int(data: object, what: str) -> int:
    if type(data) is int:  # bool and float are rejected
        return data
    raise ValueError(f"{what} must be an integer")


def _function(data: object) -> FunctionInfo:
    f = _record(data, _FUNCTION_KEYS, "function")
    return FunctionInfo(
        name=_str(f["name"], "function name"),
        start_line=_int(f["start_line"], "start_line"),
        end_line=_int(f["end_line"], "end_line"),
        cyclomatic=_int(f["cyclomatic"], "cyclomatic"),
        cognitive=_int(f["cognitive"], "cognitive"),
        max_nesting=_int(f["max_nesting"], "max_nesting"),
        params=_int(f["params"], "params"),
    )


def _comment(data: object) -> CommentInfo:
    c = _record(data, _COMMENT_KEYS, "comment")
    return CommentInfo(line=_int(c["line"], "comment line"), text=_str(c["text"], "comment text"))


def _import(data: object) -> RawImport:
    i = _record(data, _IMPORT_KEYS, "import")
    names = _list(i["names"], "import names")
    if not set(map(type, names)) <= _STR_ONLY:
        raise ValueError("import names must be strings")
    return RawImport(
        module=_str(i["module"], "import module"),
        names=tuple(names),
        level=_int(i["level"], "import level"),
        line=_int(i["line"], "import line"),
    )


def syntax_from_json(data: object) -> FileSyntax:
    """Rebuild a ``FileSyntax`` from :func:`syntax_to_json` output, validating every value.

    The object must have exactly the ``FileSyntax`` fields, and each nested function, comment and import
    exactly its dataclass fields. Strings must be ``str``, counts and line numbers ``int`` (``bool`` and
    ``float`` are rejected), ``parse_error`` a ``bool``, and ``tokens`` and ``token_lines`` must have the same
    length. For valid input the result equals ``from_jsonable(FileSyntax, data)``; it is built directly
    because the generic decoder is slower than parsing the file again. Raises ``ValueError`` for anything
    else.
    """
    s = _record(data, _SYNTAX_KEYS, "FileSyntax")
    tokens = _list(s["tokens"], "tokens")
    token_lines = _list(s["token_lines"], "token_lines")
    if not set(map(type, tokens)) <= _STR_ONLY:
        raise ValueError("tokens must be strings")
    if not set(map(type, token_lines)) <= _INT_ONLY:
        raise ValueError("token_lines must be integers")
    if len(tokens) != len(token_lines):
        raise ValueError(f"{len(tokens)} tokens but {len(token_lines)} token lines")
    parse_error = s["parse_error"]
    if type(parse_error) is not bool:
        raise ValueError("parse_error must be a boolean")
    return FileSyntax(
        path=_str(s["path"], "path"),
        language=_str(s["language"], "language"),
        sloc=_int(s["sloc"], "sloc"),
        module_cc=_int(s["module_cc"], "module_cc"),
        functions=tuple(_function(f) for f in _list(s["functions"], "functions")),
        classes=_int(s["classes"], "classes"),
        comments=tuple(_comment(c) for c in _list(s["comments"], "comments")),
        imports=tuple(_import(i) for i in _list(s["imports"], "imports")),
        tokens=tuple(tokens),
        token_lines=tuple(token_lines),
        parse_error=parse_error,
    )


# ------------------------------------------------------------------------------------------- disk cache
def _check_sha(blob_sha: str) -> None:
    if not isinstance(blob_sha, str) or _SHA.fullmatch(blob_sha) is None:
        raise ValueError(f"not a full lowercase hex blob sha: {blob_sha!r}")


class DiskStaticCache:
    """A :class:`StaticCache` with one gzip JSON file per blob under ``root``.

    File: ``root/<sha[:2]>/<sha>.<EXTRACTOR_VERSION>.json.gz`` (plan section 3.4; P2.8 passes
    ``data_dir() / "cache" / "static"`` as ``root``). Content: the JSON object ``{"format": CACHE_FORMAT,
    "extractor_version": ..., "blob_sha": ..., "syntax": syntax_to_json(...)}`` compressed with a zero
    timestamp, so the same parse always gives the same bytes. Writes go to a unique temporary file in the
    same folder and are moved into place with ``os.replace``, so readers never see a partial file. The
    cache keeps no state in memory and is safe to share between threads and processes.

    ``blob_sha`` must be a full lowercase hex sha (40 or 64 characters); anything else raises
    ``ValueError`` in every method, so a key can never leave ``root``. Nothing is created until the first
    ``put``.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The cache folder."""
        return self._root

    def path_for(self, blob_sha: str) -> Path:
        """The cache file of ``blob_sha`` (it may not exist)."""
        _check_sha(blob_sha)
        return self._root / blob_sha[:2] / f"{blob_sha}.{EXTRACTOR_VERSION}.json.gz"

    def get(self, blob_sha: str) -> FileSyntax | None:
        """The cached parse of ``blob_sha``, or None when the file is missing, unreadable or invalid.

        A file is valid when it decompresses, holds the wrapper object with exactly its four keys, names
        this format, this extractor version and this sha, and its ``syntax`` passes
        :func:`syntax_from_json`. Invalid files are logged at debug level and treated as misses (the next
        ``put`` overwrites them).
        """
        path = self.path_for(blob_sha)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.debug("static cache: cannot read %s: %s", path, exc)
            return None
        try:
            wrapper = json.loads(gzip.decompress(raw))
            if not isinstance(wrapper, dict) or wrapper.keys() != _WRAPPER_KEYS:
                raise ValueError("not a cache wrapper object")
            fmt = wrapper["format"]
            if type(fmt) is not int or fmt != CACHE_FORMAT:
                raise ValueError(f"format {fmt!r}")
            if wrapper["extractor_version"] != EXTRACTOR_VERSION or wrapper["blob_sha"] != blob_sha:
                raise ValueError("extractor version or blob sha does not match")
            return syntax_from_json(wrapper["syntax"])
        except (OSError, EOFError, zlib.error, ValueError, TypeError, KeyError) as exc:
            logger.debug("static cache: ignoring invalid %s: %s", path, exc)
            return None

    def put(self, blob_sha: str, syntax: FileSyntax) -> None:
        """Write ``syntax`` as the parse of ``blob_sha``; I/O errors are logged (debug), never raised."""
        path = self.path_for(blob_sha)
        wrapper = {
            "format": CACHE_FORMAT,
            "extractor_version": EXTRACTOR_VERSION,
            "blob_sha": blob_sha,
            "syntax": syntax_to_json(syntax),
        }
        payload = gzip.compress(
            json.dumps(wrapper, separators=(",", ":")).encode("utf-8"), compresslevel=6, mtime=0
        )
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(payload)
            os.replace(tmp, path)
        except OSError as exc:
            logger.debug("static cache: cannot write %s: %s", path, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------- parsing
@dataclass(frozen=True, slots=True)
class ParseTask:
    """One blob to parse.

    ``key`` is the blob sha (the cache key); ``path`` is the repo-relative POSIX path given to the language
    plugin (it chooses the plugin and lands in ``FileSyntax.path``); ``source`` is the blob's bytes.
    """

    key: str
    path: str
    source: bytes


def parse_blobs(
    tasks: Sequence[ParseTask],
    cache: StaticCache | None,
    *,
    timeout_s: float,
    jobs: int = 1,
) -> dict[str, FileSyntax | str]:
    """Parse the blobs of ``tasks`` through ``cache`` with a per-file time limit of ``timeout_s`` seconds.

    Tasks are de-duplicated by ``key`` (the first task of a key wins). A cache hit is used when its
    ``language`` is the language of the task's plugin; otherwise the blob counts as a miss. Misses are parsed
    in this process, or by ``min(jobs, misses // 500)`` spawned worker processes when that is at least 2 and
    this process may start children (Celery's prefork workers are daemonic and may not). In-process the
    limit is soft (an overrunning parse finishes and its result is dropped); in the pool it is hard (the
    worker is killed). If the pool cannot start, the remaining files are parsed in this process. New
    successful parses are stored in ``cache``; timeouts and failures are never cached and are logged as
    warnings. A cache whose ``get`` or ``put`` raises is dropped for the rest of the call after one warning;
    the results equal a call without a cache.

    Returns ``{key: result}`` for every distinct key in first-seen order, where ``result`` is a
    ``FileSyntax`` (a hit as stored, so its ``path`` may differ; a new parse with ``task.path``),
    :data:`PARSE_TIMEOUT` or :data:`PARSE_FAILED`. ``timeout_s`` may be ``math.inf`` (no limit). Empty
    ``tasks`` give ``{}`` without touching the cache.

    Raises ``ValueError`` when ``timeout_s`` is not positive (NaN included), ``jobs < 1``, or a task's path
    has no language plugin.
    """
    if not timeout_s > 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    languages: dict[str, str] = {}
    unique: dict[str, ParseTask] = {}
    for task in tasks:
        plugin = get_plugin(task.path)
        if plugin is None:
            raise ValueError(f"no language plugin for {task.path!r}")
        if task.key not in unique:
            unique[task.key] = task
            languages[task.key] = plugin.name
    if not unique:
        return {}

    # A cache only saves time, so one that raises is dropped instead of failing the build.
    store = cache
    results: dict[str, FileSyntax | str] = {}
    misses: list[ParseTask] = []
    for key, task in unique.items():
        hit: FileSyntax | None = None
        if store is not None:
            try:
                hit = store.get(key)
            except Exception as exc:
                _cache_failed("get", exc)
                store = None
        if hit is not None and hit.language == languages[key]:
            results[key] = hit
        else:
            misses.append(task)

    workers = min(jobs, len(misses) // _MIN_TASKS_PER_WORKER)
    if workers >= 2 and _can_spawn():
        logger.debug("parsing %d blobs in %d worker processes", len(misses), workers)
        parsed = _parse_in_pool(misses, workers, timeout_s)
    else:
        logger.debug("parsing %d blobs in this process", len(misses))
        parsed = _parse_in_process(misses, timeout_s)

    for task in misses:
        result = parsed[task.key]
        if isinstance(result, FileSyntax):
            if store is not None:
                try:
                    store.put(task.key, result)
                except Exception as exc:
                    _cache_failed("put", exc)
                    store = None
        elif result == PARSE_TIMEOUT:
            logger.warning("skipping %s (blob %s): parsing took over %g s", task.path, task.key, timeout_s)
        else:
            logger.warning("skipping %s (blob %s): the parser failed", task.path, task.key)
        results[task.key] = result
    return {key: results[key] for key in unique}


def _cache_failed(operation: str, exc: Exception) -> None:
    """Log that the static cache raised in ``operation`` ("get" or "put"); the caller stops using it."""
    logger.warning(
        "static cache %s failed (%s: %s); parsing without the cache for the rest of this batch",
        operation,
        type(exc).__name__,
        exc,
    )


def _parse_one(path: str, source: bytes) -> FileSyntax | str:
    """Parse one file; any exception (recursion and memory errors included) gives :data:`PARSE_FAILED`."""
    plugin = get_plugin(path)
    if plugin is None:
        return PARSE_FAILED
    try:
        return plugin.parse(path, source)
    except Exception:
        logger.debug("parser raised on %s", path, exc_info=True)
        return PARSE_FAILED


def _parse_in_process(tasks: Sequence[ParseTask], timeout_s: float) -> dict[str, FileSyntax | str]:
    """Parse ``tasks`` here; a parse longer than ``timeout_s`` finishes but gives :data:`PARSE_TIMEOUT`."""
    out: dict[str, FileSyntax | str] = {}
    for task in tasks:
        start = time.perf_counter()
        result = _parse_one(task.path, task.source)
        if time.perf_counter() - start > timeout_s:
            result = PARSE_TIMEOUT
        out[task.key] = result
    return out


def _can_spawn() -> bool:
    """False inside a daemonic process (for example a Celery prefork worker), which may not start children."""
    return not multiprocessing.current_process().daemon


def _worker_main(conn: _ConnectionBase[Any, Any]) -> None:
    """Entry point of a spawned parse worker.

    Importing this module in the child loads the parser before the worker reports ready. Then it answers
    each ``(key, path, source)`` request with ``("done", key, result)`` until it receives None or the parent
    goes away.
    """
    try:
        conn.send((_READY, "", None))
        while True:
            request = conn.recv()
            if request is None:
                return
            key, path, source = request
            conn.send((_DONE, key, _parse_one(path, source)))
    except (EOFError, OSError):  # the parent closed its end
        return
    finally:
        conn.close()


class _PoolFailure(Exception):
    """The worker pool cannot be used; tasks without a result are parsed in this process."""


@dataclass(slots=True, eq=False)
class _Worker:
    process: BaseProcess
    conn: _ConnectionBase[Any, Any]
    started: float  # time.perf_counter() at start (high resolution, unlike monotonic() on Windows)
    ready: bool = False
    task: ParseTask | None = None
    deadline: float = math.inf


class _Pool:
    """Spawned parse workers, one task at a time each, killed when a task overruns its deadline."""

    def __init__(self, size: int, timeout_s: float, max_starts: int) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._size = size
        self._timeout_s = timeout_s
        self._max_starts = max_starts
        self._starts = 0
        self.workers: list[_Worker] = []

    def _spawn(self) -> None:
        if self._starts >= self._max_starts:
            raise _PoolFailure(f"{self._starts} worker starts used up")
        self._starts += 1
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(target=_worker_main, args=(child,), daemon=True, name="tdp-parse")
        try:
            process.start()
        except (AssertionError, OSError) as exc:
            parent.close()
            raise _PoolFailure(f"cannot start a parse worker: {exc}") from exc
        finally:
            child.close()
        self.workers.append(_Worker(process, parent, time.perf_counter()))

    def _retire(self, worker: _Worker) -> None:
        self.workers.remove(worker)
        _stop_worker(worker, graceful=False)

    def _receive(self, worker: _Worker, message: Any, results: dict[str, FileSyntax | str]) -> None:
        try:
            kind, key, value = message
        except (TypeError, ValueError):
            raise _PoolFailure(f"malformed message from a parse worker: {message!r}") from None
        if kind == _READY and not worker.ready:
            worker.ready = True
        elif (
            kind == _DONE
            and worker.task is not None
            and key == worker.task.key
            and isinstance(value, (FileSyntax, str))
        ):
            results[key] = value
            worker.task = None
            worker.deadline = math.inf
        else:
            raise _PoolFailure(f"unexpected message from a parse worker: {kind!r}")

    def run(self, tasks: Sequence[ParseTask], results: dict[str, FileSyntax | str]) -> None:
        """Fill ``results`` for every task; raises :class:`_PoolFailure` when the pool cannot go on."""
        pending = deque(tasks)
        for _ in range(self._size):
            self._spawn()
        while pending or any(w.task is not None for w in self.workers):
            if not self.workers:
                raise _PoolFailure("no parse worker is running")
            # Hand a task to every ready, idle worker.
            for worker in list(self.workers):
                if not (pending and worker.ready and worker.task is None):
                    continue
                task = pending.popleft()
                try:
                    worker.conn.send((task.key, task.path, task.source))
                except (OSError, ValueError):
                    pending.appendleft(task)
                    self._retire(worker)
                    self._spawn()
                    continue
                worker.task = task
                worker.deadline = time.perf_counter() + self._timeout_s

            # Wait for a message or the earliest deadline.
            limits = [w.deadline for w in self.workers if w.task is not None]
            limits += [w.started + _STARTUP_TIMEOUT_S for w in self.workers if not w.ready]
            wait_s = max(0.0, min(limits) - time.perf_counter()) if limits else 0.0
            by_conn = {id(w.conn): w for w in self.workers}
            for conn in multiprocessing.connection.wait([w.conn for w in self.workers], timeout=wait_s):
                worker = by_conn[id(conn)]
                try:
                    message = worker.conn.recv()
                except (EOFError, OSError):
                    if not worker.ready:
                        raise _PoolFailure("a parse worker exited during start-up") from None
                    if worker.task is not None:
                        logger.debug("parse worker died on %s", worker.task.path)
                        results[worker.task.key] = PARSE_FAILED
                    self._retire(worker)
                    if pending:
                        self._spawn()
                    continue
                self._receive(worker, message, results)

            # Kill workers past their deadline.
            now = time.perf_counter()
            for worker in list(self.workers):
                if worker.task is not None and now >= worker.deadline:
                    results[worker.task.key] = PARSE_TIMEOUT
                    self._retire(worker)
                    if pending:
                        self._spawn()
                elif not worker.ready and now >= worker.started + _STARTUP_TIMEOUT_S:
                    raise _PoolFailure(f"a parse worker did not start within {_STARTUP_TIMEOUT_S:g} s")

    def close(self) -> None:
        """Stop every worker: idle ones are asked to exit, busy or starting ones are killed."""
        workers, self.workers = self.workers, []
        for worker in workers:
            if worker.ready and worker.task is None:
                with contextlib.suppress(OSError, ValueError):
                    worker.conn.send(None)
        for worker in workers:
            _stop_worker(worker, graceful=worker.ready and worker.task is None)


def _stop_worker(worker: _Worker, *, graceful: bool) -> None:
    """Wait for (``graceful``) or kill the worker's process, then release its pipe and process handle."""
    process = worker.process
    try:
        if graceful:
            process.join(_STOP_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(_STOP_TIMEOUT_S)
    finally:
        worker.conn.close()
        with contextlib.suppress(ValueError):  # still running after the kill: leave the handle to the GC
            process.close()


def _parse_in_pool(tasks: Sequence[ParseTask], workers: int, timeout_s: float) -> dict[str, FileSyntax | str]:
    """Parse ``tasks`` in ``workers`` spawned processes with a hard per-file limit.

    If the pool fails (a worker cannot start, does not report ready within the start-up limit, or sends
    something unexpected), every worker is stopped and the tasks without a result are parsed in this process.
    No worker outlives the call.
    """
    results: dict[str, FileSyntax | str] = {}
    pool = _Pool(workers, timeout_s, max_starts=len(tasks) + workers)
    failure: _PoolFailure | None = None
    try:
        pool.run(tasks, results)
    except _PoolFailure as exc:
        failure = exc
    finally:
        pool.close()
    if failure is not None:
        remaining = [task for task in tasks if task.key not in results]
        logger.warning("parse workers failed (%s); parsing %d files in this process", failure, len(remaining))
        results.update(_parse_in_process(remaining, timeout_s))
    return results
