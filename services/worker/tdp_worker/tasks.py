"""Celery tasks: fetch a repository, run an analysis, mark lost analyses failed and generate LLM advice.

An analysis runs as ``chain(fetch_repo.si(id), run_analysis.si(id))`` with ``mark_failed.si(id)`` as the
error callback of both links (:func:`start_analysis`):

1. :func:`fetch_repo` (queue ``analysis``) marks the analysis running and creates or refreshes the bare
   mirror ``REPO_CACHE_DIR/<host>/<owner>/<name>.git`` under the Redis lock ``lock:repo:<repository_id>``.
   Network failures, DNS failures and a busy lock are retried 3 times (10, 20, 40 s apart). The mirror
   size is checked against ``MAX_REPO_KB`` and the analysed commit must exist in it.
2. :func:`run_analysis` (queue ``analysis``) runs ``tdp_core.analyze.analyze`` on the mirror with the
   analysis's own model version, records every stage in the row and on the ``analysis:<id>`` channel,
   then stores the files, the graph and treemap, the plan for the analysis budget, the summary and the
   stats in one transaction and publishes the terminal ``done`` event.
3. :func:`mark_failed` runs only when a link fails outside its own handlers (hard time limit, lost worker,
   unexpected error) and marks a still active analysis failed.

Failures that users can act on (repository too large or gone, unknown commit, missing model version, git
errors while fetching) are stored as a full sentence, published as the terminal ``failed`` event and
**not** re-raised. Unexpected errors and timeouts are stored with a generic message (never the exception
text, which may hold paths) and re-raised for error tracking and the error callback.

:func:`generate_advice` (queue ``llm``) asks the LLM advisor about one stored file and stores the advice by
cache key; retryable LLM failures are retried 3 times.

``mirror_source`` and ``make_llm_client`` are the seams tests replace (a local repository and a fake LLM).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import redis
from celery import Task, chain
from celery.canvas import Signature
from celery.exceptions import SoftTimeLimitExceeded
from celery.result import AsyncResult

from tdp_core.advisor.advisor import Advisor
from tdp_core.advisor.client import AnthropicClient, LLMClient, LLMError
from tdp_core.advisor.schema import AdviceError
from tdp_core.analyze import analyze
from tdp_core.config import DEFAULT_SETTINGS
from tdp_core.mining.git import GitError, TransientGitError, run_git
from tdp_core.mining.mirror import ensure_mirror
from tdp_core.mining.tree import read_blobs
from tdp_core.parsing import get_plugin
from tdp_core.security.limits import RepoTooLargeError, check_mirror_size
from tdp_core.security.urls import HostLookupError, UnsafeRepoError, validate_repo_url
from tdp_platform.settings import PlatformSettings, get_settings
from tdp_worker import persist, progress
from tdp_worker.app import app
from tdp_worker.bundle_loader import BundleUnavailableError, get_bundle
from tdp_worker.persist import ACTIVE_STATUSES, AnalysisInfo, FileInfo, Store, build_payload
from tdp_worker.progress import (
    DONE_STAGE,
    FAILED_STAGE,
    AnalysisCancelled,
    EventStatus,
    ProgressEvent,
    ProgressReporter,
)

logger = logging.getLogger(__name__)

# ---------- limits (seconds) ----------

ANALYSIS_SOFT_TIME_LIMIT_S: Final = int(DEFAULT_SETTINGS.analysis_timeout_s)  # 900
ANALYSIS_TIME_LIMIT_S: Final = ANALYSIS_SOFT_TIME_LIMIT_S + 60  # hard kill
FETCH_MAX_RETRIES: Final = 3
FETCH_RETRY_BASE_S: Final = 10.0
FETCH_RETRY_MAX_S: Final = 120.0
REPO_LOCK_TIMEOUT_S: Final = 900
REPO_LOCK_WAIT_S: float = 300.0  # not Final: tests shorten it
ADVICE_MAX_RETRIES: Final = 3
ADVICE_RETRY_BASE_S: Final = 30.0
ADVICE_RETRY_MAX_S: Final = 300.0
ADVICE_MODEL: Final = DEFAULT_SETTINGS.advisor_model
ADVICE_CALL_TIMEOUT_S: Final = 180.0
ADVICE_CLIENT_RETRIES: Final = 1  # worst case 2 attempts x 180 s x 2 advisor calls = 720 s < 900 s

# ---------- stages and percentages ----------

FETCH_STAGE: Final = "fetch"
MINE_STAGE: Final = "mine"
SAVE_STAGE: Final = "save"
FETCH_DONE_PCT: Final = 5
SAVE_PCT: Final = 98

# ---------- user-facing messages ----------

GENERIC_FAILURE: Final = "The analysis failed because of an internal error."
LOST_FAILURE: Final = "The analysis stopped unexpectedly; it may have run out of time or memory."
TIMEOUT_FAILURE: Final = (
    f"The analysis took longer than {ANALYSIS_SOFT_TIME_LIMIT_S // 60} minutes and was stopped."
)
GIT_TRANSIENT_FAILURE: Final = "Git could not reach the repository in time; try again later."
GIT_FAILURE: Final = "Git could not read the repository; it may have been deleted, made private or rewritten."
BUSY_FAILURE: Final = "Another analysis is fetching this repository; try again in a few minutes."
ADVICE_GENERIC_FAILURE: Final = "The advice failed because of an internal error."
ADVICE_TIMEOUT_FAILURE: Final = "The advice took too long and was stopped."
ADVICE_DISABLED: Final = "LLM advice is disabled on this server."

_SHA = re.compile(r"^[0-9a-f]{40}$")


# ---------- errors ----------


class UserFacingError(Exception):
    """An expected failure whose ``str()`` is safe to show to users."""


class CommitNotFoundError(UserFacingError):
    """The analysed commit is not in the mirror (for example after a force push)."""


class MirrorMissingError(UserFacingError):
    """The mirror is not on this worker when the analysis starts."""


class RepositoryBusyError(UserFacingError):
    """Another task holds the repository's fetch lock."""


class SourceUnavailableError(UserFacingError):
    """A file's blob can no longer be read from the mirror."""


class AdviceDisabledError(UserFacingError):
    """LLM advice is not configured on this server."""


@dataclass(frozen=True, slots=True)
class MirrorSource:
    """Where a mirror is cloned from. Production always uses an https URL without the file protocol."""

    url: str
    allow_file_protocol: bool = False


# ---------- helpers ----------


def parse_analysis_id(value: uuid.UUID | str) -> uuid.UUID:
    """The analysis id as a UUID; ``ValueError`` for anything that is not one."""
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str):
        raise ValueError(f"analysis id must be a UUID, got {type(value).__name__}")
    return uuid.UUID(value)


def mirror_dir(repo_cache_dir: Path, url: str, host: str) -> Path:
    """The absolute mirror folder ``<repo_cache_dir>/<host>/<owner>/<name>.git`` of a stored repository URL.

    The URL is validated again without DNS (owner and name are lower-cased on github.com), so a stored
    value can never point outside the cache; an invalid one raises ``UnsafeRepoError``. The folder is
    absolute because ``ensure_mirror`` needs absolute paths.
    """
    ref = validate_repo_url(url, [host], resolve_dns=False)
    return Path(repo_cache_dir).resolve() / ref.host / ref.owner / f"{ref.name}.git"


def szz_cache_dir(mirror: Path) -> Path:
    """The SZZ cache folder beside a mirror: ``<name>.szz`` next to ``<name>.git``."""
    return mirror.with_name(mirror.name.removesuffix(".git") + ".szz")


def mirror_source(info: AnalysisInfo, settings: PlatformSettings) -> MirrorSource:
    """The clone source of an analysis: its https clone URL, validated again with a DNS check.

    Raises ``UnsafeRepoError`` (``HostLookupError`` when DNS fails) if the host is no longer allowed or
    resolves to a non-public address. Tests replace this function to clone a local repository.
    """
    ref = validate_repo_url(info.url, settings.allowed_git_hosts)
    return MirrorSource(ref.clone_url)


def make_llm_client(settings: PlatformSettings) -> LLMClient:
    """The Anthropic client for advice (180 s per call, one SDK retry). Tests replace this function.

    Raises :class:`AdviceDisabledError` without ``ANTHROPIC_API_KEY``.
    """
    key = settings.anthropic_api_key
    if key is None or not key.get_secret_value():
        raise AdviceDisabledError(ADVICE_DISABLED)
    return AnthropicClient(
        ADVICE_MODEL,
        api_key=key.get_secret_value(),
        timeout_s=ADVICE_CALL_TIMEOUT_S,
        max_retries=ADVICE_CLIENT_RETRIES,
    )


def repo_lock_name(repository_id: uuid.UUID | str) -> str:
    """The Redis key of a repository's fetch lock."""
    return f"lock:repo:{repository_id}"


class RepoLock:
    """A Redis lock owned by a random token: ``SET NX PX`` to take it, a checked delete to release it.

    ``timeout_s`` is how long the lock lives if never released (seconds), ``wait_s`` how long
    :meth:`acquire` keeps trying (0 means a single try) and ``poll_s`` the pause between tries. Release uses
    ``WATCH``/``MULTI`` instead of a Lua script, so it also works with fakeredis.
    """

    def __init__(
        self,
        client: redis.Redis,
        name: str,
        *,
        timeout_s: float = REPO_LOCK_TIMEOUT_S,
        wait_s: float = REPO_LOCK_WAIT_S,
        poll_s: float = 0.25,
    ) -> None:
        self.name = name
        self._client = client
        self._timeout_ms = max(1, int(timeout_s * 1000))
        self._wait_s = max(0.0, float(wait_s))
        self._poll_s = max(0.0, float(poll_s))
        self._token = uuid.uuid4().hex.encode("ascii")

    def acquire(self) -> bool:
        """Take the lock, retrying until ``wait_s`` has passed; True when this lock now holds it."""
        deadline = time.monotonic() + self._wait_s
        while True:
            if self._client.set(self.name, self._token, nx=True, px=self._timeout_ms):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self._poll_s, remaining))

    def release(self) -> bool:
        """Delete the key if this lock still holds it; False when it expired or another owner has it."""
        try:
            with self._client.pipeline() as pipe:
                pipe.watch(self.name)
                value = pipe.get(self.name)
                if value not in (self._token, self._token.decode("ascii")):
                    pipe.unwatch()
                    return False
                pipe.multi()
                pipe.delete(self.name)
                pipe.execute()
                return True
        except redis.WatchError:
            return False

    def __enter__(self) -> RepoLock:
        if not self.acquire():
            raise RepositoryBusyError(BUSY_FAILURE)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            released = self.release()
        except (redis.RedisError, OSError) as error:
            logger.warning("could not release lock %s: %s", self.name, error)
            return
        if not released:
            logger.warning("lock %s expired before it was released", self.name)


def _sentence(text: str, fallback: str) -> str:
    text = " ".join(text.split())
    if not text:
        return fallback
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def user_message(
    exc: BaseException, *, generic: str = GENERIC_FAILURE, timeout: str = TIMEOUT_FAILURE
) -> str:
    """The message stored and shown for ``exc``: always a full sentence that is safe for users.

    Soft time limits give ``timeout``; errors that are written for users (``UserFacingError``,
    ``BundleUnavailableError``, ``UnsafeRepoError``) give their own text as a sentence; transient git
    failures and other git failures give fixed messages; anything else gives ``generic`` (exception texts
    of unexpected errors can hold paths or internals and are never shown).
    """
    if isinstance(exc, SoftTimeLimitExceeded):
        return timeout
    if isinstance(exc, (UserFacingError, BundleUnavailableError, UnsafeRepoError)):
        return _sentence(str(exc), generic)
    if isinstance(exc, TransientGitError):
        return GIT_TRANSIENT_FAILURE
    if isinstance(exc, GitError):
        return GIT_FAILURE
    return generic


def _remove_tree(path: Path) -> None:
    """Delete a folder tree (git pack files are read-only on Windows); a missing path is fine."""

    def retry_writable(func: Callable[..., Any], target: str, error: BaseException) -> None:
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("could not delete %s", target, exc_info=True)

    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path, onexc=retry_writable)


def _commit_exists(mirror: Path, sha: str) -> bool:
    if not _SHA.fullmatch(sha):
        return False
    try:
        run_git([f"--git-dir={mirror}", "cat-file", "-e", f"{sha}^{{commit}}"], mirror)
    except TransientGitError:
        raise
    except GitError:
        return False
    return True


def ensure_repository_mirror(info: AnalysisInfo, settings: PlatformSettings) -> Path:
    """Create or refresh the mirror of an analysis's repository and return its absolute folder.

    Holds the repository's Redis lock (waiting up to ``REPO_LOCK_WAIT_S`` seconds, else
    :class:`RepositoryBusyError`). A mirror that git cannot update is deleted and cloned again once. A
    mirror over ``MAX_REPO_KB`` is deleted with its SZZ cache (``RepoTooLargeError``). The analysed commit
    must exist in the mirror (:class:`CommitNotFoundError`). Git failures propagate as ``GitError`` or
    ``TransientGitError``.
    """
    dest = mirror_dir(settings.repo_cache_dir, info.url, info.host)
    source = mirror_source(info, settings)
    lock = RepoLock(progress.get_redis(), repo_lock_name(info.repository_id), wait_s=REPO_LOCK_WAIT_S)
    with lock:
        try:
            ensure_mirror(source.url, dest, allow_file_protocol=source.allow_file_protocol)
        except TransientGitError:
            raise
        except GitError:
            if not dest.exists():
                raise
            logger.warning("mirror %s could not be updated; cloning it again", dest, exc_info=True)
            _remove_tree(dest)
            ensure_mirror(source.url, dest, allow_file_protocol=source.allow_file_protocol)
        try:
            check_mirror_size(dest, settings.max_repo_kb)
        except RepoTooLargeError:
            _remove_tree(dest)
            _remove_tree(szz_cache_dir(dest))
            raise
        if not _commit_exists(dest, info.commit_sha):
            raise CommitNotFoundError(
                f"Commit {info.commit_sha[:12]} is not in the repository; "
                "it may have been removed by a force push."
            )
    return dest


def _fail(store: Store, analysis_id: uuid.UUID, message: str, pct: int) -> bool:
    """Mark the analysis failed and publish the failure event; False when it was no longer active."""
    try:
        changed = store.set_failed(analysis_id, message)
    except Exception:
        logger.exception("could not mark analysis %s as failed", analysis_id)
        return False
    if changed:
        status: EventStatus = "failed"
        event = ProgressEvent(FAILED_STAGE, max(0, min(100, int(pct))), message, status)
        progress.publish(analysis_id, event)
    return changed


def _outcome(analysis_id: uuid.UUID | str, status: str, **extra: Any) -> dict[str, Any]:
    return {"analysis_id": str(analysis_id), "status": status, **extra}


def _start(analysis_id: str, task: str) -> tuple[uuid.UUID, Store, AnalysisInfo] | str:
    """Load an active analysis and mark it running, or return why the task has nothing to do."""
    try:
        aid = parse_analysis_id(analysis_id)
    except ValueError:
        logger.error("%s: %r is not an analysis id", task, analysis_id)
        return "missing"
    store = persist.get_store()
    info = store.load_analysis(aid)
    if info is None:
        logger.warning("%s: analysis %s does not exist", task, aid)
        return "missing"
    if info.status not in ACTIVE_STATUSES or not store.set_running(aid):
        logger.info("%s: analysis %s is %s; nothing to do", task, aid, info.status)
        return "skipped"
    return aid, store, info


# ---------- tasks ----------


@app.task(
    bind=True,
    name="tdp_worker.fetch_repo",
    acks_late=True,
    max_retries=FETCH_MAX_RETRIES,
    soft_time_limit=ANALYSIS_SOFT_TIME_LIMIT_S,
    time_limit=ANALYSIS_TIME_LIMIT_S,
)
def fetch_repo(self: Task, analysis_id: str) -> str | None:
    """Mark the analysis running and create or refresh its mirror; return the mirror folder.

    Returns None when there is nothing to do (unknown or finished analysis) or when the fetch failed in
    an expected way (the analysis is then failed with a user message). Network, DNS and lock failures are
    retried; unexpected errors fail the analysis and are re-raised.
    """
    started = _start(analysis_id, "fetch_repo")
    if isinstance(started, str):
        return None
    aid, store, info = started
    reporter = ProgressReporter(aid, store.set_progress)
    try:
        reporter(FETCH_STAGE, 0, "Fetching the repository")
        mirror = ensure_repository_mirror(info, get_settings())
        reporter(FETCH_STAGE, FETCH_DONE_PCT, "Fetched the repository")
        return str(mirror)
    except AnalysisCancelled:
        logger.info("fetch_repo: analysis %s was cancelled", aid)
        return None
    except (TransientGitError, HostLookupError, RepositoryBusyError) as exc:
        retries = int(self.request.retries or 0)
        if retries < self.max_retries:
            countdown = min(FETCH_RETRY_MAX_S, FETCH_RETRY_BASE_S * 2**retries)
            logger.warning("fetch_repo: analysis %s retry %d in %.0f s: %s", aid, retries + 1, countdown, exc)
            raise self.retry(exc=exc, countdown=countdown) from exc
        logger.warning("fetch_repo: analysis %s gave up after %d retries: %s", aid, retries, exc)
        _fail(store, aid, user_message(exc), reporter.last_pct)
        return None
    except (UserFacingError, UnsafeRepoError, GitError) as exc:
        logger.warning("fetch_repo: analysis %s failed: %s", aid, exc)
        _fail(store, aid, user_message(exc), reporter.last_pct)
        return None
    except Exception as exc:
        if isinstance(exc, SoftTimeLimitExceeded):
            logger.warning("fetch_repo: analysis %s hit the time limit", aid)
        else:
            logger.exception("fetch_repo: analysis %s failed unexpectedly", aid)
        _fail(store, aid, user_message(exc), reporter.last_pct)
        raise


@app.task(
    name="tdp_worker.run_analysis",
    acks_late=True,
    soft_time_limit=ANALYSIS_SOFT_TIME_LIMIT_S,
    time_limit=ANALYSIS_TIME_LIMIT_S,
)
def run_analysis(analysis_id: str) -> dict[str, Any]:
    """Analyse the fetched mirror at the analysis's commit and store the results.

    Returns ``{"analysis_id", "status", ...}`` with status ``succeeded`` (plus ``files`` and ``seconds``),
    ``failed`` (an expected failure, stored with its message), ``cancelled`` (the row left the running
    state during the analysis), ``skipped`` (nothing to do, or the results lost a race) or ``missing``.
    Unexpected errors and timeouts fail the analysis with a generic message and are re-raised.
    """
    t0 = time.monotonic()
    started = _start(analysis_id, "run_analysis")
    if isinstance(started, str):
        return _outcome(analysis_id, started)
    aid, store, info = started
    reporter = ProgressReporter(aid, store.set_progress)
    try:
        reporter(MINE_STAGE, FETCH_DONE_PCT, "Mining the commit history")
        settings = get_settings()
        bundle = get_bundle(info.model_version, model_dir=settings.model_dir)
        mirror = mirror_dir(settings.repo_cache_dir, info.url, info.host)
        if not mirror.is_dir():
            raise MirrorMissingError("The repository has not been fetched; start the analysis again.")
        result = analyze(
            mirror,
            info.commit_sha,
            bundle,
            cache=store.static_cache(),
            progress=reporter,
            budget_hours=info.budget_hours,
            szz_cache_dir=szz_cache_dir(mirror),
        )
        reporter(SAVE_STAGE, SAVE_PCT, "Saving the results")
        payload = build_payload(result, bundle, budget_hours=info.budget_hours, root_name=info.name)
        if not store.save_results(aid, payload):
            logger.info("run_analysis: analysis %s changed state; results discarded", aid)
            return _outcome(aid, "skipped")
    except AnalysisCancelled:
        logger.info("run_analysis: analysis %s was cancelled", aid)
        return _outcome(aid, "cancelled")
    except (UserFacingError, BundleUnavailableError, UnsafeRepoError) as exc:
        logger.warning("run_analysis: analysis %s failed: %s", aid, exc)
        _fail(store, aid, user_message(exc), reporter.last_pct)
        return _outcome(aid, "failed")
    except Exception as exc:
        if isinstance(exc, SoftTimeLimitExceeded):
            logger.warning("run_analysis: analysis %s hit the time limit", aid)
        else:
            logger.exception("run_analysis: analysis %s failed unexpectedly", aid)
        _fail(store, aid, user_message(exc), reporter.last_pct)
        raise
    status: EventStatus = "succeeded"
    progress.publish(aid, ProgressEvent(DONE_STAGE, 100, "Analysis complete", status))
    seconds = round(time.monotonic() - t0, 3)
    logger.info(
        "run_analysis: analysis %s succeeded with %d files in %.1f s", aid, len(result.files), seconds
    )
    return _outcome(aid, "succeeded", files=len(result.files), seconds=seconds)


@app.task(name="tdp_worker.mark_failed", acks_late=True)
def mark_failed(analysis_id: str) -> bool:
    """Error callback of the analysis chain: fail a still active analysis with ``LOST_FAILURE``.

    Runs as its own task (an error callback with one argument is queued, not run inside the failed
    worker). Returns True when it changed the row; False for a junk id, an unknown analysis or one that
    is already finished (its task stored its own failure).
    """
    try:
        aid = parse_analysis_id(analysis_id)
    except ValueError:
        logger.error("mark_failed: %r is not an analysis id", analysis_id)
        return False
    store = persist.get_store()
    info = store.load_analysis(aid)
    if info is None or info.status not in ACTIVE_STATUSES:
        return False
    logger.error("analysis %s stopped without storing a result; marking it failed", aid)
    return _fail(store, aid, LOST_FAILURE, info.progress)


def _read_source(info: FileInfo, settings: PlatformSettings) -> bytes:
    mirror = mirror_dir(settings.repo_cache_dir, info.analysis.url, info.analysis.host)
    if not mirror.is_dir():
        mirror = ensure_repository_mirror(info.analysis, settings)
    try:
        return read_blobs(mirror, [info.blob_sha])[info.blob_sha]
    except TransientGitError:
        raise
    except GitError as exc:
        raise SourceUnavailableError(
            "The file's source is no longer available in the repository mirror."
        ) from exc


def _advice_failed(store: Store, cache_key: str, message: str) -> None:
    try:
        store.advice_failed(cache_key, message)
    except Exception:
        logger.exception("could not mark advice %s as failed", cache_key)


@app.task(
    bind=True,
    name="tdp_worker.generate_advice",
    acks_late=True,
    max_retries=ADVICE_MAX_RETRIES,
    soft_time_limit=ANALYSIS_SOFT_TIME_LIMIT_S,
    time_limit=ANALYSIS_TIME_LIMIT_S,
)
def generate_advice(self: Task, file_result_id: int) -> str:
    """Generate LLM advice for a stored file and store it by cache key; return the advice status.

    Returns ``ready`` (stored now or earlier), ``failed`` (stored with a user message, now or earlier) or
    ``missing`` for an unknown file. Advice that already exists is not generated again; failed advice must
    be requeued first. Retryable LLM errors are retried with a 30, 60, 120 s backoff.
    """
    if isinstance(file_result_id, bool) or not isinstance(file_result_id, int):
        logger.error("generate_advice: %r is not a file result id", file_result_id)
        return "missing"
    store = persist.get_store()
    info = store.load_file(file_result_id)
    if info is None:
        logger.warning("generate_advice: file result %s does not exist", file_result_id)
        return "missing"
    key = Advisor.cache_key(info.blob_sha, ADVICE_MODEL)
    status = store.begin_advice(key, file_result_id=info.id, model=ADVICE_MODEL)
    if status != "queued":
        return status
    settings = get_settings()
    if not settings.advisor_enabled or settings.anthropic_api_key is None:
        _advice_failed(store, key, ADVICE_DISABLED)
        return "failed"
    client: LLMClient | None = None
    try:
        blob = _read_source(info, settings)
        plugin = get_plugin(info.path)
        if plugin is None:
            raise UserFacingError("Advice is only available for Python files.")
        syntax = plugin.parse(info.path, blob)
        client = make_llm_client(settings)
        advice = Advisor(client, DEFAULT_SETTINGS).advise(
            path=info.path,
            source=blob.decode("utf-8", errors="replace"),
            syntax=syntax,
            score=info.score,
            extra=info.extra,
        )
        store.advice_ready(key, advice.model_dump(mode="json"))
        return "ready"
    except AdviceError as exc:
        retries = int(self.request.retries or 0)
        if isinstance(exc, LLMError) and exc.retryable and retries < self.max_retries:
            countdown = min(ADVICE_RETRY_MAX_S, ADVICE_RETRY_BASE_S * 2**retries)
            logger.warning(
                "generate_advice: file %s retry %d in %.0f s: %s", info.id, retries + 1, countdown, exc
            )
            raise self.retry(exc=exc, countdown=countdown) from exc
        logger.warning("generate_advice: file %s failed: %s", info.id, exc)
        _advice_failed(store, key, _sentence(str(exc), ADVICE_GENERIC_FAILURE))
        return "failed"
    except (UserFacingError, UnsafeRepoError, GitError) as exc:
        logger.warning("generate_advice: file %s failed: %s", info.id, exc)
        _advice_failed(store, key, user_message(exc, generic=ADVICE_GENERIC_FAILURE))
        return "failed"
    except Exception as exc:
        timed_out = isinstance(exc, SoftTimeLimitExceeded)
        if timed_out:
            logger.warning("generate_advice: file %s hit the time limit", info.id)
        else:
            logger.exception("generate_advice: file %s failed unexpectedly", info.id)
        _advice_failed(store, key, ADVICE_TIMEOUT_FAILURE if timed_out else ADVICE_GENERIC_FAILURE)
        raise
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


# ---------- workflow ----------


def analysis_workflow(analysis_id: uuid.UUID | str) -> Signature:
    """The analysis chain ``fetch_repo -> run_analysis`` with immutable signatures (no error callback)."""
    aid = str(parse_analysis_id(analysis_id))
    return chain(fetch_repo.si(aid), run_analysis.si(aid))


def start_analysis(analysis_id: uuid.UUID | str) -> AsyncResult:
    """Queue the analysis chain with ``mark_failed`` as the error callback of every link.

    Commit the analysis row before calling this. In eager mode (``task_always_eager``) the whole analysis
    runs inside the call; expected failures return normally and only unexpected fetch errors raise.
    """
    aid = str(parse_analysis_id(analysis_id))
    return analysis_workflow(aid).apply_async(link_error=mark_failed.si(aid))
