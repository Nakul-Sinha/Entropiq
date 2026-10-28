"""The analysis tasks in Celery eager mode with an in-memory store, fakeredis and a local repository."""

from __future__ import annotations

import socket
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from celery.exceptions import SoftTimeLimitExceeded

from tdp_core.mining.git import GitError, TransientGitError
from tdp_core.security.limits import RepoTooLargeError
from tdp_core.security.urls import HostLookupError, UnsafeRepoError
from tdp_platform.settings import get_settings
from tdp_worker import tasks
from tdp_worker.bundle_loader import BundleUnavailableError
from tdp_worker.persist import AnalysisInfo
from tdp_worker.tasks import (
    GENERIC_FAILURE,
    GIT_FAILURE,
    GIT_TRANSIENT_FAILURE,
    LOST_FAILURE,
    TIMEOUT_FAILURE,
    MirrorSource,
    RepoLock,
    UserFacingError,
    analysis_workflow,
    fetch_repo,
    mark_failed,
    mirror_dir,
    mirror_source,
    parse_analysis_id,
    repo_lock_name,
    run_analysis,
    start_analysis,
    szz_cache_dir,
    user_message,
)
from tdp_worker.testing import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_REPOSITORY_ID,
    EventLog,
    MemoryStore,
    RecordingCache,
    SourceRepo,
)

EventsFactory = Callable[[uuid.UUID], EventLog]
REPOSITORY_ID = DEFAULT_REPOSITORY_ID

pytestmark = pytest.mark.usefixtures("eager", "fake_redis", "local_source")

EXPECTED_STAGES = [
    ("fetch", 0),
    ("fetch", 5),
    ("mine", 5),
    ("mine", 25),
    ("parse", 50),
    ("graphs", 60),
    ("features", 70),
    ("score", 80),
    ("explain", 90),
    ("plan", 95),
    ("save", 98),
    ("done", 100),
]


def _mirror(worker_env: dict[str, Path]) -> Path:
    return worker_env["mirrors"] / "github.com" / "acme" / "demo.git"


def _is_bare(path: Path) -> bool:
    out = subprocess.run(
        ["git", f"--git-dir={path}", "rev-parse", "--is-bare-repository"],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode().strip() == "true"


# ---------- the whole chain ----------


def test_start_analysis_end_to_end(
    store: MemoryStore,
    source_repo: SourceRepo,
    events: EventsFactory,
    worker_env: dict[str, Path],
) -> None:
    store.cache = cache = RecordingCache()
    aid = store.add_analysis(commit_sha=source_repo.head)
    log = events(aid)

    result = start_analysis(aid)

    assert result.successful()
    row = store.row(aid)
    assert (row["status"], row["stage"], row["progress"], row["error"]) == ("succeeded", "done", 100, None)
    seen = log.drain()
    assert [(e.stage, e.pct) for e in seen] == EXPECTED_STAGES
    assert [e.status for e in seen] == [None] * 11 + ["succeeded"]
    assert seen[-1].message == "Analysis complete"
    assert all(e.message and not e.message.endswith(".") for e in seen)
    assert _is_bare(_mirror(worker_env))

    paths = [r.path for r in store.files[aid]]
    assert {"pkg/core.py", "pkg/util.py", "pkg/sub/deep.py"} <= set(paths)
    assert not [p for p in paths if p.startswith(("tests/", "assets/"))]
    assert all(len(r.shap["by_feature"]) == 46 for r in store.files[aid])

    graph = store.artifacts[(aid, "graph")]
    assert {n["id"] for n in graph["nodes"]} == set(paths)
    treemap = store.artifacts[(aid, "treemap")]
    assert treemap["name"] == "demo" and treemap["path"] == ""
    plan = store.plans[(aid, 40.0)]
    assert plan["budget_hours"] == 40.0
    assert row["summary"]["files"] == len(paths)
    assert "timings_s" in row["stats"]
    assert cache.puts > 0


def test_two_commits_share_one_mirror(
    store: MemoryStore, source_repo: SourceRepo, worker_env: dict[str, Path]
) -> None:
    first = store.add_analysis(commit_sha=source_repo.first)
    head = store.add_analysis(commit_sha=source_repo.head)

    start_analysis(first)
    start_analysis(head)

    assert store.row(first)["status"] == store.row(head)["status"] == "succeeded"
    owner_dir = worker_env["mirrors"] / "github.com" / "acme"
    assert sorted(p.name for p in owner_dir.iterdir()) == ["demo.git", "demo.szz"]
    assert "pkg/sub/deep.py" not in {r.path for r in store.files[first]}
    assert "pkg/sub/deep.py" in {r.path for r in store.files[head]}


# ---------- expected failures ----------


def test_unknown_commit_fails_with_a_message(store: MemoryStore, events: EventsFactory) -> None:
    aid = store.add_analysis(commit_sha="0" * 40)
    log = events(aid)

    start_analysis(aid)

    row = store.row(aid)
    assert row["status"] == "failed"
    assert "is not in the repository" in row["error"]
    assert row["error"].startswith("Commit 000000000000 ")
    failures = [e for e in log.drain() if e.terminal]
    assert len(failures) == 1
    assert (failures[0].stage, failures[0].status, failures[0].message) == ("failed", "failed", row["error"])
    assert aid not in store.files


def test_repository_over_the_size_limit_is_removed(
    store: MemoryStore,
    source_repo: SourceRepo,
    worker_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_REPO_KB", "16")
    get_settings.cache_clear()
    aid = store.add_analysis(commit_sha=source_repo.head)

    start_analysis(aid)

    row = store.row(aid)
    assert row["status"] == "failed"
    assert "over the 16 KB limit" in row["error"]
    assert row["error"][0].isupper() and row["error"].endswith(".")
    assert not _mirror(worker_env).exists()


def test_transient_git_errors_are_retried(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tasks.ensure_mirror
    calls: list[Path] = []

    def flaky(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        calls.append(dest)
        if len(calls) < 3:
            raise TransientGitError("timed out")
        return real(url, dest, allow_file_protocol=allow_file_protocol)

    monkeypatch.setattr(tasks, "ensure_mirror", flaky)
    aid = store.add_analysis(commit_sha=source_repo.head)

    result = fetch_repo.apply(args=(str(aid),))

    assert result.successful()
    assert result.get() == str(calls[-1])
    assert len(calls) == 3
    assert store.row(aid)["status"] == "running"
    assert store.row(aid)["progress"] == 5


def test_transient_git_errors_give_up_after_three_retries(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def down(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        calls.append(dest)
        raise TransientGitError("timed out")

    monkeypatch.setattr(tasks, "ensure_mirror", down)
    aid = store.add_analysis(commit_sha=source_repo.head)

    result = start_analysis(aid)

    assert len(calls) == 4
    assert result.state == "SUCCESS"
    assert store.row(aid)["status"] == "failed"
    assert store.row(aid)["error"] == GIT_TRANSIENT_FAILURE


def test_busy_repository_lock(
    store: MemoryStore,
    source_repo: SourceRepo,
    fake_redis: fakeredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "REPO_LOCK_WAIT_S", 0.0)
    fake_redis.set(repo_lock_name(REPOSITORY_ID), b"someone-else")
    aid = store.add_analysis(commit_sha=source_repo.head)

    start_analysis(aid)

    assert store.row(aid)["status"] == "failed"
    assert store.row(aid)["error"] == tasks.BUSY_FAILURE
    assert fake_redis.get(repo_lock_name(REPOSITORY_ID)) == b"someone-else"


def test_model_version_not_installed(store: MemoryStore, source_repo: SourceRepo) -> None:
    aid = store.add_analysis(commit_sha=source_repo.head, model_version="dev-missing")

    result = start_analysis(aid)

    assert result.successful()
    assert result.get()["status"] == "failed"
    assert store.row(aid)["status"] == "failed"
    assert store.row(aid)["error"] == "Model version dev-missing is not installed on this worker."


def test_missing_mirror_fails_run_analysis(store: MemoryStore, source_repo: SourceRepo) -> None:
    aid = store.add_analysis(commit_sha=source_repo.head)

    outcome = run_analysis.apply(args=(str(aid),)).get()

    assert outcome == {"analysis_id": str(aid), "status": "failed"}
    assert store.row(aid)["error"] == "The repository has not been fetched; start the analysis again."


# ---------- unexpected failures ----------


def test_unexpected_error_is_stored_generically_and_reraised(
    store: MemoryStore,
    source_repo: SourceRepo,
    events: EventsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom C:/secret")

    monkeypatch.setattr(tasks, "analyze", boom)
    aid = store.add_analysis(commit_sha=source_repo.head)
    log = events(aid)

    result = start_analysis(aid)

    assert result.state == "FAILURE"
    row = store.row(aid)
    assert row["status"] == "failed"
    assert row["error"] == GENERIC_FAILURE
    terminal = [e for e in log.drain() if e.terminal]
    assert len(terminal) == 1
    assert terminal[0].message == GENERIC_FAILURE and terminal[0].pct == 5


def test_soft_time_limit_gives_the_timeout_message(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(*args: Any, **kwargs: Any) -> Any:
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(tasks, "analyze", slow)
    aid = store.add_analysis(commit_sha=source_repo.head)

    result = start_analysis(aid)

    assert result.state == "FAILURE"
    assert store.row(aid)["error"] == TIMEOUT_FAILURE
    assert TIMEOUT_FAILURE == "The analysis took longer than 15 minutes and was stopped."


def test_unexpected_fetch_error_fails_and_raises(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        raise OSError("disk full at C:/secret")

    monkeypatch.setattr(tasks, "ensure_mirror", broken)
    aid = store.add_analysis(commit_sha=source_repo.head)

    with pytest.raises(OSError):
        start_analysis(aid)

    assert store.row(aid)["status"] == "failed"
    assert store.row(aid)["error"] == GENERIC_FAILURE


def test_git_error_while_fetching_is_an_expected_failure(
    store: MemoryStore, source_repo: SourceRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(url: str, dest: Path, *, allow_file_protocol: bool = False) -> Path:
        raise GitError("repository not found")

    monkeypatch.setattr(tasks, "ensure_mirror", gone)
    aid = store.add_analysis(commit_sha=source_repo.head)

    result = start_analysis(aid)

    assert result.successful()
    assert store.row(aid)["error"] == GIT_FAILURE


# ---------- nothing to do ----------


def test_finished_analysis_is_skipped(
    store: MemoryStore, source_repo: SourceRepo, events: EventsFactory
) -> None:
    aid = store.add_analysis(commit_sha=source_repo.head, status="succeeded")
    log = events(aid)

    assert fetch_repo.apply(args=(str(aid),)).get() is None
    assert run_analysis.apply(args=(str(aid),)).get() == {"analysis_id": str(aid), "status": "skipped"}
    assert log.drain() == []
    assert store.row(aid)["status"] == "succeeded"
    assert store.progress_log == []


def test_unknown_analysis_is_ignored(store: MemoryStore) -> None:
    aid = str(uuid.uuid4())

    assert fetch_repo.apply(args=(aid,)).get() is None
    assert run_analysis.apply(args=(aid,)).get() == {"analysis_id": aid, "status": "missing"}
    assert run_analysis.apply(args=("junk",)).get() == {"analysis_id": "junk", "status": "missing"}
    assert mark_failed.apply(args=(aid,)).get() is False
    assert mark_failed.apply(args=("junk",)).get() is False


def test_mark_failed_marks_a_lost_analysis_once(store: MemoryStore, events: EventsFactory) -> None:
    aid = store.add_analysis(commit_sha="a" * 40, status="running")
    store.row(aid)["progress"] = 60
    log = events(aid)

    assert mark_failed.apply(args=(str(aid),)).get() is True
    assert mark_failed.apply(args=(str(aid),)).get() is False

    assert store.row(aid)["status"] == "failed"
    assert store.row(aid)["error"] == LOST_FAILURE
    seen = log.drain()
    assert [(e.stage, e.pct, e.message, e.status) for e in seen] == [("failed", 60, LOST_FAILURE, "failed")]


class CancellingStore(MemoryStore):
    """Fails the analysis from outside when the parse stage is recorded."""

    def set_progress(self, analysis_id: uuid.UUID, stage: str, pct: int) -> bool:
        if stage == "parse":
            self.set_failed(analysis_id, "Cancelled elsewhere.")
            return False
        return super().set_progress(analysis_id, stage, pct)


def test_analysis_cancelled_from_outside(
    worker_env: dict[str, Path], source_repo: SourceRepo, events: EventsFactory
) -> None:
    from tdp_worker import persist

    cancelling = CancellingStore()
    persist.set_store(cancelling)
    aid = cancelling.add_analysis(commit_sha=source_repo.head)
    log = events(aid)

    result = start_analysis(aid)

    assert result.get() == {"analysis_id": str(aid), "status": "cancelled"}
    assert cancelling.row(aid)["status"] == "failed"
    assert cancelling.row(aid)["error"] == "Cancelled elsewhere."
    assert aid not in cancelling.files
    seen = log.drain()
    assert not [e for e in seen if e.terminal]
    assert seen[-1].stage == "mine"


# ---------- workflow wiring ----------


def test_start_analysis_links_mark_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    aid = uuid.uuid4()
    captured: dict[str, Any] = {}

    class FakeWorkflow:
        def apply_async(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "queued"

    monkeypatch.setattr(tasks, "analysis_workflow", lambda analysis_id: FakeWorkflow())

    assert start_analysis(aid) == "queued"  # type: ignore[comparison-overlap]
    errback = captured["link_error"]
    assert errback.task == "tdp_worker.mark_failed"
    assert errback.args == (str(aid),)
    assert errback.immutable


def test_analysis_workflow_holds_immutable_signatures() -> None:
    aid = uuid.uuid4()
    workflow = analysis_workflow(str(aid).upper())
    links = workflow.tasks  # type: ignore[attr-defined]
    assert [s.task for s in links] == ["tdp_worker.fetch_repo", "tdp_worker.run_analysis"]
    assert all(s.args == (str(aid),) and s.immutable for s in links)
    with pytest.raises(ValueError):
        analysis_workflow("not-a-uuid")


def test_parse_analysis_id() -> None:
    aid = uuid.uuid4()
    assert parse_analysis_id(aid) is aid
    assert parse_analysis_id(str(aid)) == aid
    with pytest.raises(ValueError):
        parse_analysis_id("junk")
    with pytest.raises(ValueError):
        parse_analysis_id(42)  # type: ignore[arg-type]


# ---------- helpers ----------


def _info(**changes: Any) -> AnalysisInfo:
    values: dict[str, Any] = {
        "id": uuid.uuid4(),
        "repository_id": REPOSITORY_ID,
        "url": "https://github.com/acme/demo",
        "host": "github.com",
        "owner": "acme",
        "name": "demo",
        "commit_sha": "a" * 40,
        "model_version": DEFAULT_MODEL_VERSION,
        "budget_hours": 40.0,
        "status": "queued",
        "stage": None,
        "progress": 0,
    }
    values.update(changes)
    return AnalysisInfo(**values)


@pytest.mark.parametrize(
    ("addresses", "hosts", "error"),
    [
        (["140.82.112.3"], "github.com", None),
        (["10.0.0.7"], "github.com", UnsafeRepoError),
        (["140.82.112.3"], "gitlab.com", UnsafeRepoError),
    ],
)
def test_mirror_source_validates_again(
    worker_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    addresses: list[str],
    hosts: str,
    error: type[Exception] | None,
) -> None:
    import tdp_core.security.urls as urls

    def fake_getaddrinfo(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    monkeypatch.setattr(urls.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setenv("ALLOWED_GIT_HOSTS", hosts)
    get_settings.cache_clear()
    if error is None:
        assert mirror_source(_info(), get_settings()) == MirrorSource(
            "https://github.com/acme/demo.git", False
        )
    else:
        with pytest.raises(error):
            mirror_source(_info(), get_settings())


def test_mirror_dir_and_szz_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = mirror_dir(Path("cache"), "https://github.com/Acme/Demo", "github.com")
    assert path == tmp_path.resolve() / "cache" / "github.com" / "acme" / "demo.git"
    assert path.is_absolute()
    gitlab = mirror_dir(tmp_path, "https://gitlab.com/Acme/Demo", "gitlab.com")
    assert gitlab.parts[-2:] == ("Acme", "Demo.git")
    with pytest.raises(UnsafeRepoError):
        mirror_dir(tmp_path, "https://github.com/acme/..", "github.com")
    with pytest.raises(UnsafeRepoError):
        mirror_dir(tmp_path, "https://github.com/acme/demo", "gitlab.com")
    assert szz_cache_dir(path) == path.parent / "demo.szz"


def test_repo_lock(fake_redis: fakeredis.FakeRedis) -> None:
    name = repo_lock_name(REPOSITORY_ID)
    assert name == f"lock:repo:{REPOSITORY_ID}"
    first = RepoLock(fake_redis, name, timeout_s=60, wait_s=0)
    second = RepoLock(fake_redis, name, timeout_s=60, wait_s=0.05, poll_s=0.01)
    assert first.acquire()
    assert 0 < fake_redis.pttl(name) <= 60_000
    assert not second.acquire()
    assert not second.release()
    assert fake_redis.exists(name)
    assert first.release()
    assert not fake_redis.exists(name)
    assert not first.release()
    third = RepoLock(fake_redis, name, wait_s=0)
    with second:
        assert fake_redis.exists(name)
        with pytest.raises(tasks.RepositoryBusyError, match="Another analysis is fetching"), third:
            pass
    assert not fake_redis.exists(name)


def test_repo_lock_expired_before_release(
    fake_redis: fakeredis.FakeRedis, caplog: pytest.LogCaptureFixture
) -> None:
    name = repo_lock_name("x")
    with RepoLock(fake_redis, name, wait_s=0):
        fake_redis.set(name, b"another-owner")
    assert fake_redis.get(name) == b"another-owner"
    assert "expired before it was released" in caplog.text


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (SoftTimeLimitExceeded(), TIMEOUT_FAILURE),
        (UserFacingError("the repo is gone"), "The repo is gone."),
        (UserFacingError("Already a sentence!"), "Already a sentence!"),
        (UserFacingError("   "), GENERIC_FAILURE),
        (BundleUnavailableError("No production model is configured."), "No production model is configured."),
        (RepoTooLargeError(2000, 1000), "Repository is 2,000 KB, over the 1,000 KB limit."),
        (HostLookupError("could not resolve github.com"), "Could not resolve github.com."),
        (TransientGitError("timeout at C:/x"), GIT_TRANSIENT_FAILURE),
        (GitError("fatal: C:/x"), GIT_FAILURE),
        (RuntimeError("C:/secret"), GENERIC_FAILURE),
    ],
)
def test_user_message(exc: BaseException, expected: str) -> None:
    assert user_message(exc) == expected


def test_user_message_custom_texts() -> None:
    assert user_message(KeyError("x"), generic="Generic.") == "Generic."
    assert user_message(SoftTimeLimitExceeded(), timeout="Slow.") == "Slow."


def test_broken_mirror_is_cloned_again(
    store: MemoryStore, source_repo: SourceRepo, worker_env: dict[str, Path]
) -> None:
    junk = _mirror(worker_env)
    junk.mkdir(parents=True)
    (junk / "garbage.txt").write_text("not a repository", encoding="utf-8")
    aid = store.add_analysis(commit_sha=source_repo.head)

    path = fetch_repo.apply(args=(str(aid),)).get()

    assert path == str(junk.resolve())
    assert _is_bare(junk)
    assert not (junk / "garbage.txt").exists()
