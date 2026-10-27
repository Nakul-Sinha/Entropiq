import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tdp_core.mining import GitError, TransientGitError, run_git
from tdp_core.testing import RepoBuilder


def _fake_run(stderr: bytes, returncode: int = 128) -> Any:
    def fake(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, returncode, b"", stderr)

    return fake


def test_returns_stdout_bytes(repo_builder: RepoBuilder, t0: datetime) -> None:
    sha = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    assert run_git(["rev-parse", "HEAD"], repo_builder.path) == f"{sha}\n".encode()


def test_input_goes_to_stdin(repo_builder: RepoBuilder) -> None:
    out = run_git(["hash-object", "--stdin"], repo_builder.path, input=b"hello\n")
    assert out.strip() == b"ce013625030ba8dba906f756967f9e9ca394464a"


def test_failure_raises_git_error_with_stderr(repo_builder: RepoBuilder) -> None:
    with pytest.raises(GitError) as info:
        run_git(["rev-parse", "--verify", "no-such-ref"], repo_builder.path)
    assert not isinstance(info.value, TransientGitError)
    assert info.value.returncode != 0
    assert "no-such-ref" in str(info.value) or "Needed a single revision" in info.value.stderr


def test_refuses_local_clone_without_file_protocol(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    repo_builder.commit({"a.py": "1\n"}, "one", t0)
    for i, source in enumerate([repo_builder.path.as_uri(), str(repo_builder.path)]):
        dest = tmp_path / f"clone{i}.git"
        with pytest.raises(GitError) as info:
            run_git(["clone", "--mirror", source, str(dest)], tmp_path)
        assert not isinstance(info.value, TransientGitError)
        assert "not allowed" in info.value.stderr
        assert not dest.exists()


def test_allows_local_clone_when_opted_in(repo_builder: RepoBuilder, t0: datetime, tmp_path: Path) -> None:
    sha = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    dest = tmp_path / "clone.git"
    run_git(
        ["clone", "-q", "--mirror", repo_builder.path.as_uri(), str(dest)], tmp_path, allow_file_protocol=True
    )
    assert run_git(["rev-parse", "HEAD"], dest).strip() == sha.encode()


def test_repository_hooks_path_is_overridden(repo_builder: RepoBuilder, tmp_path: Path) -> None:
    evil = tmp_path / "evil-hooks"
    evil.mkdir()
    # Repo-local config, as a cloned repository might ship it.
    run_git(["config", "core.hooksPath", str(evil)], repo_builder.path)

    hooks = Path(run_git(["config", "--get", "core.hooksPath"], repo_builder.path).decode().strip())

    assert hooks != evil
    assert hooks.is_dir()
    assert not any(hooks.iterdir())


def test_inherited_git_dir_is_ignored(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = repo_builder.commit({"a.py": "1\n"}, "one", t0)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere.git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "index"))
    assert run_git(["rev-parse", "HEAD"], repo_builder.path).strip() == sha.encode()


def test_environment_and_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen["cmd"], seen["env"], seen["kwargs"] = cmd, kwargs["env"], kwargs
        return subprocess.CompletedProcess(cmd, 0, b"ok", b"")

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.hooksPath'='/tmp/x'")

    assert run_git(["status"], tmp_path) == b"ok"
    env, cmd = seen["env"], seen["cmd"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    assert "GIT_CONFIG_PARAMETERS" not in env
    assert seen["kwargs"]["stdin"] is subprocess.DEVNULL
    assert cmd[0] == "git"
    assert cmd[-1] == "status"
    for setting in ("protocol.file.allow=never", "core.fsmonitor=false", "submodule.recurse=false"):
        assert setting in cmd
    assert any(c.startswith("core.hooksPath=") for c in cmd)

    run_git(["status"], tmp_path, allow_file_protocol=True)
    assert "GIT_ALLOW_PROTOCOL" not in seen["env"]
    assert "protocol.file.allow=always" in seen["cmd"]


def test_timeout_is_transient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(TransientGitError, match="timed out"):
        run_git(["fetch"], tmp_path, timeout=5)


@pytest.mark.parametrize(
    "stderr",
    [
        b"fatal: unable to access 'https://nohost.invalid/x/': Could not resolve host: nohost.invalid",
        b"fatal: unable to access 'https://github.com/x/': Failed to connect to github.com port 443",
        b"error: RPC failed; curl 56 Connection reset by peer",
    ],
)
def test_network_errors_are_transient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stderr: bytes) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(stderr))
    with pytest.raises(TransientGitError):
        run_git(["fetch"], tmp_path)


def test_credentials_are_redacted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stderr = b"fatal: repository 'https://x-access-token:s3cret@github.com/a/b.git/' not found"
    monkeypatch.setattr(subprocess, "run", _fake_run(stderr))
    with pytest.raises(GitError) as info:
        run_git(["clone", "https://x-access-token:s3cret@github.com/a/b.git"], tmp_path)
    assert "s3cret" not in str(info.value)
    assert "s3cret" not in info.value.stderr
    assert all("s3cret" not in part for part in info.value.cmd)


def test_missing_git_or_cwd_raises_git_error(tmp_path: Path) -> None:
    with pytest.raises(GitError) as info:
        run_git(["status"], tmp_path / "does-not-exist")
    assert not isinstance(info.value, TransientGitError)
