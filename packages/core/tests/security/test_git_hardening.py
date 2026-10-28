"""Git never runs code from an analysed repository and never uses a transport other than https.

The first tests repeat two checks of the git runner on purpose, so this suite stands on its own. Every test
that plants a hook also runs plain git as a control, which proves the planted script was live.
"""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tdp_core.mining.git import GitError, TransientGitError, run_git
from tdp_core.mining.gitlog import mine_history
from tdp_core.mining.mirror import FETCH_REFSPEC, ensure_mirror
from tdp_core.mining.tree import list_files, read_blobs
from tdp_core.testing import RepoBuilder

HARDENED_KEYS = ("core.hooksPath", "protocol.file.allow", "core.fsmonitor", "submodule.recurse")


def _hook(marker: Path) -> bytes:
    """A shell hook that appends its own path to ``marker`` whenever git runs it."""
    return f'#!/bin/sh\necho "$0" >> "{marker.as_posix()}"\n'.encode()


def _install(path: Path, body: bytes) -> Path:
    """Write an executable script with LF line endings (Git for Windows runs it through its shebang)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(0o755)
    return path


def _plain_git_env(tmp_path: Path) -> dict[str, str]:
    """Environment for an unhardened git control run: no inherited GIT_* values, empty global config."""
    empty = tmp_path / "empty.gitconfig"
    empty.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_CONFIG_GLOBAL": str(empty), "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"})
    return env


# ---------------------------------------------------------------------------------------------- command line
def test_every_call_starts_with_the_hardening_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.hooksPath'='/tmp/evil'")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file:ext:https")

    run_git(["fetch", "origin"], tmp_path)

    cmd, kwargs = seen["cmd"], seen["kwargs"]
    assert cmd[0] == "git"
    assert cmd[1:9:2] == ["-c"] * 4
    assert cmd[2].startswith("core.hooksPath=")
    assert cmd[4:9:2] == ["protocol.file.allow=never", "core.fsmonitor=false", "submodule.recurse=false"]
    assert cmd[9:] == ["fetch", "origin"]
    hooks = Path(cmd[2].removeprefix("core.hooksPath="))
    assert hooks.is_dir()
    assert not any(hooks.iterdir())

    env = kwargs["env"]
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert "GIT_CONFIG_PARAMETERS" not in env
    assert kwargs["stdin"] is subprocess.DEVNULL

    run_git(["fetch", "origin"], tmp_path, allow_file_protocol=True)
    cmd, env = seen["cmd"], seen["kwargs"]["env"]
    assert cmd[2] == f"core.hooksPath={hooks}"
    assert cmd[4] == "protocol.file.allow=always"
    assert "GIT_ALLOW_PROTOCOL" not in env


def test_hardened_values_win_over_repository_config(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    repo = repo_builder.path
    repo_builder.commit({"a.py": "x = 1\n"}, "one", t0)
    marker = tmp_path / "marker.txt"
    evil_hooks = tmp_path / "evil-hooks"
    _install(evil_hooks / "post-checkout", _hook(marker))
    fsmonitor = _install(tmp_path / "fsmonitor.sh", _hook(marker))
    hostile = {
        "core.hooksPath": evil_hooks.as_posix(),
        "core.fsmonitor": fsmonitor.as_posix(),
        "protocol.file.allow": "always",
        "submodule.recurse": "true",
    }
    for key, value in hostile.items():
        run_git(["config", key, value], repo)  # repository-local config, as a cloned repo might ship it

    effective = [run_git(["config", "--get", key], repo).decode().strip() for key in HARDENED_KEYS]

    hooks = Path(effective[0])
    assert hooks != evil_hooks
    assert hooks.is_dir()
    assert not any(hooks.iterdir())
    assert effective[1:] == ["never", "false", "false"]
    assert not marker.exists()


# ---------------------------------------------------------------------------------------------- transports
TRANSPORTS = ["http", "ssh", "scp", "git", "ext", "fd", "file-uri", "local-path"]


def _transport_url(kind: str, upstream: Path, marker: Path) -> str:
    return {
        "http": "http://example.invalid/x/y.git",
        "ssh": "ssh://git@example.invalid/x/y.git",
        "scp": "git@example.invalid:x/y.git",
        "git": "git://example.invalid/x/y.git",
        "ext": f"ext::sh -c touch% {marker.as_posix()}",
        "fd": "fd::3",
        "file-uri": upstream.as_uri(),
        "local-path": str(upstream),
    }[kind]


@pytest.mark.parametrize("kind", TRANSPORTS)
def test_only_https_transports_are_allowed(
    kind: str, tmp_path: Path, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = RepoBuilder(tmp_path / "upstream")
    upstream.commit({"a.py": "x = 1\n"}, "one", t0)
    marker = tmp_path / "pwned.txt"
    # A hostile global config that allows every protocol: GIT_ALLOW_PROTOCOL must still win.
    hostile = tmp_path / "hostile.gitconfig"
    hostile.write_bytes(
        b"[protocol]\n\tallow = always\n"
        b'[protocol "ext"]\n\tallow = always\n'
        b'[protocol "file"]\n\tallow = always\n'
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile))
    dest = tmp_path / "m" / "x.git"

    with pytest.raises(GitError) as info:
        ensure_mirror(_transport_url(kind, upstream.path, marker), dest)

    assert not isinstance(info.value, TransientGitError)
    assert "not allowed" in info.value.stderr
    assert not dest.exists()
    assert not marker.exists()


# ---------------------------------------------------------------------------------------------- hooks
def test_repository_with_hook_scripts_is_mirrored_without_running_them(
    tmp_path: Path, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "hook-ran.txt"
    hook = _hook(marker)
    upstream = RepoBuilder(tmp_path / "upstream")
    upstream.commit(
        {"hooks/reference-transaction": hook, ".githooks/post-checkout": hook, "a.py": "x = 1\n"}, "hooks", t0
    )
    # Live hooks in a template directory, as a global init.templateDir would install them into new mirrors.
    template = tmp_path / "template"
    for name in ("reference-transaction", "post-checkout", "pre-auto-gc"):
        _install(template / "hooks" / name, hook)
    dest = tmp_path / "mirror.git"

    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(template))
    ensure_mirror(upstream.path.as_uri(), dest, allow_file_protocol=True)
    monkeypatch.delenv("GIT_TEMPLATE_DIR")
    assert (dest / "hooks" / "reference-transaction").is_file()  # git copied a live hook into the mirror

    upstream.commit({"a.py": "x = 2\n"}, "two", t0 + timedelta(days=1))
    ensure_mirror(upstream.path.as_uri(), dest, allow_file_protocol=True)  # fetch updates refs
    head = run_git(["--git-dir", str(dest), "rev-parse", "refs/heads/main"], dest).decode().strip()
    assert head == upstream.head()
    history = mine_history(dest)
    assert history.commits.num_rows == 2
    files = list_files(dest, head)
    assert "hooks/reference-transaction" in files
    assert read_blobs(dest, [files["hooks/reference-transaction"]]) == {
        files["hooks/reference-transaction"]: hook
    }
    run_git(["update-ref", "refs/heads/probe", head], dest)
    assert not marker.exists()

    # Control: plain git runs the planted reference-transaction hook on the next fetch.
    upstream.commit({"a.py": "x = 3\n"}, "three", t0 + timedelta(days=2))
    subprocess.run(
        [
            "git",
            f"--git-dir={dest}",
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "-q",
            "origin",
            FETCH_REFSPEC,
        ],
        env=_plain_git_env(tmp_path),
        check=True,
        capture_output=True,
    )
    assert marker.exists()


def test_fsmonitor_from_repository_config_never_runs(
    repo_builder: RepoBuilder, t0: datetime, tmp_path: Path
) -> None:
    repo = repo_builder.path
    head = repo_builder.commit({"a.py": "x = 1\n"}, "one", t0)
    marker = tmp_path / "fsmonitor-ran.txt"
    hook = _install(tmp_path / "outside" / "fsmonitor.sh", _hook(marker) + b"exit 1\n")
    run_git(["config", "core.fsmonitor", hook.as_posix()], repo)
    (repo / "a.py").write_bytes(b"x = 2\n")

    assert run_git(["status", "--porcelain"], repo).strip() == b"M a.py"
    assert run_git(["diff", "--name-only"], repo).strip() == b"a.py"
    assert mine_history(repo).commits.num_rows == 1
    assert set(list_files(repo, head)) == {"a.py"}
    assert not marker.exists()

    # Control: plain git runs the configured fsmonitor hook.
    subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        env=_plain_git_env(tmp_path),
        check=True,
        capture_output=True,
    )
    if not marker.exists():
        pytest.skip("this git does not run core.fsmonitor from repository config")
