"""Hardened git subprocess runner. Every git invocation in tdp_core goes through :func:`run_git`.

The runner never lets an analysed repository execute code: hooks point at an empty directory, fsmonitor and
submodule recursion are off, the terminal never prompts, and only ``https`` transports are allowed unless the
caller opts into local (``file``) repositories, which only tests and trusted local paths should do.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

# Substrings (lowercase) of git stderr that mean "the network failed", so a retry may succeed.
_TRANSIENT_MARKERS = (
    "could not resolve host",
    "connection",
    "failed to connect",
    "timed out",
    "the remote end hung up unexpectedly",
    "early eof",
    "rpc failed",
)

# Inherited variables that would point git at another repository or inject configuration from a parent
# process (for example when the caller itself runs inside a git hook).
_SCRUBBED_ENV = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_PREFIX",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_EXTERNAL_DIFF",
        "GIT_DIFF_OPTS",
    }
)
_SCRUBBED_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")

_URL_USERINFO = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]+@")


class GitError(RuntimeError):
    """A git command failed. ``stderr`` holds git's message with URL credentials redacted."""

    def __init__(
        self,
        message: str,
        *,
        cmd: Sequence[str] = (),
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.cmd = tuple(cmd)
        self.returncode = returncode
        self.stderr = stderr


class TransientGitError(GitError):
    """Network failure or timeout; the same call may succeed if retried."""


@functools.cache
def _empty_hooks_dir() -> str:
    """An empty directory (created once per process) used as ``core.hooksPath`` so no hook ever runs."""
    return tempfile.mkdtemp(prefix="tdp-git-hooks-")


def _redact(text: str) -> str:
    """Replace ``user:token@`` in URLs with ``***@`` so credentials never reach logs or exceptions."""
    return _URL_USERINFO.sub(r"\1***@", text)


def _is_transient(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def _git_env(allow_file_protocol: bool) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV and not k.startswith(_SCRUBBED_PREFIXES)
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    if allow_file_protocol:
        env.pop("GIT_ALLOW_PROTOCOL", None)
    else:
        env["GIT_ALLOW_PROTOCOL"] = "https"
    return env


def run_git(
    args: Sequence[str],
    cwd: Path | None,
    *,
    timeout: float = 600,
    allow_file_protocol: bool = False,
    input: bytes | None = None,
) -> bytes:
    """Run ``git <args>`` in ``cwd`` and return its stdout as raw bytes.

    ``args`` are the arguments after ``git`` (they may start with extra ``-c key=value`` pairs). ``timeout``
    is in seconds. ``input`` is written to git's stdin; without it stdin is closed so git can never wait for
    input. With ``allow_file_protocol=False`` (the default) only ``https`` remotes work: local paths and
    ``file://`` URLs are refused by git itself.

    Raises :class:`TransientGitError` on timeouts and network errors ("Could not resolve host", connection
    failures), and :class:`GitError` for any other non-zero exit or when git cannot be started.
    """
    cmd = [
        "git",
        "-c",
        f"core.hooksPath={_empty_hooks_dir()}",
        "-c",
        f"protocol.file.allow={'always' if allow_file_protocol else 'never'}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "submodule.recurse=false",
        *args,
    ]
    shown = _redact(" ".join(args))
    if len(shown) > 200:
        shown = shown[:200] + "..."
    try:
        if input is None:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=_git_env(allow_file_protocol),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        else:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=_git_env(allow_file_protocol),
                input=input,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise TransientGitError(
            f"git {shown} timed out after {timeout:g} s", cmd=[_redact(c) for c in cmd]
        ) from exc
    except OSError as exc:  # git not installed, or cwd missing
        raise GitError(f"could not run git {shown}: {exc}", cmd=[_redact(c) for c in cmd]) from exc
    if proc.returncode != 0:
        stderr = _redact(proc.stderr.decode("utf-8", errors="replace").strip())
        kind = TransientGitError if _is_transient(stderr) else GitError
        raise kind(
            f"git {shown} failed with exit code {proc.returncode}: {stderr}",
            cmd=[_redact(c) for c in cmd],
            returncode=proc.returncode,
            stderr=stderr,
        )
    return proc.stdout


def check_rev(rev: str) -> None:
    """Reject revisions that git could read as an option (``-x``, ``--output=...``) or that are empty."""
    if not rev or rev.startswith("-") or "\x00" in rev:
        raise ValueError(f"invalid git revision: {rev!r}")
