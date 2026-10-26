"""Build throwaway git repositories with scripted history for tests.

The builder isolates itself from the user's git configuration: it points ``GIT_CONFIG_GLOBAL`` at an
empty file, disables system config, and sets local ``user.*``, ``commit.gpgsign=false`` and
``core.autocrlf=false`` in the temporary repository only. It never touches global configuration.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

_EMPTY_CONFIG = Path(tempfile.mkdtemp(prefix="tdp-git-")) / "empty.gitconfig"
_EMPTY_CONFIG.touch()

DEFAULT_AUTHOR = ("Ana", "ana@example.com")


class RepoBuilder:
    """A git repository at ``root`` whose commits have caller-chosen authors and dates."""

    def __init__(self, root: Path, branch: str = "main") -> None:
        self._root = root
        root.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", branch)
        self._git("config", "user.name", "Test User")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "commit.gpgsign", "false")
        self._git("config", "core.autocrlf", "false")

    @property
    def path(self) -> Path:
        return self._root

    def commit(
        self,
        changes: Mapping[str, str | bytes | None],
        message: str,
        when: datetime,
        author: tuple[str, str] = DEFAULT_AUTHOR,
    ) -> str:
        """Apply ``changes`` (path -> new content, ``None`` deletes) and commit. Returns the commit sha."""
        for rel, content in changes.items():
            if content is None:
                self._git("rm", "-q", "--", rel)
            else:
                self._write(rel, content)
                self._git("add", "--", rel)
        return self._commit(message, when, author)

    def rename(
        self,
        old: str,
        new: str,
        message: str,
        when: datetime,
        author: tuple[str, str] = DEFAULT_AUTHOR,
        new_content: str | None = None,
    ) -> str:
        """``git mv old new`` (optionally rewriting the content) and commit. Returns the new commit sha."""
        target = self._root / new
        target.parent.mkdir(parents=True, exist_ok=True)
        self._git("mv", "--", old, new)
        if new_content is not None:
            self._write(new, new_content)
            self._git("add", "--", new)
        return self._commit(message, when, author)

    def write_mailmap(self, lines: list[str]) -> None:
        """Write ``.mailmap`` in the working tree (not committed; git log reads it from the work tree)."""
        (self._root / ".mailmap").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    # ------------------------------------------------------------------ internals
    def _write(self, rel: str, content: str | bytes) -> None:
        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)

    def _commit(self, message: str, when: datetime, author: tuple[str, str]) -> str:
        if when.tzinfo is None:
            raise ValueError("commit dates must be timezone-aware")
        stamp = when.isoformat()
        name, email = author
        env = {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_AUTHOR_DATE": stamp,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_COMMITTER_DATE": stamp,
        }
        self._git("commit", "-q", "--allow-empty", "-m", message, extra_env=env)
        return self.head()

    def _git(self, *args: str, extra_env: Mapping[str, str] | None = None) -> str:
        env = dict(os.environ)
        env.update(
            {
                "GIT_CONFIG_GLOBAL": str(_EMPTY_CONFIG),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(
            ["git", *args],
            cwd=self._root,
            env=env,
            check=True,
            capture_output=True,
        )
        return result.stdout.decode("utf-8", errors="replace")
