"""Read trees and blobs at a commit without a working tree (works on bare mirrors)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tdp_core.mining.git import GitError, check_rev, run_git

# Regular files only: symlinks (120000) and submodules (160000) are skipped. 100664 is a legacy regular-file
# mode that still appears in old histories.
_REGULAR_MODES = frozenset({"100644", "100755", "100664"})


def list_files(repo_dir: Path, sha: str) -> dict[str, str]:
    """Map every regular file in the tree of ``sha`` to its blob sha, as ``{repo-relative POSIX path: sha}``.

    Uses ``git ls-tree -r -z --full-tree``, so paths are verbatim (no quoting) and independent of the current
    directory. Symlinks and submodules are excluded. Keys are in git's tree order (bytewise path order).
    Raises ``ValueError`` for a revision starting with ``-`` and ``GitError`` for an unknown revision.
    """
    check_rev(sha)
    raw = run_git(["ls-tree", "-r", "-z", "--full-tree", "--end-of-options", sha], repo_dir)
    files: dict[str, str] = {}
    for entry in raw.decode("utf-8", errors="replace").split("\x00"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        mode, kind, obj = meta.split(" ")
        if kind == "blob" and mode in _REGULAR_MODES:
            files[path] = obj
    return files


def read_blobs(repo_dir: Path, blob_shas: Iterable[str]) -> dict[str, bytes]:
    """Return the exact bytes of each blob, keyed by the sha as given, using one ``git cat-file --batch``.

    Duplicate shas are read once; an empty input returns ``{}`` without running git. Raises ``ValueError`` for
    an empty sha or one containing whitespace, and ``GitError`` if any object is missing, ambiguous or not a
    blob.
    """
    wanted = list(dict.fromkeys(blob_shas))
    if not wanted:
        return {}
    for sha in wanted:
        if not sha or any(c.isspace() for c in sha):
            raise ValueError(f"invalid blob sha: {sha!r}")
    raw = run_git(["cat-file", "--batch"], repo_dir, input=("\n".join(wanted) + "\n").encode("utf-8"))

    blobs: dict[str, bytes] = {}
    pos = 0
    for sha in wanted:
        newline = raw.find(b"\n", pos)
        if newline < 0:
            raise GitError(f"truncated git cat-file output while reading {sha}")
        header = raw[pos:newline].decode("utf-8", errors="replace").split(" ")
        if len(header) != 3:
            raise GitError(f"git cat-file could not read {sha}: {' '.join(header[1:]) or 'no header'}")
        _, kind, size_s = header
        if kind != "blob":
            raise GitError(f"object {sha} is a {kind}, not a blob")
        start = newline + 1
        end = start + int(size_s)
        if end > len(raw):
            raise GitError(f"truncated git cat-file output while reading {sha}")
        blobs[sha] = raw[start:end]
        pos = end + 1  # skip the LF after the contents
    return blobs
