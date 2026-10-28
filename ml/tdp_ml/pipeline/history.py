"""Git history helpers of the pipeline: mining at a pinned sha with that sha's ``.mailmap``.

A bare mirror applies the ``.mailmap`` of ``HEAD``, which is future information for a snapshot: an edit of
``.mailmap`` after T would change the author identities (and so the ownership features) seen at T. Every
pipeline stage therefore mines with ``mailmap.blob=<sha>:.mailmap``, the mailmap of the mined commit itself,
which is what a clone checked out at that commit reads from its work tree.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tdp_core.config import DEFAULT_SETTINGS, Settings
from tdp_core.mining.git import GitError, check_rev, run_git
from tdp_core.mining.gitlog import LOG_FORMAT, MinedHistory, parse_git_log
from tdp_ml.pipeline.layout import DataLayout, PipelineError

# Same pinned options as tdp_core.mining.gitlog, so the parsed tables match mine_history exactly.
_LOG_CONFIG = ("-c", "log.showRoot=true", "-c", "log.showSignature=false", "-c", "diff.algorithm=myers")


def mine_at(repo_dir: Path, sha: str, settings: Settings = DEFAULT_SETTINGS) -> MinedHistory:
    """``mine_history(repo_dir, sha)`` with the ``.mailmap`` of ``sha`` itself (a missing file maps nothing).

    Raises ``ValueError`` for a revision that looks like an option and ``GitError`` when git fails.
    """
    check_rev(sha)
    args = [
        "-c",
        f"mailmap.blob={sha}:.mailmap",
        *_LOG_CONFIG,
        "log",
        "--no-merges",
        "-M",
        "--numstat",
        "-z",
        "--date=unix",
        "--encoding=UTF-8",
        "--no-color",
        f"--format={LOG_FORMAT}",
        "--end-of-options",
        sha,
        "--",
    ]
    return parse_git_log(run_git(args, repo_dir), settings)


def head_sha(repo_dir: Path) -> str:
    """The commit sha ``HEAD`` resolves to (``GitError`` for an empty repository)."""
    return run_git(["rev-parse", "--verify", "HEAD^{commit}"], repo_dir).decode("ascii").strip()


def is_ancestor(repo_dir: Path, ancestor: str, descendant: str) -> bool:
    """Whether ``ancestor`` is ``descendant`` or one of its ancestors (``git merge-base --is-ancestor``)."""
    check_rev(ancestor)
    check_rev(descendant)
    try:
        run_git(["merge-base", "--is-ancestor", ancestor, descendant], repo_dir)
    except GitError as exc:
        if exc.returncode == 1:
            return False
        raise
    return True


def _stored_schema(path: Path) -> pa.Schema | None:
    """The Arrow schema the file was written with (``ARROW:schema`` metadata), if present."""
    metadata = pq.read_metadata(path).metadata or {}
    raw = metadata.get(b"ARROW:schema")
    if raw is None:
        return None
    return pa.ipc.read_schema(pa.py_buffer(base64.b64decode(raw)))


def read_table(path: Path, producer: str) -> pa.Table:
    """The Parquet table at ``path``; :class:`PipelineError` naming ``tdp-ml <producer>`` if it is missing.

    Parquet stores second-resolution timestamps as milliseconds, so the column types are restored from the
    Arrow schema the file was written with: ``timestamp[s]`` columns of the core schemas read back as
    written (their ``cast(int64)`` must stay Unix seconds).
    """
    if not path.is_file():
        raise PipelineError(f"{path} is missing: run tdp-ml {producer}")
    table = pq.read_table(path)
    original = _stored_schema(path)
    same_names = original is not None and original.names == table.schema.names
    if same_names and any(a.type != b.type for a, b in zip(original, table.schema, strict=True)):
        table = table.cast(original)
    return table


def load_history(layout: DataLayout, slug: str) -> MinedHistory:
    """The full history of ``slug`` written by the ``mine`` stage."""
    return MinedHistory(
        commits=read_table(layout.commits(slug), "mine"),
        file_changes=read_table(layout.file_changes(slug), "mine"),
    )
