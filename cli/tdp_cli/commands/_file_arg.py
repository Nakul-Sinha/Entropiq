"""Find the ranked file a user means (``pkg/mod.py``, ``mod.py``, ``.\\pkg\\mod.py``, a rank or an OS path).

Shared by the commands that look at one file (``/whatif``, ``/advise``). Not a plug-in: the leading ``_``
keeps the registry from loading it as a command module.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

import typer

from tdp_cli.errors import UsageError
from tdp_cli.theme import clean

if TYPE_CHECKING:
    from tdp_cli.store import LoadedAnalysis
    from tdp_core.contracts import FileScore

FileArgument = Annotated[
    str,
    typer.Argument(
        help="File to look at: its path, a unique ending of it, or its rank (pkg/mod.py, mod.py or 3).",
        show_default=False,
    ),
]

UNKNOWN_HINT: Final = "Paths are relative to the repository root (pkg/mod.py); test files are not ranked."
MAX_LISTED: Final = 5
RANK_PATTERN: Final = re.compile(r"^#?(\d+)$")


def normalize_path(raw: str) -> str:
    """``raw`` with backslashes turned into slashes and any leading ``./`` removed."""
    norm = raw.strip().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _or_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} or {items[-1]}"


def _local_path(loaded: LoadedAnalysis, raw: str) -> str | None:
    """The repo-relative POSIX path of an OS path inside a local repository, or None."""
    if loaded.meta.kind != "local":
        return None
    try:
        path = Path(raw).expanduser()
        path = (path if path.is_absolute() else Path.cwd() / path).resolve()
        return path.relative_to(Path(loaded.meta.repo_dir).resolve()).as_posix()
    except (OSError, ValueError, RuntimeError):
        return None


def resolve_file(loaded: LoadedAnalysis, raw: str) -> FileScore:
    """The ranked file of ``loaded`` that ``raw`` names.

    Tries, in order: the exact path (backslashes and a leading ``./`` allowed), a rank in priority order
    (``3`` or ``#3``, 1 is the top file, the numbering of the ranking views), a unique case-insensitive
    match, an OS path inside a local repository (absolute or relative to the current folder), and a unique
    path suffix (``mod.py`` for ``pkg/mod.py``). Raises :class:`UsageError` for an empty argument, a rank
    outside 1 to the number of files, an ambiguous suffix or an unknown file (with close matches as a hint).
    """
    text = raw.strip()
    if not text:
        raise UsageError("give the path of a file in the analysis")
    ranked = loaded.result.files
    files = {f.path: f for f in ranked}
    norm = normalize_path(text)
    if norm in files:
        return files[norm]
    rank = RANK_PATTERN.match(norm)
    if rank is not None:
        n = int(rank.group(1))
        if not 1 <= n <= len(ranked):
            raise UsageError(f"no file has rank {n}; ranks go from 1 to {len(ranked)}")
        return ranked[n - 1]
    lowered = norm.lower()
    folded = [p for p in files if p.lower() == lowered]
    if len(folded) == 1:
        return files[folded[0]]
    local = _local_path(loaded, text)
    if local is not None:
        if local in files:
            return files[local]
        folded = [p for p in files if p.lower() == local.lower()]
        if len(folded) == 1:
            return files[folded[0]]
    suffix = [p for p in files if p.endswith("/" + norm)]
    if len(suffix) == 1:
        return files[suffix[0]]
    if suffix:
        shown = ", ".join(clean(p) for p in suffix[:MAX_LISTED])
        more = ", ..." if len(suffix) > MAX_LISTED else ""
        raise UsageError(
            f"{clean(text)} matches {len(suffix)} files: {shown}{more}", hint="Give more of the path."
        )
    close = difflib.get_close_matches(norm, list(files), n=3, cutoff=0.6)
    hint = f"Did you mean {_or_list([clean(p) for p in close])}?" if close else UNKNOWN_HINT
    raise UsageError(f"{clean(text)} is not a ranked file of this analysis", hint=hint)
