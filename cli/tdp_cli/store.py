"""Saved analyses: ``<home>/analyses/<repo slug>/<sha>.*``.

Every analysis is written as three files, each atomically (temporary file, then rename):

- ``<sha>.features.parquet``: the feature table (index ``path``, the 89 model input columns);
- ``<sha>.json``: ``{"schema": 1, "result": to_jsonable(result)}`` (the AnalysisResult without features);
- ``<sha>.meta.json`` (written last): ``{"schema": 1, ...AnalysisMeta}``, a small summary so ``/history``
  never has to import pandas.

An entry counts only when both JSON files exist, so an interrupted save is ignored.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import tdp_cli
from tdp_cli.errors import CliError

if TYPE_CHECKING:
    from tdp_cli.repos import Target
    from tdp_core.contracts import AnalysisResult, FileScore

logger = logging.getLogger(__name__)

SCHEMA: Final = 1
_NUMBER_KEY: Final = re.compile(r"^\d{1,3}$")
_SHA_KEY: Final = re.compile(r"^[0-9a-fA-F]{4,40}$")


def _iso(when: datetime) -> str:
    return when.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_time(text: str) -> datetime:
    when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("analyzed_at has no time zone")
    return when.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AnalysisMeta:
    """Summary of one saved analysis (what ``/history`` lists)."""

    slug: str
    repo: str
    source: str
    kind: Literal["local", "url"]
    repo_dir: str
    ref: str
    sha: str
    analyzed_at: datetime
    model_version: str
    demo_model: bool
    n_files: int
    counts: dict[str, int]
    critical_hours: float
    cli_version: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (``analyzed_at`` as ISO 8601 with ``Z``)."""
        return {
            "slug": self.slug,
            "repo": self.repo,
            "source": self.source,
            "kind": self.kind,
            "repo_dir": self.repo_dir,
            "ref": self.ref,
            "sha": self.sha,
            "analyzed_at": _iso(self.analyzed_at),
            "model_version": self.model_version,
            "demo_model": self.demo_model,
            "n_files": self.n_files,
            "counts": dict(self.counts),
            "critical_hours": self.critical_hours,
            "cli_version": self.cli_version,
        }

    @classmethod
    def from_dict(cls, data: Any) -> AnalysisMeta:
        """Rebuild from :meth:`to_dict` output. Raises ``ValueError``, ``KeyError`` or ``TypeError``."""
        if not isinstance(data, dict):
            raise TypeError("analysis metadata is not an object")
        kind = data["kind"]
        if kind not in ("local", "url"):
            raise ValueError(f"unknown kind {kind!r}")
        strings = ("slug", "repo", "source", "repo_dir", "ref", "sha", "model_version", "cli_version")
        for key in strings:
            if not isinstance(data[key], str):
                raise TypeError(f"{key} must be a string")
        counts = data["counts"]
        if not isinstance(counts, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) for k, v in counts.items()
        ):
            raise TypeError("counts must map band names to integers")
        n_files = data["n_files"]
        if not isinstance(n_files, int) or isinstance(n_files, bool):
            raise TypeError("n_files must be an integer")
        hours = data["critical_hours"]
        if not isinstance(hours, (int, float)) or isinstance(hours, bool):
            raise TypeError("critical_hours must be a number")
        return cls(
            slug=data["slug"],
            repo=data["repo"],
            source=data["source"],
            kind=kind,
            repo_dir=data["repo_dir"],
            ref=data["ref"],
            sha=data["sha"],
            analyzed_at=_parse_time(str(data["analyzed_at"])),
            model_version=data["model_version"],
            demo_model=bool(data["demo_model"]),
            n_files=n_files,
            counts=dict(counts),
            critical_hours=float(hours),
            cli_version=data["cli_version"],
        )


@dataclass(frozen=True, slots=True, eq=False)
class LoadedAnalysis:
    """An analysis in memory: its summary and the full result (features included)."""

    meta: AnalysisMeta
    result: AnalysisResult

    def file(self, path: str) -> FileScore | None:
        """The score of ``path`` (repo-relative POSIX), or None when it was not ranked."""
        for score in self.result.files:
            if score.path == path:
                return score
        return None


def make_meta(target: Target, result: AnalysisResult, *, demo_model: bool) -> AnalysisMeta:
    """The summary of ``result`` for the repository ``target``."""
    from tdp_cli.output import band_counts, critical_hours

    return AnalysisMeta(
        slug=target.slug,
        repo=target.display,
        source=target.source,
        kind=target.kind,
        repo_dir=str(target.repo_dir),
        ref=target.ref,
        sha=result.sha,
        analyzed_at=result.analyzed_at,
        model_version=result.model_version,
        demo_model=demo_model,
        n_files=len(result.files),
        counts=band_counts(result.files),
        critical_hours=round(critical_hours(result), 3),
        cli_version=tdp_cli.__version__,
    )


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _dumps(data: Any) -> bytes:
    return json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8")


class AnalysisStore:
    """Saved analyses under ``root`` (``<home>/analyses``)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def folder(self, slug: str) -> Path:
        """The folder of one repository's analyses."""
        return self.root / slug

    def _paths(self, slug: str, sha: str) -> tuple[Path, Path, Path]:
        folder = self.folder(slug)
        return folder / f"{sha}.meta.json", folder / f"{sha}.json", folder / f"{sha}.features.parquet"

    def save(self, loaded: LoadedAnalysis) -> Path:
        """Write ``loaded`` (features, result, then metadata) and return the result JSON path.

        Raises ``OSError`` for disk errors and ``ValueError`` or ``TypeError`` for values JSON cannot hold.
        """
        from tdp_core.serialize import to_jsonable

        meta = loaded.meta
        meta_path, result_path, features_path = self._paths(meta.slug, meta.sha)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = features_path.with_name(f"{features_path.name}.{os.getpid()}.tmp")
        try:
            loaded.result.features.to_parquet(tmp)
            os.replace(tmp, features_path)
        finally:
            tmp.unlink(missing_ok=True)
        _write_atomic(result_path, _dumps({"schema": SCHEMA, "result": to_jsonable(loaded.result)}))
        _write_atomic(meta_path, _dumps({"schema": SCHEMA, **meta.to_dict()}))
        return result_path

    def _read_meta(self, meta_path: Path) -> AnalysisMeta | None:
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") != SCHEMA:
                return None
            meta = AnalysisMeta.from_dict(data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("Skipping unreadable saved analysis %s: %s", meta_path, exc)
            return None
        if not (meta_path.parent / f"{meta.sha}.json").is_file():
            return None
        return meta

    def get(self, slug: str, sha: str) -> AnalysisMeta | None:
        """The saved analysis of commit ``sha`` (full sha) of repository ``slug``, or None."""
        meta = self._read_meta(self._paths(slug, sha)[0])
        return meta if meta is not None and meta.slug == slug and meta.sha == sha else None

    def entries(self) -> list[AnalysisMeta]:
        """Every readable saved analysis, newest first (then by repository and sha)."""
        if not self.root.is_dir():
            return []
        found = [m for p in sorted(self.root.glob("*/*.meta.json")) if (m := self._read_meta(p)) is not None]
        found.sort(key=lambda m: (m.repo, m.sha))
        found.sort(key=lambda m: m.analyzed_at, reverse=True)
        return found

    def find(self, key: str) -> AnalysisMeta:
        """A saved analysis by its ``/history`` number (``"1"``) or by a commit sha prefix (4+ hex digits).

        Raises :class:`CliError` (exit code 2) when nothing or more than one analysis matches.
        """
        key = key.strip()
        entries = self.entries()
        if _NUMBER_KEY.match(key):
            number = int(key)
            if not entries:
                raise CliError("There are no saved analyses yet.", exit_code=2, hint="Run /analyze first.")
            if not 1 <= number <= len(entries):
                raise CliError(
                    f"There is no analysis number {number}; /history lists 1 to {len(entries)}.",
                    exit_code=2,
                )
            return entries[number - 1]
        if _SHA_KEY.match(key):
            prefix = key.lower()
            matches = [m for m in entries if m.sha.lower().startswith(prefix)]
            if not matches:
                raise CliError(f"No saved analysis has a commit starting with {prefix}.", exit_code=2)
            if len(matches) > 1:
                listed = ", ".join(f"{m.repo} {m.sha[:10]}" for m in matches[:5])
                raise CliError(
                    f"{len(matches)} saved analyses match {prefix}: {listed}.",
                    exit_code=2,
                    hint="Type more characters of the sha or use the number from /history.",
                )
            return matches[0]
        raise CliError("Use a number from /history or at least 4 characters of a commit sha.", exit_code=2)

    def load(self, meta: AnalysisMeta) -> LoadedAnalysis:
        """Read the full analysis of ``meta``. Raises :class:`CliError` when its files cannot be read."""
        import pandas as pd

        from tdp_core.contracts import FEATURE_INPUT_COLUMNS, AnalysisResult
        from tdp_core.serialize import from_jsonable

        _, result_path, features_path = self._paths(meta.slug, meta.sha)
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") != SCHEMA:
                raise ValueError("unknown schema")
            result = from_jsonable(AnalysisResult, data["result"])
            if features_path.is_file():
                features = pd.read_parquet(features_path)
            else:
                features = pd.DataFrame(
                    {c: pd.Series(dtype="float32") for c in FEATURE_INPUT_COLUMNS},
                    index=pd.Index([], dtype=str, name="path"),
                )
            features = features[list(FEATURE_INPUT_COLUMNS)]
            features.index.name = "path"
        except Exception as exc:
            raise CliError(
                f"The saved analysis of {meta.repo} {meta.sha[:7]} cannot be read; run /analyze again.",
                hint=f"{type(exc).__name__}: {exc}",
            ) from exc
        return LoadedAnalysis(meta, dataclasses.replace(result, features=features))
