"""``data/manifest.json``: every repository pinned to a sha, plus the settings of one dataset build.

The manifest is written once (``tdp-ml manifest``) and every later stage reads its repositories, settings,
criteria and budgets from it, so a build has one source of truth even when it runs on another machine.
Every stage mines and selects at the manifest sha, never at the mirror's current ``HEAD``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from tdp_core.config import Settings
from tdp_core.contracts import EXTRACTOR_VERSION
from tdp_core.features.satd_model import load_default_classifier
from tdp_core.mining.mirror import ensure_mirror
from tdp_core.mining.reach import default_branch
from tdp_ml.pipeline.config import ConfigError, Criteria, PipelineConfig, settings_from_overrides
from tdp_ml.pipeline.history import head_sha
from tdp_ml.pipeline.layout import DataLayout, PipelineError, read_json, write_json

logger = logging.getLogger(__name__)

MANIFEST_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class ManifestRepo:
    """One repository of the manifest, pinned to ``sha`` on ``default_branch``."""

    id: int
    name: str
    slug: str
    url: str
    sha: str
    default_branch: str
    issues: bool

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class Manifest:
    """The pinned repositories and the settings, criteria and budgets of one dataset build."""

    created_at: datetime
    extractor_version: str
    satd_model_sha256: str | None
    settings: Settings
    criteria: Criteria
    max_snapshots: int
    max_repos: int | None
    szz_max_fixes: int | None
    repos: tuple[ManifestRepo, ...]

    def select(self, slugs: Sequence[str] | None = None) -> tuple[ManifestRepo, ...]:
        """The repositories named by ``slugs`` in manifest order; None or empty selects all of them.

        Raises :class:`PipelineError` for a slug that is not in the manifest, naming the known slugs.
        """
        if not slugs:
            return self.repos
        known = {repo.slug for repo in self.repos}
        unknown = sorted(set(slugs) - known)
        if unknown:
            raise PipelineError(
                f"unknown repository {', '.join(unknown)}; the manifest has {', '.join(sorted(known))}"
            )
        wanted = set(slugs)
        return tuple(repo for repo in self.repos if repo.slug in wanted)

    def repo(self, slug: str) -> ManifestRepo:
        return self.select([slug])[0]

    def to_json(self) -> dict[str, Any]:
        settings = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in dataclasses.asdict(self.settings).items()
        }
        return {
            "version": MANIFEST_VERSION,
            "created_at": self.created_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extractor_version": self.extractor_version,
            "satd_model_sha256": self.satd_model_sha256,
            "settings": settings,
            "criteria": self.criteria.to_json(),
            "max_snapshots": self.max_snapshots,
            "max_repos": self.max_repos,
            "szz_max_fixes": self.szz_max_fixes,
            "repos": [repo.to_json() for repo in self.repos],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Manifest:
        """Rebuild from :meth:`to_json` output; raises :class:`PipelineError` for anything malformed."""
        try:
            if data.get("version") != MANIFEST_VERSION:
                raise ValueError(f"unsupported manifest version {data.get('version')!r}")
            created = datetime.fromisoformat(str(data["created_at"]).replace("Z", "+00:00"))
            if created.tzinfo is None:
                raise ValueError("created_at has no time zone")
            repos = tuple(
                ManifestRepo(
                    id=int(r["id"]),
                    name=str(r["name"]),
                    slug=str(r["slug"]),
                    url=str(r["url"]),
                    sha=str(r["sha"]),
                    default_branch=str(r["default_branch"]),
                    issues=bool(r["issues"]),
                )
                for r in data["repos"]
            )
            return cls(
                created_at=created.astimezone(UTC),
                extractor_version=str(data["extractor_version"]),
                satd_model_sha256=data.get("satd_model_sha256"),
                settings=settings_from_overrides(data["settings"]),
                criteria=Criteria.from_json(data["criteria"]),
                max_snapshots=int(data["max_snapshots"]),
                max_repos=None if data.get("max_repos") is None else int(data["max_repos"]),
                szz_max_fixes=None if data.get("szz_max_fixes") is None else int(data["szz_max_fixes"]),
                repos=repos,
            )
        except (KeyError, TypeError, ValueError, ConfigError) as exc:
            raise PipelineError(f"the manifest is malformed: {exc}") from exc


def build_manifest(
    config: PipelineConfig, layout: DataLayout, *, fetch: bool = False, now: datetime | None = None
) -> Manifest:
    """Pin every repository of ``config`` to the current ``HEAD`` of its mirror under ``layout``.

    A missing mirror is cloned from the repository's URL; existing mirrors are refreshed only with
    ``fetch``. Raises ``GitError`` when git fails.
    """
    pinned: list[ManifestRepo] = []
    for spec in config.repos:
        mirror = layout.mirror(spec.slug)
        if fetch or not mirror.exists():
            allow_file = not spec.url.lower().startswith("https://")
            ensure_mirror(spec.url, mirror, allow_file_protocol=allow_file)
        pinned.append(
            ManifestRepo(
                id=spec.id,
                name=spec.name,
                slug=spec.slug,
                url=spec.url,
                sha=head_sha(mirror),
                default_branch=default_branch(mirror),
                issues=spec.issues,
            )
        )
    created = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return Manifest(
        created_at=created,
        extractor_version=EXTRACTOR_VERSION,
        satd_model_sha256=load_default_classifier().source_sha256,
        settings=config.settings,
        criteria=config.criteria,
        max_snapshots=config.max_snapshots,
        max_repos=config.max_repos,
        szz_max_fixes=config.szz_max_fixes,
        repos=tuple(pinned),
    )


def write_manifest(layout: DataLayout, manifest: Manifest) -> None:
    write_json(layout.manifest, manifest.to_json())


def read_manifest(layout: DataLayout) -> Manifest:
    """The manifest of ``layout``; :class:`PipelineError` when it is missing (run ``tdp-ml manifest``)."""
    if not layout.manifest.is_file():
        raise PipelineError(f"{layout.manifest} is missing: run tdp-ml manifest")
    return Manifest.from_json(read_json(layout.manifest))


def manifest_sha256(layout: DataLayout) -> str:
    """SHA-256 of the bytes of ``manifest.json`` (recorded in the dataset summary and model bundles)."""
    return hashlib.sha256(layout.manifest.read_bytes()).hexdigest()
