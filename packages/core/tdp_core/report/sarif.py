"""SARIF 2.1.0 report for code scanning: one alert per Critical or High file.

The log has one run whose driver is ``tdp`` with two rules, ``tdp/critical`` (level ``warning``) and
``tdp/high`` (level ``note``). Each result points at the file through a percent-encoded repo-relative URI
with ``uriBaseId`` ``%SRCROOT%`` and at the start line of its most complex function (1 when it has none).
No fingerprints are set: ``github/codeql-action/upload-sarif`` computes them. The output validates against
the official SARIF 2.1.0 schema (errata 01).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Literal
from urllib.parse import quote

from tdp_core.contracts import BANDS, AnalysisResult, FileExtra, FileScore, Reason
from tdp_core.report.common import TOOL_NAME, format_percent, ranked_files, tool_version
from tdp_core.serialize import to_jsonable

SARIF_VERSION: Final = "2.1.0"
SARIF_SCHEMA_URI: Final = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
)
SRCROOT: Final = "%SRCROOT%"
MAX_MESSAGE_REASONS: Final = 3

SarifBand = Literal["Critical", "High"]
SARIF_BANDS: Final[tuple[SarifBand, ...]] = ("Critical", "High")
RULE_IDS: Final[Mapping[str, str]] = MappingProxyType({"Critical": "tdp/critical", "High": "tdp/high"})
LEVELS: Final[Mapping[str, str]] = MappingProxyType({"Critical": "warning", "High": "note"})

_RULE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {"Critical": "CriticalMaintenancePriority", "High": "HighMaintenancePriority"}
)
_RULE_LIKELIHOOD: Final[Mapping[str, str]] = MappingProxyType(
    {"Critical": "among the most likely", "High": "more likely than most"}
)
_RULE_HELP: Final = (
    "Priority is a percentile within the repository, computed from code structure, change history, "
    "ownership, past bugs, dependencies, tests and admitted debt. The alert message lists the strongest "
    "reasons. Before a risky change, consider adding tests or simplifying the code those reasons point to."
)
_RULE_TAGS: Final = ("maintainability", "technical-debt")


def artifact_uri(path: str) -> str:
    """A repo-relative POSIX path as a relative URI reference: ``"a/old utils.py"`` -> ``"a/old%20utils.py"``.

    Every character except ASCII letters, digits, ``-._~`` and ``/`` is percent-encoded from UTF-8, so ``:``,
    ``?``, ``#`` and ``%`` can never read as a scheme, query, fragment or escape. A leading ``/`` is encoded
    as ``%2F`` so the reference can never be absolute or name a host. Characters that UTF-8 cannot encode
    (lone surrogates) become ``%3F``.
    """
    uri = quote(path, safe="/", errors="replace")
    return "%2F" + uri[1:] if uri.startswith("/") else uri


def region_start_line(extra: FileExtra | None) -> int:
    """The 1-based start line of the file's most complex function, or 1 when it has none.

    The most complex function has the highest cyclomatic complexity, then the highest cognitive complexity,
    then the earliest start line, then the smallest name. ``None`` or no functions gives 1, and a start line
    below 1 is raised to 1.
    """
    if extra is None or not extra.functions:
        return 1
    best = min(extra.functions, key=lambda f: (-f.cyclomatic, -f.cognitive, f.start_line, f.name))
    return max(1, best.start_line)


def to_sarif(result: AnalysisResult, min_band: SarifBand = "High") -> dict[str, Any]:
    """The SARIF 2.1.0 log for ``result`` as fresh Python values (dump with ``json.dumps``).

    ``min_band`` is ``"High"`` (Critical and High files, the default) or ``"Critical"`` (Critical files
    only); any other value, including another case, raises ``ValueError``. Both rules are always listed.
    Results follow the report order (priority descending, then path), with ``rank`` equal to the priority
    clamped to 0..100 and a message giving the band, priority, risk with its range and up to three reasons.
    Each result's ``properties`` hold the priority, band, risk, risk range (``None`` when not finite) and
    debt types; the run's ``properties`` hold the commit, analysis time, model and extractor versions.
    Raises ``ValueError`` when ``analyzed_at`` is naive.
    """
    if min_band not in SARIF_BANDS:
        raise ValueError(f"min_band must be one of {', '.join(SARIF_BANDS)}, got {min_band!r}")
    allowed = SARIF_BANDS[: SARIF_BANDS.index(min_band) + 1]
    results = [
        _result(score, result.extras.get(score.path))
        for score in ranked_files(result.files)
        if score.band in allowed
    ]
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": TOOL_NAME,
                "version": tool_version(),
                "rules": [_rule(band) for band in SARIF_BANDS],
            }
        },
        "results": results,
        "properties": {
            "sha": result.sha,
            "analyzed_at": to_jsonable(result.analyzed_at),
            "model_version": result.model_version,
            "extractor_version": result.extractor_version,
        },
    }
    return {"$schema": SARIF_SCHEMA_URI, "version": SARIF_VERSION, "runs": [run]}


def _band_range(band: SarifBand) -> tuple[int, int]:
    """Lowest and highest priority of ``band`` from ``BANDS``: Critical 90 to 100, High 75 to 89."""
    high = 100
    for name, low in BANDS:
        if name == band:
            return low, high
        high = low - 1
    raise ValueError(band)  # pragma: no cover - SARIF_BANDS is a subset of BANDS


def _rule(band: SarifBand) -> dict[str, Any]:
    low, high = _band_range(band)
    return {
        "id": RULE_IDS[band],
        "name": _RULE_NAMES[band],
        "shortDescription": {"text": f"{band} maintenance priority"},
        "fullDescription": {
            "text": f"Priority {low} to {high} out of 100, a percentile within the repository: the file is "
            f"{_RULE_LIKELIHOOD[band]} to need bug fixes and rework in the next 12 months."
        },
        "defaultConfiguration": {"level": LEVELS[band]},
        "help": {"text": _RULE_HELP},
        "properties": {"tags": list(_RULE_TAGS)},
    }


def _result(score: FileScore, extra: FileExtra | None) -> dict[str, Any]:
    return {
        "ruleId": RULE_IDS[score.band],
        "ruleIndex": SARIF_BANDS.index(score.band),
        "level": LEVELS[score.band],
        "rank": float(min(max(score.priority, 0), 100)),
        "message": {"text": _message(score)},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": artifact_uri(score.path), "uriBaseId": SRCROOT},
                    "region": {"startLine": region_start_line(extra)},
                }
            }
        ],
        "properties": {
            "priority": score.priority,
            "band": score.band,
            "risk": _finite_or_none(score.risk),
            "risk_low": _finite_or_none(score.risk_low),
            "risk_high": _finite_or_none(score.risk_high),
            "debt_types": list(score.debt_types),
        },
    }


def _message(score: FileScore) -> str:
    text = (
        f"{score.band} maintenance priority: {score.priority} of 100. Estimated chance that a bug in its "
        f"current code is fixed within 12 months: {format_percent(score.risk)} "
        f"({format_percent(score.risk_low)} to {format_percent(score.risk_high)})."
    )
    reasons = _reason_texts(score.reasons)
    if len(reasons) == 1:
        text += f" Top reason: {reasons[0]}."
    elif reasons:
        text += f" Top reasons: {'; '.join(reasons)}."
    return text


def _reason_texts(reasons: tuple[Reason, ...]) -> list[str]:
    texts = (" ".join(reason.text.split()).rstrip(".") for reason in reasons)
    return [text for text in texts if text][:MAX_MESSAGE_REASONS]


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
