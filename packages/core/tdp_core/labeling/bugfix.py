"""Decide which commits fix a bug, from linked bug issues or from the commit message.

A commit is a bug fix when it is not a bulk commit, changes at least one source file, and either references a
GitHub issue labelled as a bug (source ``"issue"``) or, without such a link, its message says it fixes
something (source ``"keyword"``). The output is the ``data/bugfix/<slug>.parquet`` layout of plan section 3.4
(``BUGFIX_SCHEMA``): one row for every commit, bug fix or not.

Message rules, refined from the plan after reading the histories of the 23 training repositories:

- The subject decides when it has a fix keyword or a non-fix phrase; the body matters only for a neutral
  subject (squash merges list "fix tests" or "fix typo" under a real fix).
- Non-fix phrases ("fix typo", "fix the failing tests", "fixing docstrings", ...) and declared non-fix commit
  types (``docs:``, ``chore(deps):``, ``feat:``, documentation or tooling gitmoji) veto the keyword source;
  bug gitmoji count as a fix keyword.
- ``Revert "X"`` claims nothing from X (its own body still counts); a revert of a revert or ``Reapply "X"``
  claims X.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import Literal

import pyarrow as pa

from tdp_core.parsing import get_plugin
from tdp_core.parsing.filters import VENDOR_DIRS

BUGFIX_SCHEMA = pa.schema(
    [
        ("sha", pa.string()),
        ("is_bugfix", pa.bool_()),
        ("source", pa.string()),  # "issue" | "keyword" | "none"
        ("issue_number", pa.int32()),  # null unless source == "issue"
        ("issue_created_at", pa.timestamp("s", tz="UTC")),  # null unless source == "issue"
    ]
)

BugfixSource = Literal["issue", "keyword", "none"]
BUGFIX_SOURCES: tuple[BugfixSource, ...] = ("issue", "keyword", "none")

# Fix keywords (the plan's rule 3).
_POSITIVE = re.compile(
    r"\b(fix(e[sd])?|bug(s|fix)?|defect|crash(es|ed)?|regression|fault|incorrect|wrong|broken|hotfix)\b",
    re.IGNORECASE,
)
# "fix" + up to two filler words + a target that is not a defect in the code. Keeps the plan's targets.
_NOT_A_FIX = re.compile(
    r"\bfix(?:es|ed|ing)?\s+"
    r"(?:(?:a|an|the|some|minor|small|few|more|several|various|remaining|couple\s+of|failing|flaky|broken)\s+)"
    r"{0,2}"
    r"(?:typos?|lint(?:ing)?|docs?|documentation|formatting|format|style|spelling|whitespace|tests?|ci|build"
    r"|readme|changelog|docstrings?|grammar|wording|mypy|flake8|pre-commit|typing(?![.\w])|type\s+hints?"
    r"|examples?|(?:broken|dead)\s+links?)\b",
    re.IGNORECASE,
)
# Conventional-commit types and gitmoji that declare a commit is not a fix (docs, tests, CI, chores, style,
# build, refactoring, features, version bumps, translations, lint, config, releases).
_DECLARED_NON_FIX = re.compile(
    r"^\s*(?:(?:docs?|ci|tests?|chore|style|build|refactor|feat)(?:\([^)\n]*\))?!?:"
    r"|(?:\U0001F4DD|\u270F\uFE0F?|\U0001F3A8|\u2705|\U0001F477|\u2B06\uFE0F?|\U0001F310|\U0001F6A8"
    r"|\u267B\uFE0F?|\u2728|\U0001F527|\U0001F516)"
    r"|:(?:memo|pencil2|art|white_check_mark|construction_worker|arrow_up|globe_with_meridians"
    r"|rotating_light|recycle|sparkles|wrench|bookmark):)",
    re.IGNORECASE,
)
# Gitmoji for bug fixes: bug, ambulance (hotfix), adhesive bandage (simple fix).
_BUG_MARKER = re.compile(
    r"^\s*(?:\U0001F41B|\U0001F691\uFE0F?|\U0001FA79|:(?:bug|ambulance|adhesive_bandage):)"
)
_REVERT = re.compile(r'^\s*revert\s+"(?P<inner>.*)"', re.IGNORECASE)  # greedy: inner runs to the last quote
_REAPPLY = re.compile(r'^\s*reapply\s+"(?P<inner>.*)"', re.IGNORECASE)
_REVERTS_LINE = re.compile(r"^\s*this reverts commit [0-9a-f]{7,64}\b.*$", re.IGNORECASE | re.MULTILINE)

# Issue references.
_URL_REF = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+)/issues/(\d{1,7})\b", re.IGNORECASE)
_CROSS_REPO_REF = re.compile(r"([\w.-]+)/([\w.-]+)#(\d{1,7})\b")
_HASH_REF = re.compile(r"(?<![\w&])#(\d{1,7})\b")
_GH_REF = re.compile(r"\bGH-(\d{1,7})\b", re.IGNORECASE)

_ISSUE_COLUMNS = ("number", "created_at", "is_bug")
_Row = tuple[bool, str, int | None, datetime | None]
_NOT_A_BUGFIX: _Row = (False, "none", None, None)


def default_is_source(path: str) -> bool:
    """True for a file a language plugin handles (``.py``) that is neither a test file nor vendored.

    Test files follow the plugin's rule (``test_*.py``, ``*_test.py``, ``conftest.py``, files under ``tests/``
    or ``test/``); vendored files have a path component in ``VENDOR_DIRS``. ``path`` is repo-relative POSIX.
    It matches the files the ranker scores, so SZZ should use the same predicate.
    """
    plugin = get_plugin(path)
    if plugin is None or plugin.is_test_file(path):
        return False
    return not any(part in VENDOR_DIRS for part in path.split("/"))


def source_commits(
    commits: pa.Table,
    file_changes: pa.Table,
    is_source: Callable[[str], bool] = default_is_source,
) -> frozenset[str]:
    """Shas of the non-bulk commits in ``commits`` that change at least one source file.

    A ``file_changes`` row counts when it is not binary, has ``added + deleted > 0`` and ``is_source`` holds
    for its path or, for a rename, its old path. Pure renames (0/0), mode changes and binary changes do not
    count. Rows whose sha is not in ``commits`` are ignored. ``is_source`` is called once per distinct path.
    """
    non_bulk = {
        sha
        for sha, bulk in zip(
            commits.column("sha").to_pylist(), commits.column("is_bulk").to_pylist(), strict=True
        )
        if not bulk
    }
    memo: dict[str, bool] = {}

    def source(path: str | None) -> bool:
        if path is None:
            return False
        hit = memo.get(path)
        if hit is None:
            hit = memo[path] = bool(is_source(path))
        return hit

    eligible: set[str] = set()
    for sha, path, old_path, added, deleted, binary in zip(
        file_changes.column("sha").to_pylist(),
        file_changes.column("path").to_pylist(),
        file_changes.column("old_path").to_pylist(),
        file_changes.column("added").to_pylist(),
        file_changes.column("deleted").to_pylist(),
        file_changes.column("is_binary").to_pylist(),
        strict=True,
    ):
        if sha in eligible or sha not in non_bulk or binary or (added or 0) + (deleted or 0) <= 0:
            continue
        if source(path) or source(old_path):
            eligible.add(sha)
    return frozenset(eligible)


def effective_message(message: str) -> tuple[str, str]:
    """``(subject, body)`` that the commit claims for itself, both stripped.

    The subject is the first line, the body the rest. Quoted reverts are unwrapped: ``Revert "X"`` undoes X,
    so its subject is ``""`` (the quoted words are not its own claim) while its body still counts (for
    example "X caused a regression"). ``Revert "Revert "X""`` and ``Reapply "X"`` re-apply X and claim X.
    For any revert, "This reverts commit <sha>" lines are removed from the body. ``""`` gives ``("", "")``.
    """
    subject, _, body = message.partition("\n")
    depth = 0
    while True:
        revert = _REVERT.match(subject)
        if revert:
            depth += 1
            subject = revert.group("inner")
            continue
        reapply = _REAPPLY.match(subject)
        if reapply:
            subject = reapply.group("inner")
            continue
        break
    if depth:
        body = _REVERTS_LINE.sub("", body)
        if depth % 2:
            subject = ""
    return subject.strip(), body.strip()


def _repo_key(repo: str | None) -> tuple[str, str] | None:
    if repo is None:
        return None
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")
    return owner.lower(), name.lower()


def _references(text: str, repo: tuple[str, str] | None) -> tuple[int, ...]:
    refs: set[int] = set()
    if repo is not None:
        for m in _URL_REF.finditer(text):
            if (m.group(1).lower(), m.group(2).lower()) == repo:
                refs.add(int(m.group(3)))
        for m in _CROSS_REPO_REF.finditer(text):
            if (m.group(1).lower(), m.group(2).lower()) == repo:
                refs.add(int(m.group(3)))
    text = _CROSS_REPO_REF.sub(" ", _URL_REF.sub(" ", text))
    refs.update(int(m.group(1)) for m in _HASH_REF.finditer(text))
    refs.update(int(m.group(1)) for m in _GH_REF.finditer(text))
    refs.discard(0)
    return tuple(sorted(refs))


def issue_references(message: str, repo: str | None = None) -> tuple[int, ...]:
    """Issue numbers a commit message refers to, sorted and unique.

    Looks at the effective message (see ``effective_message``, so a revert does not inherit the references of
    the commit it reverts) for ``#N`` (not inside a word or an HTML entity such as ``&#39;``; ``fixes #N``,
    ``closes #N`` and ``resolves #N`` are covered) and ``GH-N``. References into another repository
    (``owner/name#N``, ``https://github.com/owner/name/issues/N``) are ignored unless ``repo``
    (``"owner/name"``, case insensitive) names this repository, in which case they count. 0 is never a
    reference. Raises ``ValueError`` when ``repo`` is not ``owner/name``.
    """
    subject, body = effective_message(message)
    return _references(f"{subject}\n{body}", _repo_key(repo))


def _keyword(subject: str, body: str) -> bool:
    if _DECLARED_NON_FIX.match(subject):
        return False
    if _BUG_MARKER.match(subject):
        return True
    if _NOT_A_FIX.search(subject):
        return False
    if _POSITIVE.search(subject):
        return True
    if _NOT_A_FIX.search(body):
        return False
    return bool(_POSITIVE.search(body))


def keyword_bugfix(message: str) -> bool:
    """True when the message says the commit fixes something (the keyword rule, ignoring eligibility).

    On the effective message (see ``effective_message``): a declared non-fix type (``docs:``, ``chore(x):``,
    ``feat:``, ... or a documentation, test, tooling or feature gitmoji) gives False; a bug gitmoji gives
    True; otherwise the subject decides when it contains a non-fix phrase ("fix typo", "fix the failing
    tests") or a fix keyword (fix, bug, crash, regression, wrong, broken, ...), and only a neutral subject
    lets the body decide the same way. For a one-line message this is the plan's keyword rule with a wider
    non-fix list.
    """
    return _keyword(*effective_message(message))


def _bug_issues(issues: pa.Table | None) -> dict[int, datetime]:
    """``number -> created_at`` of the bug issues in ``issues``."""
    if issues is None:
        return {}
    missing = [c for c in _ISSUE_COLUMNS if c not in issues.column_names]
    if missing:
        raise ValueError(f"issues table lacks columns: {', '.join(missing)}")
    bugs: dict[int, datetime] = {}
    for number, created, is_bug in zip(
        issues.column("number").to_pylist(),
        issues.column("created_at").to_pylist(),
        issues.column("is_bug").to_pylist(),
        strict=True,
    ):
        if is_bug and number is not None and created is not None:
            bugs[int(number)] = created
    return bugs


def classify_bugfix(
    commits: pa.Table,
    file_changes: pa.Table,
    issues: pa.Table | None = None,
    is_source: Callable[[str], bool] = default_is_source,
    *,
    repo: str | None = None,
) -> pa.Table:
    """Classify every commit as a bug fix or not. Returns a ``BUGFIX_SCHEMA`` table.

    Output row i describes input row i of ``commits`` (``COMMITS_SCHEMA``), so the table has one row per
    commit in input order, bug fix or not. Bulk commits and commits that change no source file (see
    ``source_commits``) are never bug fixes. Otherwise:

    1. ``source="issue"`` when the message references (``issue_references``) an issue whose ``is_bug`` is
       true in ``issues`` (columns ``number``, ``created_at``, ``is_bug``, as from ``fetch_issues``). Of
       several, the earliest created one (ties: smaller number) gives ``issue_number`` and
       ``issue_created_at``; the latter is meant as SZZ's ``not_after``.
    2. ``source="keyword"`` when there is no linked bug issue and ``keyword_bugfix(message)`` holds. A
       message that references only non-bug issues may still be a keyword fix (plan rule 3).
    3. Otherwise ``source="none"`` and ``is_bugfix`` false.

    ``issue_number`` and ``issue_created_at`` are null unless ``source == "issue"``. ``issues=None`` and an
    empty issues table behave the same (keyword source only, no network). ``repo`` (``"owner/name"``) also
    lets same-repository issue URLs and ``owner/name#N`` references count. Raises ``ValueError`` when
    ``issues`` lacks a required column or ``repo`` is malformed. Runs in one pass over each table.
    """
    repo_key = _repo_key(repo)
    eligible = source_commits(commits, file_changes, is_source)
    bug_created = _bug_issues(issues)
    shas = commits.column("sha").to_pylist()
    messages = commits.column("message").to_pylist()
    subjects = commits.column("subject").to_pylist()

    rows: list[_Row] = []
    for sha, message, subject_line in zip(shas, messages, subjects, strict=True):
        if sha not in eligible:
            rows.append(_NOT_A_BUGFIX)
            continue
        subject, body = effective_message(message or subject_line or "")
        if bug_created:
            refs = _references(f"{subject}\n{body}", repo_key)
            linked = sorted((bug_created[n], n) for n in refs if n in bug_created)
            if linked:
                created, number = linked[0]
                rows.append((True, "issue", number, created))
                continue
        rows.append((True, "keyword", None, None) if _keyword(subject, body) else _NOT_A_BUGFIX)

    return pa.Table.from_pydict(
        {
            "sha": shas,
            "is_bugfix": [r[0] for r in rows],
            "source": [r[1] for r in rows],
            "issue_number": [r[2] for r in rows],
            "issue_created_at": [r[3] for r in rows],
        },
        schema=BUGFIX_SCHEMA,
    )
