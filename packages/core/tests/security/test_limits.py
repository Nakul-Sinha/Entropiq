import copy
import json
import random
import urllib.error
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pytest

from tdp_core.labeling.github_issues import HttpResponse
from tdp_core.mining.git import GitError
from tdp_core.mining.mirror import ensure_mirror
from tdp_core.security.limits import (
    DEFAULT_MAX_REPO_KB,
    GITHUB_API_URL,
    RepoCheckUnavailableError,
    RepoNotFoundError,
    RepoTooLargeError,
    check_mirror_size,
    check_repo_size,
)
from tdp_core.security.urls import DEFAULT_ALLOWED_HOSTS, RepoRef, UnsafeRepoError, validate_repo_url
from tdp_core.testing import RepoBuilder

TOKEN = "s3cret"


class FakeTransport:
    """Records each request and answers with one scripted response or exception."""

    def __init__(self, answer: HttpResponse | Exception) -> None:
        self.answer = answer
        self.requests: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.requests.append((url, dict(headers), timeout))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _ok(body: object, headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(200, dict(headers or {}), json.dumps(body).encode())


def _ref(url: str = "https://github.com/psf/requests") -> RepoRef:
    return validate_repo_url(url, DEFAULT_ALLOWED_HOSTS, resolve_dns=False)


def _never_called(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    raise AssertionError("no request expected")


# ---------------------------------------------------------------------------------------------- GitHub API
def test_public_repository_size_is_returned() -> None:
    transport = FakeTransport(_ok({"size": 1234, "private": False, "visibility": "public"}))

    assert check_repo_size(_ref(), TOKEN, 2000, transport=transport, timeout=7.5) == 1234

    assert len(transport.requests) == 1
    url, headers, timeout = transport.requests[0]
    assert url == f"{GITHUB_API_URL}/repos/psf/requests" == "https://api.github.com/repos/psf/requests"
    assert headers == {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "entropiq-tdp",
        "Authorization": f"Bearer {TOKEN}",
    }
    assert timeout == 7.5


@pytest.mark.parametrize("token", [None, ""])
def test_anonymous_request_has_no_authorization_header(token: str | None) -> None:
    transport = FakeTransport(_ok({"size": 1}))

    assert check_repo_size(_ref(), token, 10, transport=transport) == 1

    _, headers, timeout = transport.requests[0]
    assert "Authorization" not in headers
    assert timeout == 10.0


def test_size_equal_to_the_limit_passes() -> None:
    assert check_repo_size(_ref(), None, 500, transport=FakeTransport(_ok({"size": 500}))) == 500


def test_size_over_the_limit_raises() -> None:
    with pytest.raises(RepoTooLargeError) as info:
        check_repo_size(_ref(), None, 1_000, transport=FakeTransport(_ok({"size": 2_000})))

    err = info.value
    assert isinstance(err, UnsafeRepoError)
    assert isinstance(err, ValueError)
    assert (err.size_kb, err.max_kb) == (2_000, 1_000)
    assert str(err) == "repository is 2,000 KB, over the 1,000 KB limit"


def test_zero_limit_allows_only_empty_repositories() -> None:
    assert check_repo_size(_ref(), None, 0, transport=FakeTransport(_ok({"size": 0}))) == 0
    with pytest.raises(RepoTooLargeError):
        check_repo_size(_ref(), None, 0, transport=FakeTransport(_ok({"size": 1})))


def test_other_hosts_are_skipped_without_a_request() -> None:
    ref = _ref("https://gitlab.com/Group/Repo")
    assert check_repo_size(ref, TOKEN, DEFAULT_MAX_REPO_KB, transport=_never_called) == -1


def test_owner_and_name_are_quoted_in_the_api_path() -> None:
    transport = FakeTransport(_ok({"size": 3}))
    check_repo_size(_ref("https://github.com/a-b_c.d/.github"), None, 10, transport=transport)
    assert transport.requests[0][0] == "https://api.github.com/repos/a-b_c.d/.github"


NOT_FOUND = [
    HttpResponse(404, {}, b'{"message": "Not Found"}'),
    HttpResponse(410, {}, b'{"message": "Repository access blocked"}'),
    HttpResponse(451, {}, b'{"message": "Repository access blocked"}'),
    _ok({"size": 10, "private": True, "visibility": "private"}),
    _ok({"size": 10, "private": False, "visibility": "internal"}),
]


@pytest.mark.parametrize("response", NOT_FOUND)
def test_missing_or_non_public_repositories_look_the_same(response: HttpResponse) -> None:
    with pytest.raises(RepoNotFoundError) as info:
        check_repo_size(_ref(), TOKEN, DEFAULT_MAX_REPO_KB, transport=FakeTransport(response))

    assert str(info.value) == "repository psf/requests was not found on github.com or is not public"
    assert isinstance(info.value, UnsafeRepoError)


UNAVAILABLE = [
    (HttpResponse(403, {"x-ratelimit-remaining": "0"}, b"{}"), 403, "rate limit"),
    (HttpResponse(429, {}, b""), 429, "rate limit"),
    (HttpResponse(403, {}, b'{"message": "API rate limit exceeded for 1.2.3.4."}'), 403, "rate limit"),
    (HttpResponse(403, {}, b'{"message": "You have exceeded a secondary rate limit"}'), 403, "rate limit"),
    (HttpResponse(403, {"x-ratelimit-remaining": "12"}, b'{"message": "Forbidden"}'), 403, "HTTP 403"),
    (HttpResponse(401, {}, b'{"message": "Bad credentials"}'), 401, "HTTP 401"),
    (HttpResponse(500, {}, b""), 500, "HTTP 500"),
    (HttpResponse(502, {}, b"<html>bad gateway</html>"), 502, "HTTP 502"),
    (HttpResponse(301, {"location": "https://elsewhere.example/"}, b""), 301, "HTTP 301"),
]


@pytest.mark.parametrize(("response", "status", "fragment"), UNAVAILABLE)
def test_unusable_answers_mean_the_check_is_unavailable(
    response: HttpResponse, status: int, fragment: str
) -> None:
    with pytest.raises(RepoCheckUnavailableError) as info:
        check_repo_size(_ref(), TOKEN, DEFAULT_MAX_REPO_KB, transport=FakeTransport(response))

    assert info.value.status == status
    assert fragment in str(info.value)
    assert "psf/requests" in str(info.value)
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize(
    "exc", [TimeoutError("timed out"), ConnectionResetError("reset"), urllib.error.URLError("down")]
)
def test_network_failure_has_no_status(exc: Exception) -> None:
    with pytest.raises(RepoCheckUnavailableError) as info:
        check_repo_size(_ref(), TOKEN, DEFAULT_MAX_REPO_KB, transport=FakeTransport(exc))

    assert info.value.status is None
    assert "could not reach the GitHub API" in str(info.value)
    assert type(exc).__name__ in str(info.value)
    assert info.value.__cause__ is None
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff\xfe\x00garbage",
        b"[1, 2]",
        b"{}",
        b'{"size": true}',
        b'{"size": -1}',
        b'{"size": "12"}',
        b'{"size": 1.5}',
        b'{"size": null}',
        pytest.param(b"[" * 100_000, id="deeply-nested"),
    ],
)
def test_bodies_without_a_valid_size_are_rejected(body: bytes) -> None:
    with pytest.raises(RepoCheckUnavailableError, match="unexpected GitHub API response") as info:
        check_repo_size(
            _ref(), TOKEN, DEFAULT_MAX_REPO_KB, transport=FakeTransport(HttpResponse(200, {}, body))
        )

    assert info.value.status == 200
    assert TOKEN not in str(info.value)


def test_negative_limit_is_a_value_error_before_any_request() -> None:
    with pytest.raises(ValueError, match="max_kb"):
        check_repo_size(_ref(), TOKEN, -1, transport=_never_called)


def test_errors_rebuild_from_their_args() -> None:
    too_large = RepoTooLargeError(2_000, 1_000)
    rebuilt = type(too_large)(*too_large.args)
    assert str(rebuilt) == str(too_large)
    assert (rebuilt.size_kb, rebuilt.max_kb) == (2_000, 1_000)

    unavailable = RepoCheckUnavailableError("GitHub API returned HTTP 502", status=502)
    assert str(type(unavailable)(*unavailable.args)) == str(unavailable)
    copied = copy.deepcopy(unavailable)
    assert copied.status == 502
    assert str(copied) == str(unavailable)

    not_found = RepoNotFoundError("repository a/b was not found on github.com or is not public")
    assert str(type(not_found)(*not_found.args)) == str(not_found)


# ---------------------------------------------------------------------------------------------- mirrors
def test_mirror_size_counts_loose_and_packed_objects(tmp_path: Path, t0: datetime) -> None:
    upstream = RepoBuilder(tmp_path / "upstream")
    upstream.commit({"blob.bin": random.Random(42).randbytes(300_000)}, "big blob", t0)
    mirror = ensure_mirror(upstream.path.as_uri(), tmp_path / "m.git", allow_file_protocol=True)

    size = check_mirror_size(mirror, DEFAULT_MAX_REPO_KB)

    assert 250 <= size <= 1000
    assert check_mirror_size(mirror, size) == size
    with pytest.raises(RepoTooLargeError) as info:
        check_mirror_size(mirror, size - 1)
    assert (info.value.size_kb, info.value.max_kb) == (size, size - 1)


def test_mirror_size_accepts_a_relative_path(
    tmp_path: Path, t0: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = RepoBuilder(tmp_path / "upstream")
    upstream.commit({"a.py": "x = 1\n"}, "one", t0)
    mirror = ensure_mirror(upstream.path.as_uri(), tmp_path / "m.git", allow_file_protocol=True)
    expected = check_mirror_size(mirror, DEFAULT_MAX_REPO_KB)

    monkeypatch.chdir(tmp_path)

    assert check_mirror_size(Path("m.git"), DEFAULT_MAX_REPO_KB) == expected


def test_mirror_size_of_a_plain_directory_is_a_git_error(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitError):
        check_mirror_size(plain, DEFAULT_MAX_REPO_KB)
    with pytest.raises(GitError):
        check_mirror_size(tmp_path / "missing", DEFAULT_MAX_REPO_KB)


def test_mirror_size_rejects_a_negative_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_kb"):
        check_mirror_size(tmp_path, -1)


def test_mirror_size_sums_loose_packed_and_garbage_kib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[list[str]] = []

    def fake_run_git(args: list[str], cwd: Path | None, **kwargs: object) -> bytes:
        seen.append(list(args))
        return (
            b"count: 3\nsize: 10\nin-pack: 5\npacks: 1\nsize-pack: 20\nprune-packable: 0\ngarbage: 1\n"
            b"size-garbage: 2\nalternate: /elsewhere/objects\nnot a stat line\n"
        )

    monkeypatch.setattr("tdp_core.security.limits.run_git", fake_run_git)

    assert check_mirror_size(tmp_path, 32) == 32
    assert seen == [[f"--git-dir={tmp_path}", "count-objects", "-v"]]
    with pytest.raises(RepoTooLargeError):
        check_mirror_size(tmp_path, 31)
