import json
import threading
from collections import deque
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tdp_core.labeling.github_issues import (
    ISSUES_SCHEMA,
    GitHubError,
    HttpResponse,
    fetch_issues,
    is_bug_issue,
    is_bug_label,
    is_not_bug_label,
    issues_table,
    read_issues,
    urllib_transport,
)

TOKEN = "ghp_secret_test_token_value"
NOW = 1_700_000_000.0
FIRST = "https://api.github.com/repos/psf/requests/issues?state=all&per_page=100&sort=created&direction=asc"
SECOND = "https://api.github.com/repositories/1/issues?state=all&per_page=100&after=abc"


class FakeTransport:
    """Serves queued responses (or raises queued exceptions) per URL and records every request."""

    def __init__(self, script: Mapping[str, list[HttpResponse | Exception]] | None = None) -> None:
        self.queues: dict[str, deque[HttpResponse | Exception]] = {
            url: deque(items) for url, items in (script or {}).items()
        }
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        self.calls.append((url, dict(headers)))
        queue = self.queues.get(url)
        if not queue:
            raise AssertionError(f"unexpected request to {url}")
        item = queue.popleft() if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _clock() -> float:
    return NOW


def _issue(number: int, labels: list[Any] | None = None, *, pr: bool = False, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "number": number,
        "created_at": f"2020-01-{1 + number % 28:02d}T10:00:00Z",
        "closed_at": None,
        "labels": [{"name": lb} for lb in (labels or [])],
    }
    if pr:
        item["pull_request"] = {"url": "x"}
    item.update(extra)
    return item


def _ok(items: list[dict[str, Any]], headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(200, dict(headers or {}), json.dumps(items).encode())


def _status(code: int, headers: Mapping[str, str] | None = None, message: str = "nope") -> HttpResponse:
    return HttpResponse(code, dict(headers or {}), json.dumps({"message": message}).encode())


def _fetch(transport: FakeTransport, sleep: FakeSleep | None = None, **kw: Any) -> pa.Table:
    return fetch_issues(
        "psf",
        "requests",
        kw.pop("token", TOKEN),
        transport=transport,
        sleep=sleep or FakeSleep(),
        clock=_clock,
        **kw,
    )


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize(
    "label",
    [
        "bug",
        "Bug :beetle:",
        "type: bug",
        "kind/bug",
        "T: bug",
        "C: crash",
        "regression",
        "Issue Type: Bug Report",
        "Issue Type: Regression",
        "bug:minor",
        "upstream bug",
        "Defect",
    ],
)
def test_bug_labels(label: str) -> None:
    assert is_bug_label(label)


@pytest.mark.parametrize(
    "label",
    [
        "debug",
        "debugger",
        "PR Type: Bugfix",
        "bugfix",
        "enhancement",
        "Status: Not a Bug",
        "R: not a bug",
        "Question/Not a bug",
        "resolution: not a bug",
        "non-bug",
        "not-a-bug",
        "nobug",
    ],
)
def test_not_bug_labels(label: str) -> None:
    assert not is_bug_label(label)


def test_not_a_bug_verdict_overrides_bug_label() -> None:
    assert is_not_bug_label("resolution: not a bug")
    assert not is_not_bug_label("type: bug")
    assert is_bug_issue(["type: bug", "help wanted"])
    assert not is_bug_issue(["type: bug", "resolution: not a bug"])
    assert not is_bug_issue([])


# ------------------------------------------------------------------ issues_table
def test_issues_table_parses_items() -> None:
    items: list[dict[str, Any]] = [
        {
            "number": 5,
            "created_at": "2020-02-03T04:05:06Z",
            "closed_at": "2020-03-01T00:00:00+02:00",
            "labels": [{"name": "Bug"}, "needs info", None, {"name": ""}, {"color": "red"}],
        },
        {"number": 2, "created_at": "2019-12-31T23:59:59Z", "closed_at": None, "labels": []},
        {"number": 3, "created_at": "2020-01-01T00:00:00Z", "pull_request": {}, "labels": []},
        {"number": 2, "created_at": "2020-01-01T00:00:00Z", "closed_at": None, "labels": ["crash"]},
        {"number": "7", "created_at": "2020-01-01T00:00:00Z"},
        {"number": 8, "created_at": "not a date"},
        {"number": True, "created_at": "2020-01-01T00:00:00Z"},
    ]
    table = issues_table(items)
    assert table.schema.equals(ISSUES_SCHEMA)
    assert table.column("number").to_pylist() == [2, 5]
    rows = table.to_pylist()
    assert rows[0]["created_at"] == datetime(2020, 1, 1, tzinfo=UTC)  # duplicate: the last one wins
    assert rows[0]["labels"] == ["crash"]
    assert rows[0]["is_bug"] is True
    assert rows[0]["closed_at"] is None
    assert rows[1]["created_at"] == datetime(2020, 2, 3, 4, 5, 6, tzinfo=UTC)
    assert rows[1]["closed_at"] == datetime(2020, 2, 29, 22, tzinfo=UTC)
    assert rows[1]["labels"] == ["Bug", "needs info"]


def test_issues_table_empty() -> None:
    table = issues_table([])
    assert table.num_rows == 0
    assert table.schema.equals(ISSUES_SCHEMA)


# ------------------------------------------------------------------ pagination
def test_pagination_follows_link_header() -> None:
    page1 = [_issue(n, ["bug"] if n % 10 == 0 else [], pr=n % 3 == 0) for n in range(100, 0, -1)]
    page2 = [_issue(101, ["enhancement"]), _issue(102, pr=True)]
    transport = FakeTransport(
        {
            FIRST: [_ok(page1, {"link": f'<{SECOND}>; rel="next", <{SECOND}&x=1>; rel="last"'})],
            SECOND: [_ok(page2)],
        }
    )
    table = _fetch(transport)
    numbers = table.column("number").to_pylist()
    assert table.schema.equals(ISSUES_SCHEMA)
    assert numbers == sorted(n for n in range(1, 102) if n % 3 != 0)
    assert sum(table.column("is_bug").to_pylist()) == len([n for n in range(10, 101, 10) if n % 3 != 0])
    assert [url for url, _ in transport.calls] == [FIRST, SECOND]
    first_query = transport.calls[0][0].split("?", 1)[1].split("&")
    assert sorted(first_query) == ["direction=asc", "per_page=100", "sort=created", "state=all"]
    for url, headers in transport.calls:
        assert TOKEN not in url
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert headers["Accept"] == "application/vnd.github+json"
        assert headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_anonymous_requests_send_no_authorization() -> None:
    transport = FakeTransport({FIRST: [_ok([_issue(1)])]})
    _fetch(transport, token=None)
    _fetch(transport, token="")
    assert all("Authorization" not in headers for _, headers in transport.calls)


def test_next_link_to_another_host_is_refused() -> None:
    evil = "https://evil.example.com/issues?after=abc"
    transport = FakeTransport({FIRST: [_ok([_issue(1)], {"link": f'<{evil}>; rel="next"'})]})
    with pytest.raises(GitHubError, match="another host"):
        _fetch(transport)
    assert [url for url, _ in transport.calls] == [FIRST]


def test_max_pages_gives_partial_result_without_cache(tmp_path: Path) -> None:
    cache = tmp_path / "issues" / "psf__requests.parquet"
    transport = FakeTransport({FIRST: [_ok([_issue(1)], {"link": f'<{SECOND}>; rel="next"'})]})
    table = _fetch(transport, cache_path=cache, max_pages=1)
    assert table.column("number").to_pylist() == [1]
    assert not cache.exists()
    assert len(transport.calls) == 1


# ------------------------------------------------------------------ rate limits and retries
def test_used_up_rate_limit_waits_before_next_page() -> None:
    headers = {
        "link": f'<{SECOND}>; rel="next"',
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": str(int(NOW) + 30),
    }
    transport = FakeTransport({FIRST: [_ok([_issue(1)], headers)], SECOND: [_ok([_issue(2)])]})
    sleep = FakeSleep()
    table = _fetch(transport, sleep)
    assert table.num_rows == 2
    assert sleep.waits == [31.0]


def test_primary_rate_limit_403_waits_and_retries_same_url() -> None:
    limited = _status(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW) + 100)})
    transport = FakeTransport({FIRST: [limited, _ok([_issue(1)])]})
    sleep = FakeSleep()
    table = _fetch(transport, sleep, max_retries=0)
    assert table.num_rows == 1
    assert sleep.waits == [101.0]
    assert [url for url, _ in transport.calls] == [FIRST, FIRST]


def test_primary_rate_limit_gives_up_after_three_waits() -> None:
    limited = _status(403, {"x-ratelimit-remaining": "0"})
    transport = FakeTransport({FIRST: [limited]})
    sleep = FakeSleep()
    with pytest.raises(GitHubError) as info:
        _fetch(transport, sleep)
    assert info.value.status == 403
    assert sleep.waits == [60.0, 60.0, 60.0]


def test_secondary_rate_limit_honours_retry_after() -> None:
    transport = FakeTransport({FIRST: [_status(429, {"retry-after": "7"}), _ok([_issue(1)])]})
    sleep = FakeSleep()
    assert _fetch(transport, sleep).num_rows == 1
    assert sleep.waits == [7.0]


def test_secondary_rate_limit_403_without_retry_after() -> None:
    blocked = _status(403, message="You have exceeded a secondary rate limit.")
    transport = FakeTransport({FIRST: [blocked, _ok([_issue(1)])]})
    sleep = FakeSleep()
    assert _fetch(transport, sleep).num_rows == 1
    assert sleep.waits == [60.0]


def test_network_errors_are_retried_with_backoff() -> None:
    transport = FakeTransport({FIRST: [OSError("reset"), ConnectionError("dropped"), _ok([_issue(1)])]})
    sleep = FakeSleep()
    assert _fetch(transport, sleep).num_rows == 1
    assert sleep.waits == [5.0, 10.0]


def test_persistent_network_error_raises_after_max_retries() -> None:
    transport = FakeTransport({FIRST: [TimeoutError("timed out")]})
    sleep = FakeSleep()
    with pytest.raises(GitHubError, match="after 4 attempts"):
        _fetch(transport, sleep, max_retries=3)
    assert sleep.waits == [5.0, 10.0, 20.0]
    assert len(transport.calls) == 4


def test_server_error_then_success() -> None:
    transport = FakeTransport({FIRST: [_status(502), _ok([_issue(1)])]})
    sleep = FakeSleep()
    assert _fetch(transport, sleep).num_rows == 1
    assert sleep.waits == [5.0]


def test_truncated_body_is_retried() -> None:
    truncated = HttpResponse(200, {}, b'[{"number": 1, "creat')
    not_a_list = HttpResponse(200, {}, b'{"message": "odd"}')
    transport = FakeTransport({FIRST: [truncated, not_a_list, _ok([_issue(1)])]})
    sleep = FakeSleep()
    assert _fetch(transport, sleep).num_rows == 1
    assert sleep.waits == [5.0, 10.0]


@pytest.mark.parametrize("code", [401, 404, 422])
def test_client_errors_raise_without_retry(code: int) -> None:
    transport = FakeTransport({FIRST: [_status(code, message="Bad credentials")]})
    sleep = FakeSleep()
    with pytest.raises(GitHubError) as info:
        _fetch(transport, sleep)
    assert info.value.status == code
    assert "Bad credentials" in str(info.value)
    assert TOKEN not in str(info.value)
    assert len(transport.calls) == 1
    assert sleep.waits == []


def test_issues_disabled_gives_empty_table() -> None:
    transport = FakeTransport({FIRST: [_status(410, message="Issues are disabled for this repo")]})
    table = _fetch(transport)
    assert table.num_rows == 0
    assert table.schema.equals(ISSUES_SCHEMA)


# ------------------------------------------------------------------ cache
def test_cache_is_written_and_reused(tmp_path: Path) -> None:
    cache = tmp_path / "issues" / "psf__requests.parquet"
    first = _fetch(FakeTransport({FIRST: [_ok([_issue(2, ["bug"]), _issue(1)])]}), cache_path=cache)
    assert cache.exists()
    assert list(cache.parent.iterdir()) == [cache]  # no .tmp file left behind

    offline = FakeTransport({FIRST: [OSError("offline")]})
    second = _fetch(offline, cache_path=cache)
    assert second.equals(first)
    assert offline.calls == []

    refreshed = _fetch(FakeTransport({FIRST: [_ok([_issue(3)])]}), cache_path=cache, refresh=True)
    assert refreshed.column("number").to_pylist() == [3]
    assert read_issues(cache).column("number").to_pylist() == [3]


def test_corrupt_cache_is_refetched(tmp_path: Path) -> None:
    cache = tmp_path / "psf__requests.parquet"
    cache.write_bytes(b"not parquet")
    table = _fetch(FakeTransport({FIRST: [_ok([_issue(1)])]}), cache_path=cache)
    assert table.num_rows == 1
    assert read_issues(cache).equals(table)


def test_read_issues_recomputes_is_bug(tmp_path: Path) -> None:
    path = tmp_path / "old.parquet"
    stale = pa.Table.from_pydict(
        {
            "number": [2, 1],
            "created_at": [datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)],
            "closed_at": [None, None],
            "labels": [["Question/Not a bug"], ["kind/bug"]],
            "is_bug": [True, False],
        },
        schema=ISSUES_SCHEMA,
    )
    pq.write_table(stale, path)
    table = read_issues(path)
    assert table.schema.equals(ISSUES_SCHEMA)
    assert table.column("number").to_pylist() == [1, 2]
    assert table.column("is_bug").to_pylist() == [True, False]


def test_read_issues_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"number": [1]}), path)
    with pytest.raises(ValueError, match="created_at"):
        read_issues(path)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    ("owner", "name"), [("..", "x"), ("psf", "../x"), ("a b", "x"), ("", "x"), ("psf", ".")]
)
def test_invalid_names_are_rejected(owner: str, name: str) -> None:
    with pytest.raises(ValueError):
        fetch_issues(owner, name, None, transport=FakeTransport())


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        fetch_issues("psf", "requests", None, transport=FakeTransport(), max_pages=0)
    with pytest.raises(ValueError):
        fetch_issues("psf", "requests", None, transport=FakeTransport(), max_retries=-1)
    with pytest.raises(ValueError):
        fetch_issues("psf", "requests", None, transport=FakeTransport(), api_url="file:///etc")


# ------------------------------------------------------------------ urllib transport (loopback only)
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        host = self.headers.get("Host", "")  # 127.0.0.1:<port>
        if self.path == "/same":
            self._reply(301, {"Location": f"http://{host}/ok"})
        elif self.path == "/other":
            self._reply(302, {"Location": f"http://{host.replace('127.0.0.1', 'localhost')}/ok"})
        elif self.path == "/ok":
            auth = self.headers.get("Authorization", "")
            self._reply(200, {"Link": '<x>; rel="next"'}, json.dumps([{"auth": auth}]).encode())
        else:
            self._reply(404, {}, b'{"message": "Not Found"}')

    def _reply(self, code: int, headers: Mapping[str, str], body: bytes = b"") -> None:
        self.send_response(code)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def loopback() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_urllib_transport_follows_same_host_redirects(loopback: str) -> None:
    resp = urllib_transport(f"{loopback}/same", {"Authorization": "Bearer t"}, 10.0)
    assert resp.status == 200
    assert resp.headers["link"] == '<x>; rel="next"'
    assert json.loads(resp.body) == [{"auth": "Bearer t"}]


def test_urllib_transport_refuses_redirects_to_another_host(loopback: str) -> None:
    resp = urllib_transport(f"{loopback}/other", {"Authorization": "Bearer t"}, 10.0)
    assert resp.status == 302


def test_urllib_transport_returns_error_statuses(loopback: str) -> None:
    resp = urllib_transport(f"{loopback}/missing", {}, 10.0)
    assert resp.status == 404
    assert json.loads(resp.body) == {"message": "Not Found"}
