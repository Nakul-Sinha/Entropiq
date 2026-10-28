import logging
import re
import socket
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from tdp_core.security.urls import (
    CASE_INSENSITIVE_HOSTS,
    DEFAULT_ALLOWED_HOSTS,
    MAX_URL_LENGTH,
    HostLookupError,
    RepoRef,
    UnsafeRepoError,
    is_public_address,
    validate_ref,
    validate_repo_url,
)

GITHUB_V4 = "140.82.112.3"


class FakeDns:
    """Replacement for ``socket.getaddrinfo`` with scripted answers; records every looked-up host."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def getaddrinfo(self, host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        self.calls.append(host)
        if host not in self.answers:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        infos: list[Any] = []
        for addr in self.answers[host]:
            if ":" in addr:
                infos.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (addr, 443, 0, 0)))
            else:
                infos.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 443)))
        return infos


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> FakeDns:
    fake = FakeDns({"github.com": [GITHUB_V4, "2606:50c0:8000::153"], "gitlab.com": ["172.65.251.78"]})
    monkeypatch.setattr(socket, "getaddrinfo", fake.getaddrinfo)
    return fake


# ---------------------------------------------------------------------------------------------- accepted URLs
LONG_OWNER = "o" * 100
LONG_NAME = "n" * 100


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/psf/requests", ("github.com", "psf", "requests")),
        ("https://github.com/psf/requests.git", ("github.com", "psf", "requests")),
        ("https://github.com/psf/requests/", ("github.com", "psf", "requests")),
        ("https://github.com/psf/requests.git/", ("github.com", "psf", "requests")),
        ("https://github.com:443/psf/requests", ("github.com", "psf", "requests")),
        ("HTTPS://GitHub.com/PSF/Requests.GIT", ("github.com", "psf", "requests")),
        ("  https://github.com/psf/requests\n", ("github.com", "psf", "requests")),
        ("https://github.com/pallets/.github", ("github.com", "pallets", ".github")),
        ("https://gitlab.com/Some-Group/Mixed.Case", ("gitlab.com", "Some-Group", "Mixed.Case")),
        ("https://github.com/a-b_c.d/x.y-z_1", ("github.com", "a-b_c.d", "x.y-z_1")),
        (f"https://github.com/{LONG_OWNER}/{LONG_NAME}.git", ("github.com", LONG_OWNER, LONG_NAME)),
    ],
)
def test_accepts_normal_repository_urls(url: str, expected: tuple[str, str, str], dns: FakeDns) -> None:
    ref = validate_repo_url(url, DEFAULT_ALLOWED_HOSTS)

    host, owner, name = expected
    assert (ref.host, ref.owner, ref.name) == expected
    assert ref.clone_url == f"https://{host}/{owner}/{name}.git"
    assert ref.canonical_url == f"https://{host}/{owner}/{name}"
    assert dns.calls == [host]


# ---------------------------------------------------------------------------------------------- rejected URLs
GH = "https://github.com"
SCHEME = "must start with https://"
CREDS = "must not contain credentials"
SHAPE = "must look like https://<host>/<owner>/<name>"
OWNER = "invalid repository owner"
NAME = "invalid repository name"
QUERY = "must not contain a query or fragment"
HOST = "is not allowed"
PORT = "may only use port 443"
NO_HOST = "has no valid host"
ASCII = "may only contain visible ASCII characters"


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        # Other transports and scheme tricks.
        ("http://github.com/psf/requests", SCHEME),
        ("file:///tmp/repo", SCHEME),
        ("ssh://git@github.com/psf/requests.git", SCHEME),
        ("git@github.com:psf/requests.git", SCHEME),
        ("git://github.com/psf/requests.git", SCHEME),
        ("ext::sh -c touch% /tmp/pwned", ASCII),
        ("ext::sh%20-c%20touch%25%20/tmp/pwned", SCHEME),
        ("https:github.com/psf/requests", SCHEME),
        ("//github.com/psf/requests", SCHEME),
        ("https:\\\\github.com\\psf\\requests", SCHEME),
        ("https:///github.com/psf/requests", NO_HOST),
        # Credentials and anything else before the host.
        ("https://user:tok@github.com/psf/requests", CREDS),
        ("https://tok@github.com/psf/requests", CREDS),
        ("https://@github.com/psf/requests", CREDS),
        ("https://github.com:443@evil.example/psf/requests", CREDS),
        # Path shape.
        (f"{GH}/psf/requests/tree/main", SHAPE),
        (f"{GH}/psf", SHAPE),
        (f"{GH}/", SHAPE),
        (GH, SHAPE),
        (f"{GH}//requests", SHAPE),
        (f"{GH}/psf/requests//", SHAPE),
        (f"{GH}\\psf\\requests", NO_HOST),
        # Dot names, double suffixes and escapes.
        (f"{GH}/../requests", OWNER),
        (f"{GH}/psf/..", NAME),
        (f"{GH}/psf/.", NAME),
        (f"{GH}/psf/...", NAME),
        (f"{GH}/psf/..git", NAME),
        (f"{GH}/psf/.git", NAME),
        (f"{GH}/psf/requests.git.git", NAME),
        (f"{GH}/%2e%2e/requests", OWNER),
        (f"{GH}/psf/requests;x", NAME),
        (f"{GH}/psf/requests%00", NAME),
        (f"{GH}/{'o' * 101}/requests", OWNER),
        (f"{GH}/psf/{'n' * 101}", NAME),
        # Query strings and fragments.
        (f"{GH}/psf/requests?tab=readme", QUERY),
        (f"{GH}/psf/requests#readme", QUERY),
        (f"{GH}/psf/requests?", QUERY),
        (f"{GH}/psf/requests#", QUERY),
        ("https://github.com#@evil.example/x/y", QUERY),
        ("https://evil.example#@github.com/x/y", HOST),
        # Hosts and ports.
        ("https://evil.example/psf/requests", HOST),
        ("https://github.com.evil.example/psf/requests", HOST),
        ("https://github.com./psf/requests", HOST),
        ("https://127.0.0.1/psf/requests", HOST),
        ("https://[::1]/psf/requests", NO_HOST),
        ("https://github.com:8443/psf/requests", PORT),
        ("https://github.com:/psf/requests", PORT),
        ("https://github.com:0443/psf/requests", PORT),
        ("https://github.com:443:80/psf/requests", NO_HOST),
        # Characters.
        (f"{GH}/psf/re quests", ASCII),
        (f"{GH}/psf/re\nquests", ASCII),
        (f"{GH}/psf\t/requests", ASCII),
        (f"{GH}/psf/requests\x00", ASCII),
        (f"{GH}/psf/réquests", ASCII),
        ("https://gïthub.com/psf/requests", ASCII),
        # Empty and too long.
        ("", "is empty"),
        ("   ", "is empty"),
        (f"{GH}/psf/" + "a" * (MAX_URL_LENGTH + 1 - len(f"{GH}/psf/")), "longer than 2048 characters"),
    ],
)
def test_rejects_unsafe_or_malformed_urls(url: str, fragment: str, dns: FakeDns) -> None:
    with pytest.raises(UnsafeRepoError) as info:
        validate_repo_url(url, DEFAULT_ALLOWED_HOSTS)

    assert not isinstance(info.value, HostLookupError)
    assert fragment in str(info.value)
    assert dns.calls == []


def test_length_limit_is_inclusive(dns: FakeDns) -> None:
    prefix = f"{GH}/psf/"
    url = prefix + "a" * (MAX_URL_LENGTH - len(prefix))
    with pytest.raises(UnsafeRepoError, match=NAME):  # 2048 characters pass the length check
        validate_repo_url(url, DEFAULT_ALLOWED_HOSTS)


def test_credentials_are_never_echoed(dns: FakeDns, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="tdp_core.security.urls")
    with pytest.raises(UnsafeRepoError) as info:
        validate_repo_url("https://x-access-token:s3cret@github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)

    assert "s3cret" not in str(info.value)
    assert "x-access-token" not in str(info.value)
    assert caplog.records
    assert "s3cret" not in caplog.text


@pytest.mark.parametrize("url", [None, b"https://github.com/psf/requests", 42])
def test_non_string_url_is_rejected(url: Any) -> None:
    with pytest.raises(UnsafeRepoError, match="must be a string"):
        validate_repo_url(url, DEFAULT_ALLOWED_HOSTS)


# ---------------------------------------------------------------------------------------------- identity
def test_github_urls_differing_in_case_are_the_same_repository(dns: FakeDns) -> None:
    a = validate_repo_url("https://github.com/PSF/Requests", DEFAULT_ALLOWED_HOSTS)
    b = validate_repo_url("https://GITHUB.com/psf/requests.git/", DEFAULT_ALLOWED_HOSTS)

    assert a == b
    assert a == RepoRef(
        host="github.com",
        owner="psf",
        name="requests",
        clone_url="https://github.com/psf/requests.git",
        canonical_url="https://github.com/psf/requests",
    )
    assert "github.com" in CASE_INSENSITIVE_HOSTS


def test_gitlab_keeps_case(dns: FakeDns) -> None:
    a = validate_repo_url("https://gitlab.com/Group/Repo", DEFAULT_ALLOWED_HOSTS)
    b = validate_repo_url("https://gitlab.com/group/repo", DEFAULT_ALLOWED_HOSTS)

    assert a != b
    assert a.canonical_url == "https://gitlab.com/Group/Repo"
    assert a.clone_url == "https://gitlab.com/Group/Repo.git"


def test_repo_ref_is_immutable(dns: FakeDns) -> None:
    ref = validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)
    with pytest.raises(AttributeError):
        ref.owner = "evil"  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------- allowlist
def test_allowlist_entries_are_normalised(dns: FakeDns) -> None:
    ref = validate_repo_url("https://github.com/psf/requests", [" GitHub.COM ", ""])
    assert ref.host == "github.com"

    with pytest.raises(UnsafeRepoError, match=r"'gitlab\.com' is not allowed \(allowed hosts: github\.com\)"):
        validate_repo_url("https://gitlab.com/a/b", [" GitHub.COM ", ""])


def test_empty_allowlist_rejects_everything(dns: FakeDns) -> None:
    with pytest.raises(UnsafeRepoError, match=r"allowed hosts: none"):
        validate_repo_url("https://github.com/psf/requests", [])
    assert dns.calls == []


def test_string_allowlist_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="not a string"):
        validate_repo_url("https://github.com/psf/requests", "github.com,gitlab.com")


def test_custom_host_is_accepted_when_allowlisted(dns: FakeDns) -> None:
    dns.answers["git.example.org"] = ["93.184.215.14"]
    ref = validate_repo_url("https://git.example.org/team/tool.git", {"git.example.org"})

    assert ref.clone_url == "https://git.example.org/team/tool.git"
    assert dns.calls == ["git.example.org"]


# ---------------------------------------------------------------------------------------------- DNS
NON_PUBLIC = [
    "127.0.0.1",
    "10.0.0.8",
    "172.16.5.4",
    "192.168.1.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "224.0.0.1",
    "240.0.0.1",
    "255.255.255.255",
    "192.0.2.1",
    "::1",
    "::",
    "fe80::1%eth0",
    "fc00::1",
    "fec0::1",
    "ff02::1",
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "2002:7f00:1::",
    "64:ff9b::7f00:1",
    "2001:db8::1",
]
PUBLIC = ["140.82.112.3", "8.8.8.8", "2606:4700:4700::1111", "::ffff:140.82.112.3", "64:ff9b::8c52:7003"]


def test_host_resolving_to_loopback_is_rejected(dns: FakeDns) -> None:
    dns.answers["git.internal.example"] = ["127.0.0.1"]
    with pytest.raises(UnsafeRepoError) as info:
        validate_repo_url("https://git.internal.example/team/tool", {"git.internal.example"})

    assert not isinstance(info.value, HostLookupError)
    assert "private or reserved" in str(info.value)
    assert "127.0.0.1" not in str(info.value)
    assert dns.calls == ["git.internal.example"]


@pytest.mark.parametrize("bad", NON_PUBLIC)
def test_any_non_public_address_in_the_answer_is_rejected(bad: str, dns: FakeDns) -> None:
    public = "2606:50c0:8000::153" if ":" in bad else GITHUB_V4
    dns.answers["github.com"] = [public, bad]
    with pytest.raises(UnsafeRepoError, match="private or reserved") as info:
        validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)

    assert not isinstance(info.value, HostLookupError)
    assert bad.split("%")[0] not in str(info.value)


def test_public_nat64_answer_is_accepted(dns: FakeDns) -> None:
    dns.answers["github.com"] = ["64:ff9b::8c52:7003", GITHUB_V4]
    assert validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS).name == "requests"


def test_lookup_failure_raises_host_lookup_error(dns: FakeDns) -> None:
    del dns.answers["github.com"]
    with pytest.raises(HostLookupError, match=r"could not resolve host 'github\.com'"):
        validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)


def test_empty_answer_raises_host_lookup_error(dns: FakeDns) -> None:
    dns.answers["github.com"] = []
    with pytest.raises(HostLookupError, match="has no addresses"):
        validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)


def test_idna_failure_raises_host_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(host: str, *args: Any, **kwargs: Any) -> Any:
        raise UnicodeError("label too long")

    monkeypatch.setattr(socket, "getaddrinfo", broken)
    with pytest.raises(HostLookupError):
        validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS)


def test_resolve_dns_false_makes_no_lookup() -> None:
    # The autouse fixture fails the test if anything resolves a host.
    ref = validate_repo_url("https://github.com/psf/requests", DEFAULT_ALLOWED_HOSTS, resolve_dns=False)
    assert ref.canonical_url == "https://github.com/psf/requests"


@pytest.mark.parametrize("address", PUBLIC)
def test_public_addresses(address: str) -> None:
    assert is_public_address(address)


@pytest.mark.parametrize("address", [*NON_PUBLIC, "not-an-ip", ""])
def test_non_public_addresses(address: str) -> None:
    assert not is_public_address(address)


# ---------------------------------------------------------------------------------------------- refs
@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("main", "main"),
        ("HEAD", "HEAD"),
        ("v1.2.3", "v1.2.3"),
        ("release/2.x", "release/2.x"),
        ("refs/heads/main", "refs/heads/main"),
        ("refs/tags/v1.0", "refs/tags/v1.0"),
        ("feature_x-y", "feature_x-y"),
        ("0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567"),
        (" main ", "main"),
        ("r" * 255, "r" * 255),
    ],
)
def test_validate_ref_accepts_branch_and_tag_names(ref: str, expected: str) -> None:
    assert validate_ref(ref) == expected


@pytest.mark.parametrize(
    "ref",
    [
        "",
        "   ",
        "-x",
        "--upload-pack=evil",
        "a..b",
        "a//b",
        "/main",
        "main/",
        "main.",
        ".hidden",
        "a/.b",
        "x.lock",
        "a/b.lock/c",
        "ma in",
        "main~1",
        "main^",
        "@{u}",
        "*",
        "ref?",
        "[a]",
        "a:b",
        "a\\b",
        "é",
        "a\x00",
        "r" * 256,
        "refs/pull/1/head",
        "refs/remotes/origin/main",
        "refs/heads",
    ],
)
def test_validate_ref_rejects_unsafe_names(ref: str) -> None:
    with pytest.raises(UnsafeRepoError, match="invalid git ref"):
        validate_ref(ref)


def test_validate_ref_explains_unsupported_full_refs() -> None:
    with pytest.raises(UnsafeRepoError, match="only branches and tags"):
        validate_ref("refs/pull/1/head")


@pytest.mark.parametrize("ref", [None, b"main", 1])
def test_validate_ref_rejects_non_strings(ref: Any) -> None:
    with pytest.raises(UnsafeRepoError, match="must be a string"):
        validate_ref(ref)


# ---------------------------------------------------------------------------------------------- property
_SCHEMES = st.sampled_from(["https://", "HTTPS://", "http://", "ssh://", "git@", "", "file://", "https:/"])
_HOSTS = st.sampled_from(
    [
        "github.com",
        "GitHub.com",
        "gitlab.com",
        "evil.example",
        "github.com.",
        "[::1]",
        "user@github.com",
        "127.0.0.1",
    ]
)
_PORTS = st.sampled_from(["", ":443", ":80", ":", ":0443"])
_PARTS = st.from_regex(r"[A-Za-z0-9_.-]{1,8}", fullmatch=True) | st.text(
    alphabet="aZ09_.-%/@?#: \\", max_size=12
)
_SUFFIXES = st.sampled_from(["", "/", ".git", ".git/", ".GIT", ".git.git", "?x=1", "#top", "/tree/main"])
_CLONE_URL = re.compile(r"https://(github\.com|gitlab\.com)/[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}\.git")


@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(scheme=_SCHEMES, host=_HOSTS, port=_PORTS, owner=_PARTS, name=_PARTS, suffix=_SUFFIXES)
@example(scheme="https://", host="github.com", port="", owner="a", name="a.git", suffix=".git")
@example(scheme="https://", host="GitHub.com", port=":443", owner="Ab", name="C.d", suffix=".GIT")
def test_property_accepted_urls_are_canonical_and_idempotent(
    scheme: str, host: str, port: str, owner: str, name: str, suffix: str
) -> None:
    url = f"{scheme}{host}{port}/{owner}/{name}{suffix}"
    try:
        ref = validate_repo_url(url, DEFAULT_ALLOWED_HOSTS, resolve_dns=False)
    except UnsafeRepoError:
        return  # the only exception type allowed

    assert _CLONE_URL.fullmatch(ref.clone_url)
    assert ref.clone_url == ref.canonical_url + ".git"
    assert validate_repo_url(ref.canonical_url, DEFAULT_ALLOWED_HOSTS, resolve_dns=False) == ref
    assert validate_repo_url(ref.clone_url, DEFAULT_ALLOWED_HOSTS, resolve_dns=False) == ref
