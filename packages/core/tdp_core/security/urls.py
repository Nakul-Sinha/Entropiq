"""Validate user-supplied repository URLs and git refs before anything is cloned or fetched.

Only ``https://<host>[:443]/<owner>/<name>[.git][/]`` on an allowlisted host is accepted. The URL is checked
with an explicit grammar on a string that is already known to be visible ASCII (``urllib.parse`` silently
drops tabs, newlines and control characters and reads ``https:host/x`` as a path), and the result is always
rebuilt from the validated parts, so nothing the user typed reaches git unless it was checked. Error
messages never repeat the raw URL, which may hold a token, and are safe to show to users.

When ``resolve_dns`` is on, an allowlisted host must resolve only to public addresses. The lookup runs after
every syntax and allowlist check, so a host that is not allowlisted is never looked up.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from collections.abc import Collection
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({"github.com", "gitlab.com"})
"""The platform's default allowlist (the same hosts as ``ALLOWED_GIT_HOSTS`` in ``.env.example``)."""

CASE_INSENSITIVE_HOSTS: frozenset[str] = frozenset({"github.com"})
"""Hosts whose owner and repository names are case-insensitive; :class:`RepoRef` lower-cases them there."""

MAX_URL_LENGTH = 2048
"""Longest accepted repository URL, in characters (after surrounding whitespace is stripped)."""

MAX_REF_LENGTH = 255
"""Longest accepted git ref, in characters."""

_NETLOC = re.compile(r"(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>[^:]*))?")
_REPO_PATH = re.compile(r"/(?P<owner>[^/]+)/(?P<name>[^/]+)/?")
_PART = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_REF = re.compile(rf"[A-Za-z0-9._/-]{{1,{MAX_REF_LENGTH}}}")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")

_OWNER_RULE = "invalid repository owner: use 1 to 100 letters, digits, '.', '-' or '_' (not only dots)"
_NAME_RULE = (
    "invalid repository name: use 1 to 100 letters, digits, '.', '-' or '_'"
    " (not only dots, not ending in .git)"
)
_REF_RULE = "invalid git ref: use letters, digits, '.', '-', '_' and '/' (at most 255 characters)"


class UnsafeRepoError(ValueError):
    """The repository URL or ref is not acceptable. The message is safe to show to users."""


class HostLookupError(UnsafeRepoError):
    """DNS lookup of an allowed host failed (often transient; an API may answer 503)."""


@dataclass(frozen=True, slots=True)
class RepoRef:
    """A validated repository on an allowlisted host.

    ``host`` is lower-case and one of the allowed hosts. ``owner`` and ``name`` (``name`` without the
    ``.git`` suffix) contain only ``[A-Za-z0-9_.-]``, are 1 to 100 characters long and are never made only of
    dots, so they are safe as single path components; on :data:`CASE_INSENSITIVE_HOSTS` they are lower-cased.
    ``clone_url`` is ``https://<host>/<owner>/<name>.git`` and ``canonical_url`` is the same without
    ``.git``; ``canonical_url`` is the identity key of a repository (for example ``repositories.url``).
    """

    host: str
    owner: str
    name: str
    clone_url: str
    canonical_url: str


def _reject(message: str, error: type[UnsafeRepoError] = UnsafeRepoError) -> UnsafeRepoError:
    # Log the reason only: the raw URL may contain a token.
    logger.debug("rejected repository input: %s", message)
    return error(message)


def _valid_part(part: str) -> bool:
    return _PART.fullmatch(part) is not None and part.strip(".") != ""


def validate_repo_url(url: str, allowed_hosts: Collection[str], *, resolve_dns: bool = True) -> RepoRef:
    """Check a user-supplied repository URL and return its canonical :class:`RepoRef`.

    Accepted form: ``https://<host>[:443]/<owner>/<name>[.git][/]`` after stripping surrounding whitespace,
    at most :data:`MAX_URL_LENGTH` characters of visible ASCII. The scheme is case-insensitive; the host is
    lower-cased and must equal one of ``allowed_hosts`` exactly (no trailing dot, no subdomain). Every entry
    of ``allowed_hosts`` is stripped and lower-cased and empty entries are dropped. One ``.git`` suffix
    (any case) is removed from the name; a name that still ends in ``.git`` is rejected, so ``clone_url`` and
    ``canonical_url`` validate again to the same :class:`RepoRef`.

    Rejected: other schemes (``http``, ``ssh``, ``git``, ``file``, ``ext::``, scp-like ``git@host:x/y``),
    credentials or any ``@`` before the path, ports other than 443, IP literals and hosts that are not
    allowlisted, query strings and fragments, extra or missing path segments, ``%``-escapes, and owner or
    name parts that are made only of dots or contain other characters.

    With ``resolve_dns=True`` (the default) the host is resolved (port 443, TCP) after every other check
    passed, and every address must be public (:func:`is_public_address`); a mixed answer fails.

    Raises :class:`UnsafeRepoError` with a user-safe message that never contains the URL or a resolved
    address, :class:`HostLookupError` (a subclass) when the lookup itself fails or returns nothing, and
    ``TypeError`` when ``allowed_hosts`` is a single string (a comma-separated value passed by mistake).
    """
    if isinstance(allowed_hosts, str):
        raise TypeError("allowed_hosts must be a collection of host names, not a string")
    allowed = frozenset(h.strip().lower() for h in allowed_hosts if h.strip())

    if not isinstance(url, str):
        raise _reject("repository URL must be a string")
    text = url.strip()
    if not text:
        raise _reject("repository URL is empty")
    if len(text) > MAX_URL_LENGTH:
        raise _reject(f"repository URL is longer than {MAX_URL_LENGTH} characters")
    if any(not 0x21 <= ord(ch) <= 0x7E for ch in text):
        raise _reject("repository URL may only contain visible ASCII characters")

    scheme, sep, rest = text.partition("://")
    if not sep or scheme.lower() != "https":
        raise _reject("repository URL must start with https://")

    end = len(rest)
    for stop in "/?#":
        index = rest.find(stop)
        if index != -1:
            end = min(end, index)
    netloc, tail = rest[:end], rest[end:]

    if "@" in netloc:
        raise _reject("repository URL must not contain credentials")
    match = _NETLOC.fullmatch(netloc)
    if match is None:
        raise _reject("repository URL has no valid host")
    port = match.group("port")
    if port is not None and port != "443":
        raise _reject("repository URL may only use port 443")
    host = match.group("host").lower()
    if host not in allowed:
        raise _reject(f"host {host!r} is not allowed (allowed hosts: {', '.join(sorted(allowed)) or 'none'})")

    if "?" in tail or "#" in tail:
        raise _reject("repository URL must not contain a query or fragment")
    path = _REPO_PATH.fullmatch(tail)
    if path is None:
        raise _reject("repository URL must look like https://<host>/<owner>/<name>")
    owner, name = path.group("owner"), path.group("name")
    if name[-4:].lower() == ".git":
        name = name[:-4]
    if not _valid_part(owner):
        raise _reject(_OWNER_RULE)
    if not _valid_part(name) or name[-4:].lower() == ".git":
        raise _reject(_NAME_RULE)
    if host in CASE_INSENSITIVE_HOSTS:
        owner, name = owner.lower(), name.lower()

    if resolve_dns:
        _check_resolves_public(host)
    canonical = f"https://{host}/{owner}/{name}"
    return RepoRef(host=host, owner=owner, name=name, clone_url=f"{canonical}.git", canonical_url=canonical)


def _check_resolves_public(host: str) -> None:
    """Resolve ``host`` (port 443, TCP) and require every address to be public."""
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        logger.debug("DNS lookup of %r failed: %s", host, type(exc).__name__)
        raise _reject(f"could not resolve host {host!r}", HostLookupError) from None
    addresses = sorted({str(info[4][0]) for info in infos})
    if not addresses:
        raise _reject(f"host {host!r} has no addresses", HostLookupError)
    if not all(is_public_address(address) for address in addresses):
        logger.warning("host %r resolves to non-public addresses %s", host, addresses)
        raise _reject(f"host {host!r} resolves to a private or reserved address")


def is_public_address(address: str) -> bool:
    """True when ``address`` (an IPv4 or IPv6 literal, IPv6 may carry a ``%scope``) is publicly routable.

    False for anything that is not an IP literal, and for private, loopback, link-local, multicast, reserved,
    unspecified, shared (CGNAT ``100.64.0.0/10``), documentation, benchmarking and deprecated site-local
    (``fec0::/10``) addresses. IPv4-mapped (``::ffff:a.b.c.d``) and NAT64 (``64:ff9b::a.b.c.d``) addresses
    are judged by their embedded IPv4 address; a 6to4 (``2002::/16``) address is never public when its
    embedded IPv4 address is not.
    """
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        else:
            # fec0::/10 (deprecated site-local) is still reported as global by Python.
            embedded_private = ip.sixtofour is not None and not is_public_address(str(ip.sixtofour))
            if ip.is_site_local or embedded_private:
                return False
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_ref(ref: str) -> str:
    """Check a user-supplied branch, tag or ``HEAD`` name and return it without surrounding whitespace.

    Accepted: 1 to :data:`MAX_REF_LENGTH` characters from ``[A-Za-z0-9._/-]`` that follow git's ref-name
    rules: no leading ``-``, ``/`` or ``.``; no trailing ``/``, ``.`` or ``.lock``; no ``..``, ``//``, ``/.``
    or ``.lock/``. Full refs must be ``refs/heads/...`` or ``refs/tags/...``, because mirrors hold only
    branches and tags. ``HEAD`` and a 40-character hex sha pass as names (``git ls-remote`` lists refs only,
    so a caller decides what a sha means). Glob characters, ``~``, ``^``, ``@{``, ``:``, spaces, backslashes
    and non-ASCII characters never pass, so the result is safe as a single git argument after ``--``.

    Raises :class:`UnsafeRepoError` with a user-safe message.
    """
    if not isinstance(ref, str):
        raise _reject("git ref must be a string")
    text = ref.strip()
    if _REF.fullmatch(text) is None:
        raise _reject(_REF_RULE)
    if (
        text.startswith(("-", "/", "."))
        or text.endswith(("/", ".", ".lock"))
        or any(bad in text for bad in ("..", "//", "/.", ".lock/"))
    ):
        raise _reject("invalid git ref")
    if text.startswith("refs/") and not text.startswith(("refs/heads/", "refs/tags/")):
        raise _reject("invalid git ref: only branches and tags can be analysed")
    return text
