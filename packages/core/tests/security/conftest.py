import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _block_real_dns(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any offline test that would resolve a real host name (network tests are exempt).

    ``AssertionError`` is deliberately not an ``OSError``, so ``validate_repo_url`` cannot swallow it.
    """
    if request.node.get_closest_marker("network") is not None:
        return

    def blocked(host: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"unexpected DNS lookup of {host!r} in an offline test")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
