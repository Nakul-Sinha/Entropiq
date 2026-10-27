"""Language plugin protocol (IMPLEMENTATION_PLAN.md step P1.2)."""

from __future__ import annotations

from typing import Protocol

from tdp_core.contracts import FileSyntax


class LanguagePlugin(Protocol):
    """A parser for one source language.

    Attributes:
        name: language name stored in ``FileSyntax.language`` (for example ``"python"``).
        extensions: file suffixes the plugin handles, lower case with the dot (for example ``(".py",)``).
    """

    name: str
    extensions: tuple[str, ...]

    def parse(self, path: str, source: bytes) -> FileSyntax:
        """Parse ``source`` (raw file bytes) of the repo-relative POSIX ``path`` into a ``FileSyntax``.

        Never raises on malformed code: syntax errors set ``parse_error`` and metrics are best effort.
        """
        ...

    def is_test_file(self, path: str) -> bool:
        """True if the repo-relative POSIX ``path`` is test code by the language's naming conventions."""
        ...
