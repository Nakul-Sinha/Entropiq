"""Source parsing: language plugins and file filters (IMPLEMENTATION_PLAN.md step P1.2)."""

from __future__ import annotations

from tdp_core.parsing.base import LanguagePlugin
from tdp_core.parsing.filters import skip_reason
from tdp_core.parsing.python import PythonPlugin, is_test_file, parse_python

__all__ = ["LanguagePlugin", "PythonPlugin", "get_plugin", "is_test_file", "parse_python", "skip_reason"]

_PLUGINS: tuple[LanguagePlugin, ...] = (PythonPlugin(),)


def get_plugin(path: str) -> LanguagePlugin | None:
    """The plugin for ``path`` chosen by its extension (case sensitive), or None if no plugin handles it.

    ``".py"`` maps to the shared ``PythonPlugin`` instance.
    """
    for plugin in _PLUGINS:
        if path.endswith(plugin.extensions):
            return plugin
    return None
