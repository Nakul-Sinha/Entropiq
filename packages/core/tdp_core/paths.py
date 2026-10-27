"""Locations of data, model and result folders shared by every tool.

Each folder can be moved with an environment variable so several worktrees can share one copy of the
mined data, trained models and experiment results:

- ``TDP_DATA_DIR`` (default ``./data``): mirrors, mined tables, features, labels, the dataset, caches
- ``TDP_MODELS_DIR`` (default ``./models``): model bundles and ``production.json``
- ``TDP_RESULTS_DIR`` (default ``./results``): experiment tables and figures

Relative defaults are resolved against the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def _dir(var: str, default: str) -> Path:
    value = os.environ.get(var, "").strip()
    return Path(value) if value else Path.cwd() / default


def data_dir() -> Path:
    """Root of mined data (mirrors, tables, features, labels, dataset, caches)."""
    return _dir("TDP_DATA_DIR", "data")


def models_dir() -> Path:
    """Folder holding model bundles and ``production.json``."""
    return _dir("TDP_MODELS_DIR", "models")


def results_dir() -> Path:
    """Folder for experiment tables and figures."""
    return _dir("TDP_RESULTS_DIR", "results")
