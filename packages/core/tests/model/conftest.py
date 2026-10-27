"""Shared fixtures for the model runtime tests (also used by the explainer tests).

The fixtures are session scoped: tests must copy a frame (``frame.copy()``) or a bundle directory
(``shutil.copytree``) before changing it, and never mutate the loaded bundle's dicts.
"""

from pathlib import Path

import pandas as pd
import pytest

from tdp_core.model.bundle import Bundle, load_bundle, make_dev_bundle
from tdp_core.model.synthetic import synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """The default synthetic dataset (30 queries x 200 rows, seed 42) in the dataset.parquet layout."""
    return synthetic_dataset(seed=42)


@pytest.fixture(scope="session")
def dev_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A dev bundle directory built once per test session with seed 42."""
    return make_dev_bundle(tmp_path_factory.mktemp("dev-bundle") / "bundle")


@pytest.fixture(scope="session")
def dev_bundle(dev_bundle_dir: Path) -> Bundle:
    """The dev bundle loaded with ``load_bundle``."""
    return load_bundle(dev_bundle_dir)
