import importlib

import pytest


@pytest.mark.parametrize("module", ["tdp_core", "tdp_platform", "tdp_api", "tdp_worker", "tdp_cli", "tdp_ml"])
def test_package_imports(module: str) -> None:
    mod = importlib.import_module(module)
    assert mod.__version__ == "0.1.0"
