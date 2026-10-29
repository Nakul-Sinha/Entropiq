from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tdp_cli.commands._file_arg import normalize_path, resolve_file
from tdp_cli.errors import UsageError
from tdp_cli.store import LoadedAnalysis


def _stand_in(*paths: str) -> Any:
    files = [SimpleNamespace(path=p) for p in paths]
    meta = SimpleNamespace(kind="url", repo_dir="/nowhere")
    return SimpleNamespace(meta=meta, result=SimpleNamespace(files=files))


def test_normalize_path() -> None:
    assert normalize_path(" .\\pkg\\mu.py ") == "pkg/mu.py"
    assert normalize_path("././pkg/mu.py") == "pkg/mu.py"


def test_exact_and_normalized_paths(sample_loaded: LoadedAnalysis) -> None:
    assert resolve_file(sample_loaded, "pkg/mu.py").path == "pkg/mu.py"
    assert resolve_file(sample_loaded, ".\\pkg\\mu.py").path == "pkg/mu.py"
    assert resolve_file(sample_loaded, "./pkg/mu.py").path == "pkg/mu.py"
    assert resolve_file(sample_loaded, "PKG/MU.PY").path == "pkg/mu.py"
    assert resolve_file(sample_loaded, "mu.py").path == "pkg/mu.py"


def test_ambiguous_suffix() -> None:
    loaded = _stand_in("a/x.py", "b/x.py", "c/y.py")
    with pytest.raises(UsageError, match=r"matches 2 files: a/x.py, b/x.py") as caught:
        resolve_file(loaded, "x.py")
    assert caught.value.hint == "Give more of the path."
    assert resolve_file(loaded, "y.py").path == "c/y.py"


def test_os_path_inside_a_local_repository(sample_loaded: LoadedAnalysis, sample_repo: Path) -> None:
    assert resolve_file(sample_loaded, str(sample_repo / "pkg" / "lam.py")).path == "pkg/lam.py"


def test_ranks(sample_loaded: LoadedAnalysis) -> None:
    files = sample_loaded.result.files
    assert resolve_file(sample_loaded, "1").path == files[0].path == "pkg/mu.py"
    assert resolve_file(sample_loaded, "#2").path == files[1].path
    assert resolve_file(sample_loaded, str(len(files))).path == files[-1].path
    for bad in ("0", "99"):
        with pytest.raises(UsageError, match=f"no file has rank {int(bad)}; ranks go from 1 to {len(files)}"):
            resolve_file(sample_loaded, bad)


def test_a_path_that_looks_like_a_rank_wins() -> None:
    loaded = _stand_in("a.py", "3")
    assert resolve_file(loaded, "3").path == "3"
    assert resolve_file(loaded, "1").path == "a.py"


def test_unknown_files(sample_loaded: LoadedAnalysis) -> None:
    with pytest.raises(UsageError, match="give the path"):
        resolve_file(sample_loaded, "  ")
    with pytest.raises(UsageError, match=r"mux.py is not a ranked file") as caught:
        resolve_file(sample_loaded, "pkg/mux.py")
    assert caught.value.hint is not None
    assert caught.value.hint.startswith("Did you mean pkg/mu.py")
    with pytest.raises(UsageError) as unknown:
        resolve_file(sample_loaded, "zzzzzz/qqqq.txt")
    assert unknown.value.hint is not None
    assert "relative to the repository root" in unknown.value.hint
