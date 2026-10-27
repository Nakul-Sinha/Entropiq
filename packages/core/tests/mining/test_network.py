import time
from pathlib import Path

import pytest

from tdp_core.contracts import COMMITS_SCHEMA, FILE_CHANGES_SCHEMA
from tdp_core.mining import ensure_mirror, list_files, make_snapshots, mine_history, path_lineage

pytestmark = pytest.mark.network


def test_mine_psf_requests(tmp_path: Path) -> None:
    mirror = ensure_mirror("https://github.com/psf/requests.git", tmp_path / "psf__requests.git")

    start = time.perf_counter()
    h = mine_history(mirror)
    elapsed = time.perf_counter() - start

    assert elapsed < 60
    assert h.commits.num_rows > 4000
    assert h.commits.schema.equals(COMMITS_SCHEMA)
    assert h.file_changes.schema.equals(FILE_CHANGES_SCHEMA)
    snaps = make_snapshots("psf__requests", h.commits)
    assert snaps
    lineage = path_lineage(h.file_changes, h.commits, snaps[-1].time)
    files = list_files(mirror, snaps[-1].sha)
    assert any(len(lineage.get(p, frozenset({p}))) > 1 for p in files)
