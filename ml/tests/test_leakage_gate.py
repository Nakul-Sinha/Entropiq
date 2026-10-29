"""The leakage promotion gate on the real dataset: ``uv run pytest -m leakage``.

Rebuilds 10 sampled queries of ``$TDP_DATA_DIR/dataset.parquet`` in clones truncated at each snapshot and
compares them with the stored rows (P2.8's ``check-leakage``), which rewrites ``runs/check-leakage.json``
for ``tdp-ml promote`` to read. It takes a few minutes, so it runs only when ``leakage`` is selected with
``-m`` and the dataset and manifest exist; the default test run skips it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tdp_ml.pipeline.layout import DataLayout
from tdp_ml.registry import LEAKAGE_MIN_QUERIES


@pytest.mark.leakage
def test_leakage_on_sampled_queries(request: pytest.FixtureRequest) -> None:
    if "leakage" not in str(request.config.getoption("markexpr") or ""):
        pytest.skip("select it with `pytest -m leakage`")
    layout = DataLayout.from_env()
    if not layout.manifest.is_file() or not layout.dataset.is_file():
        pytest.skip(f"no manifest.json or dataset.parquet in {layout.root}")
    from tdp_ml.pipeline.leakage import check_leakage

    queries = pd.read_parquet(layout.dataset, columns=["qid"])["qid"].nunique()
    report = check_leakage(layout, sample=LEAKAGE_MIN_QUERIES, jobs=4, threads=2)
    failing = [q for q in report["queries"] if not q["passed"]][:3]
    assert report["checked"] == min(LEAKAGE_MIN_QUERIES, queries)
    assert report["passed"], failing
