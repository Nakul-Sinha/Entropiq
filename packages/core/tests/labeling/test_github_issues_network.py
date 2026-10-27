import os

import pytest

from tdp_core.labeling.github_issues import ISSUES_SCHEMA, fetch_issues

pytestmark = pytest.mark.network


def test_fetch_first_page_of_psf_requests() -> None:
    table = fetch_issues("psf", "requests", os.environ.get("GITHUB_TOKEN") or None, max_pages=1)
    numbers = table.column("number").to_pylist()
    assert table.schema.equals(ISSUES_SCHEMA)
    assert table.num_rows > 0
    assert len(set(numbers)) == len(numbers)
