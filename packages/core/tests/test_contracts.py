import pytest

from tdp_core.contracts import (
    BINARY_FEATURES,
    FAMILY_ORDER,
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_INPUT_COLUMNS,
    PCT_COLUMNS,
    band_for,
    family_of,
)


def test_feature_column_count_and_uniqueness() -> None:
    assert len(FEATURE_COLUMNS) == 46
    assert len(set(FEATURE_INPUT_COLUMNS)) == len(FEATURE_INPUT_COLUMNS)
    assert len(PCT_COLUMNS) == 46 - len(BINARY_FEATURES)
    assert set(FAMILY_ORDER) == set(FEATURE_FAMILIES)


def test_family_of_every_column() -> None:
    for col in FEATURE_INPUT_COLUMNS:
        fam = family_of(col)
        base = col.removeprefix("pct_")
        assert base in FEATURE_FAMILIES[fam]
    with pytest.raises(KeyError):
        family_of("nope")


@pytest.mark.parametrize(
    ("priority", "band"),
    [
        (100, "Critical"),
        (90, "Critical"),
        (89, "High"),
        (75, "High"),
        (74, "Medium"),
        (50, "Medium"),
        (49, "Low"),
        (0, "Low"),
    ],
)
def test_band_for(priority: int, band: str) -> None:
    assert band_for(priority) == band
