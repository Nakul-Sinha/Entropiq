import itertools
import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tdp_core.model.priority import priority_from_scores

FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
TIED = st.integers(min_value=-3, max_value=3).map(float)
SCORES = st.one_of(st.lists(FINITE, min_size=1, max_size=60), st.lists(TIED, min_size=1, max_size=60))


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], [0, 25, 50, 75]),
        ([3.0, 1.0, 2.0], [67, 0, 33]),
        ([5.0, 5.0, 1.0], [33, 33, 0]),
        ([2.5, 2.5, 2.5, 2.5], [0, 0, 0, 0]),
        ([7.0], [100]),
        ([-0.3, 0.1, 0.7, 1.2, 1.9, 2.4, 3.3, 4.0], [0, 13, 25, 38, 50, 63, 75, 88]),
    ],
)
def test_known_vectors(scores: list[float], expected: list[int]) -> None:
    out = priority_from_scores(scores)
    assert out.dtype == np.int64
    assert out.tolist() == expected


def test_empty_input_gives_empty_int64() -> None:
    out = priority_from_scores(np.zeros(0))
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_top_of_200_gets_100_and_top_of_100_gets_99() -> None:
    assert priority_from_scores(np.arange(200.0)).max() == 100
    assert priority_from_scores(np.arange(100.0)).max() == 99


def test_accepts_float32_and_lists() -> None:
    a = priority_from_scores(np.array([0.5, -1.0, 2.0], dtype=np.float32))
    b = priority_from_scores([0.5, -1.0, 2.0])
    assert a.tolist() == b.tolist() == [33, 0, 67]


@pytest.mark.parametrize(
    "bad",
    [[1.0, float("nan")], [float("inf"), 1.0], [[1.0, 2.0], [3.0, 4.0]], 3.0],
)
def test_rejects_non_finite_and_wrong_shape(bad: object) -> None:
    with pytest.raises(ValueError):
        priority_from_scores(bad)  # type: ignore[arg-type]


@settings(max_examples=200, deadline=None, derandomize=True)
@given(SCORES)
def test_properties(scores: list[float]) -> None:
    s = np.array(scores)
    p = priority_from_scores(s)
    assert p.shape == s.shape
    assert ((p >= 0) & (p <= 100)).all()
    order = np.argsort(s, kind="stable")
    for i, j in itertools.pairwise(order):
        if s[i] == s[j]:
            assert p[i] == p[j]
        else:
            assert p[i] <= p[j]
    if s.size >= 2:
        assert p[np.argmin(s)] == 0


@settings(max_examples=200, deadline=None, derandomize=True)
@given(SCORES, st.randoms(use_true_random=False))
def test_permutation_equivariant(scores: list[float], rnd: random.Random) -> None:
    s = np.array(scores)
    order = list(range(s.size))
    rnd.shuffle(order)
    perm = np.array(order)
    assert priority_from_scores(s[perm]).tolist() == priority_from_scores(s)[perm].tolist()
