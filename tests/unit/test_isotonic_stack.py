import numpy as np
import pytest

from sepsis_most.training.calibration import isotonic_fit


@pytest.mark.parametrize("seed", range(8))
def test_weighted_tied_knots_match_the_isotonic_minmax_solution(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 6, 25).astype(float)
    y = rng.integers(0, 2, 25).astype(float)
    levels, inverse, weights = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y)
    expected = []
    for i in range(len(levels)):
        expected.append(max(min(sums[a:b+1].sum() / weights[a:b+1].sum()
                                for b in range(i, len(levels))) for a in range(i + 1)))
    knots, fitted = isotonic_fit(x, y)
    np.testing.assert_array_equal(knots, levels)
    np.testing.assert_allclose(fitted, expected, atol=1e-12, rtol=0)


def test_large_descending_fit_pools_to_a_single_mean():
    x = np.linspace(0, 1, 12000)
    knots, fitted = isotonic_fit(x, 1 - x)
    np.testing.assert_array_equal(knots, x)
    np.testing.assert_allclose(fitted, 0.5, atol=1e-12, rtol=0)
