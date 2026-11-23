import numpy as np
import pytest

from sepsis_most.training import isotonic_apply, isotonic_fit


def test_decreasing_output_knots_are_rejected():
    with pytest.raises(ValueError, match="non-decreasing outputs"):
        isotonic_apply([0., 1.], [.8, .2], [.25, .75])


def test_fitted_maps_and_roundoff_tolerance_are_preserved():
    x, y = isotonic_fit([.1, .2, .3, .4], [1., 0., 1., 0.])
    np.testing.assert_array_equal(isotonic_apply(x, y, x), y)
    np.testing.assert_allclose(isotonic_apply([0., 1.], [.5, .5 - 1e-13], [.5]), [.5], atol=1e-12)
