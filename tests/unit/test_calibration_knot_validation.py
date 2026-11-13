import numpy as np
import pytest

from sepsis_most.training import isotonic_apply


@pytest.mark.parametrize("knots, fitted", [
    ([0.8, 0.2], [0.1, 0.9]), ([0.2, 0.2], [0.1, 0.9]),
    ([0.2, np.nan], [0.1, 0.9]), ([0.2, 0.8], [0.1, np.inf]),
])
def test_interpolation_rejects_malformed_calibration_maps(knots, fitted):
    with pytest.raises(ValueError, match="calibration knots"):
        isotonic_apply(knots, fitted, [0.5])


def test_valid_single_knot_and_interpolated_maps_are_preserved():
    np.testing.assert_allclose(isotonic_apply([0.3], [0.4], [0.0, 1.0]), [0.4, 0.4])
    np.testing.assert_allclose(isotonic_apply([0.2, 0.8], [0.1, 0.9], [0, 0.5, 1]), [0.1, 0.5, 0.9])
