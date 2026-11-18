import numpy as np
import pytest

from test_inference_inputs import ensemble


@pytest.mark.parametrize("count", [0, 1, 5, 7])
def test_calibration_cannot_drop_or_add_organs(count):
    maps = [(np.array([0., 1.]), np.array([0., 1.]))] * count
    with pytest.raises(ValueError, match="one map per organ"):
        ensemble(isotonic=maps)


def test_invalid_calibration_knots_fail_even_before_empty_predictions():
    maps = [(np.array([0., 1.]), np.array([0., 1.]))] * 6
    maps[0] = (np.array([1., 0.]), np.array([0., 1.]))
    with pytest.raises(ValueError, match="calibration knots"):
        ensemble(isotonic=maps)
