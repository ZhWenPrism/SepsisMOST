import numpy as np
import pytest

from sepsis_most.training.calibration import isotonic_apply, isotonic_fit


def test_tied_predictions_share_one_calibrated_value():
    x, y = np.array([0.2, 0.2, 0.8]), np.array([0, 1, 1])
    knots, fitted = isotonic_fit(x, y)
    assert knots.tolist() == [0.2, 0.8]
    assert fitted.tolist() == pytest.approx([0.5, 1.0])
    assert isotonic_apply(knots, fitted, np.array([0.2]))[0] == pytest.approx(0.5)


def test_calibration_is_invariant_to_order_within_a_tie():
    x = np.array([0.2, 0.2, 0.8])
    first = isotonic_fit(x, np.array([0, 1, 1]))
    second = isotonic_fit(x, np.array([1, 0, 1]))
    assert np.allclose(isotonic_apply(*first, x), isotonic_apply(*second, x))


def test_pooled_groups_keep_observation_weights():
    knots, fitted = isotonic_fit(np.array([0.1, 0.1, 0.1, 0.6]), np.array([1, 1, 1, 0]))
    assert isotonic_apply(knots, fitted, np.array([0.1, 0.6])).tolist() == pytest.approx([0.75, 0.75])


@pytest.mark.parametrize("x,y", [([], []), ([0.1], [0, 1]), ([np.nan], [0]), ([0.1], [np.inf])])
def test_invalid_calibration_inputs_are_rejected(x, y):
    with pytest.raises(ValueError):
        isotonic_fit(np.asarray(x), np.asarray(y))

