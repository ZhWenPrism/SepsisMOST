import numpy as np
import pytest

from sepsis_most.training import Standardiser


@pytest.mark.parametrize("name,value", [
    ("mu", [[0., 1.]]), ("mu", [np.nan, 0.]), ("sd", [1.]), ("sd", [0., 1.]),
    ("sd", [-1., 1.]), ("sd", [np.inf, 1.]), ("smu", [np.nan]), ("ssd", [0.]),
])
def test_invalid_standardisation_state_is_rejected(name, value):
    values = dict(mu=np.zeros(2), sd=np.ones(2), smu=np.zeros(1), ssd=np.ones(1))
    values[name] = np.asarray(value)
    with pytest.raises(ValueError, match="aligned finite vectors"):
        Standardiser(**values)


def test_valid_statistics_preserve_dtype_values_and_caller_isolation():
    arrays = [np.array(v, dtype=np.float32) for v in ([1., 2.], [2., 4.], [3.], [2.])]
    state = Standardiser(*arrays)
    before = state.transform(np.array([[[3., 6.]]], dtype=np.float32), np.array([[5.]], dtype=np.float32))
    arrays[0][:] = 99.
    np.testing.assert_array_equal(state.mu, [1., 2.])
    assert state.mu.dtype == state.sd.dtype == np.float32
    np.testing.assert_array_equal(before[0], [[[1., 1.]]])
    np.testing.assert_array_equal(before[1], [[1.]])
