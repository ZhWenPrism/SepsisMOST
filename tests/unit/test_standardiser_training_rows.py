import numpy as np
import pytest
from sepsis_most.training import Standardiser


@pytest.mark.parametrize("indices", [[], [-1], [3], [0, 0], [0.5], [True, False, False]])
def test_invalid_training_rows_cannot_fit_preprocessing(indices):
    with pytest.raises(ValueError, match="train_idx"):
        Standardiser.fit(np.ones((3, 2, 7)), np.ones((3, 2)), indices)


def test_nonfinite_training_values_fail_without_using_other_rows():
    x, s = np.ones((3, 2, 7)), np.ones((3, 2))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="training features"):
        Standardiser.fit(x, s, [0, 1])
    result = Standardiser.fit(x, s, [1, 2])
    assert np.isfinite(result.mu).all()
