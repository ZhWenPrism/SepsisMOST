import numpy as np
import pytest

from sepsis_most.training import Standardiser


@pytest.mark.parametrize("n_raw", [-1, 3, 0.5, True])
def test_raw_channel_count_cannot_silently_select_wrong_columns(n_raw):
    X = np.arange(12, dtype=float).reshape(3, 2, 2)
    with pytest.raises(ValueError, match="n_raw"):
        Standardiser.fit(X, np.ones((3, 1)), [0, 1], n_raw=n_raw)


def test_raw_channel_count_applies_only_to_temporal_features():
    X = np.arange(12, dtype=float).reshape(3, 2, 2)
    standardiser = Standardiser.fit(X, np.ones((3, 1)), [0, 1], n_raw=1)
    assert standardiser.mu.tolist() == [0.0, 4.0]
    assert standardiser.sd[0] == 1.0
