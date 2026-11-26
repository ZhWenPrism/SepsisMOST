import numpy as np
import pytest
from conftest import synthetic_cohort
from test_cohort_validation import load
from sepsis_most.training.data import load_scoring_inputs


@pytest.mark.parametrize("key", ["X", "S"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_training_features_are_rejected(tmp_path, key, value):
    temporal, static = synthetic_cohort(n=3)
    arrays = temporal if key == "X" else static
    arrays[key].flat[0] = value
    with pytest.raises(ValueError, match="finite training features"):
        load(tmp_path, temporal, static)


def test_scoring_inputs_keep_the_existing_missing_value_contract(tmp_path):
    temporal, static = synthetic_cohort(n=3)
    static["S"][0, 0] = np.nan
    tp, sp = tmp_path / "temporal.npz", tmp_path / "static.npz"
    np.savez(tp, X=temporal["X"])
    np.savez(sp, S=static["S"])
    _, values = load_scoring_inputs(tp, sp)
    assert np.isnan(values[0, 0])
