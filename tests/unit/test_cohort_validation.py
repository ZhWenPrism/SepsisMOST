import numpy as np
import pytest

from conftest import synthetic_cohort
from sepsis_most.training import load_cohort


def load(tmp_path, temporal, static):
    tp, sp = tmp_path / "temporal.npz", tmp_path / "static.npz"
    np.savez(tp, **temporal)
    np.savez(sp, **static)
    return load_cohort(tp, sp)


@pytest.mark.parametrize("key,value", [
    ("cause", 0.5), ("cause", 2), ("ebin", -1), ("ebin", 8), ("ebin", 0.5),
    ("evt", 2), ("atrisk", np.nan), ("death_h", 0.5),
])
def test_invalid_label_codes_are_rejected_before_integer_conversion(tmp_path, key, value):
    temporal, static = synthetic_cohort(n=3)
    temporal[key] = temporal[key].astype(float)
    temporal[key].flat[0] = value
    with pytest.raises(ValueError, match=key):
        load(tmp_path, temporal, static)


@pytest.mark.parametrize("key", ["cause", "etime", "evt", "shock_h"])
def test_label_rows_and_dimensions_match_cohort(tmp_path, key):
    temporal, static = synthetic_cohort(n=3)
    temporal[key] = temporal[key][:-1]
    with pytest.raises(ValueError, match=key):
        load(tmp_path, temporal, static)


@pytest.mark.parametrize("edges", [[], [12, 12], [24, 12], [12, np.inf], [0, 12]])
def test_bin_edges_are_finite_positive_and_increasing(tmp_path, edges):
    temporal, static = synthetic_cohort(n=3)
    temporal["bin_edges"] = np.asarray(edges)
    with pytest.raises(ValueError, match="bin_edges"):
        load(tmp_path, temporal, static)


def test_valid_cohort_retains_values_and_optional_ids(tmp_path):
    temporal, static = synthetic_cohort(n=3)
    result = load(tmp_path, temporal, static)
    np.testing.assert_array_equal(result.X, temporal["X"])
    np.testing.assert_array_equal(result.S, static["S"])
    np.testing.assert_array_equal(result.cause, temporal["cause"])
    assert result.cause.dtype == np.int64 and result.n_bins == 8
