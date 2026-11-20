import numpy as np
import pytest
from conftest import synthetic_cohort
from test_cohort_validation import load


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_at_risk_event_times_are_rejected(tmp_path, value):
    temporal, static = synthetic_cohort(n=3)
    temporal["atrisk"][0, 0] = 1
    temporal["etime"][0, 0] = value
    with pytest.raises(ValueError, match="etime"):
        load(tmp_path, temporal, static)


def test_masked_organ_times_can_remain_unspecified(tmp_path):
    temporal, static = synthetic_cohort(n=3)
    temporal["atrisk"][0, 0] = 0
    temporal["etime"][0, 0] = np.nan
    assert np.isnan(load(tmp_path, temporal, static).etime[0, 0])
