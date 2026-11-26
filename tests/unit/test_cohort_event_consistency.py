import pytest
from conftest import synthetic_cohort
from test_cohort_validation import load


@pytest.mark.parametrize("cause,event", [(0, 0), (1, 1), (-1, 1)])
def test_at_risk_event_codes_must_describe_the_same_outcome(tmp_path, cause, event):
    temporal, static = synthetic_cohort(n=3)
    temporal["atrisk"][0, 0] = 1
    temporal["cause"][0, 0] = cause
    temporal["evt"][0, 0] = event
    with pytest.raises(ValueError, match="cause and evt"):
        load(tmp_path, temporal, static)


def test_masked_organ_outcomes_remain_excluded_from_consistency_checks(tmp_path):
    temporal, static = synthetic_cohort(n=3)
    temporal["atrisk"][0, 0] = 0
    temporal["cause"][0, 0] = 0
    temporal["evt"][0, 0] = 0
    assert load(tmp_path, temporal, static).atrisk[0, 0] == 0
