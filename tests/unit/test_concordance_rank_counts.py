import numpy as np
import pytest

from sepsis_most.evaluation.discrimination import cause_specific_cindex


def _reference(risk, time, event, at_risk):
    rows = np.flatnonzero(at_risk)
    numerator = denominator = 0.
    for a in rows:
        if not event[a]:
            continue
        for b in rows:
            if time[b] > time[a]:
                denominator += 1
                numerator += float(risk[a] > risk[b]) + 0.5 * float(risk[a] == risk[b])
    return numerator / denominator if denominator else float("nan")


@pytest.mark.parametrize("seed", range(10))
def test_rank_counts_preserve_censoring_time_and_risk_ties(seed):
    rng = np.random.default_rng(seed)
    risk = rng.integers(0, 7, 60) / 7
    time = rng.integers(0, 12, 60)
    event, at_risk = rng.integers(0, 2, (2, 60))
    expected = _reference(risk, time, event, at_risk)
    assert cause_specific_cindex(risk, time, event, at_risk) == expected


def test_empty_groups_lists_and_invalid_selected_rows():
    assert np.isnan(cause_specific_cindex([], [], [], []))
    assert cause_specific_cindex([0.9, 0.1], [1, 2], [1, 0], [1, 1]) == 1.
    assert cause_specific_cindex([np.nan, 0.9, 0.1], [np.nan, 1, 2], [np.nan, 1, 0], [0, 1, 1]) == 1.
    for values in [([0.9], [1, 2], [1], [1]), ([np.nan], [1], [1], [1]),
                   ([0.9], [1], [0.5], [1]), ([0.9], [1], [1], [2])]:
        with pytest.raises(ValueError):
            cause_specific_cindex(*values)


def test_large_monotone_cohort_has_exact_concordance():
    n = 15000
    assert cause_specific_cindex(np.arange(n)[::-1], np.arange(n), np.ones(n), np.ones(n)) == 1.
