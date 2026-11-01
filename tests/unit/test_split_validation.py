import numpy as np
import pytest

from sepsis_most.training.data import stratified_split


def test_list_labels_are_split_without_losing_rows():
    splits = stratified_split([0, 1] * 10, seed=3)
    assert sorted(np.concatenate(splits).tolist()) == list(range(20))


def test_empty_partitions_are_valid_integer_index_arrays():
    splits = stratified_split(np.array([0, 1]), seed=3)
    for part in splits:
        assert np.issubdtype(part.dtype, np.integer)
        np.arange(2)[part]


@pytest.mark.parametrize("death", [np.array([0, 2]), np.array([0, np.nan]), np.array([[0, 1]])])
def test_invalid_outcomes_cannot_be_silently_omitted(death):
    with pytest.raises(ValueError):
        stratified_split(death)


@pytest.mark.parametrize("fractions", [(0.8, 0.3, -0.1), (0.5, 0.2, 0.2), (0.7, 0.3), (0.7, np.nan, 0.3)])
def test_invalid_split_fractions_are_rejected(fractions):
    with pytest.raises(ValueError):
        stratified_split(np.array([0, 1] * 10), fractions=fractions)


def test_valid_split_is_disjoint_complete_and_reproducible():
    first = stratified_split(np.array([0, 1] * 20), seed=8)
    second = stratified_split(np.array([0, 1] * 20), seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(set(np.concatenate(first).tolist())) == 40

