import numpy as np
import pytest

from sepsis_most.graph.relations import add_mods_onset, prune, relation_matrices


@pytest.mark.parametrize("min_support", [True, -1, 2.5, np.nan])
def test_pruning_validates_patient_count(onsets, min_support):
    with pytest.raises(ValueError, match="min_support must be a non-negative integer"):
        prune(relation_matrices(add_mods_onset(onsets)), min_support)


def test_zero_support_threshold_leaves_weights_unchanged(onsets):
    relations = relation_matrices(add_mods_onset(onsets))
    unpruned = prune(relations, np.int64(0))
    for name in ("precedence", "cooccurrence", "trigger"):
        np.testing.assert_array_equal(getattr(unpruned, name), getattr(relations, name))
