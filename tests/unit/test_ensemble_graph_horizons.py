import numpy as np
import pytest

from test_inference_inputs import DummyModel
from sepsis_most.training import SepsisMOSTEnsemble, Standardiser, TrainingConfig


@pytest.mark.parametrize("field,value,message", [
    ("bin_edges", [], "bin_edges"),
    ("bin_edges", [12., 6.], "bin_edges"),
    ("bin_edges", [6., float("inf")], "bin_edges"),
    ("organ_embeddings", np.zeros((5, 4)), "frozen graph"),
    ("coupling", np.eye(7), "frozen graph"),
])
def test_frozen_array_shapes_cannot_describe_different_outputs(field, value, message):
    options = dict(organ_embeddings=np.zeros((6, 4)), coupling=np.eye(6), bin_edges=[6., 12.])
    options[field] = value
    with pytest.raises(ValueError, match=message):
        SepsisMOSTEnsemble([DummyModel()], Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2)),
                           config=TrainingConfig(), **options)


def test_time_bins_must_match_the_network_heads():
    member = DummyModel()
    member.n_bins = 3
    with pytest.raises(ValueError, match="every ensemble member"):
        SepsisMOSTEnsemble([member], Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2)),
                           np.zeros((6, 4)), np.eye(6), [6., 12.], TrainingConfig())
