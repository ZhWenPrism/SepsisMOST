import numpy as np
import pytest

from sepsis_most.constants import GRAPH_NODES
from sepsis_most.graph import build_sofae_graph


@pytest.mark.parametrize("dim", [0, -1, False, 2.5])
def test_invalid_embedding_dimension_is_rejected(onsets, dim):
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        build_sofae_graph(onsets, dim=dim)


def test_embedding_dimension_can_be_numpy_integer(onsets):
    assert build_sofae_graph(onsets, dim=np.int64(3)).embeddings.shape == (len(GRAPH_NODES), 3)
