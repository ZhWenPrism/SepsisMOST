import numpy as np
import pytest
import torch

from sepsis_most.training import SepsisMOSTEnsemble, Standardiser, TrainingConfig


class DummyModel(torch.nn.Module):
    def predict(self, X, S):
        return X.new_zeros((len(X), 6, 2)), S.new_full((len(S), 3), .5)


def ensemble(members=None, isotonic=None):
    return SepsisMOSTEnsemble([DummyModel()] if members is None else members,
                             Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2)),
                             np.zeros((6, 4)), np.eye(6), [6., 12.], TrainingConfig(), isotonic)


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="member"):
        ensemble([])


@pytest.mark.parametrize("X,S", [(np.zeros((2, 3, 7)), np.zeros((2, 1))),
                                  (np.zeros((2, 3, 7)), np.zeros((1, 2))),
                                  (np.zeros((2, 7)), np.zeros((2, 2)))])
def test_predict_rejects_broadcasting_and_row_mismatches(X, S):
    with pytest.raises(ValueError, match="inputs"):
        ensemble().predict(X, S)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        ensemble().predict(np.zeros((2, 3, 7)), np.zeros((2, 2)), batch_size=batch_size)


def test_empty_cohort_returns_typed_empty_predictions():
    maps = [(np.array([0., 1.]), np.array([0., 1.]))] * 6
    result = ensemble(isotonic=maps).predict(np.empty((0, 3, 7)), np.empty((0, 2)), calibrated=True)
    assert result["cif"].shape == (0, 6, 2)
    assert result["event_prob"].shape == (0, 3)
    assert result["cif_7d_calibrated"].shape == (0, 6)
