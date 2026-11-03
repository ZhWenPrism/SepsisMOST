import numpy as np
import pytest
import torch

from conftest import synthetic_cohort
from sepsis_most.training import CohortTensors, TrainingConfig, train_network


def test_training_cannot_return_an_unselected_model(monkeypatch):
    temporal, static = synthetic_cohort(n=8)
    data = CohortTensors(X=temporal["X"], S=static["S"], atrisk=temporal["atrisk"].astype(np.int64),
                         cause=temporal["cause"].astype(np.int64), ebin=temporal["ebin"].astype(np.int64), evt=temporal["evt"].astype(np.int64),
                         etime=temporal["etime"], events=np.zeros((8, 3)), bin_edges=temporal["bin_edges"])
    cfg = TrainingConfig.from_dict({"model": {"d_model": 16, "d_state": 4, "d_static": 8,
                                             "expert_hidden": 16, "tcn_dilations": [1]},
                                    "optimisation": {"epochs": 1}})
    monkeypatch.setattr("sepsis_most.training.trainer.mean_cause_specific_cindex", lambda *a: float("nan"))
    with pytest.raises(ValueError, match="validation"):
        train_network(3, data.X, data.S, data, np.arange(6), np.arange(6, 8),
                      np.zeros((6, 4), np.float32), np.eye(6, dtype=np.float32), cfg, torch.device("cpu"))


@pytest.mark.parametrize("epochs,min_delta", [(0, 1e-4), (1, 2.)])
def test_training_without_a_selected_epoch_is_rejected(monkeypatch, epochs, min_delta):
    temporal, static = synthetic_cohort(n=8)
    data = CohortTensors(X=temporal["X"], S=static["S"], atrisk=temporal["atrisk"].astype(np.int64),
                         cause=temporal["cause"].astype(np.int64), ebin=temporal["ebin"].astype(np.int64),
                         evt=temporal["evt"].astype(np.int64), etime=temporal["etime"],
                         events=np.zeros((8, 3)), bin_edges=temporal["bin_edges"])
    cfg = TrainingConfig.from_dict({"model": {"d_model": 16, "d_state": 4, "d_static": 8,
                                             "expert_hidden": 16, "tcn_dilations": [1]},
                                    "optimisation": {"epochs": epochs, "min_delta": min_delta}})
    monkeypatch.setattr("sepsis_most.training.trainer.mean_cause_specific_cindex", lambda *a: .5)
    with pytest.raises(ValueError, match="validation"):
        train_network(3, data.X, data.S, data, np.arange(6), np.arange(6, 8),
                      np.zeros((6, 4), np.float32), np.eye(6, dtype=np.float32), cfg, torch.device("cpu"))
