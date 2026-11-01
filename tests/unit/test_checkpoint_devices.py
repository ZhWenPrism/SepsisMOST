import numpy as np
import pytest
import torch

from sepsis_most.training.config import TrainingConfig
from sepsis_most.training.data import Standardiser
from sepsis_most.training.ensemble import SepsisMOSTEnsemble
from sepsis_most.training.trainer import build_model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_loading_keeps_numpy_metadata_on_cpu(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(3)
    cfg = TrainingConfig.from_dict({"model": {"d_model": 16, "d_state": 4, "d_static": 8,
                                              "expert_hidden": 16, "tcn_dilations": [1]}})
    embeddings = np.zeros((6, 4), dtype=np.float32)
    coupling = np.eye(6, dtype=np.float32)
    model = build_model(7, 2, 2, embeddings, coupling, cfg)
    standardiser = Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2))
    ensemble = SepsisMOSTEnsemble([model], standardiser, embeddings, coupling, [6.0, 12.0], cfg)
    X, S = np.zeros((2, 3, 7), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)
    expected = ensemble.predict(X, S)
    checkpoint = tmp_path / "model.pt"
    ensemble.save(checkpoint)
    restored = SepsisMOSTEnsemble.load(checkpoint, device=device)
    assert next(restored.members[0].parameters()).device.type == device
    assert isinstance(restored.standardiser.mu, np.ndarray)
    actual = restored.predict(X, S, device=device)
    np.testing.assert_allclose(actual["cif"], expected["cif"], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual["event_prob"], expected["event_prob"], atol=1e-6, rtol=1e-5)

