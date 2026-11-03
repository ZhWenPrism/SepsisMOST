import numpy as np
import pytest
import torch

from test_inference_inputs import ensemble


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_input_allocations_are_bounded_by_batch_size(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    original_tensor, original_as_tensor = torch.tensor, torch.as_tensor
    allocations = []
    def record(call):
        def wrapper(value, *args, **kwargs):
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                allocations.append(value.shape[0])
            return call(value, *args, **kwargs)
        return wrapper
    model = ensemble()
    monkeypatch.setattr(torch, "tensor", record(original_tensor))
    monkeypatch.setattr(torch, "as_tensor", record(original_as_tensor))
    result = model.predict(np.zeros((7, 3, 7)), np.zeros((7, 2)), batch_size=2, device=device)
    assert allocations and max(allocations) <= 2
    np.testing.assert_array_equal(result["event_prob"], np.full((7, 3), .5))
    assert result["cif"].shape == (7, 6, 2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_network_predictions_agree_across_batch_sizes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from sepsis_most.training import SepsisMOSTEnsemble, Standardiser, TrainingConfig, build_model
    torch.manual_seed(7)
    cfg = TrainingConfig.from_dict({"model": {"d_model": 16, "d_state": 4, "d_static": 8,
                                             "expert_hidden": 16, "tcn_dilations": [1]}})
    embeddings, coupling = np.zeros((6, 4), np.float32), np.eye(6, dtype=np.float32)
    model = build_model(7, 2, 2, embeddings, coupling, cfg)
    std = Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2))
    ens = SepsisMOSTEnsemble([model], std, embeddings, coupling, [6., 12.], cfg)
    rng = np.random.default_rng(7)
    X, S = rng.normal(size=(5, 3, 7)), rng.normal(size=(5, 2))
    whole = ens.predict(X, S, batch_size=5, device=device)
    batched = ens.predict(X, S, batch_size=2, device=device)
    for key in whole:
        np.testing.assert_allclose(batched[key], whole[key], atol=1e-6, rtol=1e-5)
