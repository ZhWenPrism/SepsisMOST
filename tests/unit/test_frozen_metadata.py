import numpy as np
import torch

from sepsis_most.training import SepsisMOSTEnsemble, Standardiser, TrainingConfig, build_model


def test_frozen_models_and_saved_metadata_do_not_alias_constructor_inputs(tmp_path):
    config = TrainingConfig()
    config.model.d_model = 8
    config.model.d_state = 2
    config.model.d_static = 4
    config.model.expert_hidden = 8
    config.model.dropout = 0
    embeddings = np.arange(18, dtype=np.float32).reshape(6, 3) / 20
    coupling = np.eye(6, dtype=np.float32)
    edges = np.array([12., 24.])
    standardiser = Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2))
    isotonic = [(np.array([0., 1.]), np.array([0., 1.])) for _ in range(6)]
    model = build_model(7, 2, 2, embeddings, coupling, config).eval()
    ensemble = SepsisMOSTEnsemble([model], standardiser, embeddings, coupling, edges, config, isotonic)
    x = np.ones((2, 3, 7), dtype=np.float32)
    s = np.ones((2, 2), dtype=np.float32)
    before = ensemble.predict(x, s, calibrated=True)

    embeddings[:] = 0
    coupling[:] = 0
    edges[:] = 999
    standardiser.mu[:] = 10
    isotonic[0][1][:] = 0
    config.model.d_model = 32

    after = ensemble.predict(x, s, calibrated=True)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    path = tmp_path / "frozen.pt"
    ensemble.save(path)
    restored = SepsisMOSTEnsemble.load(path)
    assert restored.config.model.d_model == 8
    np.testing.assert_array_equal(restored.bin_edges, [12, 24])
    for name, expected in before.items():
        np.testing.assert_allclose(restored.predict(x, s, calibrated=True)[name], expected, atol=1e-7)
    torch.testing.assert_close(model.graph_coupling, torch.eye(6))
