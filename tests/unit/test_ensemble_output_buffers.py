import tracemalloc
import numpy as np
import torch
from sepsis_most.training import SepsisMOSTEnsemble, Standardiser, TrainingConfig


class Member(torch.nn.Module):
    n_bins = 8

    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def predict(self, x, s):
        return (x[:, 0, 0, None, None].expand(-1, 6, 8) + self.bias,
                s[:, 0, None].expand(-1, 3) + self.bias)


def ensemble():
    return SepsisMOSTEnsemble(
        [Member(i * .01) for i in range(8)],
        Standardiser(np.zeros(7), np.ones(7), np.zeros(2), np.ones(2)),
        np.zeros((6, 4)), np.eye(6), np.arange(1., 9.), TrainingConfig())


def test_ensemble_output_buffers_do_not_scale_with_member_count():
    model = ensemble()
    x, s = np.zeros((5000, 1, 7), np.float32), np.zeros((5000, 2), np.float32)
    model.predict(x[:1], s[:1], batch_size=128)
    tracemalloc.start()
    try:
        result = model.predict(x, s, batch_size=128)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8_000_000
    np.testing.assert_allclose(result["cif"], .035, atol=1e-7)
    np.testing.assert_allclose(result["event_prob"], .035, atol=1e-7)


def test_batch_averaging_preserves_values_dtype_and_row_order():
    model = ensemble()
    x, s = np.zeros((9, 1, 7), np.float32), np.zeros((9, 2), np.float32)
    x[:, 0, 0] = np.arange(9) / 20
    s[:, 0] = np.arange(9) / 30
    result = model.predict(x, s, batch_size=4)
    expected_cif = np.mean([x[:, 0, 0] + i * .01 for i in range(8)], axis=0)
    expected_event = np.mean([s[:, 0] + i * .01 for i in range(8)], axis=0)
    assert result["cif"].dtype == np.float32 and result["event_prob"].dtype == np.float32
    for organ in range(6):
        np.testing.assert_array_equal(result["cif"][:, organ, -1], expected_cif)
    for event in range(3):
        np.testing.assert_array_equal(result["event_prob"][:, event], expected_event)
