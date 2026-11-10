import pytest
import torch

from sepsis_most.model.routing.kg_moe import KGMoE


def test_double_precision_routing_preserves_dtype_and_gradients():
    torch.manual_seed(2)
    router = KGMoE(8, 3, d_hidden=12, dropout=0).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    graph = torch.randn(2, 4, 3, dtype=torch.float64)
    shared, _, gate, probabilities, _ = router(x, graph)
    assert gate.dtype == probabilities.dtype == torch.float64
    torch.testing.assert_close(gate.sum(-1), torch.ones(2, 4, dtype=torch.float64))
    shared.square().mean().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device, dtype", [("cpu", torch.bfloat16), ("cuda", torch.float16)])
def test_autocast_routing_can_scatter_its_probabilities(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    router = KGMoE(8, 3, d_hidden=12, dropout=0).to(device).eval()
    x, graph = torch.randn(2, 4, 8, device=device), torch.randn(2, 4, 3, device=device)
    with torch.autocast(device_type=device, dtype=dtype):
        shared, _, gate, probabilities, _ = router(x, graph)
    assert gate.dtype == probabilities.dtype
    assert torch.isfinite(shared).all()
