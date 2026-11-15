import pytest
import torch

from sepsis_most.model.routing import KGMoE


@pytest.mark.parametrize(("n_experts", "top_k", "match"), [
    (0, 1, "n_experts"), (True, 1, "n_experts"), (6, 0, "top_k"),
    (6, 7, "top_k"), (6, 1.5, "top_k"),
])
def test_router_fails_at_construction_for_invalid_expert_counts(n_experts, top_k, match):
    with pytest.raises(ValueError, match=match):
        KGMoE(4, 3, n_experts=n_experts, top_k=top_k)


def test_router_keeps_requested_experts_and_normalizes_weights():
    router = KGMoE(4, 3, n_experts=3, top_k=2, dropout=0).eval()
    _, _, gate, _, _ = router(torch.ones(1, 2, 4), torch.ones(1, 2, 3))
    assert (gate.count_nonzero(dim=-1) == 2).all()
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones(1, 2))
