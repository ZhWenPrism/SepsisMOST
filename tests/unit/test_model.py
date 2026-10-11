import numpy as np
import torch

from sepsis_most.model import SepsisMOST, cumulative_incidence
from sepsis_most.model.encoders import TCNBlock


def make_model(**kw):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    coup = rng.random((6, 6)).astype(np.float32)
    coup /= coup.sum(0, keepdims=True)
    return SepsisMOST(21, 116, emb, coup, **kw)


def inputs(b=5):
    torch.manual_seed(0)
    x = torch.randn(b, 12, 21)
    x[:, :, :6] = torch.randint(0, 5, (b, 12, 6)).float()
    return x, torch.randn(b, 116)


def test_output_shapes():
    m = make_model().eval()
    out = m(*inputs())
    assert out["cr_logits"].shape == (5, 6, 8, 3)
    assert out["event_logits"].shape == (5, 3)
    assert out["route"].shape == (5, 6) and out["gate"].shape == (5, 6)
    assert out["load_balance"].ndim == 0


def test_head_input_width():
    m = make_model()
    assert all(h.in_features == 256 for h in m.heads)      # 96 + 96 + 64
    assert m.event_head.in_features == 160                  # 96 + 64


def test_cif_is_monotone_and_bounded():
    m = make_model().eval()
    cif, prob = m.predict(*inputs())
    assert cif.shape == (5, 6, 8)
    assert torch.all(cif[..., 1:] >= cif[..., :-1] - 1e-7)
    assert torch.all((cif >= 0) & (cif <= 1)) and torch.all((prob >= 0) & (prob <= 1))


def test_failure_and_death_cif_do_not_exceed_one():
    logits = torch.randn(10, 8, 3)
    death = cumulative_incidence(logits[..., [1, 0, 2]])
    assert torch.all(cumulative_incidence(logits) + death <= 1 + 1e-6)


def test_tcn_is_causal():
    blk = TCNBlock(16, 3, 2, dropout=0.0).eval()
    x = torch.randn(2, 12, 16)
    y = x.clone()
    y[:, 8:] += 5.0
    assert torch.allclose(blk(x)[:, :8], blk(y)[:, :8])


def test_top_k_gate():
    m = make_model(top_k=2).eval()
    out = m(*inputs())
    # gate weights are averaged over time; at each hour exactly two experts are active
    x, s = inputs()
    h = m.ssm(torch.nn.Sequential(*m.tcn)(m.in_proj(x)))
    _, _, gate, _, _ = m.moe(h, m.graph_state(x))
    assert torch.all((gate > 0).sum(-1) == 2)
    torch.testing.assert_close(gate.sum(-1), torch.ones_like(gate.sum(-1)))
    assert torch.all(out["gate"] >= 0)


def test_coupling_only_raises_failure_logits():
    m = make_model().eval()
    x, s = inputs()
    base = m(x, s)["cr_logits"]
    with torch.no_grad():
        m.coupling_lambda.fill_(3.0)
    raised = m(x, s)["cr_logits"]
    assert torch.all(raised[..., 0] >= base[..., 0])
    torch.testing.assert_close(raised[..., 1:], base[..., 1:])
