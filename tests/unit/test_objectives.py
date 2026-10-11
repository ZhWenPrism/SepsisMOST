import torch

from sepsis_most.objectives import LossWeights, cr_nll, joint_objective, ranking_loss, specialisation_loss


def test_cr_nll_masks_patients_not_at_risk():
    torch.manual_seed(0)
    logits = torch.randn(6, 8, 3, requires_grad=True)
    cause = torch.tensor([0, 1, -1, 0, 1, -1])
    ebin = torch.tensor([0, 3, 7, 2, 5, 1])
    full = cr_nll(logits, cause, ebin, torch.ones(6))
    part = cr_nll(logits, cause, ebin, torch.tensor([1, 1, 1, 0, 0, 0]))
    assert torch.isfinite(full) and full > 0 and not torch.isclose(full, part)
    assert cr_nll(logits, cause, ebin, torch.zeros(6)) == 0


def test_ranking_loss_prefers_correct_order():
    etime = torch.tensor([[10.0] * 6, [100.0] * 6])
    evt = torch.tensor([[1] * 6, [0] * 6])
    atrisk = torch.ones(2, 6)
    good = ranking_loss(torch.tensor([[0.9] * 6, [0.1] * 6]), etime, evt, atrisk)
    bad = ranking_loss(torch.tensor([[0.1] * 6, [0.9] * 6]), etime, evt, atrisk)
    assert good < bad


def test_specialisation_loss_is_zero_at_target():
    sev = torch.rand(4, 6)
    target = torch.softmax(sev * 2.0, -1)
    assert specialisation_loss(target, sev, tau=2.0).abs() < 1e-6


def test_joint_objective_backpropagates():
    from test_model import inputs, make_model
    m = make_model()
    x, s = inputs(8)
    g = torch.Generator().manual_seed(1)
    cause = torch.randint(-1, 2, (8, 6), generator=g)
    ebin = torch.randint(0, 8, (8, 6), generator=g)
    atrisk = torch.ones(8, 6, dtype=torch.long)
    evt = (cause == 0).long()
    etime = torch.rand(8, 6, generator=g) * 168
    events = torch.randint(0, 2, (8, 3), generator=g).float()
    out = m(x, s)
    loss = joint_objective(out, cause, ebin, atrisk, etime, evt, events, x[:, :, :6].mean(1), LossWeights())
    loss.backward()
    assert torch.isfinite(loss)
    assert m.heads[0].weight.grad is not None and m.moe.kg_gate.weight.grad is not None
