import torch
import torch.nn.functional as F

from sepsis_most.model.heads import cr_decode, cumulative_incidence
from sepsis_most.objectives import cr_nll


def test_rare_event_likelihood_keeps_its_gradient():
    logits = torch.tensor([[[-100., 0., 0.]]], dtype=torch.float64, requires_grad=True)
    loss = cr_nll(logits, torch.tensor([0]), torch.tensor([0]), torch.ones(1))
    expected = -F.log_softmax(logits, -1)[0, 0, 0]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 0, 0] < -0.99


def test_log_survival_matches_multinomial_even_at_extreme_logits():
    logits = torch.tensor([[[100., 0., -100.], [0., 0., 100.]]], dtype=torch.float64)
    _, logs, pre = cr_decode(logits)
    expected = F.log_softmax(logits, -1)[..., 2]
    torch.testing.assert_close(logs, expected)
    torch.testing.assert_close(pre[:, 1], expected[:, 0])
    loss = cr_nll(logits, torch.tensor([-1]), torch.tensor([1]), torch.ones(1))
    torch.testing.assert_close(loss, -expected.sum())


def test_failure_and_death_cif_remain_bounded_after_extreme_survival():
    logits = torch.tensor([[[-100., -100., 100.], [100., -100., -100.]]], dtype=torch.float64)
    failure = cumulative_incidence(logits)
    death = cumulative_incidence(logits[..., [1, 0, 2]])
    assert torch.all(failure + death <= 1. + 1e-12)


def test_moderate_logits_match_manual_likelihood():
    logits = torch.tensor([[[1., 2., 3.], [2., 1., 3.]]], dtype=torch.float64)
    probabilities = logits.softmax(-1)
    expected = -torch.log(probabilities[0, 0, 2] * probabilities[0, 1, 1])
    loss = cr_nll(logits, torch.tensor([1]), torch.tensor([1]), torch.ones(1))
    torch.testing.assert_close(loss, expected)
