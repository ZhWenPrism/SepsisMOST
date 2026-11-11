import torch

from sepsis_most.objectives import ranking_loss


def test_incomparable_pairs_cannot_overflow_loss_or_gradients():
    risks = torch.tensor([[1.0] * 6, [0.0] * 6, [0.0] * 6], requires_grad=True)
    times = torch.tensor([[1.0] * 6, [2.0] * 6, [3.0] * 6])
    events = torch.tensor([[0] * 6, [1] * 6, [0] * 6])

    loss = ranking_loss(risks, times, events, torch.ones_like(events), sigma=0.01)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert torch.isfinite(risks.grad).all()
    assert not risks.grad[0].any()



def test_ranking_accumulates_half_precision_pairs_in_float32():
    risks = torch.tensor([[0.0] * 6, [1.0] * 6, [1.0] * 6, [1.0] * 6, [1.0] * 6],
                         dtype=torch.float16, requires_grad=True)
    times = torch.arange(5).unsqueeze(-1).expand(-1, 6)
    events = torch.zeros_like(times)
    events[0] = 1

    loss = ranking_loss(risks, times, events, torch.ones_like(events))
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.exp(torch.tensor(10.0, dtype=torch.float16)).float())
    assert torch.isfinite(risks.grad).all()
