from unittest.mock import patch
import torch
import torch.nn.functional as F
from sepsis_most.objectives import cr_nll


def test_likelihood_computes_log_probabilities_once():
    logits = torch.zeros(3, 2, 3)
    with patch.object(F, "log_softmax", wraps=F.log_softmax) as softmax:
        loss = cr_nll(logits, torch.tensor([0, 1, -1]),
                      torch.tensor([0, 1, 1]), torch.ones(3))
    assert torch.isfinite(loss)
    assert softmax.call_count == 1


def test_likelihood_values_and_gradients_match_the_multinomial_definition():
    values = torch.arange(18, dtype=torch.float64).reshape(3, 2, 3) / 7
    actual, reference = values.clone().requires_grad_(), values.clone().requires_grad_()
    loss = cr_nll(actual, torch.tensor([0, 1, -1]),
                  torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0]))
    probabilities = F.log_softmax(reference, -1)
    expected = -(probabilities[0, 0, 0] + probabilities[1, 0, 2]
                 + probabilities[1, 1, 1]) / 2
    loss.backward()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(actual.grad, reference.grad)
