"""Expert specialisation loss (manuscript Eq. 23).

The target distribution softmax(tau * sev) puts most weight on the organs with
the highest mean SOFA subscore in the observation window. The loss is
KL(target || route), where route is the mean routing distribution of the
patient over the window; it aligns expert k with organ k.
"""
from __future__ import annotations

import torch.nn.functional as F


def specialisation_loss(route, sofa_mean, tau: float = 2.0):
    """route: (B, 6) mean routing probabilities; sofa_mean: (B, 6) mean organ SOFA subscores."""
    target = F.softmax(sofa_mean * tau, dim=-1)
    return F.kl_div(route.clamp_min(1e-8).log(), target, reduction="batchmean")
