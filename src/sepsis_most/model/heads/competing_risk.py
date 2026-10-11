"""Discrete-time multinomial competing-risks decoding (manuscript Eqs. 15-16).

Within each of the K time bins an organ head gives a distribution over three
mutually exclusive states: failure of the organ, ICU death before failure, and
survival free of both. The probability of remaining free of both up to the
start of a bin is the product of the survival probabilities of the earlier
bins, and the cumulative incidence function (CIF) of organ failure adds up the
failure probability of each bin weighted by the survival before it. Failure and
death share one distribution, so their CIFs together cannot exceed one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cr_decode(logits):
    """logits (*, K, 3) -> state probabilities, log P(survive bin), log S before bin."""
    pr = F.softmax(logits, dim=-1)                                  # [fail, death, survive]
    logs = torch.log(pr[..., 2].clamp(1e-6, 1 - 1e-6))             # log survival in each bin
    pre = torch.cat([torch.zeros_like(logs[..., :1]),
                     torch.cumsum(logs, -1)[..., :-1]], -1)         # log survival before each bin
    return pr, logs, pre


def cumulative_incidence(logits):
    """Cause-specific CIF of organ failure at the end of each bin, (*, K)."""
    pr, _, pre = cr_decode(logits)
    return torch.cumsum(torch.exp(pre) * pr[..., 0], -1)
