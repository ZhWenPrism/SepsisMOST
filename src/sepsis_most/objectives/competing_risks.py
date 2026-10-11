"""Discrete-time competing-risks negative log likelihood (manuscript Eq. 19).

For a patient at risk for organ k whose event or censoring falls in bin b with
cause c (0 = organ failure, 1 = ICU death before failure, -1 = censored), the
log likelihood is the log survival over the bins before b plus

* log P(cause c in bin b) for an observed event, or
* log P(survive bin b) for a censored patient.

Patients not at risk for the organ at the time of prediction are masked.
"""
from __future__ import annotations

import torch

from ..model.heads import cr_decode


def cr_nll(logits, cause, ebin, atrisk):
    """Mean negative log likelihood over the patients at risk for one organ.

    logits: (B, K, 3); cause: (B,) in {0, 1, -1}; ebin: (B,) bin index; atrisk: (B,) mask.
    """
    m = atrisk.bool()
    if m.sum() == 0:
        return logits.sum() * 0.0
    pr, logs, pre = cr_decode(logits[m])
    cz, bz = cause[m], ebin[m]
    ar = torch.arange(int(m.sum()), device=logits.device)
    ev = cz >= 0
    ll = pre[ar, bz].clone()                                   # survived up to the event bin
    ll = torch.where(ev, ll + torch.log(pr[ar, bz, cz.clamp(min=0)].clamp_min(1e-8)),
                     pre[ar, bz] + logs[ar, bz])               # censored: survived the last bin
    return -ll.mean()
