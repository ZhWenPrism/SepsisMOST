"""Ranking loss as in DeepHit (manuscript Eq. 21).

For each organ, every patient i with an observed failure is paired with every
at-risk patient j whose event or censoring time is later, and the loss
exp(-(F_i - F_j) / sigma) on the 7-day CIF encourages earlier events to receive
a higher predicted risk. The loss is averaged over the comparable pairs of each
organ and then over the organs that have at least one event in the batch.
"""
from __future__ import annotations

import torch

from ..constants import N_ORGANS


def ranking_loss(cif_h, etime, evt, atrisk, sigma: float = 0.1):
    """cif_h: (B, 6) CIF at the horizon; etime, evt, atrisk: (B, 6)."""
    loss = cif_h.new_zeros(())
    cnt = 0
    for k in range(N_ORGANS):
        m = atrisk[:, k].bool()
        e = evt[:, k].bool() & m
        if e.sum() == 0 or m.sum() < 2:
            continue
        r, t = cif_h[:, k], etime[:, k]
        ei = torch.where(e)[0]
        ri, ti = r[ei].unsqueeze(1), t[ei].unsqueeze(1)
        rj, tj = r[m].unsqueeze(0), t[m].unsqueeze(0)
        A = (tj > ti).float()                                  # comparable pairs
        loss = loss + (torch.exp(-(ri - rj) / sigma) * A).sum() / A.sum().clamp(min=1)
        cnt += 1
    return loss / max(cnt, 1)
