"""Input augmentation applied to training batches.

Gaussian jitter is added to the standardised inputs (scaled by 0.3 on the SOFA
channels, which keep their native scale), and each patient's temporal and
static inputs are rescaled by a random factor around one.
"""
from __future__ import annotations

import torch

from ..constants import N_RAW_CHANNELS


def jitter(xb, sb, sigma: float):
    """Return augmented copies of a temporal batch (B, T, F) and a static batch (B, P)."""
    nz = torch.randn_like(xb) * sigma
    nz[:, :, :N_RAW_CHANNELS] *= 0.3
    scale = 1.0 + torch.randn(xb.shape[0], 1, 1, device=xb.device) * 0.05
    xb = (xb + nz) * scale
    sb = sb * (1.0 + torch.randn(sb.shape[0], 1, device=sb.device) * 0.05) + torch.randn_like(sb) * (sigma * 0.5)
    return xb, sb
