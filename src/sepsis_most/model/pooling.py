"""Attention pooling over the hourly steps (manuscript Eq. 13)."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """p = sum_t a_t x_t with a = softmax(proj(x) q)."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.proj = nn.Linear(d, d)

    def forward(self, x):                              # x: (B, T, D) -> (B, D)
        a = (self.proj(x) @ self.q).softmax(-1)
        return (x * a.unsqueeze(-1)).sum(1)
