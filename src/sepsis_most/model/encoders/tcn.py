"""Dilated causal temporal convolution blocks (manuscript Eq. 6)."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D convolution padded on the left by (k - 1) d, so no future step leaks."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):                              # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    """H_l = LayerNorm(H_{l-1} + Conv_d(GELU(Conv_d(H_{l-1}))))."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, kernel_size, dilation), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, kernel_size, dilation), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):                              # x: (B, T, C)
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + h)
