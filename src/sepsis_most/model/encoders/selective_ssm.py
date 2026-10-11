"""Selective state space (Mamba) layer (manuscript Eq. 7).

For each channel the layer produces input-dependent parameters (Delta_t, B_t,
C_t) and runs the selective scan

    h_t = exp(Delta_t A) * h_{t-1} + (Delta_t x_t) B_t,   y_t = C_t h_t + D x_t.

The output is gated by SiLU and added back as a residual. The observation
window has only 12 hourly steps, so the scan is written as an explicit
recurrence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_in = expand * d_model
        self.d_in, self.d_state = d_in, d_state
        self.in_proj = nn.Linear(d_model, 2 * d_in)
        self.x_proj = nn.Linear(d_in, d_state * 2 + d_in)
        self.dt_bias = nn.Parameter(torch.zeros(d_in))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_in, 1)))
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                              # x: (B, T, D)
        B, T, _ = x.shape
        res = x
        xin, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        A = -torch.exp(self.A_log)
        Bm, Cm, delta = torch.split(self.x_proj(xin),
                                    [self.d_state, self.d_state, self.d_in], dim=-1)
        delta = F.softplus(delta + self.dt_bias)
        h = x.new_zeros(B, self.d_in, self.d_state)
        ys = []
        for t in range(T):
            dt = delta[:, t]
            dA = torch.exp(dt.unsqueeze(-1) * A)
            dBx = (dt * xin[:, t]).unsqueeze(-1) * Bm[:, t].unsqueeze(1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1) + self.D * xin[:, t])
        y = torch.stack(ys, dim=1) * F.silu(z)          # SiLU gate
        return res + self.out_proj(y)                    # residual
