"""Graph-biased expert routing of the KG-MoE Predictor (manuscript Eqs. 9-12, 22).

At each hour the router adds the graph signal e_t (Eq. 8) to the data signal
before the softmax, so that gating is biased toward the experts that the SOFAE
graph marks as relevant (Eq. 9). Only the two experts with the highest scores
are kept and their gate weights are renormalised to sum to one (Eq. 10). The
gated mixture forms the shared context U across organs (Eq. 12), while the
outputs of all six experts are returned as organ streams regardless of the
gate, so every organ head always receives its own expert stream.

The module also returns the Switch-Transformer load-balancing term (Eq. 22),
N_E * sum_e f_e p_e, where f_e is the fraction of tokens routed to expert e and
p_e its mean routing probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..experts import organ_experts


class KGMoE(nn.Module):
    def __init__(self, d_model: int, graph_dim: int, n_experts: int = 6, top_k: int = 2,
                 d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.experts = organ_experts(n_experts, d_model, d_hidden, dropout)
        self.gate = nn.Linear(d_model, n_experts)                     # data signal W_g
        self.kg_gate = nn.Linear(graph_dim, n_experts, bias=False)    # graph signal W_kg
        self.alpha = nn.Parameter(torch.tensor(1.0))                  # weight of the graph signal
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, e_kg):                        # x: (B, T, D), e_kg: (B, T, graph_dim)
        B, T, D = x.shape
        h = self.norm(x)
        logits = self.gate(h) + self.alpha * self.kg_gate(e_kg)
        probs = F.softmax(logits, dim=-1)                              # Eq. 9
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)              # Eq. 10
        ex = torch.stack([e(h) for e in self.experts], dim=2)          # (B, T, E, D), Eq. 11
        gate_full = torch.zeros(B, T, self.n_experts, device=x.device)
        gate_full.scatter_(-1, topi, topv)
        shared = x + (ex * gate_full.unsqueeze(-1)).sum(2)             # U, Eq. 12
        imp = probs.mean(dim=(0, 1))
        load = gate_full.gt(0).float().mean(dim=(0, 1))
        lb = self.n_experts * (imp * load).sum()                       # Eq. 22
        return shared, ex, gate_full, probs, lb
