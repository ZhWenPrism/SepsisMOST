"""The Sepsis-MOST network.

    X (B, 12, 21) -> input projection (Eq. 5)
                  -> three dilated causal TCN blocks, dilations 1, 2, 4 (Eq. 6)
                  -> selective state space (Mamba) layer (Eq. 7)
                  -> KG-MoE: six organ experts, top-2 gate biased by the SOFAE graph (Eqs. 8-12)
                  -> attention pooling of the shared context and of each expert stream,
                     plus a static branch for S (B, 116) (Eq. 13)
                  -> six competing-risks organ heads, K bins x {fail, death, survive},
                     with the graph-derived coupling bias on the failure logit (Eqs. 14-16)
                  -> three adverse-event heads: ICU death, septic shock, MODS (Eq. 17)

Every organ output is a cause-specific cumulative incidence function in which
failure of the organ competes with ICU death.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import ADVERSE_EVENTS, N_BINS, N_ORGANS
from .encoders import SelectiveSSM, TCNBlock
from .heads import cumulative_incidence
from .pooling import AttentionPooling
from .routing import KGMoE


class SepsisMOST(nn.Module):
    """Sepsis-MOST network.

    Parameters
    ----------
    n_features, n_static
        Number of temporal features per hour and of static features.
    organ_embeddings
        SOFAE organ embeddings (6 x graph_dim), frozen.
    coupling
        Column-normalised SOFAE precedence matrix (6 x 6), frozen.
    """

    def __init__(self, n_features: int, n_static: int, organ_embeddings, coupling,
                 n_bins: int = N_BINS, d_model: int = 96, d_state: int = 16, d_static: int = 64,
                 expert_hidden: int = 128, tcn_dilations=(1, 2, 4), kernel_size: int = 3,
                 top_k: int = 2, dropout: float = 0.15):
        super().__init__()
        self.n_bins = n_bins
        # Temporal Feature Extractor
        self.in_proj = nn.Linear(n_features, d_model)
        self.tcn = nn.ModuleList([TCNBlock(d_model, kernel_size, d, dropout) for d in tcn_dilations])
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        # KG-MoE Predictor
        self.moe = KGMoE(d_model, graph_dim=organ_embeddings.shape[1], n_experts=N_ORGANS,
                         top_k=top_k, d_hidden=expert_hidden, dropout=dropout)
        self.pool_shared = AttentionPooling(d_model)
        self.pool_expert = nn.ModuleList([AttentionPooling(d_model) for _ in range(N_ORGANS)])
        # frozen SOFAE graph: organ embeddings (router) and coupling matrix
        self.register_buffer("graph_organ_emb", torch.as_tensor(organ_embeddings, dtype=torch.float32))
        self.register_buffer("graph_coupling", torch.as_tensor(coupling, dtype=torch.float32))
        self.coupling_lambda = nn.Parameter(torch.zeros(N_ORGANS))   # lambda_k = softplus(.) >= 0
        # static branch
        self.static_net = nn.Sequential(
            nn.LayerNorm(n_static), nn.Linear(n_static, d_static), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_static, d_static), nn.GELU())
        self.drop = nn.Dropout(dropout)
        # organ heads: [expert stream || shared context || static embedding] -> K bins x 3 states
        self.heads = nn.ModuleList([nn.Linear(2 * d_model + d_static, n_bins * 3)
                                    for _ in range(N_ORGANS)])
        # adverse-event heads: ICU death, septic shock, MODS
        self.event_head = nn.Linear(d_model + d_static, len(ADVERSE_EVENTS))

    def graph_state(self, x):
        """Graph vector e_t of each hour (Eq. 8): SOFA-2 domain scores / 4 times the organ embeddings."""
        sofa = x[:, :, :N_ORGANS].clamp(0, 4) / 4.0
        return sofa @ self.graph_organ_emb

    def forward(self, x, s):
        """x: (B, T, F) standardised temporal tensor, s: (B, P) standardised static vector."""
        e_kg = self.graph_state(x)
        h = self.in_proj(x)
        for blk in self.tcn:
            h = blk(h)
        h = self.ssm(h)
        shared, experts, gate, route, lb = self.moe(h, e_kg)
        ps = self.drop(self.pool_shared(shared))
        se = self.static_net(s)
        # organ coupling (Eq. 14): upstream dysfunction raises the downstream failure logit
        sev = (x[:, :, :N_ORGANS].clamp(0, 4) / 4.0).mean(1)                # (B, 6)
        coupling = F.softplus(self.coupling_lambda) * (sev @ self.graph_coupling)
        logits = []
        for k, head in enumerate(self.heads):
            pk = self.drop(self.pool_expert[k](experts[:, :, k, :]))
            lg = head(torch.cat([pk, ps, se], -1)).view(-1, self.n_bins, 3)   # Eq. 15
            lg = lg.clone()
            lg[..., 0] = lg[..., 0] + coupling[:, k:k + 1]                    # failure logit, all bins
            logits.append(lg)
        cr = torch.stack(logits, dim=1)                                       # (B, 6, K, 3)
        events = self.event_head(torch.cat([ps, se], -1))                     # (B, 3), Eq. 17
        return {"cr_logits": cr, "event_logits": events, "gate": gate.mean(1),
                "route": route.mean(1), "load_balance": lb}

    @torch.no_grad()
    def predict(self, x, s):
        """Return the organ CIFs (B, 6, K) and the adverse-event probabilities (B, 3)."""
        out = self.forward(x, s)
        cif = torch.stack([cumulative_incidence(out["cr_logits"][:, k]) for k in range(N_ORGANS)], 1)
        return cif, torch.sigmoid(out["event_logits"])
