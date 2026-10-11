"""Organ-specific experts of the KG-MoE Predictor (manuscript Eq. 11).

Each of the six experts is a two-layer feedforward network with its own
parameters, E_e(H_t) = W_e2 GELU(W_e1 H_t), so that expert e can specialise in
the physiology of organ e.
"""
from __future__ import annotations

import torch.nn as nn


def organ_expert(d_model: int, d_hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_hidden, d_model))


def organ_experts(n_experts: int, d_model: int, d_hidden: int, dropout: float) -> nn.ModuleList:
    return nn.ModuleList([organ_expert(d_model, d_hidden, dropout) for _ in range(n_experts)])
