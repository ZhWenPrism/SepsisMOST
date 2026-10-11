"""Spectral node embeddings of the SOFAE graph (manuscript Eq. 4).

The three relation matrices and their transposes are stacked into
M = [R_prec, R_co, R_trig, R_prec^T, R_co^T, R_trig^T] of size N x 6N, so that
each node is described by both its outgoing and incoming relations. A truncated
singular value decomposition M = U S V^T keeps the top eight directions, and
each node embedding U[:, :8] S[:8] is l2-normalised so that the routing bias it
produces has a stable scale.
"""
from __future__ import annotations

import numpy as np

from ..relations import RelationMatrices


def stacked_relations(relations: RelationMatrices) -> np.ndarray:
    """Return M = [R_prec, R_co, R_trig, R_prec^T, R_co^T, R_trig^T]."""
    r_prec, r_co, r_tr = relations.precedence, relations.cooccurrence, relations.trigger
    return np.concatenate([r_prec, r_co, r_tr, r_prec.T, r_co.T, r_tr.T], axis=1)


def spectral_embeddings(relations: RelationMatrices, dim: int = 8):
    """Return the l2-normalised node embeddings (N x dim) and all singular values."""
    m = stacked_relations(relations)
    u, sv, _ = np.linalg.svd(m, full_matrices=False)
    k = min(dim, m.shape[0])
    emb = (u[:, :k] * sv[:k]).astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb, sv.astype(np.float32)
