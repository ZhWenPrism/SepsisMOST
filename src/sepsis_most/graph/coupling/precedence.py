"""Organ coupling matrix derived from the SOFAE graph.

The organ-to-organ block of the temporal-precedence relation is normalised by
column, so that entry (j, k) is the share of the inflow into organ k that comes
from organ j. The model multiplies the current organ dysfunction by this matrix
to obtain the coupling bias of each downstream organ (manuscript Eq. 14).
"""
from __future__ import annotations

import numpy as np

from ...constants import N_ORGANS


def coupling_matrix(precedence: np.ndarray) -> np.ndarray:
    """Column-normalise the 6 x 6 organ block of the precedence matrix."""
    prec = np.asarray(precedence)[:N_ORGANS, :N_ORGANS].astype("float32")
    return prec / (prec.sum(0, keepdims=True) + 1e-8)
