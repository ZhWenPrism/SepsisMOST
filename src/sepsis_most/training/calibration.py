"""Optional post-hoc isotonic calibration of the 7-day organ CIF.

A monotone map from the predicted 7-day CIF to the observed failure rate is
fitted with the pool-adjacent-violators algorithm on the validation patients at
risk for each organ, and applied by linear interpolation between the knots.
"""
from __future__ import annotations

import numpy as np


def isotonic_fit(x, y):
    """Pool-adjacent-violators fit; returns the sorted knots and fitted values."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    vals, wts = list(ys), list(np.ones_like(ys))
    blocks = [[i] for i in range(len(ys))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:              # merge violating neighbours
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            wts[i] += wts[i + 1]
            blocks[i] += blocks[i + 1]
            del vals[i + 1]
            del wts[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = np.zeros(len(ys))
    for v, blk in zip(vals, blocks):
        fitted[blk] = v
    return xs, fitted


def isotonic_apply(knots_x, knots_y, x):
    return np.interp(x, knots_x, knots_y)
