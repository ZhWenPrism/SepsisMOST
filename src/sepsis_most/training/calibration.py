"""Optional post-hoc isotonic calibration of the 7-day organ CIF.

A monotone map from the predicted 7-day CIF to the observed failure rate is
fitted with the pool-adjacent-violators algorithm on the validation patients at
risk for each organ, and applied by linear interpolation between the knots.
"""
from __future__ import annotations

import numpy as np


def isotonic_fit(x, y):
    """Pool-adjacent-violators fit; returns the sorted knots and fitted values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("calibration inputs must be non-empty aligned vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration inputs must be finite")
    order = np.argsort(x, kind="stable")
    xs, starts, counts = np.unique(x[order], return_index=True, return_counts=True)
    ys = np.add.reduceat(y[order], starts) / counts
    blocks = []
    for index, (value, weight) in enumerate(zip(ys, counts)):
        blocks.append((index, index + 1, float(value), float(weight)))
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2] + 1e-12:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[3] + right[3]
            value = (left[2] * left[3] + right[2] * right[3]) / weight
            blocks.append((left[0], right[1], value, weight))
    fitted = np.zeros(len(ys))
    for start, stop, value, _ in blocks:
        fitted[start:stop] = value
    return xs, fitted


def isotonic_apply(knots_x, knots_y, x):
    knots_x, knots_y = np.asarray(knots_x, dtype=float), np.asarray(knots_y, dtype=float)
    if (knots_x.ndim != 1 or not len(knots_x) or knots_y.shape != knots_x.shape
            or not np.isfinite(knots_x).all() or not np.isfinite(knots_y).all()
            or np.any(knots_x[1:] <= knots_x[:-1])):
        raise ValueError("calibration knots must be finite aligned vectors with strictly increasing inputs")
    return np.interp(x, knots_x, knots_y)
