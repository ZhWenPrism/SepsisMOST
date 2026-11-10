"""Cause-specific concordance index used for model selection.

For one organ, each patient a with an observed failure is compared with every
at-risk patient whose event or censoring time is later. A pair is concordant
when patient a has the higher predicted 7-day CIF, and ties count one half.
The mean of this index over the six organ systems on the validation split is the
early-stopping criterion.
"""
from __future__ import annotations

import numpy as np


def cause_specific_cindex(risk, etime, evt, atrisk) -> float:
    """risk, etime, evt, atrisk: (N,) arrays for one organ."""
    risk, etime, evt, atrisk = [np.asarray(values) for values in (risk, etime, evt, atrisk)]
    if risk.ndim != 1 or any(values.shape != risk.shape for values in (etime, evt, atrisk)):
        raise ValueError("concordance inputs must be aligned one-dimensional vectors")
    if not np.isin(atrisk, (0, 1)).all():
        raise ValueError("atrisk must contain binary indicators")
    mask = atrisk.astype(bool)
    r, t, e = risk[mask], etime[mask], evt[mask]
    if not np.isfinite(r).all() or not np.isfinite(t).all() or not np.isin(e, (0, 1)).all():
        raise ValueError("at-risk rows require finite risks and times and binary event indicators")
    if not len(r):
        return float("nan")
    levels = np.unique(r)
    ranks = np.searchsorted(levels, r) + 1
    tree = np.zeros(len(levels) + 1, dtype=np.int64)
    order = np.argsort(t, kind="stable")[::-1]
    concordant, comparable, later = 0.0, 0, 0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and t[order[stop]] == t[order[start]]:
            stop += 1
        group = order[start:stop]
        for row in group:
            if e[row]:
                below = _prefix_count(tree, ranks[row] - 1)
                tied = _prefix_count(tree, ranks[row]) - below
                concordant += below + 0.5 * tied
                comparable += later
        for row in group:
            cursor = ranks[row]
            while cursor < len(tree):
                tree[cursor] += 1
                cursor += cursor & -cursor
        later += len(group)
        start = stop
    return concordant / comparable if comparable else float("nan")


def _prefix_count(tree, end):
    count = 0
    while end > 0:
        count += int(tree[end])
        end -= end & -end
    return count


def mean_cause_specific_cindex(cif_h, etime, evt, atrisk) -> float:
    """Mean over organs; cif_h, etime, evt, atrisk: (N, 6)."""
    return float(np.nanmean([cause_specific_cindex(cif_h[:, k], etime[:, k], evt[:, k], atrisk[:, k])
                             for k in range(cif_h.shape[1])]))
