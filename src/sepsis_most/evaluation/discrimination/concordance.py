"""Cause-specific concordance index used for model selection.

For one organ, each patient a with an observed failure is compared with every
at-risk patient whose event or censoring time is later. A pair is concordant
when patient a has the higher predicted 7-day CIF, and ties count one half.
The mean of this index over the six organs on the validation split is the
early-stopping criterion.
"""
from __future__ import annotations

import numpy as np


def cause_specific_cindex(risk, etime, evt, atrisk) -> float:
    """risk, etime, evt, atrisk: (N,) arrays for one organ."""
    m = atrisk.astype(bool)
    r, t, e = risk[m], etime[m], evt[m].astype(bool)
    conc = comp = 0.0
    for a in np.where(e)[0]:
        longer = t > t[a]
        n = longer.sum()
        if n:
            comp += n
            conc += (r[a] > r[longer]).sum() + 0.5 * (r[a] == r[longer]).sum()
    return conc / comp if comp else float("nan")


def mean_cause_specific_cindex(cif_h, etime, evt, atrisk) -> float:
    """Mean over organs; cif_h, etime, evt, atrisk: (N, 6)."""
    return float(np.nanmean([cause_specific_cindex(cif_h[:, k], etime[:, k], evt[:, k], atrisk[:, k])
                             for k in range(cif_h.shape[1])]))
