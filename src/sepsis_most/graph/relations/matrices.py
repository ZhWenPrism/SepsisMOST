"""Relation matrices of the SOFAE graph (manuscript Eqs. 1-3).

For a pair of nodes (a, b), let O_a be the set of patients in whom event a
occurred and t_a its onset time. Three relations are computed:

* temporal precedence, organ -> organ (Eq. 1):
  w_prec[a, b] = |{i in O_a and O_b : t_b > t_a}| / |O_a|
* co-occurrence, symmetric Jaccard overlap (Eq. 2):
  w_co[a, b] = |O_a and O_b| / |O_a or O_b|
* syndrome trigger, syndrome -> organ (Eq. 3):
  w_trig[a, b] = |{i in O_a and O_b : t_b >= t_a - 1 h}| / |O_a|

The number of supporting patients is stored with every entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...constants import GRAPH_NODES, ORGANS, SYNDROME_NODES

#: An organ failure recorded up to one hour before the syndrome onset is
#: counted as triggered by the syndrome (Eq. 3).
TRIGGER_SLACK_H = 1.0


@dataclass
class RelationMatrices:
    """The three SOFAE relations over ``GRAPH_NODES`` and their support counts."""

    precedence: np.ndarray            # (9, 9) P(b fails after a | a), organ -> organ
    cooccurrence: np.ndarray          # (9, 9) Jaccard overlap
    trigger: np.ndarray               # (9, 9) P(organ b at or after syndrome a | a)
    support_precedence: np.ndarray    # (9, 9) supporting patients
    support_cooccurrence: np.ndarray
    support_trigger: np.ndarray
    counts: np.ndarray                # (9,) patients in whom each node occurred


def add_mods_onset(onsets: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``onsets`` with a ``mods`` column.

    MODS onset is the second earliest organ-failure onset; it is missing when
    fewer than two organs failed.
    """
    table = onsets.copy()
    organ_times = table[list(ORGANS)]
    if all(pd.api.types.is_datetime64_any_dtype(organ_times[c]) for c in ORGANS):
        values = np.stack([organ_times[c].to_numpy(dtype="datetime64[ns]") for c in ORGANS], axis=1)
    else:
        values = organ_times.to_numpy(dtype=float)
    ordered = np.sort(values, axis=1)                 # missing values sort last
    table["mods"] = ordered[:, 1]
    return table


def _hours_between(t_a: pd.Series, t_b: pd.Series) -> pd.Series:
    """Signed time from a to b in hours (datetime or numeric hour columns)."""
    if pd.api.types.is_datetime64_any_dtype(t_a):
        return (t_b - t_a).dt.total_seconds() / 3600.0
    return t_b - t_a


def relation_matrices(onsets: pd.DataFrame) -> RelationMatrices:
    """Compute the three SOFAE relation matrices.

    Parameters
    ----------
    onsets
        One row per patient and one column per node in ``GRAPH_NODES``, holding
        the onset time of that event (datetime, or hours on a common time axis
        within each patient). A missing value means the event did not occur.
    """
    missing = [n for n in GRAPH_NODES if n not in onsets.columns]
    if missing:
        raise ValueError(f"onset table is missing columns: {missing}")
    organ_idx = {GRAPH_NODES.index(o) for o in ORGANS}
    syndrome_idx = {GRAPH_NODES.index(s) for s in SYNDROME_NODES}
    n = len(GRAPH_NODES)
    occurred = onsets[list(GRAPH_NODES)].notna().to_numpy()

    r_prec, r_co, r_tr = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    s_prec, s_co, s_tr = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for a in range(n):
        t_a = onsets[GRAPH_NODES[a]]
        n_a = int(occurred[:, a].sum())
        for b in range(n):
            if a == b:
                continue
            both = occurred[:, a] & occurred[:, b]
            n_both = int(both.sum())
            n_union = int(occurred[:, a].sum() + occurred[:, b].sum() - n_both)
            r_co[a, b] = n_both / max(n_union, 1)                    # Eq. 2
            s_co[a, b] = n_both
            if n_a == 0 or n_both == 0:
                continue
            dt = _hours_between(t_a[both], onsets[GRAPH_NODES[b]][both])
            if a in organ_idx and b in organ_idx:                    # Eq. 1
                n_after = int((dt > 0).sum())                        # b strictly after a
                r_prec[a, b] = n_after / n_a
                s_prec[a, b] = n_after
            if a in syndrome_idx and b in organ_idx:                 # Eq. 3
                n_trig = int((dt >= -TRIGGER_SLACK_H).sum())
                r_tr[a, b] = n_trig / n_a
                s_tr[a, b] = n_trig
    return RelationMatrices(precedence=r_prec, cooccurrence=r_co, trigger=r_tr,
                            support_precedence=s_prec, support_cooccurrence=s_co,
                            support_trigger=s_tr, counts=occurred.sum(0).astype(int))


def prune(relations: RelationMatrices, min_support: int = 50) -> RelationMatrices:
    """Set to zero every edge supported by fewer than ``min_support`` patients."""
    def keep(r, s):
        return r * (s >= min_support)

    return RelationMatrices(
        precedence=keep(relations.precedence, relations.support_precedence),
        cooccurrence=keep(relations.cooccurrence, relations.support_cooccurrence),
        trigger=keep(relations.trigger, relations.support_trigger),
        support_precedence=relations.support_precedence,
        support_cooccurrence=relations.support_cooccurrence,
        support_trigger=relations.support_trigger,
        counts=relations.counts)
