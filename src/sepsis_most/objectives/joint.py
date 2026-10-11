"""Training objective of Sepsis-MOST (manuscript Eq. 18).

    L = L_cr + sum_{a in A} lambda_a L_a,   A = {syn, rank, lb, spec}

The main term L_cr is the competing-risks negative log likelihood summed over
the six organ systems (Eq. 19). The four auxiliary terms are the binary cross entropy
of the three adverse events (L_syn, Eq. 20), the ranking loss on the 7-day CIF
(L_rank, Eq. 21), the load-balancing term of the router (L_lb, Eq. 22) and the
expert specialisation loss (L_spec, Eq. 23).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..constants import N_ORGANS
from ..model.heads import cumulative_incidence
from .competing_risks import cr_nll
from .ranking import ranking_loss
from .specialisation import specialisation_loss


@dataclass(frozen=True)
class LossWeights:
    rank: float = 0.15          # lambda_rank
    syn: float = 0.5            # lambda_syn
    lb: float = 0.01            # lambda_lb
    spec: float = 0.2           # lambda_spec
    rank_sigma: float = 0.1     # sigma of the ranking loss
    spec_tau: float = 2.0       # tau of the specialisation target


def joint_objective(out, cause, ebin, atrisk, etime, evt, events, sofa_mean,
                    w: LossWeights = LossWeights()):
    """Return the scalar training loss for one batch.

    out        output dictionary of :class:`~sepsis_most.model.SepsisMOST`
    cause, ebin, atrisk, evt, etime
               (B, 6) organ labels; see docs/DATA_FORMAT.md
    events     (B, 3) float adverse-event labels (ICU death, septic shock, MODS)
    sofa_mean  (B, 6) mean SOFA-2 domain scores over the observation window
    """
    nll = sum(cr_nll(out["cr_logits"][:, k], cause[:, k], ebin[:, k], atrisk[:, k])
              for k in range(N_ORGANS))
    cif_h = torch.stack([cumulative_incidence(out["cr_logits"][:, k])[:, -1]
                         for k in range(N_ORGANS)], 1)
    return (nll + w.rank * ranking_loss(cif_h, etime, evt, atrisk, sigma=w.rank_sigma)
            + w.lb * out["load_balance"]
            + w.syn * F.binary_cross_entropy_with_logits(out["event_logits"], events)
            + w.spec * specialisation_loss(out["route"], sofa_mean, tau=w.spec_tau))
