"""Optimisation of one Sepsis-MOST network.

All loss terms are optimised jointly with AdamW under a cosine learning-rate
schedule. After every epoch the mean cause-specific C-index of the six organs
is computed on the validation split, the best state is kept, and training stops
when the validation C-index has not improved for ``patience`` epochs.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..constants import N_ORGANS
from ..evaluation.discrimination import mean_cause_specific_cindex
from ..model import SepsisMOST, cumulative_incidence
from ..objectives import joint_objective
from .augment import jitter
from .config import TrainingConfig
from .data import CohortTensors


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(n_features: int, n_static: int, n_bins: int, organ_embeddings, coupling,
                cfg: TrainingConfig) -> SepsisMOST:
    m = cfg.model
    return SepsisMOST(n_features=n_features, n_static=n_static, organ_embeddings=organ_embeddings,
                      coupling=coupling, n_bins=n_bins, d_model=m.d_model, d_state=m.d_state,
                      d_static=m.d_static, expert_hidden=m.expert_hidden,
                      tcn_dilations=tuple(m.tcn_dilations), kernel_size=m.kernel_size,
                      top_k=m.top_k, dropout=m.dropout)


def train_network(seed: int, Xn: np.ndarray, Sn: np.ndarray, data: CohortTensors,
                  train_idx: np.ndarray, val_idx: np.ndarray, organ_embeddings, coupling,
                  cfg: TrainingConfig, device: torch.device,
                  log: Optional[Callable[[str], None]] = None):
    """Train one network with a fixed random seed.

    ``Xn`` and ``Sn`` are the standardised inputs of the whole cohort
    only the
    rows in ``train_idx`` are used for gradient updates and only the rows in
    ``val_idx`` for model selection. Returns the model restored to its best
    validation epoch, the best validation mean C-index and the per-epoch history.
    """
    oc = cfg.optimisation
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(Xn.shape[-1], Sn.shape[-1], data.n_bins, organ_embeddings, coupling,
                        cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=oc.learning_rate, weight_decay=oc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, oc.epochs)

    cause = torch.tensor(data.cause).to(device)
    ebin = torch.tensor(data.ebin).to(device)
    atr = torch.tensor(data.atrisk).to(device)
    evt = torch.tensor(data.evt).to(device)
    etime = torch.tensor(data.etime).to(device)
    events = torch.tensor(data.events).float().to(device)
    Xt, St = torch.tensor(Xn).to(device), torch.tensor(Sn).to(device)

    def batches(ids, shuffle):
        ids = ids.copy()
        if shuffle:
            rng.shuffle(ids)
        for i in range(0, len(ids), oc.batch_size):
            yield ids[i:i + oc.batch_size]

    best, best_state, bad, history = -1, None, 0, []
    for epoch in range(1, oc.epochs + 1):
        model.train()
        for b in batches(train_idx, True):
            bb = torch.tensor(b).to(device)
            xb, sb = Xt[bb], St[bb]
            if cfg.augmentation.jitter > 0:
                xb, sb = jitter(xb, sb, cfg.augmentation.jitter)
            out = model(xb, sb)
            sofa_mean = Xt[bb][:, :, :N_ORGANS].mean(1)              # un-augmented SOFA subscores
            loss = joint_objective(out, cause[bb], ebin[bb], atr[bb], etime[bb], evt[bb],
                                   events[bb], sofa_mean, cfg.loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), oc.grad_clip)
            opt.step()
        sched.step()

        # model selection on the validation mean C-index of the 7-day CIF
        model.eval()
        with torch.no_grad():
            cv = []
            for b in batches(val_idx, False):
                bb = torch.tensor(b).to(device)
                cr = model(Xt[bb], St[bb])["cr_logits"]
                cv.append(torch.stack([cumulative_incidence(cr[:, k])[:, -1]
                                       for k in range(N_ORGANS)], 1).cpu().numpy())
        cv = np.concatenate(cv)
        vc = mean_cause_specific_cindex(cv, data.etime[val_idx], data.evt[val_idx],
                                        data.atrisk[val_idx])
        history.append(vc)
        if log:
            log(f"seed {seed} epoch {epoch:2d} validation mean C-index {vc:.4f}")
        if vc > best + oc.min_delta:
            best, bad = vc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= oc.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best, history
