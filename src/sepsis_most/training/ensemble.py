"""Frozen Sepsis-MOST model: trained networks, standardisation and SOFAE graph.

The networks trained with the configured seeds share the same graph and the
same standardisation. Their organ CIFs and adverse-event probabilities are
averaged at prediction time. The checkpoint stores only tensors and plain
Python values and can be loaded with ``torch.load(weights_only=True)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .calibration import isotonic_apply
from .config import TrainingConfig
from .data import Standardiser
from .trainer import build_model

CHECKPOINT_FORMAT = "sepsis-most/1"


class SepsisMOSTEnsemble:
    def __init__(self, members, standardiser: Standardiser, organ_embeddings, coupling,
                 bin_edges, config: TrainingConfig, isotonic: Optional[list] = None):
        self.members = list(members)
        self.standardiser = standardiser
        self.organ_embeddings = np.asarray(organ_embeddings, dtype=np.float32)
        self.coupling = np.asarray(coupling, dtype=np.float32)
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        self.config = config
        self.isotonic = isotonic            # optional per-organ (knots_x, knots_y)

    @torch.no_grad()
    def predict(self, X, S, batch_size: int = 512, device: str = "cpu", calibrated: bool = False):
        """Score raw (unstandardised) inputs.

        Returns a dictionary with ``cif`` (N, 6, K), the cause-specific CIF of
        each organ at the end of each bin, and ``event_prob`` (N, 3), the 7-day
        probabilities of ICU death, septic shock and MODS. With
        ``calibrated=True`` and fitted isotonic maps, ``cif_7d_calibrated``
        (N, 6) is added.
        """
        Xn, Sn = self.standardiser.transform_for_scoring(np.asarray(X, np.float32),
                                                         np.asarray(S, np.float32))
        Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
        St = torch.tensor(Sn, dtype=torch.float32, device=device)
        cifs, probs = [], []
        for model in self.members:
            model.to(device).eval()
            c_all, p_all = [], []
            for a in range(0, len(Xt), batch_size):
                c, p = model.predict(Xt[a:a + batch_size], St[a:a + batch_size])
                c_all.append(c.cpu().numpy())
                p_all.append(p.cpu().numpy())
            cifs.append(np.concatenate(c_all))
            probs.append(np.concatenate(p_all))
        out = {"cif": np.mean(cifs, 0), "event_prob": np.mean(probs, 0)}
        if calibrated:
            if self.isotonic is None:
                raise ValueError("no isotonic maps were fitted for this model")
            out["cif_7d_calibrated"] = np.stack(
                [isotonic_apply(kx, ky, out["cif"][:, k, -1]) for k, (kx, ky) in enumerate(self.isotonic)], 1)
        return out

    def save(self, path) -> None:
        st = self.standardiser
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "states": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in self.members],
            "n_features": int(st.mu.shape[0]), "n_static": int(st.smu.shape[0]),
            "standardiser": {k: torch.from_numpy(np.asarray(getattr(st, k)))
                             for k in ("mu", "sd", "smu", "ssd")},
            "organ_embeddings": torch.from_numpy(self.organ_embeddings),
            "coupling": torch.from_numpy(self.coupling),
            "bin_edges": torch.from_numpy(self.bin_edges),
            "isotonic": None if self.isotonic is None else
            [(torch.from_numpy(np.asarray(kx)), torch.from_numpy(np.asarray(ky))) for kx, ky in self.isotonic],
        }, Path(path))

    @classmethod
    def load(cls, path, device: str = "cpu") -> "SepsisMOSTEnsemble":
        ck = torch.load(Path(path), map_location=device, weights_only=True)
        if ck.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format: {ck.get('format')}")
        cfg = TrainingConfig.from_dict(ck["config"])
        st = Standardiser(**{k: v.numpy() for k, v in ck["standardiser"].items()})
        emb, coup, edges = ck["organ_embeddings"].numpy(), ck["coupling"].numpy(), ck["bin_edges"].numpy()
        members = []
        for state in ck["states"]:
            model = build_model(ck["n_features"], ck["n_static"], len(edges), emb, coup, cfg)
            model.load_state_dict(state)
            members.append(model.to(device).eval())
        iso = None if ck["isotonic"] is None else [(kx.numpy(), ky.numpy()) for kx, ky in ck["isotonic"]]
        return cls(members, st, emb, coup, edges, cfg, iso)
