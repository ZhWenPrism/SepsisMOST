"""Model inputs, data split and standardisation.

The input files are described in docs/DATA_FORMAT.md. Standardisation
parameters are estimated on the training split only and applied unchanged to
the validation, internal test and external data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import N_RAW_CHANNELS

TEMPORAL_LABEL_KEYS = ("atrisk", "cause", "ebin", "evt", "etime", "death_h", "shock_h", "mods_h")


@dataclass
class CohortTensors:
    """Inputs and labels of one cohort."""

    X: np.ndarray            # (N, T, F) float32 hourly temporal tensor
    S: np.ndarray            # (N, P) float32 static vector
    atrisk: np.ndarray       # (N, 6) int64, 1 = at risk for the organ at the time of prediction
    cause: np.ndarray        # (N, 6) int64, 0 = organ failure, 1 = ICU death first, -1 = censored
    ebin: np.ndarray         # (N, 6) int64, bin of the event or censoring time
    evt: np.ndarray          # (N, 6) int64, 1 = organ failure observed
    etime: np.ndarray        # (N, 6) float32, hours from the time of prediction
    events: np.ndarray       # (N, 3) int64, ICU death, septic shock, MODS within 7 days
    bin_edges: np.ndarray    # (K,) right-closed bin edges in hours

    @property
    def n_bins(self) -> int:
        return int(len(self.bin_edges))


def load_cohort(temporal_path, static_path) -> CohortTensors:
    """Load the temporal tensor with labels and the static matrix."""
    z = np.load(Path(temporal_path), allow_pickle=False)
    s = np.load(Path(static_path), allow_pickle=False)
    missing = [k for k in ("X", "bin_edges") + TEMPORAL_LABEL_KEYS if k not in z.files]
    if missing or "S" not in s.files:
        raise KeyError(f"missing arrays: {missing + ([] if 'S' in s.files else ['S'])}")
    if "stay_id" in z.files and "stay_id" in s.files:
        if not np.array_equal(z["stay_id"], s["stay_id"]):
            raise ValueError("temporal and static files are not aligned by row")
    X, S = z["X"].astype(np.float32), s["S"].astype(np.float32)
    if len(X) != len(S):
        raise ValueError("temporal and static files have different numbers of rows")
    return CohortTensors(
        X=X, S=S,
        atrisk=z["atrisk"].astype(np.int64), cause=z["cause"].astype(np.int64),
        ebin=z["ebin"].astype(np.int64), evt=z["evt"].astype(np.int64),
        etime=z["etime"].astype(np.float32),
        events=np.stack([z["death_h"], z["shock_h"], z["mods_h"]], 1).astype(np.int64),
        bin_edges=z["bin_edges"].astype(float))


def stratified_split(death, seed: int = 42, fractions=(0.70, 0.15, 0.15)):
    """Random split into training, validation and internal test rows, stratified by ICU death."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(death))
    cut_train, cut_val = fractions[0], round(fractions[0] + fractions[1], 10)
    tr, va, te = [], [], []
    for cls in (0, 1):
        c = idx[death == cls]
        rng.shuffle(c)
        n1, n2 = int(cut_train * len(c)), int(cut_val * len(c))
        tr += list(c[:n1])
        va += list(c[n1:n2])
        te += list(c[n2:])
    return np.array(tr), np.array(va), np.array(te)


@dataclass
class Standardiser:
    """z-score standardisation fitted on the training split.

    The first ``N_RAW_CHANNELS`` temporal channels (organ SOFA subscores and
    total SOFA) keep their native scale.
    """

    mu: np.ndarray
    sd: np.ndarray
    smu: np.ndarray
    ssd: np.ndarray

    @classmethod
    def fit(cls, X, S, train_idx, n_raw: int = N_RAW_CHANNELS) -> "Standardiser":
        flat = X[train_idx].reshape(-1, X.shape[-1])
        mu = flat.mean(0)
        sd = flat.std(0) + 1e-6
        mu[:n_raw] = 0.0
        sd[:n_raw] = 1.0
        smu = S[train_idx].mean(0)
        ssd = S[train_idx].std(0) + 1e-6
        return cls(mu=mu, sd=sd, smu=smu, ssd=ssd)

    def transform(self, X, S):
        """Standardise development data."""
        return (X - self.mu) / self.sd, (S - self.smu) / self.ssd

    def transform_for_scoring(self, X, S):
        """Standardise data scored by a frozen model.

        A missing static value is replaced with the training mean, and any
        non-finite standardised value is set to 0 (the training mean).
        """
        S = np.where(np.isnan(S), self.smu, S)
        Xn, Sn = self.transform(X, S)
        return (np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                np.nan_to_num(Sn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
