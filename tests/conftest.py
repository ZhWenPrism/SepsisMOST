"""Synthetic inputs shared by the tests. No patient data are used."""
import numpy as np
import pandas as pd
import pytest

from sepsis_most.constants import BIN_EDGES_H, ORGANS


def synthetic_cohort(n=240, t=12, f=21, p=116, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, t, f)).astype(np.float32)
    X[:, :, :6] = rng.integers(0, 5, size=(n, t, 6))
    X[:, :, 6] = X[:, :, :6].sum(-1)
    S = rng.normal(size=(n, p)).astype(np.float32)
    atrisk = (rng.random((n, 6)) < 0.85).astype(np.int8)
    cause = rng.choice([-1, 0, 1], size=(n, 6), p=[0.6, 0.3, 0.1]).astype(np.int8)
    ebin = rng.integers(0, 8, size=(n, 6)).astype(np.int8)
    edges = np.array(BIN_EDGES_H, dtype=np.float32)
    etime = (edges[ebin] - rng.random((n, 6)) * 6).astype(np.float32)
    evt = (cause == 0).astype(np.int8)
    death = (rng.random(n) < 0.2).astype(np.int8)
    shock = (rng.random(n) < 0.3).astype(np.int8)
    mods = (rng.random(n) < 0.4).astype(np.int8)
    temporal = dict(X=X, atrisk=atrisk, cause=cause, ebin=ebin, etime=etime, evt=evt, death_h=death,
                    shock_h=shock, mods_h=mods, bin_edges=edges, stay_id=np.arange(n))
    static = dict(S=S, stay_id=np.arange(n))
    return temporal, static


def synthetic_onsets(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for c in list(ORGANS) + ["sepsis", "septic_shock"]:
        h = rng.integers(0, 200, n).astype(float)
        h[rng.random(n) < 0.5] = np.nan
        cols[c] = h
    return pd.DataFrame(cols)


@pytest.fixture
def cohort_files(tmp_path):
    temporal, static = synthetic_cohort()
    tp, sp = tmp_path / "temporal.npz", tmp_path / "static.npz"
    np.savez(tp, **temporal)
    np.savez(sp, **static)
    return tp, sp


@pytest.fixture
def onsets():
    return synthetic_onsets()
