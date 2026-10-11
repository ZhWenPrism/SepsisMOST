import numpy as np
import pandas as pd

from sepsis_most.constants import GRAPH_NODES
from sepsis_most.graph import SOFAEGraph, build_sofae_graph
from sepsis_most.graph.relations import add_mods_onset, prune, relation_matrices


def test_mods_onset_is_second_organ_failure():
    table = pd.DataFrame({"kidney": [5.0, np.nan], "lung": [2.0, 3.0], "heart": [9.0, np.nan],
                          "brain": [np.nan] * 2, "coagulation": [np.nan] * 2, "liver": [np.nan] * 2})
    out = add_mods_onset(table)
    assert out["mods"].iloc[0] == 5.0
    assert np.isnan(out["mods"].iloc[1])


def test_datetime_and_hour_inputs_agree(onsets):
    base = pd.Timestamp("2020-01-01")
    as_dt = onsets.apply(lambda c: base + pd.to_timedelta(c, unit="h"))
    g_h, g_dt = build_sofae_graph(onsets), build_sofae_graph(as_dt)
    np.testing.assert_array_equal(g_h.relations.precedence, g_dt.relations.precedence)
    np.testing.assert_array_equal(g_h.relations.trigger, g_dt.relations.trigger)


def test_relation_structure(onsets):
    r = relation_matrices(add_mods_onset(onsets))
    n = len(GRAPH_NODES)
    assert r.precedence.shape == (n, n)
    assert np.all(r.precedence[6:, :] == 0) and np.all(r.precedence[:, 6:] == 0)   # organ -> organ only
    assert np.all(r.trigger[:6, :] == 0) and np.all(r.trigger[:, 6:] == 0)         # syndrome -> organ only
    np.testing.assert_allclose(r.cooccurrence, r.cooccurrence.T)
    assert np.all((r.precedence >= 0) & (r.precedence <= 1))


def test_pruning_removes_weak_edges(onsets):
    r = relation_matrices(add_mods_onset(onsets))
    p = prune(r, min_support=10 ** 6)
    assert not p.precedence.any() and not p.cooccurrence.any() and not p.trigger.any()


def test_embeddings_and_coupling(onsets, tmp_path):
    g = build_sofae_graph(onsets, min_support=5)
    assert g.embeddings.shape == (9, 8) and g.organ_embeddings.shape == (6, 8)
    np.testing.assert_allclose(np.linalg.norm(g.embeddings, axis=1), 1.0, atol=1e-5)
    c = g.coupling
    assert c.shape == (6, 6) and c.dtype == np.float32
    np.testing.assert_allclose(c.sum(0)[c.sum(0) > 0], 1.0, atol=1e-5)
    g.save(tmp_path / "g.npz")
    g2 = SOFAEGraph.load(tmp_path / "g.npz")
    np.testing.assert_array_equal(g2.embeddings, g.embeddings)
    assert g2.nodes == g.nodes and g2.min_support == 5
