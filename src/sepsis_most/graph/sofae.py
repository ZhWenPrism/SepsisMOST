"""SOFAE Graph Constructor.

The SOFAE graph is a directed, weighted graph over N = 9 nodes (six organ
failures and the sepsis, septic shock and MODS syndrome nodes). It is learned
only from how often organ failures occur together, and in what order, in the
development data. Edges supported by fewer than 50 patients are pruned, the
retained relations are compressed into eight-dimensional spectral node
embeddings, and the graph is then frozen.

The graph enters Sepsis-MOST in two ways: the first six rows of the embedding
matrix are the organ embeddings that bias the KG-MoE router (Eq. 8), and the
column-normalised precedence matrix drives the organ coupling (Eq. 14).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import GRAPH_NODES, N_ORGANS
from .coupling import coupling_matrix
from .embedding import spectral_embeddings
from .relations import RelationMatrices, add_mods_onset, prune, relation_matrices

_RELATIONS = (("precedes", "precedence"), ("co_occurs", "cooccurrence"), ("triggers", "trigger"))


@dataclass
class SOFAEGraph:
    """A frozen SOFAE graph."""

    relations: RelationMatrices
    embeddings: np.ndarray            # (9, dim) l2-normalised node embeddings
    singular_values: np.ndarray
    min_support: int
    nodes: tuple = GRAPH_NODES

    @property
    def organ_embeddings(self) -> np.ndarray:
        """Organ rows of the embedding matrix (6 x dim), used by the router."""
        return self.embeddings[:N_ORGANS].astype("float32")

    @property
    def coupling(self) -> np.ndarray:
        """Column-normalised organ precedence matrix (6 x 6), used for coupling."""
        return coupling_matrix(self.relations.precedence)

    def edges(self) -> pd.DataFrame:
        """Edge list of the retained relations, strongest first within each relation."""
        rows = []
        for name, attr in _RELATIONS:
            weight = getattr(self.relations, attr)
            support = getattr(self.relations, f"support_{attr}")
            for a in range(len(self.nodes)):
                for b in range(len(self.nodes)):
                    if a == b or weight[a, b] == 0 or support[a, b] < self.min_support:
                        continue
                    rows.append(dict(head=self.nodes[a], relation=name, tail=self.nodes[b],
                                     weight=round(float(weight[a, b]), 4), n_pairs=int(support[a, b])))
        edges = pd.DataFrame(rows, columns=["head", "relation", "tail", "weight", "n_pairs"])
        return edges.sort_values(["relation", "weight"], ascending=[True, False]).reset_index(drop=True)

    def save(self, path) -> None:
        r = self.relations
        np.savez_compressed(
            Path(path), nodes=np.array(self.nodes), embeddings=self.embeddings,
            singular_values=self.singular_values, min_support=np.array(self.min_support),
            R_prec=r.precedence, R_co=r.cooccurrence, R_tr=r.trigger,
            S_prec=r.support_precedence, S_co=r.support_cooccurrence, S_tr=r.support_trigger,
            counts=r.counts)

    @classmethod
    def load(cls, path) -> "SOFAEGraph":
        z = np.load(Path(path), allow_pickle=False)
        relations = RelationMatrices(
            precedence=z["R_prec"], cooccurrence=z["R_co"], trigger=z["R_tr"],
            support_precedence=z["S_prec"], support_cooccurrence=z["S_co"],
            support_trigger=z["S_tr"], counts=z["counts"])
        return cls(relations=relations, embeddings=z["embeddings"],
                   singular_values=z["singular_values"], min_support=int(z["min_support"]),
                   nodes=tuple(str(n) for n in z["nodes"]))


def build_sofae_graph(onsets: pd.DataFrame, min_support: int = 50, dim: int = 8) -> SOFAEGraph:
    """Build the SOFAE graph from per-patient event onset times.

    Parameters
    ----------
    onsets
        One row per patient with the columns ``kidney``, ``respiratory``, ``cardiovascular``,
        ``brain``, ``haemostasis``, ``liver``, ``sepsis`` and ``septic_shock``
        holding onset times (datetime, or hours on a common time axis within
        each patient); missing values mean the event did not occur. A ``mods``
        column, if absent, is derived as the second earliest organ failure.
    min_support
        Edges supported by fewer than this many patients are pruned.
    dim
        Number of singular directions kept for the node embeddings.
    """
    table = onsets if "mods" in onsets.columns else add_mods_onset(onsets)
    relations = prune(relation_matrices(table), min_support)
    embeddings, singular_values = spectral_embeddings(relations, dim)
    return SOFAEGraph(relations=relations, embeddings=embeddings,
                      singular_values=singular_values, min_support=min_support)
