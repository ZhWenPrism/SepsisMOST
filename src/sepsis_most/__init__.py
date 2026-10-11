"""Sepsis-MOST: graph-guided competing-risks prediction of organ failure and
adverse events in sepsis.

Subpackages
-----------
graph        SOFAE Graph Constructor (relations, spectral embedding, organ coupling)
model        Temporal Feature Extractor, KG-MoE Predictor and prediction heads
objectives   joint training objective
training     data interface, optimisation and inference
evaluation   concordance index used for model selection
"""
from .constants import ADVERSE_EVENTS, BIN_EDGES_H, GRAPH_NODES, ORGANS
from .graph import SOFAEGraph, build_sofae_graph
from .model import SepsisMOST
from .training import SepsisMOSTEnsemble, TrainingConfig

__version__ = "1.0.0"

__all__ = [
    "ADVERSE_EVENTS", "BIN_EDGES_H", "GRAPH_NODES", "ORGANS", "SOFAEGraph", "SepsisMOST",
    "SepsisMOSTEnsemble", "TrainingConfig", "build_sofae_graph",
]
