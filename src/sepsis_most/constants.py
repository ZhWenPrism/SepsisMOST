"""Constants shared by the Sepsis-MOST modules."""
from __future__ import annotations

#: Organ systems, in the order used by every organ-indexed array.
ORGANS = ("kidney", "lung", "heart", "brain", "coagulation", "liver")
N_ORGANS = len(ORGANS)

#: Adverse events predicted by the three adverse-event heads, in output order.
ADVERSE_EVENTS = ("icu_death", "septic_shock", "mods")

#: Syndrome nodes of the SOFAE graph. MODS onset is the second organ failure.
SYNDROME_NODES = ("sepsis", "septic_shock", "mods")

#: The nine SOFAE graph nodes: six organ failures followed by three syndromes.
GRAPH_NODES = ORGANS + SYNDROME_NODES

#: The first six temporal channels are the organ SOFA subscores (in ``ORGANS``
#: order) and the seventh is the total SOFA score. These channels keep their
#: native scale and are not standardised.
N_RAW_CHANNELS = 7

#: Right-closed edges (hours after the time of prediction) of the K = 8
#: discrete time bins covering the 7-day horizon.
BIN_EDGES_H = (12.0, 24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0)
N_BINS = len(BIN_EDGES_H)

#: States of each organ head in every time bin.
CR_STATES = ("fail", "death", "survive")
