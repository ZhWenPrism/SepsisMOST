import numpy as np
import pandas as pd

from sepsis_most.constants import GRAPH_NODES, ORGANS
from sepsis_most.graph import build_sofae_graph
from sepsis_most.graph.relations import add_mods_onset, relation_matrices


def test_mods_onset_handles_nullable_numeric_columns():
    times = pd.DataFrame({name: pd.array([pd.NA, pd.NA], dtype="Int64") for name in ORGANS})
    times["kidney"] = pd.array([8, 3], dtype="Int64")
    times["respiratory"] = pd.array([5, pd.NA], dtype="Int64")

    out = add_mods_onset(times)

    assert out["mods"].iloc[0] == 8
    assert pd.isna(out["mods"].iloc[1])
    assert "mods" not in times


def test_unsigned_hour_onsets_preserve_temporal_direction_and_trigger_slack():
    numeric = {name: [8] for name in GRAPH_NODES}
    numeric.update(kidney=[5], respiratory=[8], sepsis=[6], septic_shock=[7])
    signed = relation_matrices(pd.DataFrame(numeric, dtype=np.int64))
    unsigned = relation_matrices(pd.DataFrame(numeric, dtype=np.uint8))

    np.testing.assert_array_equal(unsigned.precedence, signed.precedence)
    np.testing.assert_array_equal(unsigned.trigger, signed.trigger)
    assert unsigned.precedence[GRAPH_NODES.index("respiratory"), GRAPH_NODES.index("kidney")] == 0
    assert unsigned.trigger[GRAPH_NODES.index("sepsis"), GRAPH_NODES.index("kidney")] == 1
    assert unsigned.trigger[GRAPH_NODES.index("septic_shock"), GRAPH_NODES.index("kidney")] == 0


