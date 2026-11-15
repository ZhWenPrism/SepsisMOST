import numpy as np
import pandas as pd
import pytest

from sepsis_most.constants import GRAPH_NODES
from sepsis_most.graph.relations import add_mods_onset, relation_matrices


def onset_table():
    return pd.DataFrame({node: [1., np.nan] for node in GRAPH_NODES})


@pytest.mark.parametrize("func", [add_mods_onset, relation_matrices])
def test_infinite_onset_cannot_count_as_an_event(func):
    table = onset_table()
    table["kidney"] = [np.inf, 2.]
    with pytest.raises(ValueError, match="must be finite or missing"):
        func(table)


@pytest.mark.parametrize("func", [add_mods_onset, relation_matrices])
def test_mixed_datetime_and_hour_columns_fail_at_input_validation(func):
    table = onset_table()
    table["kidney"] = pd.to_datetime(["2026-01-01", None])
    with pytest.raises(ValueError, match="all be datetime values or all numeric hours"):
        func(table)
