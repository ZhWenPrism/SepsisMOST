import numpy as np
import pytest

from sepsis_most.graph import SOFAEGraph, build_sofae_graph


@pytest.mark.parametrize("defect", [None, "node order", "arrays"])
def test_graph_archive_checks_indexing_and_closes_files(tmp_path, monkeypatch, onsets, defect):
    path = tmp_path / "graph.npz"
    graph = build_sofae_graph(onsets, min_support=5)
    graph.save(path)
    if defect:
        with np.load(path) as saved:
            arrays = dict(saved)
        if defect == "node order":
            arrays["nodes"] = arrays["nodes"][::-1]
        else:
            arrays["R_co"] = arrays["R_co"][:-1]
        np.savez(path, **arrays)
    opened = []
    load = np.load

    def record(*args, **kwargs):
        stream = load(*args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(np, "load", record)
    try:
        if defect:
            with pytest.raises(ValueError, match=defect):
                SOFAEGraph.load(path)
        else:
            restored = SOFAEGraph.load(path)
            np.testing.assert_array_equal(restored.embeddings, graph.embeddings)
        assert opened and opened[0].zip is None
    finally:
        for archive in opened:
            archive.close()
