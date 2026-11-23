import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def command(monkeypatch, destination):
    source = Path(__file__).resolve().parents[2] / "scripts/predict/predict_sepsis_most.py"
    spec = importlib.util.spec_from_file_location("predict_command_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictions = {"cif": np.zeros((2, 6, 2)), "event_prob": np.full((2, 3), .5)}
    ensemble = SimpleNamespace(bin_edges=np.array([6., 12.]), predict=lambda *a, **k: predictions)
    monkeypatch.setattr(module, "load_scoring_inputs", lambda *a: (np.zeros((2, 3, 7)), np.zeros((2, 2))))
    monkeypatch.setattr(module.SepsisMOSTEnsemble, "load", lambda *a, **k: ensemble)
    monkeypatch.setattr(sys, "argv", ["predict", "--checkpoint", "synthetic.pt", "--temporal", "synthetic.npz",
                                     "--static", "synthetic-static.npz", "--out", str(destination)])
    return module


def test_failed_prediction_serialization_keeps_existing_archive(tmp_path, monkeypatch):
    target = tmp_path / "predictions.npz"
    target.write_bytes(b"previous archive")
    module = command(monkeypatch, target)

    def fail(destination, **arrays):
        if hasattr(destination, "write"):
            destination.write(b"partial archive")
        else:
            Path(destination).write_bytes(b"partial archive")
        raise OSError("archive serialization failed")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="archive serialization failed"):
        module.main()
    assert target.read_bytes() == b"previous archive"
    assert [p.name for p in tmp_path.iterdir()] == ["predictions.npz"]


@pytest.mark.parametrize("name", ["predictions", "predictions.npz", "predictions.other"])
def test_successful_predictions_preserve_npz_extension_rules(tmp_path, monkeypatch, capsys, name):
    module = command(monkeypatch, tmp_path / name)
    module.main()
    target = tmp_path / (name if name.endswith(".npz") else name + ".npz")
    with np.load(target, allow_pickle=False) as values:
        np.testing.assert_array_equal(values["bin_edges"], [6., 12.])
        assert values["cif"].shape == (2, 6, 2)
    assert str(target) in capsys.readouterr().out
