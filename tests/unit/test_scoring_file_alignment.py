import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sepsis_most.training import SepsisMOSTEnsemble


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/predict/predict_sepsis_most.py"


def test_scoring_cli_rejects_reordered_stays_before_loading_model(tmp_path, monkeypatch):
    temporal = tmp_path / "temporal.npz"
    static = tmp_path / "static.npz"
    output = tmp_path / "pred.npz"
    np.savez(temporal, X=np.zeros((2, 3, 7)), stay_id=[10, 20])
    np.savez(static, S=np.zeros((2, 2)), stay_id=[20, 10])
    spec = importlib.util.spec_from_file_location("score_cli", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--temporal", str(temporal), "--static", str(static),
                                     "--checkpoint", str(tmp_path / "model.pt"), "--out", str(output)])

    def no_load(*args, **kwargs):
        pytest.fail("misaligned rows reached model loading")

    monkeypatch.setattr(SepsisMOSTEnsemble, "load", no_load)
    with pytest.raises(ValueError, match="not aligned by row"):
        script.main()
    assert not output.exists()
