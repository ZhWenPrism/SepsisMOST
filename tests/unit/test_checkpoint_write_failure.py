from pathlib import Path

import pytest
import torch

from test_inference_inputs import ensemble


def test_partial_serialization_preserves_existing_checkpoint_and_cleans_tempfile(tmp_path, monkeypatch):
    path = tmp_path / "existing.pt"
    path.write_bytes(b"previous completed checkpoint")

    def fail(state, destination):
        Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        ensemble().save(path)
    assert path.read_bytes() == b"previous completed checkpoint"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["existing.pt"]
