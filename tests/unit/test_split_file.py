import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("train_cli", ROOT / "scripts/train/train_sepsis_most.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("train,val,test", [
    ([0, 1], [1, 2], [3, 4, 5]), ([0, 0], [1, 2], [3, 4, 5]),
    ([-1, 0], [1, 2], [3, 4, 5]), ([0, 6], [1, 2], [3, 4, 5]),
    ([0., 1.], [2, 3], [4, 5]), ([[0, 1]], [2, 3], [4, 5]),
    ([0, 1], [2], [3, 4]), ([], [0, 1], [2, 3, 4, 5]),
])
def test_invalid_split_is_rejected_before_preprocessing(tmp_path, monkeypatch, train, val, test):
    path = tmp_path / "split.npz"
    np.savez(path, train=train, val=val, test=test)
    monkeypatch.setattr(cli, "load_cohort", lambda *a: SimpleNamespace(X=np.zeros((6, 3, 7)), S=np.zeros((6, 2))))
    monkeypatch.setattr(cli.TrainingConfig, "from_yaml", lambda *a: cli.TrainingConfig())
    monkeypatch.setattr(cli.Standardiser, "fit", lambda *a: pytest.fail("invalid split reached preprocessing"))
    monkeypatch.setattr("sys.argv", ["train", "--temporal", "unused", "--static", "unused",
                                    "--graph", "unused", "--out", str(tmp_path / "m.pt"),
                                    "--split-file", str(path), "--device", "cpu"])
    with pytest.raises(ValueError, match="split"):
        cli.main()


def test_valid_partitions_retain_row_order():
    from sepsis_most.training import validate_split_indices
    train, val, test = validate_split_indices(6, [1, 0], [3, 2], [5, 4])
    np.testing.assert_array_equal(train, [1, 0])
    np.testing.assert_array_equal(val, [3, 2])
    np.testing.assert_array_equal(test, [5, 4])
    assert train.dtype == val.dtype == test.dtype == np.int64


def test_library_training_partitions_reject_overlap_without_requiring_test_rows():
    from sepsis_most.training import validate_split_indices
    train, val = validate_split_indices(6, [0, 1], [2, 3])
    assert train.tolist() == [0, 1] and val.tolist() == [2, 3]
    with pytest.raises(ValueError, match="split"):
        validate_split_indices(6, [0, 1], [1, 2])
