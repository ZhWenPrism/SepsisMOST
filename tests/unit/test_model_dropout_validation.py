import pytest
from sepsis_most.training.config import ModelConfig, TrainingConfig


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_dropout_is_rejected_by_configuration(value):
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=value)
    with pytest.raises(ValueError, match="dropout"):
        TrainingConfig.from_dict({"model": {"dropout": value}})


@pytest.mark.parametrize("value", [0.0, 0.15, 1.0])
def test_supported_dropout_probabilities_are_preserved(value):
    assert ModelConfig(dropout=value).dropout == value
