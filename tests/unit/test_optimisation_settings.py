import pytest
from sepsis_most.training import TrainingConfig


@pytest.mark.parametrize("section,name,value", [
    ("optimisation", "epochs", 0), ("optimisation", "batch_size", 0),
    ("optimisation", "patience", True), ("optimisation", "learning_rate", float("nan")),
    ("optimisation", "weight_decay", -1.), ("optimisation", "grad_clip", 0.),
    ("augmentation", "jitter", -1.), ("augmentation", "jitter", float("inf"))])
def test_invalid_settings_fail_at_configuration(section, name, value):
    with pytest.raises(ValueError, match=name):
        TrainingConfig.from_dict({section: {name: value}})


def test_valid_configuration_round_trips():
    config = TrainingConfig.from_dict({"optimisation": {"epochs": 2}, "augmentation": {"jitter": 0.}})
    assert TrainingConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
