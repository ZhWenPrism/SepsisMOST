import pytest
from sepsis_most.objectives import LossWeights
from sepsis_most.training.config import TrainingConfig


@pytest.mark.parametrize("name,value", [
    ("rank", -1), ("syn", float("nan")), ("lb", float("inf")), ("spec", -0.1),
    ("rank_sigma", 0), ("rank_sigma", float("nan")), ("spec_tau", -1), ("spec_tau", 0),
])
def test_invalid_loss_parameters_fail_before_training(name, value):
    for factory in (lambda: LossWeights(**{name: value}),
                    lambda: TrainingConfig.from_dict({"loss": {name: value}})):
        with pytest.raises(ValueError, match=name):
            factory()


def test_auxiliary_weights_can_be_disabled_without_changing_scale_defaults():
    value = LossWeights(rank=0, syn=0, lb=0, spec=0)
    assert value.rank_sigma == 0.1 and value.spec_tau == 2.0
