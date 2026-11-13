import pytest

from sepsis_most.training import TrainingConfig


@pytest.mark.parametrize("seeds", ["42", [], [True], [2.5], [-1], [2**64], [7, 7]])
def test_config_rejects_invalid_seed_lists(seeds):
    with pytest.raises(ValueError, match="seeds must"):
        TrainingConfig.from_dict({"seeds": seeds})


def test_valid_seeds_keep_their_order_in_config_round_trip():
    config = TrainingConfig.from_dict({"seeds": [42, 7, 0]})
    assert config.seeds == (42, 7, 0)
    assert TrainingConfig.from_dict(config.to_dict()) == config
