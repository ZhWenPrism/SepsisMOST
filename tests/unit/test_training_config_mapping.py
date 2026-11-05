import pytest

from sepsis_most.training import TrainingConfig


@pytest.mark.parametrize("content", ["false", "[]", "0", "'data'", "model: []", "loss: false",
                                     "optimisation: [[epochs, 2]]"])
def test_yaml_rejects_non_mapping_sections(tmp_path, content):
    path = tmp_path / "training.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="configuration must be a mapping"):
        TrainingConfig.from_yaml(path)


def test_defaults_and_valid_partial_config_round_trip(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("", encoding="utf-8")
    assert TrainingConfig.from_yaml(path) == TrainingConfig()
    path.write_text("model: null\noptimisation: {epochs: 2}\nseeds: [3, 4]", encoding="utf-8")
    cfg = TrainingConfig.from_yaml(path)
    assert cfg.model == TrainingConfig().model
    assert cfg.optimisation.epochs == 2
    assert cfg.seeds == (3, 4)
    assert TrainingConfig.from_dict(cfg.to_dict()) == cfg
