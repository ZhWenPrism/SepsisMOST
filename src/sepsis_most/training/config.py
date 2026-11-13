"""Training configuration.

The defaults reproduce ``configs/model/sepsis_most.yaml``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral
from pathlib import Path

from ..objectives import LossWeights


@dataclass
class ModelConfig:
    d_model: int = 96                   # hidden size D of the Temporal Feature Extractor
    d_state: int = 16                   # state size of the selective SSM
    d_static: int = 64                  # size of the static embedding
    expert_hidden: int = 128            # hidden size of each two-layer expert
    tcn_dilations: tuple = (1, 2, 4)    # one causal TCN block per dilation
    kernel_size: int = 3
    top_k: int = 2                      # experts kept by the gate
    dropout: float = 0.15


@dataclass
class OptimConfig:
    learning_rate: float = 2e-3         # AdamW
    weight_decay: float = 2e-4
    epochs: int = 60                    # cosine schedule length and epoch limit
    batch_size: int = 256
    patience: int = 10                  # early stopping on the validation mean C-index
    min_delta: float = 1e-4
    grad_clip: float = 1.0


@dataclass
class AugmentConfig:
    jitter: float = 0.1                 # standard deviation of the additive input noise


@dataclass
class DataConfig:
    split_seed: int = 42
    split_fractions: tuple = (0.70, 0.15, 0.15)   # train, validation, internal test
    n_raw_channels: int = 7             # SOFA-2 domain scores and total SOFA are not standardised


@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    optimisation: OptimConfig = field(default_factory=OptimConfig)
    augmentation: AugmentConfig = field(default_factory=AugmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seeds: tuple = (42, 7, 2024)

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingConfig":
        if not isinstance(d, Mapping):
            raise ValueError("training configuration must be a mapping")

        def build(kind, values):
            if values is None:
                values = {}
            if not isinstance(values, Mapping):
                raise ValueError(f"{kind.__name__} configuration must be a mapping")
            unknown = set(values) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"unknown {kind.__name__} keys: {sorted(unknown)}")
            values = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
            return kind(**values)

        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        seeds = d.get("seeds", cls.seeds)
        if (not isinstance(seeds, (list, tuple)) or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                       or not 0 <= seed < 2 ** 64 for seed in seeds)):
            raise ValueError("seeds must be a non-empty list of non-negative integer seeds below 2**64")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return cls(model=build(ModelConfig, d.get("model")),
                   loss=build(LossWeights, d.get("loss")),
                   optimisation=build(OptimConfig, d.get("optimisation")),
                   augmentation=build(AugmentConfig, d.get("augmentation")),
                   data=build(DataConfig, d.get("data")),
                   seeds=tuple(map(int, seeds)))

    @classmethod
    def from_yaml(cls, path) -> "TrainingConfig":
        import yaml
        with open(Path(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict({} if data is None else data)

    def to_dict(self) -> dict:
        def plain(v):
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            return v
        return plain(asdict(self))
