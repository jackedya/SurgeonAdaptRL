from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    image_height: int = 224
    image_width: int = 224
    window: int = 30
    stride: int = 5
    horizon: int = 15
    phases: int = 10
    instruments: int = 19
    fps: int = 30
    detector_frames: int = 3500
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    workers: int = 32
    prefetch_factor: int = 8


@dataclass(frozen=True)
class PolicyConfig:
    hidden: int = 512
    heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 4
    feedforward: int = 2048
    adapter_bottleneck: int = 64
    adapter_scale: float = 0.1
    spatial_bins: int = 256
    displacement_min: float = -0.05
    displacement_max: float = 0.05
    dropout: float = 0.1
    visual_features: int = 2048
    position_features: int = 2
    imagenet_pretrained: bool = True
    adapters_enabled: bool = True
    phase_conditioning_enabled: bool = True
    primitive_conditioning_enabled: bool = True


@dataclass(frozen=True)
class PrimitiveConfig:
    feature_dim: int = 3
    latent_dim: int = 32
    hidden_dim: int = 128
    tcn_layers: int = 8
    tcn_kernel: int = 5
    minimum_components: int = 4
    maximum_components: int = 12
    beta: float = 0.1


@dataclass(frozen=True)
class StyleConfig:
    velocity_weight: float = 0.25
    acceleration_weight: float = 0.20
    trajectory_weight: float = 0.30
    phase_weight: float = 0.25
    velocity_scale: float = 0.05
    trajectory_scale: float = 0.02
    loss_weight: float = 0.3


@dataclass(frozen=True)
class SafetyConfig:
    maximum_velocity_mm_s: float = 15.0
    maximum_force_mn: float = 30.0
    minimum_clearance_mm: float = 0.2
    policy_learning_rate: float = 3e-4
    dual_learning_rate: float = 1e-3
    refinement_iterations: int = 100000
    discount: float = 0.99


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    world_size: int = 4
    gradient_accumulation: int = 1
    pretrain_epochs: int = 200
    meta_iterations: int = 50000
    adaptation_steps: int = 200
    learning_rate: float = 3e-4
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    precision: str = "bf16"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 0
    seeds: tuple[int, ...] = (13, 29, 47, 71, 101)
    save_every: int = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    primitives: PrimitiveConfig = field(default_factory=PrimitiveConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: str = "runs/main"
    variant: str = "full"
    disabled_components: tuple[str, ...] = ()

    @staticmethod
    def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ExperimentConfig._merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @classmethod
    def _read_values(cls, path: Path, seen: set[Path]) -> dict[str, Any]:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError("configuration inheritance contains a cycle")
        seen.add(resolved)
        with resolved.open("r", encoding="utf-8") as stream:
            values: dict[str, Any] = yaml.safe_load(stream) or {}
        base_name = values.pop("base", None)
        if base_name is None:
            return values
        base_values = cls._read_values(resolved.parent / str(base_name), seen)
        return cls._merge(base_values, values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        values = cls._read_values(Path(path), set())
        disabled = tuple(str(value) for value in values.get("disabled_components", ()))
        policy = PolicyConfig(**values.get("policy", {}))
        policy = replace(
            policy,
            adapters_enabled="adapter_modules" not in disabled,
            phase_conditioning_enabled="phase_conditioning" not in disabled,
            primitive_conditioning_enabled="skill_primitives" not in disabled,
        )
        return cls(
            data=DataConfig(**values.get("data", {})),
            policy=policy,
            primitives=PrimitiveConfig(**values.get("primitives", {})),
            style=StyleConfig(**values.get("style", {})),
            safety=SafetyConfig(**values.get("safety", {})),
            train=TrainConfig(**values.get("train", {})),
            output=values.get("output", "runs/main"),
            variant=values.get("variant", "full"),
            disabled_components=disabled,
        )
