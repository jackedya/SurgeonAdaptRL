from __future__ import annotations

from pathlib import Path

import pytest

from surgeonadaptrl.configuration import ExperimentConfig


ROOT = Path(__file__).resolve().parents[1]


def test_main_configuration_preserves_reported_training_values() -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs/main.yaml")
    assert config.train.batch_size == 256
    assert config.train.world_size == 4
    assert config.train.pretrain_epochs == 200
    assert config.train.meta_iterations == 50000
    assert config.train.adaptation_steps == 200
    assert config.train.learning_rate == pytest.approx(3e-4)
    assert config.train.inner_learning_rate == pytest.approx(0.01)
    assert config.train.outer_learning_rate == pytest.approx(3e-4)
    assert config.train.weight_decay == pytest.approx(0.01)
    assert config.safety.refinement_iterations == 100000
    assert config.data.horizon == 16


@pytest.mark.parametrize(
    "filename,component,flag",
    [
        ("ablation_no_adapters.yaml", "adapter_modules", "adapters_enabled"),
        ("ablation_no_phase.yaml", "phase_conditioning", "phase_conditioning_enabled"),
        ("ablation_no_primitives.yaml", "skill_primitives", "primitive_conditioning_enabled"),
    ],
)
def test_model_ablation_inheritance_disables_real_forward_component(
    filename: str,
    component: str,
    flag: str,
) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs" / filename)
    assert component in config.disabled_components
    assert getattr(config.policy, flag) is False
    assert config.policy.hidden == 512
    assert config.train.pretrain_epochs == 200


@pytest.mark.parametrize(
    "filename,component",
    [
        ("ablation_no_meta.yaml", "meta_learning"),
        ("ablation_no_safety.yaml", "safety_constraints"),
        ("ablation_sim_rl.yaml", "video_pretraining"),
        ("ablation_video_il.yaml", "simulation_refinement"),
    ],
)
def test_stage_ablation_inheritance_is_explicit(filename: str, component: str) -> None:
    config = ExperimentConfig.from_yaml(ROOT / "configs" / filename)
    assert config.disabled_components == (component,)
    assert config.variant != "full"
    assert config.train.batch_size == 256


def test_dataset_configs_inherit_model_and_training_settings() -> None:
    cataract = ExperimentConfig.from_yaml(ROOT / "configs/cataract_1k.yaml")
    cadis = ExperimentConfig.from_yaml(ROOT / "configs/cadis_extraction.yaml")
    assert cataract.variant == "cataract_1k"
    assert cadis.variant == "cadis_extraction"
    assert cataract.policy.encoder_layers == 6
    assert cadis.primitives.latent_dim == 32


def test_configuration_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base: second.yaml\n")
    second.write_text("base: first.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        ExperimentConfig.from_yaml(first)
