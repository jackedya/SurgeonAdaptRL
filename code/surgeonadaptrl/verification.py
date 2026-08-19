"""Independent executable checks for the scientific training path."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .checkpointing import Progress, load_checkpoint, save_checkpoint
from .configuration import ExperimentConfig, PolicyConfig
from .policy import SurgeonPolicy
from .style import SurgeonStyleFidelity
from .training import calculate_loss
from .trajectory import motion_features


class VerificationError(RuntimeError):
    """An executable verification invariant did not hold."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class VerificationSummary:
    status: str
    checks: tuple[Check, ...]
    passed: int
    failed: int
    not_run: int
    blocked: int


class SyntheticMotionDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, samples: int, observation: int, horizon: int, seed: int = 17) -> None:
        if samples <= 0 or observation < 3 or horizon <= 0:
            raise VerificationError("invalid synthetic verification dimensions")
        generator = torch.Generator().manual_seed(seed)
        self.records: list[dict[str, Tensor]] = []
        for index in range(samples):
            phase = index % 4
            time = torch.arange(observation + horizon, dtype=torch.float32)
            speed = 0.0025 * (phase + 1)
            angle = 0.3 * phase
            start = torch.rand(2, generator=generator) * 0.25 + 0.25
            direction = torch.tensor((np.cos(angle), np.sin(angle)), dtype=torch.float32)
            curve = torch.stack((torch.zeros_like(time), 0.0002 * torch.sin(time / 3 + phase)), dim=-1)
            positions = (start + speed * time.unsqueeze(-1) * direction + curve).clamp(0, 1)
            features = torch.cat(
                (
                    positions,
                    torch.nn.functional.one_hot(torch.full((time.numel(),), phase), 4).float(),
                ),
                dim=-1,
            )
            self.records.append(
                {
                    "features": features[:observation],
                    "positions": positions[:observation],
                    "future": positions[observation:],
                    "phase": torch.tensor(phase),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.records[index]


class MinimalTrajectoryPolicy(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.constructor_arguments = (feature_dim, hidden, horizon)
        self.horizon = horizon
        self.encoder = nn.GRU(feature_dim, hidden, batch_first=True)
        self.phase = nn.Linear(hidden, 4)
        self.decoder = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2))

    def forward(self, features: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        encoded, _ = self.encoder(features)
        context = encoded[:, -1]
        changes = self.decoder(torch.cat((context, positions[:, -1]), dim=-1)).reshape(-1, self.horizon, 2)
        trajectory = positions[:, -1:] + torch.cumsum(changes, dim=1)
        return trajectory, self.phase(context)


class VerificationVisualEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, hidden)

    def forward(self, frames: Tensor) -> Tensor:
        return self.projection(frames.mean(dim=(-1, -2)))


def independent_loss(predicted: Tensor, target: Tensor, phase_logits: Tensor, phases: Tensor) -> Tensor:
    if predicted.shape != target.shape:
        raise VerificationError("prediction and target shapes differ")
    squared = (predicted - target).pow(2).sum(dim=-1).mean()
    classification = torch.nn.functional.cross_entropy(phase_logits, phases)
    predicted_velocity = predicted[:, 1:] - predicted[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_penalty = (predicted_velocity - target_velocity).abs().mean()
    return squared + 0.1 * classification + 0.05 * velocity_penalty


def clone_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.detach().clone().to(device) for name, value in batch.items()}


def parameter_vector(model: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])


def gradient_vector(model: nn.Module) -> Tensor:
    values = [parameter.grad.detach().flatten().cpu() for parameter in model.parameters() if parameter.grad is not None]
    return torch.cat(values) if values else torch.empty(0)


def run_data_check(loader: DataLoader[dict[str, Tensor]]) -> Check:
    try:
        batch = next(iter(loader))
        expected = {"features", "positions", "future", "phase"}
        if set(batch) != expected:
            raise VerificationError("batch fields differ from the verification contract")
        if batch["positions"].shape[-1] != 2 or batch["future"].shape[-1] != 2:
            raise VerificationError("trajectory coordinate dimension is not two")
        if not all(torch.isfinite(value).all() for value in batch.values()):
            raise VerificationError("batch contains non-finite values")
        evidence = {name: list(value.shape) for name, value in batch.items()}
        return Check("data_reading", "PASS", "batch decoded and validated", evidence)
    except Exception as error:
        return Check("data_reading", "FAIL", str(error), {})


def run_forward_check(model: MinimalTrajectoryPolicy, batch: Mapping[str, Tensor], device: torch.device) -> Check:
    try:
        values = clone_batch(batch, device)
        predicted, phases = model(values["features"], values["positions"])
        if predicted.shape != values["future"].shape:
            raise VerificationError("forward trajectory shape differs from target")
        if phases.shape != (values["phase"].size(0), 4):
            raise VerificationError("phase-logit shape is incorrect")
        if not torch.isfinite(predicted).all() or not torch.isfinite(phases).all():
            raise VerificationError("forward output contains non-finite values")
        return Check(
            "forward_propagation",
            "PASS",
            "independent shape and finiteness checks passed",
            {"trajectory_shape": list(predicted.shape), "phase_shape": list(phases.shape)},
        )
    except Exception as error:
        return Check("forward_propagation", "FAIL", str(error), {})


def run_backward_update_check(
    model: MinimalTrajectoryPolicy,
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> Check:
    try:
        values = clone_batch(batch, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
        before = parameter_vector(model)
        optimizer.zero_grad(set_to_none=True)
        predicted, phase_logits = model(values["features"], values["positions"])
        loss = independent_loss(predicted, values["future"], phase_logits, values["phase"])
        loss.backward()
        gradients = gradient_vector(model)
        if gradients.numel() == 0 or not torch.isfinite(gradients).all() or float(gradients.norm()) == 0:
            raise VerificationError("backward pass did not produce finite nonzero gradients")
        optimizer.step()
        after = parameter_vector(model)
        change = float(torch.linalg.vector_norm(after - before))
        if change == 0:
            raise VerificationError("optimizer did not update parameters")
        return Check(
            "loss_backward_update",
            "PASS",
            "loss, backward pass, and parameter update completed",
            {"loss": float(loss.detach()), "gradient_norm": float(gradients.norm()), "parameter_change": change},
        )
    except Exception as error:
        return Check("loss_backward_update", "FAIL", str(error), {})


def run_checkpoint_check(
    model: MinimalTrajectoryPolicy,
    batch: Mapping[str, Tensor],
    directory: Path,
    device: torch.device,
) -> Check:
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        values = clone_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predicted, phase_logits = model(values["features"], values["positions"])
        independent_loss(predicted, values["future"], phase_logits, values["phase"]).backward()
        optimizer.step()
        scheduler.step()
        expected = copy.deepcopy(model.state_dict())
        path = directory / "roundtrip.pt"
        digest = save_checkpoint(path, model, optimizer, scheduler, None, Progress("verification", 1, 1, len(values["phase"])), 31)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter))
        result = load_checkpoint(path, model, optimizer, scheduler, device=device)
        for name, value in model.state_dict().items():
            if not torch.equal(value, expected[name]):
                raise VerificationError(f"restored parameter differs: {name}")
        if result.seed != 31 or result.progress.stage != "verification":
            raise VerificationError("checkpoint metadata did not round-trip")
        return Check("checkpoint_roundtrip", "PASS", "atomic save and exact load completed", {"sha256": digest})
    except Exception as error:
        return Check("checkpoint_roundtrip", "FAIL", str(error), {})


def run_overfit_check(
    model: MinimalTrajectoryPolicy,
    batch: Mapping[str, Tensor],
    device: torch.device,
    steps: int = 80,
) -> Check:
    try:
        values = clone_batch(batch, device)
        small = {name: value[:2] for name, value in values.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            predicted, phase_logits = model(small["features"], small["positions"])
            loss = independent_loss(predicted, small["future"], phase_logits, small["phase"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        ratio = losses[-1] / max(losses[0], 1e-12)
        if not np.isfinite(losses).all() or ratio >= 0.35:
            raise VerificationError(f"single-batch loss ratio {ratio:.4f} did not meet 0.35 threshold")
        return Check(
            "single_batch_overfit",
            "PASS",
            "independent training loop reduced the fixed-batch loss",
            {"initial_loss": losses[0], "final_loss": losses[-1], "ratio": ratio, "steps": steps},
        )
    except Exception as error:
        return Check("single_batch_overfit", "FAIL", str(error), {})


def run_motion_formula_check() -> Check:
    try:
        time = torch.arange(8, dtype=torch.float64)
        straight = torch.stack((0.1 + 0.01 * time, 0.2 + 0.02 * time), dim=-1)
        values = motion_features(straight)
        if not torch.allclose(values.acceleration, torch.zeros_like(values.acceleration), atol=1e-12):
            raise VerificationError("constant-velocity path has nonzero acceleration")
        if not torch.allclose(values.curvature, torch.zeros_like(values.curvature), atol=1e-12):
            raise VerificationError("straight path has nonzero curvature")
        expected_speed = np.sqrt(0.01**2 + 0.02**2)
        if not np.isclose(float(values.velocity.mean()), expected_speed):
            raise VerificationError("velocity formula differs from analytic value")
        return Check(
            "motion_formulas",
            "PASS",
            "velocity, acceleration, and curvature match analytic trajectory",
            {"speed": float(values.velocity.mean()), "expected_speed": expected_speed},
        )
    except Exception as error:
        return Check("motion_formulas", "FAIL", str(error), {})


def run_science_core_check() -> Check:
    try:
        torch.manual_seed(37)
        policy_config = PolicyConfig(
            hidden=32,
            heads=4,
            encoder_layers=2,
            decoder_layers=2,
            feedforward=64,
            adapter_bottleneck=8,
            spatial_bins=17,
            visual_features=32,
            imagenet_pretrained=False,
        )
        experiment = ExperimentConfig(policy=policy_config)
        model = SurgeonPolicy(policy_config, phases=4, primitives=6)
        model.visual = VerificationVisualEncoder(policy_config.hidden)
        batch = 2
        observation = 5
        horizon = 3
        frames = torch.randn(batch, observation, 3, 12, 12)
        positions = torch.rand(batch, observation, 2) * 0.4 + 0.3
        future = positions[:, -1:] + torch.cumsum(torch.randn(batch, horizon, 2) * 0.005, dim=1)
        observed_phases = torch.randint(0, 4, (batch, observation))
        future_phases = torch.randint(0, 4, (batch, horizon))
        primitives = observed_phases.remainder(6)
        decoder = positions[:, -1:].expand(batch, horizon, 2)
        model.add_surgeon("verification")
        before = parameter_vector(model)
        output = model(frames, positions, observed_phases, primitives, decoder, surgeon="verification")
        if output.x_logits.shape != (batch, horizon, policy_config.spatial_bins):
            raise VerificationError("trajectory-logit shape differs from the configured horizon")
        if output.phase_logits.shape != (batch, horizon, 4):
            raise VerificationError("phase logits do not align with the prediction horizon")
        metric = SurgeonStyleFidelity(experiment.style)
        losses = calculate_loss(output, future, positions, future_phases, experiment, metric)
        losses.total.backward()
        gradients = gradient_vector(model)
        if gradients.numel() == 0 or not torch.isfinite(gradients).all():
            raise VerificationError("science-core backward pass produced invalid gradients")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        change = float(torch.linalg.vector_norm(parameter_vector(model) - before))
        if change == 0 or not torch.isfinite(losses.total):
            raise VerificationError("science-core loss or parameter update is invalid")
        manual_phase = torch.nn.functional.cross_entropy(output.phase_logits.flatten(0, 1), future_phases.flatten())
        if not torch.allclose(manual_phase, losses.phase):
            raise VerificationError("phase loss differs from an independent cross-entropy calculation")
        return Check(
            "science_core_training_path",
            "PASS",
            "policy, adapter, trajectory loss, style loss, phase loss, backward pass, and update completed",
            {
                "total_loss": float(losses.total.detach()),
                "trajectory_loss": float(losses.trajectory.detach()),
                "style_loss": float(losses.style.detach()),
                "phase_loss": float(losses.phase.detach()),
                "gradient_norm": float(gradients.norm()),
                "parameter_change": change,
            },
        )
    except Exception as error:
        return Check("science_core_training_path", "FAIL", str(error), {})


def summarize(checks: list[Check]) -> VerificationSummary:
    counts = {status: sum(check.status == status for check in checks) for status in ("PASS", "FAIL", "NOT_RUN", "BLOCKED")}
    if counts["FAIL"]:
        status = "UNVERIFIED"
    elif counts["NOT_RUN"] or counts["BLOCKED"]:
        status = "PARTIALLY_VERIFIED"
    else:
        status = "PARTIALLY_VERIFIED"
    return VerificationSummary(status, tuple(checks), counts["PASS"], counts["FAIL"], counts["NOT_RUN"], counts["BLOCKED"])


def run_verification(directory: str | Path, device: str = "cpu") -> VerificationSummary:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(23)
    selected_device = torch.device(device)
    dataset = SyntheticMotionDataset(16, 8, 4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    data_check = run_data_check(loader)
    batch = next(iter(loader))
    checks = [data_check, run_motion_formula_check(), run_science_core_check()]
    model = MinimalTrajectoryPolicy(6, 24, 4).to(selected_device)
    checks.append(run_forward_check(model, batch, selected_device))
    checks.append(run_backward_update_check(model, batch, selected_device))
    checks.append(run_checkpoint_check(model, batch, output, selected_device))
    overfit_model = MinimalTrajectoryPolicy(6, 24, 4).to(selected_device)
    checks.append(run_overfit_check(overfit_model, batch, selected_device))
    checks.extend(
        (
            Check(
                "full_dataset_training",
                "NOT_RUN",
                "official video archives and the reported four-GPU compute target are not present",
                {},
            ),
            Check(
                "ambf_safety_refinement",
                "NOT_RUN",
                "the AMBF simulator and surgical scene assets are not present",
                {},
            ),
            Check(
                "reported_metric_reproduction",
                "NOT_RUN",
                "reported tables require complete datasets and five full training seeds",
                {},
            ),
        )
    )
    return summarize(checks)


def write_summary(summary: VerificationSummary, destination: str | Path) -> None:
    payload = asdict(summary)
    Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_json_payload(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
