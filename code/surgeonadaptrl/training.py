from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler

from .configuration import ExperimentConfig
from .data import SurgicalSequenceDataset, quantize_displacements
from .policy import PolicyOutput, SurgeonPolicy
from .runtime import DistributedContext, atomic_save, capture_state, reduce_mean, set_seed
from .safety import DualVariables, PrimalDualObjective, Simulation
from .style import SurgeonStyleFidelity


logger = logging.getLogger("surgeonadaptrl")


@dataclass(frozen=True)
class BatchLoss:
    total: Tensor
    trajectory: Tensor
    style: Tensor
    phase: Tensor


def trajectory_cross_entropy(output: PolicyOutput, target: Tensor) -> Tensor:
    x = torch.nn.functional.cross_entropy(output.x_logits.flatten(0, 1), target[..., 0].flatten())
    y = torch.nn.functional.cross_entropy(output.y_logits.flatten(0, 1), target[..., 1].flatten())
    return x + y


def expected_displacements(output: PolicyOutput, low: float, high: float) -> Tensor:
    bins = output.x_logits.size(-1)
    values = torch.linspace(low, high, bins, device=output.x_logits.device)
    x = (torch.softmax(output.x_logits, dim=-1) * values).sum(dim=-1)
    y = (torch.softmax(output.y_logits, dim=-1) * values).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def calculate_loss(
    output: PolicyOutput,
    future: Tensor,
    current: Tensor,
    future_phases: Tensor,
    config: ExperimentConfig,
    style_metric: SurgeonStyleFidelity,
) -> BatchLoss:
    displacement = torch.cat((future[:, :1] - current[:, -1:], torch.diff(future, dim=1)), dim=1)
    target = quantize_displacements(
        displacement,
        config.policy.spatial_bins,
        config.policy.displacement_min,
        config.policy.displacement_max,
    )
    trajectory = trajectory_cross_entropy(output, target)
    predicted_displacement = expected_displacements(
        output,
        config.policy.displacement_min,
        config.policy.displacement_max,
    )
    predicted = current[:, -1:] + torch.cumsum(predicted_displacement, dim=1)
    if output.phase_logits.shape[:2] != future_phases.shape:
        raise ValueError("phase predictions must align with the future trajectory")
    predicted_phase = output.phase_logits.argmax(dim=-1)
    style_values = [
        style_metric.loss(predicted[index], future[index], predicted_phase[index], future_phases[index])
        for index in range(predicted.size(0))
    ]
    style = torch.stack(style_values).mean()
    phase = torch.nn.functional.cross_entropy(output.phase_logits.flatten(0, 1), future_phases.flatten())
    total = trajectory + config.style.loss_weight * style + phase
    return BatchLoss(total, trajectory, style, phase)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def decoder_seed(positions: Tensor, horizon: int) -> Tensor:
    batch = positions.size(0)
    start = positions[:, -1:].expand(batch, horizon, 2)
    return start


def primitive_seed(phases: Tensor, maximum: int = 12) -> Tensor:
    return phases.remainder(maximum)


class BasePolicyTrainer:
    def __init__(
        self,
        model: SurgeonPolicy,
        config: ExperimentConfig,
        context: DistributedContext,
    ) -> None:
        self.model = model.to(context.device)
        self.config = config
        self.context = context
        self.style = SurgeonStyleFidelity(config.style)
        self.optimizer = torch.optim.AdamW(
            model.base_parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train.pretrain_epochs,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.train.precision == "fp16")

    def step(self, batch: dict[str, Tensor]) -> BatchLoss:
        values = move_batch(batch, self.context.device)
        horizon = values["future"].size(1)
        decoder = decoder_seed(values["positions"], horizon)
        primitives = primitive_seed(values["phases"])
        self.optimizer.zero_grad(set_to_none=True)
        dtype = torch.bfloat16 if self.config.train.precision == "bf16" else torch.float16
        enabled = self.context.device.type == "cuda" and self.config.train.precision in {"bf16", "fp16"}
        with torch.autocast(device_type=self.context.device.type, dtype=dtype, enabled=enabled):
            output = self.model(values["frames"], values["positions"], values["phases"], primitives, decoder)
            loss = calculate_loss(
                output,
                values["future"],
                values["positions"],
                values["future_phases"],
                self.config,
                self.style,
            )
        self.scaler.scale(loss.total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def train(self, loader: DataLoader[dict[str, Tensor]], seed: int) -> None:
        set_seed(seed)
        for epoch in range(self.config.train.pretrain_epochs):
            sampler = loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            self.model.train()
            total = torch.zeros((), device=self.context.device)
            for batch in loader:
                total += self.step(batch).total.detach()
            self.scheduler.step()
            average = reduce_mean(total / max(len(loader), 1), self.context)
            if self.context.primary:
                logger.info("pretrain epoch=%d loss=%.6f", epoch + 1, average.item())
                if (epoch + 1) % 10 == 0:
                    payload = capture_state(self.model, self.optimizer, self.scheduler, seed, "pretrain", epoch + 1)
                    atomic_save(payload, Path(self.config.output) / f"pretrain_{epoch + 1:04d}.pt")


class MetaTrainer:
    def __init__(self, model: SurgeonPolicy, config: ExperimentConfig, device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.style = SurgeonStyleFidelity(config.style)
        self.outer = torch.optim.AdamW(
            model.base_parameters(),
            lr=config.train.outer_learning_rate,
            weight_decay=config.train.weight_decay,
        )

    def surgeon_loss(self, batch: dict[str, Tensor], surgeon: str) -> Tensor:
        values = move_batch(batch, self.device)
        horizon = values["future"].size(1)
        output = self.model(
            values["frames"],
            values["positions"],
            values["future_phases"],
            primitive_seed(values["phases"]),
            decoder_seed(values["positions"], horizon),
            surgeon,
        )
        return calculate_loss(
            output,
            values["future"],
            values["positions"],
            values["phases"],
            self.config,
            self.style,
        ).total

    def adapt(self, support: Iterable[dict[str, Tensor]], surgeon: str) -> None:
        parameters = self.model.adapter_parameters(surgeon)
        optimizer = torch.optim.SGD(parameters, lr=self.config.train.inner_learning_rate)
        for batch in support:
            optimizer.zero_grad(set_to_none=True)
            loss = self.surgeon_loss(batch, surgeon)
            loss.backward()
            optimizer.step()

    def iteration(
        self,
        tasks: dict[str, tuple[Iterable[dict[str, Tensor]], Iterable[dict[str, Tensor]]]],
    ) -> Tensor:
        self.outer.zero_grad(set_to_none=True)
        query_losses: list[Tensor] = []
        for surgeon, (support, query) in tasks.items():
            self.adapt(support, surgeon)
            query_losses.extend(self.surgeon_loss(batch, surgeon) for batch in query)
        meta_loss = torch.stack(query_losses).mean()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.base_parameters(), self.config.train.gradient_clip)
        self.outer.step()
        return meta_loss.detach()


class SafetyRefiner:
    def __init__(
        self,
        policy: nn.Module,
        simulation: Simulation,
        config: ExperimentConfig,
        tolerances: Tensor,
    ) -> None:
        self.policy = policy
        self.simulation = simulation
        self.config = config
        self.dual = DualVariables().to(tolerances.device)
        self.objective = PrimalDualObjective(config.safety, tolerances)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.safety.policy_learning_rate)
        self.dual_optimizer = torch.optim.Adam(self.dual.parameters(), lr=config.safety.dual_learning_rate)

    def update(self, log_probabilities: Tensor, rewards: Tensor, costs: Tensor) -> tuple[Tensor, Tensor]:
        dual_values = self.dual()
        policy_loss = self.objective.policy_loss(log_probabilities, rewards, costs, dual_values)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        dual_loss = self.objective.dual_loss(costs, self.dual())
        self.dual_optimizer.zero_grad(set_to_none=True)
        dual_loss.backward()
        self.dual_optimizer.step()
        return policy_loss.detach(), dual_loss.detach()


def make_loader(
    manifest: str,
    config: ExperimentConfig,
    context: DistributedContext,
) -> DataLoader[dict[str, Tensor]]:
    dataset = SurgicalSequenceDataset(
        manifest,
        config.data.image_height,
        config.data.image_width,
        config.data.window,
        config.data.horizon,
        config.data.stride,
    )
    sampler = DistributedSampler(dataset) if context.world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config.data.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=True,
    )
