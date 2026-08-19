from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from torch import Tensor

from .configuration import StyleConfig
from .primitives import acceleration, velocity


def dynamic_time_warping(first: Tensor, second: Tensor) -> Tensor:
    rows = first.numel() + 1
    columns = second.numel() + 1
    zero = first.new_zeros(())
    infinity = first.new_full((), float("inf"))
    previous = [zero, *[infinity for _ in range(columns - 1)]]
    for row in range(1, rows):
        current = [infinity]
        for column in range(1, columns):
            cost = torch.abs(first[row - 1] - second[column - 1])
            current.append(cost + torch.min(torch.stack((previous[column], current[column - 1], previous[column - 1]))))
        previous = current
    return previous[-1]


def discrete_frechet(first: Tensor, second: Tensor) -> Tensor:
    rows = first.size(0)
    columns = second.size(0)
    previous: list[Tensor] = []
    for row in range(rows):
        current: list[Tensor] = []
        for column in range(columns):
            distance = torch.linalg.vector_norm(first[row] - second[column])
            if row == 0 and column == 0:
                value = distance
            elif row == 0:
                value = torch.maximum(current[column - 1], distance)
            elif column == 0:
                value = torch.maximum(previous[column], distance)
            else:
                preceding = torch.min(
                    torch.stack((previous[column], previous[column - 1], current[column - 1]))
                )
                value = torch.maximum(preceding, distance)
            current.append(value)
        previous = current
    return previous[-1]


def phase_durations(phases: Tensor, count: int) -> Tensor:
    return torch.stack([(phases == phase).sum() for phase in range(count)]).float()


def phase_similarity(predicted: Tensor, target: Tensor, count: int) -> Tensor:
    predicted_durations = phase_durations(predicted, count)
    target_durations = phase_durations(target, count)
    ratio = torch.abs(predicted_durations - target_durations).sum() / target_durations.sum().clamp_min(1.0)
    return (1.0 - ratio).clamp(0.0, 1.0)


@dataclass(frozen=True)
class StyleScores:
    velocity: Tensor
    acceleration: Tensor
    trajectory: Tensor
    phase: Tensor
    total: Tensor


class SurgeonStyleFidelity:
    def __init__(self, config: StyleConfig) -> None:
        self.config = config

    def compare(
        self,
        predicted: Tensor,
        target: Tensor,
        predicted_phases: Tensor,
        target_phases: Tensor,
        phase_count: int = 10,
    ) -> StyleScores:
        predicted_velocity = velocity(predicted)
        target_velocity = velocity(target)
        velocity_score = torch.exp(
            -dynamic_time_warping(predicted_velocity.flatten(), target_velocity.flatten())
            / self.config.velocity_scale
        )
        predicted_acceleration = acceleration(predicted)
        target_acceleration = acceleration(target)
        acceleration_score = torch.exp(
            -dynamic_time_warping(predicted_acceleration.flatten(), target_acceleration.flatten())
            / self.config.velocity_scale
        )
        trajectory_score = torch.exp(-discrete_frechet(predicted, target) / self.config.trajectory_scale)
        phase_score = phase_similarity(predicted_phases, target_phases, phase_count)
        total = (
            self.config.velocity_weight * velocity_score
            + self.config.acceleration_weight * acceleration_score
            + self.config.trajectory_weight * trajectory_score
            + self.config.phase_weight * phase_score
        )
        return StyleScores(velocity_score, acceleration_score, trajectory_score, phase_score, total)

    def loss(
        self,
        predicted: Tensor,
        target: Tensor,
        predicted_phases: Tensor,
        target_phases: Tensor,
    ) -> Tensor:
        return 1.0 - self.compare(predicted, target, predicted_phases, target_phases).total


def normalized_trajectory_error(predicted: Tensor, target: Tensor) -> Tensor:
    return torch.linalg.vector_norm(predicted - target, dim=-1).mean()


def tracking_accuracy(predicted: Tensor, target: Tensor, threshold: float = 0.02) -> Tensor:
    distances = torch.linalg.vector_norm(predicted - target, dim=-1)
    return (distances <= threshold).float().mean()


def trajectory_hausdorff(predicted: Tensor, target: Tensor) -> float:
    first = predicted.detach().cpu().numpy()
    second = target.detach().cpu().numpy()
    return max(directed_hausdorff(first, second)[0], directed_hausdorff(second, first)[0])


def bootstrap_mean(
    values: np.ndarray,
    samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    lower = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(estimates, lower)), float(np.quantile(estimates, 1.0 - lower))
