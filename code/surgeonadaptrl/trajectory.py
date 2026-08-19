"""Trajectory representation and motion descriptors for Sections 3.1 and 3.3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor


class TrajectoryError(ValueError):
    """A trajectory violates coordinate or temporal invariants."""


class BoundaryMode(IntEnum):
    CLAMP = 0
    REFLECT = 1
    REJECT = 2


@dataclass(frozen=True)
class ImageGeometry:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise TrajectoryError("image dimensions must be positive")

    def normalize(self, pixels: Tensor) -> Tensor:
        _check_last_coordinate(pixels)
        scale = pixels.new_tensor((self.width - 1, self.height - 1)).clamp_min(1)
        return pixels / scale

    def denormalize(self, coordinates: Tensor) -> Tensor:
        _check_last_coordinate(coordinates)
        scale = coordinates.new_tensor((self.width - 1, self.height - 1)).clamp_min(1)
        return coordinates * scale


@dataclass(frozen=True)
class TrajectoryWindow:
    observations: Tensor
    positions: Tensor
    future: Tensor
    phases: Tensor
    surgeon: int
    video: str
    start: int

    def validate(self) -> None:
        _check_trajectory(self.positions)
        _check_trajectory(self.future)
        if self.observations.size(0) != self.positions.size(0):
            raise TrajectoryError("observations and positions must have equal time length")
        if self.phases.ndim != 1 or self.phases.size(0) != self.future.size(0):
            raise TrajectoryError("future phases must be a one-dimensional aligned sequence")
        if self.start < 0:
            raise TrajectoryError("window start cannot be negative")


@dataclass(frozen=True)
class MotionFeatures:
    velocity: Tensor
    acceleration: Tensor
    curvature: Tensor

    def stack(self) -> Tensor:
        return torch.stack((self.velocity, self.acceleration, self.curvature), dim=-1)


@dataclass(frozen=True)
class ResamplingResult:
    positions: Tensor
    source_times: Tensor
    target_times: Tensor


def _check_last_coordinate(values: Tensor) -> None:
    if values.ndim < 1 or values.size(-1) != 2:
        raise TrajectoryError("the final dimension must contain x and y")
    if not torch.isfinite(values).all():
        raise TrajectoryError("coordinates must be finite")


def _check_trajectory(values: Tensor, minimum: int = 1) -> None:
    _check_last_coordinate(values)
    if values.ndim != 2:
        raise TrajectoryError("a trajectory must have shape [time, 2]")
    if values.size(0) < minimum:
        raise TrajectoryError(f"trajectory requires at least {minimum} samples")


def validate_normalized(values: Tensor, tolerance: float = 1e-6) -> None:
    _check_last_coordinate(values)
    if bool((values < -tolerance).any()) or bool((values > 1.0 + tolerance).any()):
        raise TrajectoryError("normalized coordinates must lie in [0, 1]")


def displacement(positions: Tensor) -> Tensor:
    _check_trajectory(positions, 2)
    return positions[1:] - positions[:-1]


def velocity(positions: Tensor, delta_time: float = 1.0) -> Tensor:
    if delta_time <= 0:
        raise TrajectoryError("delta_time must be positive")
    return torch.linalg.vector_norm(displacement(positions), dim=-1) / delta_time


def velocity_vectors(positions: Tensor, delta_time: float = 1.0) -> Tensor:
    if delta_time <= 0:
        raise TrajectoryError("delta_time must be positive")
    return displacement(positions) / delta_time


def acceleration(positions: Tensor, delta_time: float = 1.0) -> Tensor:
    vectors = velocity_vectors(positions, delta_time)
    if vectors.size(0) < 2:
        return vectors.new_empty((0,))
    return torch.linalg.vector_norm(vectors[1:] - vectors[:-1], dim=-1) / delta_time


def acceleration_vectors(positions: Tensor, delta_time: float = 1.0) -> Tensor:
    vectors = velocity_vectors(positions, delta_time)
    if vectors.size(0) < 2:
        return vectors.new_empty((0, 2))
    return (vectors[1:] - vectors[:-1]) / delta_time


def curvature(positions: Tensor, epsilon: float = 1e-8) -> Tensor:
    _check_trajectory(positions, 3)
    first = displacement(positions)
    change = first[1:] - first[:-1]
    tangent = first[:-1]
    cross = tangent[:, 0] * change[:, 1] - tangent[:, 1] * change[:, 0]
    denominator = torch.linalg.vector_norm(tangent, dim=-1).pow(3).clamp_min(epsilon)
    return cross.abs() / denominator


def motion_features(positions: Tensor, delta_time: float = 1.0) -> MotionFeatures:
    speed = velocity(positions, delta_time)
    accel = acceleration(positions, delta_time)
    curve = curvature(positions)
    common = min(speed.numel(), accel.numel(), curve.numel())
    return MotionFeatures(speed[-common:], accel[-common:], curve[-common:])


def cumulative_positions(start: Tensor, changes: Tensor) -> Tensor:
    if start.shape != (2,):
        raise TrajectoryError("start must have shape [2]")
    _check_trajectory(changes)
    return start.unsqueeze(0) + torch.cumsum(changes, dim=0)


def apply_boundary(values: Tensor, mode: BoundaryMode) -> Tensor:
    _check_last_coordinate(values)
    if mode is BoundaryMode.CLAMP:
        return values.clamp(0.0, 1.0)
    if mode is BoundaryMode.REFLECT:
        folded = torch.remainder(values, 2.0)
        return torch.where(folded <= 1.0, folded, 2.0 - folded)
    validate_normalized(values)
    return values


def path_length(positions: Tensor) -> Tensor:
    return torch.linalg.vector_norm(displacement(positions), dim=-1).sum()


def straightness(positions: Tensor, epsilon: float = 1e-8) -> Tensor:
    _check_trajectory(positions, 2)
    chord = torch.linalg.vector_norm(positions[-1] - positions[0])
    return chord / path_length(positions).clamp_min(epsilon)


def signed_area(positions: Tensor) -> Tensor:
    _check_trajectory(positions, 3)
    closed = torch.cat((positions, positions[:1]), dim=0)
    return 0.5 * (closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1]).sum()


def bounding_box(positions: Tensor) -> Tensor:
    _check_trajectory(positions)
    minimum = positions.amin(dim=0)
    maximum = positions.amax(dim=0)
    return torch.cat((minimum, maximum))


def center_trajectory(positions: Tensor) -> tuple[Tensor, Tensor]:
    _check_trajectory(positions)
    center = positions.mean(dim=0)
    return positions - center, center


def standardize_trajectory(positions: Tensor, epsilon: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    centered, mean = center_trajectory(positions)
    deviation = centered.square().mean(dim=0).sqrt().clamp_min(epsilon)
    return centered / deviation, mean, deviation


def restore_standardized(values: Tensor, mean: Tensor, deviation: Tensor) -> Tensor:
    _check_last_coordinate(values)
    if mean.shape != (2,) or deviation.shape != (2,):
        raise TrajectoryError("mean and deviation must have shape [2]")
    return values * deviation + mean


def linear_resample(positions: Tensor, source_rate: float, target_rate: float) -> ResamplingResult:
    _check_trajectory(positions, 2)
    if source_rate <= 0 or target_rate <= 0:
        raise TrajectoryError("sampling rates must be positive")
    duration = (positions.size(0) - 1) / source_rate
    target_count = int(round(duration * target_rate)) + 1
    source_times = torch.arange(positions.size(0), device=positions.device, dtype=positions.dtype) / source_rate
    target_times = torch.linspace(0, duration, target_count, device=positions.device, dtype=positions.dtype)
    right = torch.searchsorted(source_times, target_times, right=True).clamp(1, positions.size(0) - 1)
    left = right - 1
    span = (source_times[right] - source_times[left]).clamp_min(torch.finfo(positions.dtype).eps)
    weight = ((target_times - source_times[left]) / span).unsqueeze(-1)
    values = positions[left] * (1.0 - weight) + positions[right] * weight
    return ResamplingResult(values, source_times, target_times)


def phase_boundaries(phases: Tensor) -> Tensor:
    if phases.ndim != 1:
        raise TrajectoryError("phases must be one-dimensional")
    if phases.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=phases.device)
    changes = torch.nonzero(phases[1:] != phases[:-1], as_tuple=False).flatten() + 1
    ends = torch.tensor((0, phases.numel()), dtype=torch.long, device=phases.device)
    return torch.cat((ends[:1], changes, ends[1:]))


def segment_by_phase(positions: Tensor, phases: Tensor, minimum_length: int = 3) -> list[tuple[int, Tensor]]:
    _check_trajectory(positions)
    if phases.ndim != 1 or phases.numel() != positions.size(0):
        raise TrajectoryError("phases must align with positions")
    boundaries = phase_boundaries(phases)
    segments: list[tuple[int, Tensor]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        begin = int(left)
        end = int(right)
        if end - begin >= minimum_length:
            segments.append((int(phases[begin]), positions[begin:end]))
    return segments


def sliding_indices(length: int, observation: int, horizon: int, stride: int) -> Iterator[tuple[slice, slice]]:
    if min(length, observation, horizon, stride) <= 0:
        raise TrajectoryError("window dimensions must be positive")
    final_start = length - observation - horizon
    for start in range(0, final_start + 1, stride):
        pivot = start + observation
        yield slice(start, pivot), slice(pivot, pivot + horizon)


def interpolate_missing(positions: Tensor, valid: Tensor) -> Tensor:
    _check_trajectory(positions)
    if valid.ndim != 1 or valid.numel() != positions.size(0):
        raise TrajectoryError("validity mask must align with positions")
    if not bool(valid.any()):
        raise TrajectoryError("at least one valid position is required")
    indices = torch.arange(positions.size(0), device=positions.device)
    valid_indices = indices[valid]
    result = positions.clone()
    for index in indices[~valid]:
        before = valid_indices[valid_indices < index]
        after = valid_indices[valid_indices > index]
        if before.numel() == 0:
            result[index] = positions[after[0]]
        elif after.numel() == 0:
            result[index] = positions[before[-1]]
        else:
            left = before[-1]
            right = after[0]
            weight = (index - left).to(positions.dtype) / (right - left).to(positions.dtype)
            result[index] = positions[left] * (1 - weight) + positions[right] * weight
    return result


def moving_average(positions: Tensor, radius: int) -> Tensor:
    _check_trajectory(positions)
    if radius < 0:
        raise TrajectoryError("radius cannot be negative")
    if radius == 0:
        return positions.clone()
    padded = torch.nn.functional.pad(positions.transpose(0, 1).unsqueeze(0), (radius, radius), mode="replicate")
    return torch.nn.functional.avg_pool1d(padded, 2 * radius + 1, stride=1).squeeze(0).transpose(0, 1)


def finite_difference_matrix(length: int, order: int = 1, dtype: torch.dtype = torch.float32) -> Tensor:
    if length <= order or order not in {1, 2}:
        raise TrajectoryError("difference order must be one or two and smaller than length")
    if order == 1:
        matrix = torch.zeros((length - 1, length), dtype=dtype)
        rows = torch.arange(length - 1)
        matrix[rows, rows] = -1
        matrix[rows, rows + 1] = 1
        return matrix
    first = finite_difference_matrix(length, 1, dtype)
    return finite_difference_matrix(length - 1, 1, dtype) @ first


def smooth_least_squares(positions: Tensor, strength: float) -> Tensor:
    _check_trajectory(positions, 3)
    if strength < 0:
        raise TrajectoryError("smoothing strength cannot be negative")
    difference = finite_difference_matrix(positions.size(0), 2, positions.dtype).to(positions.device)
    identity = torch.eye(positions.size(0), device=positions.device, dtype=positions.dtype)
    system = identity + strength * difference.transpose(0, 1) @ difference
    return torch.linalg.solve(system, positions)


def procrustes_align(source: Tensor, target: Tensor) -> Tensor:
    _check_trajectory(source, 2)
    _check_trajectory(target, 2)
    if source.shape != target.shape:
        raise TrajectoryError("aligned trajectories must have equal shape")
    source_centered, source_mean = center_trajectory(source)
    target_centered, target_mean = center_trajectory(target)
    covariance = source_centered.transpose(0, 1) @ target_centered
    left, _, right = torch.linalg.svd(covariance)
    rotation = left @ right
    if torch.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    scale = (target_centered * (source_centered @ rotation)).sum() / source_centered.square().sum().clamp_min(1e-8)
    return source_centered @ rotation * scale + target_mean


def trajectory_from_numpy(values: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
    if values.ndim != 2 or values.shape[1] != 2:
        raise TrajectoryError("array must have shape [time, 2]")
    return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype)


def stack_equal_length(trajectories: Sequence[Tensor]) -> Tensor:
    if not trajectories:
        raise TrajectoryError("at least one trajectory is required")
    for values in trajectories:
        _check_trajectory(values)
    lengths = {values.size(0) for values in trajectories}
    if len(lengths) != 1:
        raise TrajectoryError("all trajectories must have equal length")
    return torch.stack(tuple(trajectories))
