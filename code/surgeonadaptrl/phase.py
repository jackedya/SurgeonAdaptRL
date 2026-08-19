"""Temporal phase models and phase-duration analysis for Sections 3.3 and 3.5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class PhaseError(ValueError):
    """Phase labels or temporal model inputs are invalid."""


@dataclass(frozen=True)
class PhaseRun:
    phase: int
    start: int
    stop: int

    @property
    def duration(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class PhaseMetrics:
    accuracy: Tensor
    macro_recall: Tensor
    transition_f1: Tensor
    edit_score: Tensor


class Chomp1d(nn.Module):
    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, values: Tensor) -> Tensor:
        if self.amount == 0:
            return values
        return values[..., : -self.amount]


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.first = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.first_chomp = Chomp1d(padding)
        self.first_norm = ChannelLayerNorm(channels)
        self.second = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.second_chomp = Chomp1d(padding)
        self.second_norm = ChannelLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.first_norm(self.first_chomp(self.first(values)))
        hidden = self.dropout(torch.nn.functional.gelu(hidden))
        hidden = self.second_norm(self.second_chomp(self.second(hidden)))
        return torch.nn.functional.gelu(values + self.dropout(hidden))


class TemporalPhaseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phases: int = 10,
        channels: int = 256,
        layers: int = 8,
        kernel: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or phases <= 1 or channels <= 0 or layers <= 0:
            raise PhaseError("classifier dimensions must be positive")
        if kernel <= 1 or kernel % 2 == 0:
            raise PhaseError("kernel must be an odd integer greater than one")
        self.input_projection = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(channels, kernel, 2**index, dropout) for index in range(layers)
        )
        self.output_projection = nn.Conv1d(channels, phases, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise PhaseError("features must have shape [batch, time, channels]")
        hidden = self.input_projection(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden).transpose(1, 2)


class MultiStagePhaseClassifier(nn.Module):
    def __init__(self, input_dim: int, phases: int, stages: int = 3, **kwargs: float | int) -> None:
        super().__init__()
        if stages <= 0:
            raise PhaseError("at least one stage is required")
        self.first = TemporalPhaseClassifier(input_dim, phases, **kwargs)
        self.refiners = nn.ModuleList(TemporalPhaseClassifier(phases, phases, **kwargs) for _ in range(stages - 1))

    def forward(self, features: Tensor) -> list[Tensor]:
        outputs = [self.first(features)]
        for refiner in self.refiners:
            outputs.append(refiner(torch.softmax(outputs[-1], dim=-1)))
        return outputs


def phase_runs(labels: Tensor) -> list[PhaseRun]:
    if labels.ndim != 1:
        raise PhaseError("labels must be one-dimensional")
    if labels.numel() == 0:
        return []
    changes = torch.nonzero(labels[1:] != labels[:-1], as_tuple=False).flatten() + 1
    boundaries = [0, *(int(value) for value in changes), labels.numel()]
    return [PhaseRun(int(labels[left]), left, right) for left, right in zip(boundaries[:-1], boundaries[1:])]


def collapse_phases(labels: Tensor) -> Tensor:
    runs = phase_runs(labels)
    return torch.tensor([run.phase for run in runs], device=labels.device, dtype=labels.dtype)


def phase_durations(labels: Tensor, count: int) -> Tensor:
    if count <= 0:
        raise PhaseError("phase count must be positive")
    durations = torch.zeros(count, device=labels.device, dtype=torch.float32)
    for run in phase_runs(labels):
        if run.phase < 0 or run.phase >= count:
            raise PhaseError("phase label outside configured range")
        durations[run.phase] += run.duration
    return durations


def transition_indices(labels: Tensor) -> Tensor:
    if labels.ndim != 1:
        raise PhaseError("labels must be one-dimensional")
    return torch.nonzero(labels[1:] != labels[:-1], as_tuple=False).flatten() + 1


def transition_matrix(labels: Tensor, count: int, smoothing: float = 0.0) -> Tensor:
    if smoothing < 0:
        raise PhaseError("smoothing cannot be negative")
    matrix = torch.full((count, count), smoothing, dtype=torch.float64, device=labels.device)
    if labels.numel() > 1:
        flat = labels[:-1].long() * count + labels[1:].long()
        matrix += torch.bincount(flat, minlength=count * count).reshape(count, count)
    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)


def phase_similarity(predicted: Tensor, target: Tensor, count: int) -> Tensor:
    predicted_duration = phase_durations(predicted, count)
    target_duration = phase_durations(target, count)
    difference = (predicted_duration - target_duration).abs().sum()
    return (1.0 - difference / target_duration.sum().clamp_min(1)).clamp(0.0, 1.0)


def confusion_matrix(predicted: Tensor, target: Tensor, count: int) -> Tensor:
    if predicted.shape != target.shape:
        raise PhaseError("predictions and targets must have equal shape")
    predicted = predicted.flatten().long()
    target = target.flatten().long()
    if bool((predicted < 0).any()) or bool((predicted >= count).any()):
        raise PhaseError("predicted phase outside configured range")
    if bool((target < 0).any()) or bool((target >= count).any()):
        raise PhaseError("target phase outside configured range")
    return torch.bincount(target * count + predicted, minlength=count * count).reshape(count, count)


def macro_recall(matrix: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise PhaseError("confusion matrix must be square")
    denominator = matrix.sum(dim=1)
    valid = denominator > 0
    recalls = matrix.diagonal().float() / denominator.clamp_min(1)
    return recalls[valid].mean() if bool(valid.any()) else recalls.new_zeros(())


def transition_f1(predicted: Tensor, target: Tensor, tolerance: int = 5) -> Tensor:
    if tolerance < 0:
        raise PhaseError("transition tolerance cannot be negative")
    predicted_points = transition_indices(predicted).tolist()
    target_points = transition_indices(target).tolist()
    matched: set[int] = set()
    true_positive = 0
    for point in predicted_points:
        candidates = [(abs(point - target), index) for index, target in enumerate(target_points) if index not in matched]
        if candidates:
            distance, index = min(candidates)
            if distance <= tolerance:
                matched.add(index)
                true_positive += 1
    precision = true_positive / max(len(predicted_points), 1)
    recall = true_positive / max(len(target_points), 1)
    if precision + recall == 0:
        return torch.tensor(0.0, device=predicted.device)
    return torch.tensor(2 * precision * recall / (precision + recall), device=predicted.device)


def levenshtein(first: Tensor, second: Tensor) -> int:
    if first.ndim != 1 or second.ndim != 1:
        raise PhaseError("edit sequences must be one-dimensional")
    previous = list(range(second.numel() + 1))
    for row, first_value in enumerate(first.tolist(), start=1):
        current = [row]
        for column, second_value in enumerate(second.tolist(), start=1):
            substitution = previous[column - 1] + int(first_value != second_value)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitution))
        previous = current
    return previous[-1]


def edit_score(predicted: Tensor, target: Tensor) -> Tensor:
    first = collapse_phases(predicted)
    second = collapse_phases(target)
    distance = levenshtein(first, second)
    return torch.tensor(1.0 - distance / max(first.numel(), second.numel(), 1), device=predicted.device).clamp_min(0)


def phase_metrics(predicted: Tensor, target: Tensor, count: int, tolerance: int = 5) -> PhaseMetrics:
    matrix = confusion_matrix(predicted, target, count)
    accuracy = matrix.diagonal().sum().float() / matrix.sum().clamp_min(1)
    return PhaseMetrics(
        accuracy,
        macro_recall(matrix),
        transition_f1(predicted.flatten(), target.flatten(), tolerance),
        edit_score(predicted.flatten(), target.flatten()),
    )


def class_weights(labels: Tensor, count: int) -> Tensor:
    frequencies = torch.bincount(labels.flatten().long(), minlength=count).float()
    present = frequencies > 0
    weights = torch.zeros_like(frequencies)
    weights[present] = frequencies[present].sum() / (present.sum() * frequencies[present])
    return weights


def temporal_smoothing_loss(logits: Tensor, mask: Tensor | None = None, limit: float = 4.0) -> Tensor:
    if logits.ndim != 3:
        raise PhaseError("logits must have shape [batch, time, phases]")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    differences = (log_probabilities[:, 1:] - log_probabilities[:, :-1]).square().clamp_max(limit)
    values = differences.mean(dim=-1)
    if mask is not None:
        if mask.shape != logits.shape[:2]:
            raise PhaseError("mask must align with batch and time")
        pair_mask = mask[:, 1:] & mask[:, :-1]
        return values[pair_mask].mean() if bool(pair_mask.any()) else values.new_zeros(())
    return values.mean()


def supervised_phase_loss(outputs: list[Tensor], target: Tensor, smoothing_weight: float = 0.15) -> Tensor:
    if not outputs:
        raise PhaseError("at least one classifier stage is required")
    losses = []
    for logits in outputs:
        classification = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
        losses.append(classification + smoothing_weight * temporal_smoothing_loss(logits))
    return torch.stack(losses).mean()


class OnlinePhaseDecoder:
    def __init__(self, transition: Tensor, minimum_duration: int = 1) -> None:
        if transition.ndim != 2 or transition.size(0) != transition.size(1):
            raise PhaseError("transition matrix must be square")
        if minimum_duration <= 0:
            raise PhaseError("minimum duration must be positive")
        self.transition = transition
        self.minimum_duration = minimum_duration
        self.phase: int | None = None
        self.duration = 0

    def update(self, logits: Tensor) -> int:
        if logits.ndim != 1 or logits.numel() != self.transition.size(0):
            raise PhaseError("logits must contain one value per phase")
        likelihood = torch.log_softmax(logits, dim=0)
        previous = self.phase
        if previous is None:
            self.phase = int(likelihood.argmax())
        elif self.duration >= self.minimum_duration:
            prior = self.transition[previous].clamp_min(1e-12).log()
            self.phase = int((likelihood + prior).argmax())
        self.duration = self.duration + 1 if self.phase == previous else 1
        return self.phase

    def reset(self) -> None:
        self.phase = None
        self.duration = 0
