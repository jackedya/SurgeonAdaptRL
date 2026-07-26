from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import Tensor, nn

from .configuration import PrimitiveConfig


def velocity(trajectory: Tensor, delta_time: float = 1.0) -> Tensor:
    return torch.linalg.vector_norm(torch.diff(trajectory, dim=-2), dim=-1) / delta_time


def acceleration(trajectory: Tensor, delta_time: float = 1.0) -> Tensor:
    differences = torch.diff(trajectory, dim=-2) / delta_time
    return torch.linalg.vector_norm(torch.diff(differences, dim=-2), dim=-1) / delta_time


def curvature(trajectory: Tensor, epsilon: float = 1e-8) -> Tensor:
    movement = torch.diff(trajectory, dim=-2)
    velocity_change = torch.diff(movement, dim=-2)
    first = movement[..., 1:, :]
    numerator = torch.abs(first[..., 0] * velocity_change[..., 1] - first[..., 1] * velocity_change[..., 0])
    denominator = torch.linalg.vector_norm(first, dim=-1).pow(3).clamp_min(epsilon)
    return numerator / denominator


def motion_features(trajectory: Tensor) -> Tensor:
    speed = velocity(trajectory)[..., 1:]
    change = acceleration(trajectory)
    bend = curvature(trajectory)
    length = min(speed.size(-1), change.size(-1), bend.size(-1))
    return torch.stack((speed[..., :length], change[..., :length], bend[..., :length]), dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel - 1)
        self.padding = padding
        self.first = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.norm_first = nn.GroupNorm(1, channels)
        self.norm_second = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def trim(self, values: Tensor) -> Tensor:
        return values[..., : -self.padding] if self.padding else values

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.dropout(torch.nn.functional.gelu(self.norm_first(self.trim(self.first(values)))))
        values = self.dropout(torch.nn.functional.gelu(self.norm_second(self.trim(self.second(values)))))
        return values + residual


class PhaseClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, phases: int, layers: int = 8, kernel: int = 5) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden, 1)
        self.blocks = nn.ModuleList([TemporalBlock(hidden, kernel, 2**index) for index in range(layers)])
        self.output = nn.Conv1d(hidden, phases, 1)

    def forward(self, values: Tensor) -> Tensor:
        values = self.input(values.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        return self.output(values).transpose(1, 2)


class PrimitiveVAE(nn.Module):
    def __init__(self, config: PrimitiveConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(config.feature_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.mean = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.log_variance = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.initial = nn.Linear(config.latent_dim, config.hidden_dim)
        self.decoder = nn.GRU(config.feature_dim, config.hidden_dim, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, config.feature_dim)

    def encode(self, values: Tensor) -> tuple[Tensor, Tensor]:
        _, hidden = self.encoder(values)
        joined = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self.mean(joined), self.log_variance(joined)

    def sample(self, mean: Tensor, log_variance: Tensor) -> Tensor:
        deviation = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(deviation) * deviation

    def decode(self, latent: Tensor, template: Tensor) -> Tensor:
        initial = self.initial(latent).unsqueeze(0)
        decoded, _ = self.decoder(template, initial)
        return self.output(decoded)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(values)
        reconstruction = self.decode(self.sample(mean, log_variance), torch.zeros_like(values))
        return reconstruction, mean, log_variance

    def loss(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        reconstruction, mean, log_variance = self(values)
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, values)
        divergence = -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())
        return reconstruction_loss + self.config.beta * divergence, reconstruction_loss, divergence


@dataclass
class PhaseMixture:
    phase: int
    model: GaussianMixture

    @property
    def count(self) -> int:
        return self.model.n_components

    @property
    def means(self) -> np.ndarray:
        return self.model.means_

    @property
    def covariances(self) -> np.ndarray:
        return self.model.covariances_

    @property
    def weights(self) -> np.ndarray:
        return self.model.weights_


class PrimitiveLibrary:
    def __init__(self, minimum: int = 4, maximum: int = 12, seed: int = 13) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.seed = seed
        self.mixtures: dict[int, PhaseMixture] = {}
        self.surgeon_logits: dict[tuple[int, int], Tensor] = {}

    def fit_phase(self, phase: int, latent: np.ndarray) -> PhaseMixture:
        candidates = [
            GaussianMixture(
                n_components=count,
                covariance_type="full",
                random_state=self.seed,
                reg_covar=1e-6,
                max_iter=1000,
            ).fit(latent)
            for count in range(self.minimum, min(self.maximum, latent.shape[0]) + 1)
        ]
        selected = min(candidates, key=lambda model: model.bic(latent))
        mixture = PhaseMixture(phase, selected)
        self.mixtures[phase] = mixture
        return mixture

    def fit(self, latent: np.ndarray, phases: np.ndarray) -> None:
        for phase in np.unique(phases):
            selection = latent[phases == phase]
            if len(selection) >= self.minimum:
                self.fit_phase(int(phase), selection)

    def assign(self, latent: np.ndarray, phase: int) -> np.ndarray:
        return self.mixtures[phase].model.predict(latent)

    def probabilities(self, surgeon: int, phase: int) -> Tensor:
        key = (surgeon, phase)
        count = self.mixtures[phase].count
        if key not in self.surgeon_logits:
            self.surgeon_logits[key] = torch.zeros(count, requires_grad=True)
        return torch.softmax(self.surgeon_logits[key], dim=0)

    def sample(self, surgeon: int, phase: int, samples: int, device: torch.device) -> Tensor:
        probabilities = self.probabilities(surgeon, phase).to(device)
        return torch.multinomial(probabilities, samples, replacement=True)

    def state(self) -> dict[str, object]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "seed": self.seed,
            "mixtures": self.mixtures,
            "surgeon_logits": self.surgeon_logits,
        }

    def load_state(self, values: dict[str, object]) -> None:
        self.minimum = int(values["minimum"])
        self.maximum = int(values["maximum"])
        self.seed = int(values["seed"])
        self.mixtures = values["mixtures"]
        self.surgeon_logits = values["surgeon_logits"]
