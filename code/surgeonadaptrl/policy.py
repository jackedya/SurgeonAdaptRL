from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from .configuration import PolicyConfig


class PositionEncoding(nn.Module):
    def __init__(self, hidden: int, maximum: int = 2048) -> None:
        super().__init__()
        locations = torch.arange(maximum).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        encoding = torch.zeros(maximum, hidden)
        encoding[:, 0::2] = torch.sin(locations * divisor)
        encoding[:, 1::2] = torch.cos(locations * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.encoding[: values.size(1)].unsqueeze(0)


class ResidualAdapter(nn.Module):
    def __init__(self, hidden: int, bottleneck: int, scale: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, hidden)
        self.scale = nn.Parameter(torch.tensor(scale))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.scale * self.up(torch.nn.functional.gelu(self.down(values)))


class AdaptedEncoderLayer(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden, config.heads, config.dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, config.hidden),
        )
        self.first_norm = nn.LayerNorm(config.hidden)
        self.second_norm = nn.LayerNorm(config.hidden)
        self.first_dropout = nn.Dropout(config.dropout)
        self.second_dropout = nn.Dropout(config.dropout)
        self.adapters = nn.ModuleDict()
        self.config = config

    def add_surgeon(self, surgeon: str) -> None:
        if surgeon not in self.adapters:
            self.adapters[surgeon] = ResidualAdapter(
                self.config.hidden,
                self.config.adapter_bottleneck,
                self.config.adapter_scale,
            )

    def forward(self, values: Tensor, surgeon: str | None = None) -> Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        values = self.first_norm(values + self.first_dropout(attended))
        values = self.second_norm(values + self.second_dropout(self.feedforward(values)))
        if surgeon is not None:
            self.add_surgeon(surgeon)
            values = self.adapters[surgeon](values)
        return values


class VisualEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(2048, hidden)

    def forward(self, frames: Tensor) -> Tensor:
        batch, steps, channels, height, width = frames.shape
        features = self.features(frames.reshape(batch * steps, channels, height, width)).flatten(1)
        return self.projection(features).reshape(batch, steps, -1)


class PositionEncoder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, positions: Tensor) -> Tensor:
        return self.layers(positions)


@dataclass(frozen=True)
class PolicyOutput:
    x_logits: Tensor
    y_logits: Tensor
    phase_logits: Tensor
    hidden: Tensor


class SurgeonPolicy(nn.Module):
    def __init__(self, config: PolicyConfig, phases: int = 10, primitives: int = 12) -> None:
        super().__init__()
        self.config = config
        self.visual = VisualEncoder(config.hidden)
        self.position = PositionEncoder(config.hidden)
        self.temporal = PositionEncoding(config.hidden)
        self.phase_embedding = nn.Embedding(phases, config.hidden)
        self.primitive_embedding = nn.Embedding(primitives, config.hidden)
        self.encoders = nn.ModuleList([AdaptedEncoderLayer(config) for _ in range(config.encoder_layers)])
        decoder_layer = nn.TransformerDecoderLayer(
            config.hidden,
            config.heads,
            config.feedforward,
            config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.action_embedding = nn.Linear(2, config.hidden)
        self.x_head = nn.Linear(config.hidden, config.spatial_bins)
        self.y_head = nn.Linear(config.hidden, config.spatial_bins)
        self.phase_head = nn.Linear(config.hidden, phases)

    def add_surgeon(self, surgeon: str) -> None:
        for layer in self.encoders:
            layer.add_surgeon(surgeon)

    def base_parameters(self) -> list[nn.Parameter]:
        adapter_ids = {id(parameter) for layer in self.encoders for parameter in layer.adapters.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in adapter_ids]

    def adapter_parameters(self, surgeon: str) -> list[nn.Parameter]:
        self.add_surgeon(surgeon)
        return [parameter for layer in self.encoders for parameter in layer.adapters[surgeon].parameters()]

    def encode(
        self,
        frames: Tensor,
        positions: Tensor,
        phases: Tensor,
        primitive_indices: Tensor,
        surgeon: str | None,
    ) -> Tensor:
        values = self.visual(frames)
        values = values + self.position(positions)
        values = values + self.phase_embedding(phases)
        values = values + self.primitive_embedding(primitive_indices)
        values = self.temporal(values)
        for layer in self.encoders:
            values = layer(values, surgeon)
        return values

    def forward(
        self,
        frames: Tensor,
        positions: Tensor,
        phases: Tensor,
        primitive_indices: Tensor,
        decoder_positions: Tensor,
        surgeon: str | None = None,
    ) -> PolicyOutput:
        memory = self.encode(frames, positions, phases, primitive_indices, surgeon)
        targets = self.temporal(self.action_embedding(decoder_positions))
        length = targets.size(1)
        mask = torch.triu(torch.full((length, length), float("-inf"), device=targets.device), diagonal=1)
        hidden = self.decoder(targets, memory, tgt_mask=mask)
        return PolicyOutput(self.x_head(hidden), self.y_head(hidden), self.phase_head(memory), hidden)
