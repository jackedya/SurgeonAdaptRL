from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


class BehavioralCloning(nn.Module):
    def __init__(self, observation_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.network(observations).reshape(observations.size(0), self.horizon, 2)

    def loss(self, observations: Tensor, actions: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(self(observations), actions)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden

    def forward(self, time: Tensor) -> Tensor:
        half = self.hidden // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=time.device, dtype=time.dtype) / max(half - 1, 1)
        )
        angles = time.unsqueeze(-1) * frequency
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(channels, channels, 3, padding=1)
        self.second = nn.Conv1d(channels, channels, 3, padding=1)
        self.condition = nn.Linear(condition, channels)
        self.first_norm = nn.GroupNorm(8, channels)
        self.second_norm = nn.GroupNorm(8, channels)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        residual = values
        values = torch.nn.functional.silu(self.first_norm(self.first(values)))
        values = values + self.condition(condition).unsqueeze(-1)
        values = self.second(self.second_norm(torch.nn.functional.silu(values)))
        return values + residual


class DiffusionPolicy(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 256, steps: int = 100) -> None:
        super().__init__()
        self.steps = steps
        self.input = nn.Conv1d(2, hidden, 3, padding=1)
        self.observation = nn.Linear(observation_dim, hidden)
        self.time = SinusoidalTimeEmbedding(hidden)
        self.blocks = nn.ModuleList([ConditionalResidualBlock(hidden, hidden * 2) for _ in range(8)])
        self.output = nn.Conv1d(hidden, 2, 3, padding=1)
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1.0 - beta
        cumulative = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("cumulative", cumulative)

    def noise_prediction(self, actions: Tensor, observations: Tensor, time: Tensor) -> Tensor:
        values = self.input(actions.transpose(1, 2))
        condition = torch.cat((self.observation(observations), self.time(time.float())), dim=-1)
        for block in self.blocks:
            values = block(values, condition)
        return self.output(values).transpose(1, 2)

    def loss(self, observations: Tensor, actions: Tensor) -> Tensor:
        time = torch.randint(0, self.steps, (actions.size(0),), device=actions.device)
        noise = torch.randn_like(actions)
        cumulative = self.cumulative[time].reshape(-1, 1, 1)
        noisy = cumulative.sqrt() * actions + (1.0 - cumulative).sqrt() * noise
        prediction = self.noise_prediction(noisy, observations, time)
        return torch.nn.functional.mse_loss(prediction, noise)


class ActionChunkingTransformer(nn.Module):
    def __init__(self, observation_dim: int, hidden: int, heads: int, layers: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.observation = nn.Linear(observation_dim, hidden)
        self.queries = nn.Parameter(torch.randn(horizon, hidden) * 0.02)
        layer = nn.TransformerDecoderLayer(hidden, heads, hidden * 4, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.output = nn.Linear(hidden, 2)

    def forward(self, observations: Tensor) -> Tensor:
        memory = self.observation(observations).unsqueeze(1)
        queries = self.queries.unsqueeze(0).expand(observations.size(0), -1, -1)
        return self.output(self.decoder(queries, memory))


class ImplicitQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.layers(torch.cat((states, actions), dim=-1)).squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.layers(states).squeeze(-1)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_deviation = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, states: Tensor) -> torch.distributions.Normal:
        hidden = self.body(states)
        return torch.distributions.Normal(self.mean(hidden), self.log_deviation.exp())

    def forward(self, states: Tensor) -> Tensor:
        return self.distribution(states).mean


@dataclass(frozen=True)
class IQLLoss:
    value: Tensor
    critic: Tensor
    actor: Tensor


class ImplicitQLearning(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, expectile: float = 0.7, temperature: float = 3.0) -> None:
        super().__init__()
        self.first_q = ImplicitQNetwork(state_dim, action_dim, hidden)
        self.second_q = ImplicitQNetwork(state_dim, action_dim, hidden)
        self.value = ValueNetwork(state_dim, hidden)
        self.actor = GaussianActor(state_dim, action_dim, hidden)
        self.expectile = expectile
        self.temperature = temperature

    def losses(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        terminated: Tensor,
        discount: float = 0.99,
    ) -> IQLLoss:
        first = self.first_q(states, actions)
        second = self.second_q(states, actions)
        minimum = torch.minimum(first, second).detach()
        difference = minimum - self.value(states)
        weight = torch.where(difference > 0, self.expectile, 1.0 - self.expectile)
        value_loss = (weight * difference.pow(2)).mean()
        target = rewards + discount * (1.0 - terminated.float()) * self.value(next_states).detach()
        critic_loss = torch.nn.functional.mse_loss(first, target) + torch.nn.functional.mse_loss(second, target)
        advantage = (minimum - self.value(states)).detach()
        actor_weight = torch.exp(self.temperature * advantage).clamp_max(100.0)
        actor_loss = -(actor_weight * self.actor.distribution(states).log_prob(actions).sum(dim=-1)).mean()
        return IQLLoss(value_loss, critic_loss, actor_loss)


class Discriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.layers(torch.cat((states, actions), dim=-1)).squeeze(-1)


class GAILObjective:
    def __init__(self, discriminator: Discriminator) -> None:
        self.discriminator = discriminator

    def discriminator_loss(
        self,
        expert_states: Tensor,
        expert_actions: Tensor,
        policy_states: Tensor,
        policy_actions: Tensor,
    ) -> Tensor:
        expert = self.discriminator(expert_states, expert_actions)
        policy = self.discriminator(policy_states, policy_actions)
        return torch.nn.functional.softplus(-expert).mean() + torch.nn.functional.softplus(policy).mean()

    def reward(self, states: Tensor, actions: Tensor) -> Tensor:
        return torch.nn.functional.softplus(self.discriminator(states, actions))


@dataclass
class AggregatedDataset:
    states: list[Tensor]
    actions: list[Tensor]

    def append(self, states: Tensor, actions: Tensor) -> None:
        self.states.append(states.detach().cpu())
        self.actions.append(actions.detach().cpu())

    def tensors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        return torch.cat(self.states).to(device), torch.cat(self.actions).to(device)


class DAgger:
    def __init__(self, policy: nn.Module, expert: Callable[[Tensor], Tensor]) -> None:
        self.policy = policy
        self.expert = expert
        self.dataset = AggregatedDataset([], [])

    def collect(self, states: Tensor) -> None:
        with torch.no_grad():
            labels = self.expert(states)
        self.dataset.append(states, labels)

    def loss(self, states: Tensor, labels: Tensor) -> Tensor:
        prediction = self.policy(states)
        return torch.nn.functional.mse_loss(prediction, labels)


class MAMLImitation:
    def __init__(self, model: nn.Module, inner_rate: float = 0.01) -> None:
        self.model = model
        self.inner_rate = inner_rate

    def inner_update(self, loss: Tensor) -> dict[str, Tensor]:
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        return {name: parameter - self.inner_rate * gradient for name, parameter, gradient in zip(names, parameters, gradients)}


class Trans4Trans(nn.Module):
    def __init__(self, observation_dim: int, hidden: int = 512, heads: int = 8, layers: int = 6, horizon: int = 15) -> None:
        super().__init__()
        self.horizon = horizon
        self.input = nn.Linear(observation_dim, hidden)
        encoder = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.queries = nn.Parameter(torch.randn(horizon, hidden) * 0.02)
        decoder = nn.TransformerDecoderLayer(hidden, heads, hidden * 4, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder, 4)
        self.output = nn.Linear(hidden, 2)

    def forward(self, observations: Tensor) -> Tensor:
        memory = self.encoder(self.input(observations))
        queries = self.queries.unsqueeze(0).expand(observations.size(0), -1, -1)
        return self.output(self.decoder(queries, memory))
