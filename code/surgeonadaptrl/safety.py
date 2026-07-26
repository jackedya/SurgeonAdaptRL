from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from .configuration import SafetyConfig


@dataclass(frozen=True)
class SimulationState:
    observation: Tensor
    position: Tensor
    velocity: Tensor
    contact_force: Tensor
    clearance: Tensor


@dataclass(frozen=True)
class SimulationStep:
    state: SimulationState
    reward: Tensor
    terminated: Tensor


class Simulation(Protocol):
    def reset(self, batch_size: int) -> SimulationState: ...

    def step(self, actions: Tensor) -> SimulationStep: ...


def velocity_cost(values: Tensor, maximum: float) -> Tensor:
    return (values.abs().amax(dim=-1) - maximum).clamp_min(0.0)


def force_cost(values: Tensor, maximum: float) -> Tensor:
    return (torch.linalg.vector_norm(values, dim=-1) - maximum).clamp_min(0.0)


def clearance_cost(values: Tensor, minimum: float) -> Tensor:
    return (minimum - values).clamp_min(0.0)


@dataclass(frozen=True)
class ConstraintCosts:
    velocity: Tensor
    force: Tensor
    clearance: Tensor

    def stacked(self) -> Tensor:
        return torch.stack((self.velocity, self.force, self.clearance), dim=-1)


def constraint_costs(state: SimulationState, config: SafetyConfig) -> ConstraintCosts:
    return ConstraintCosts(
        velocity_cost(state.velocity, config.maximum_velocity_mm_s),
        force_cost(state.contact_force, config.maximum_force_mn),
        clearance_cost(state.clearance, config.minimum_clearance_mm),
    )


class DualVariables(nn.Module):
    def __init__(self, constraints: int = 3) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(constraints))

    def forward(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw)


def discounted_returns(values: Tensor, discount: float) -> Tensor:
    returns = torch.zeros_like(values)
    running = torch.zeros_like(values[-1])
    for index in range(values.size(0) - 1, -1, -1):
        running = values[index] + discount * running
        returns[index] = running
    return returns


class PrimalDualObjective:
    def __init__(self, config: SafetyConfig, tolerances: Tensor) -> None:
        self.config = config
        self.tolerances = tolerances

    def policy_loss(
        self,
        log_probabilities: Tensor,
        rewards: Tensor,
        costs: Tensor,
        dual: Tensor,
    ) -> Tensor:
        reward_returns = discounted_returns(rewards, self.config.discount)
        cost_returns = discounted_returns(costs, self.config.discount)
        advantage = reward_returns - (cost_returns * dual).sum(dim=-1)
        normalized = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        return -(log_probabilities * normalized.detach()).mean()

    def dual_loss(self, costs: Tensor, dual: Tensor) -> Tensor:
        expected = discounted_returns(costs, self.config.discount)[0].mean(dim=0)
        return -(dual * (expected - self.tolerances).detach()).sum()


@dataclass
class SafetyReport:
    velocity_violations: int = 0
    force_violations: int = 0
    clearance_violations: int = 0
    steps: int = 0

    def update(self, costs: ConstraintCosts) -> None:
        self.velocity_violations += int((costs.velocity > 0).sum().item())
        self.force_violations += int((costs.force > 0).sum().item())
        self.clearance_violations += int((costs.clearance > 0).sum().item())
        self.steps += int(costs.velocity.numel())

    def rates(self) -> dict[str, float]:
        denominator = max(self.steps, 1)
        return {
            "velocity": self.velocity_violations / denominator,
            "force": self.force_violations / denominator,
            "clearance": self.clearance_violations / denominator,
            "overall": (
                self.velocity_violations + self.force_violations + self.clearance_violations
            )
            / (3 * denominator),
        }
