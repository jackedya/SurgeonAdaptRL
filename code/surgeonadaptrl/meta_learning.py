"""Support-query adaptation and differentiable meta-updates for Algorithm 1."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn


class MetaLearningError(ValueError):
    """A support-query task or meta-update is not well formed."""


Batch = Mapping[str, Tensor]
LossFunction = Callable[[nn.Module, Batch], Tensor]


@dataclass(frozen=True)
class SurgeonTask:
    surgeon: str
    support: tuple[Batch, ...]
    query: tuple[Batch, ...]

    def validate(self) -> None:
        if not self.surgeon:
            raise MetaLearningError("surgeon identifier cannot be empty")
        if not self.support or not self.query:
            raise MetaLearningError("each task requires support and query batches")


@dataclass(frozen=True)
class AdaptationTrace:
    initial_loss: float
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]


@dataclass(frozen=True)
class MetaStepResult:
    loss: float
    task_losses: dict[str, float]
    task_count: int


def split_support_query(
    indices: Sequence[int],
    support_size: int,
    query_size: int,
    generator: torch.Generator,
) -> tuple[list[int], list[int]]:
    if support_size <= 0 or query_size <= 0:
        raise MetaLearningError("support and query sizes must be positive")
    if len(indices) < support_size + query_size:
        raise MetaLearningError("task does not contain enough examples")
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    support = [indices[index] for index in permutation[:support_size]]
    query = [indices[index] for index in permutation[support_size : support_size + query_size]]
    if set(support).intersection(query):
        raise MetaLearningError("support and query sets must be disjoint")
    return support, query


class TaskSampler:
    def __init__(
        self,
        surgeon_indices: Mapping[str, Sequence[int]],
        support_size: int,
        query_size: int,
        seed: int,
    ) -> None:
        if not surgeon_indices:
            raise MetaLearningError("at least one surgeon is required")
        self.surgeon_indices = {name: tuple(indices) for name, indices in surgeon_indices.items()}
        self.support_size = support_size
        self.query_size = query_size
        self.generator = torch.Generator().manual_seed(seed)
        for name, indices in self.surgeon_indices.items():
            if not name or len(indices) < support_size + query_size:
                raise MetaLearningError("every surgeon must provide enough indexed examples")

    def sample(self, surgeons: int) -> dict[str, tuple[list[int], list[int]]]:
        if surgeons <= 0 or surgeons > len(self.surgeon_indices):
            raise MetaLearningError("invalid surgeon batch size")
        names = sorted(self.surgeon_indices)
        selection = torch.randperm(len(names), generator=self.generator)[:surgeons].tolist()
        return {
            names[index]: split_support_query(
                self.surgeon_indices[names[index]], self.support_size, self.query_size, self.generator
            )
            for index in selection
        }


def named_trainable_parameters(module: nn.Module) -> OrderedDict[str, Tensor]:
    return OrderedDict((name, value) for name, value in module.named_parameters() if value.requires_grad)


def clone_parameters(parameters: Mapping[str, Tensor]) -> OrderedDict[str, Tensor]:
    return OrderedDict((name, value.clone()) for name, value in parameters.items())


def gradient_norm(gradients: Iterable[Tensor | None]) -> Tensor:
    squares = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return torch.zeros(())
    return torch.stack(squares).sum().sqrt()


def apply_gradient_step(
    parameters: Mapping[str, Tensor],
    gradients: Sequence[Tensor | None],
    learning_rate: float,
) -> OrderedDict[str, Tensor]:
    if learning_rate <= 0:
        raise MetaLearningError("inner learning rate must be positive")
    if len(parameters) != len(gradients):
        raise MetaLearningError("parameter and gradient counts differ")
    updated: OrderedDict[str, Tensor] = OrderedDict()
    for (name, value), gradient in zip(parameters.items(), gradients):
        updated[name] = value if gradient is None else value - learning_rate * gradient
    return updated


def functional_forward(module: nn.Module, parameters: Mapping[str, Tensor], *args: Tensor, **kwargs: Tensor) -> object:
    return torch.func.functional_call(module, parameters, args, kwargs)


class FunctionalAdapter:
    def __init__(self, module: nn.Module, learning_rate: float, first_order: bool = False) -> None:
        if learning_rate <= 0:
            raise MetaLearningError("learning rate must be positive")
        self.module = module
        self.learning_rate = learning_rate
        self.first_order = first_order
        self.parameters = named_trainable_parameters(module)
        if not self.parameters:
            raise MetaLearningError("adapter has no trainable parameters")

    def forward(self, *args: Tensor, **kwargs: Tensor) -> object:
        return functional_forward(self.module, self.parameters, *args, **kwargs)

    def step(self, loss: Tensor) -> float:
        gradients = torch.autograd.grad(
            loss,
            tuple(self.parameters.values()),
            create_graph=not self.first_order,
            allow_unused=True,
        )
        norm = float(gradient_norm(gradients))
        if self.first_order:
            gradients = tuple(None if value is None else value.detach() for value in gradients)
        self.parameters = apply_gradient_step(self.parameters, gradients, self.learning_rate)
        return norm


def mean_batch_loss(module: nn.Module, batches: Iterable[Batch], loss_function: LossFunction) -> Tensor:
    losses = [loss_function(module, batch) for batch in batches]
    if not losses:
        raise MetaLearningError("cannot compute loss over an empty batch collection")
    return torch.stack(losses).mean()


class AdapterOptimizer:
    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        learning_rate: float = 0.01,
        steps: int = 200,
        gradient_clip: float | None = None,
    ) -> None:
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise MetaLearningError("adapter parameter collection cannot be empty")
        if learning_rate <= 0 or steps <= 0:
            raise MetaLearningError("adaptation settings must be positive")
        self.learning_rate = learning_rate
        self.steps = steps
        self.gradient_clip = gradient_clip

    def adapt(self, closure: Callable[[], Tensor]) -> AdaptationTrace:
        optimizer = torch.optim.SGD(self.parameters, lr=self.learning_rate)
        losses: list[float] = []
        norms: list[float] = []
        initial = float(closure().detach())
        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.gradient_clip or float("inf"))
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        return AdaptationTrace(initial, tuple(losses), tuple(norms))


def cycle_batches(batches: Sequence[Batch]) -> Iterator[Batch]:
    if not batches:
        raise MetaLearningError("cannot cycle an empty batch list")
    while True:
        yield from batches


class FirstOrderMetaOptimizer:
    def __init__(
        self,
        model: nn.Module,
        outer_learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        inner_learning_rate: float = 0.01,
        inner_steps: int = 1,
    ) -> None:
        if outer_learning_rate <= 0 or inner_learning_rate <= 0 or inner_steps <= 0:
            raise MetaLearningError("meta-optimization settings must be positive")
        self.model = model
        self.inner_learning_rate = inner_learning_rate
        self.inner_steps = inner_steps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=outer_learning_rate, weight_decay=weight_decay)

    def task_gradient(self, task: SurgeonTask, loss_function: LossFunction) -> tuple[Tensor, list[Tensor | None]]:
        task.validate()
        clone = type(self.model)(*self._constructor_arguments())
        clone.load_state_dict(self.model.state_dict())
        clone.to(next(self.model.parameters()).device)
        inner = torch.optim.SGD(clone.parameters(), lr=self.inner_learning_rate)
        support_cycle = cycle_batches(task.support)
        for _ in range(self.inner_steps):
            inner.zero_grad(set_to_none=True)
            loss_function(clone, next(support_cycle)).backward()
            inner.step()
        query_loss = mean_batch_loss(clone, task.query, loss_function)
        gradients = torch.autograd.grad(query_loss, tuple(clone.parameters()), allow_unused=True)
        return query_loss.detach(), list(gradients)

    def _constructor_arguments(self) -> tuple[object, ...]:
        arguments = getattr(self.model, "constructor_arguments", None)
        if arguments is None:
            raise MetaLearningError("model must expose constructor_arguments for isolated task adaptation")
        return tuple(arguments)

    def step(self, tasks: Sequence[SurgeonTask], loss_function: LossFunction) -> MetaStepResult:
        if not tasks:
            raise MetaLearningError("meta-step requires at least one task")
        aggregate: list[Tensor | None] = [None for _ in self.model.parameters()]
        task_losses: dict[str, float] = {}
        for task in tasks:
            query_loss, gradients = self.task_gradient(task, loss_function)
            task_losses[task.surgeon] = float(query_loss)
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    aggregate[index] = gradient.detach() if aggregate[index] is None else aggregate[index] + gradient.detach()
        self.optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(self.model.parameters(), aggregate):
            if gradient is not None:
                parameter.grad = gradient / len(tasks)
        self.optimizer.step()
        mean_loss = sum(task_losses.values()) / len(task_losses)
        return MetaStepResult(mean_loss, task_losses, len(tasks))


def adaptation_parameter_ratio(shared: nn.Module, adapter: nn.Module) -> float:
    shared_count = sum(value.numel() for value in shared.parameters())
    adapter_count = sum(value.numel() for value in adapter.parameters())
    if shared_count == 0:
        raise MetaLearningError("shared module contains no parameters")
    return adapter_count / shared_count


def freeze_except(module: nn.Module, prefixes: Sequence[str]) -> list[str]:
    if not prefixes:
        raise MetaLearningError("at least one trainable prefix is required")
    trainable: list[str] = []
    for name, parameter in module.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise MetaLearningError("no parameters matched the requested prefixes")
    return trainable


def restore_trainability(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True


def task_weighted_mean(losses: Mapping[str, Tensor], weights: Mapping[str, float] | None = None) -> Tensor:
    if not losses:
        raise MetaLearningError("at least one task loss is required")
    selected = weights or {name: 1.0 for name in losses}
    if set(selected) != set(losses) or any(value < 0 for value in selected.values()):
        raise MetaLearningError("task weights must be nonnegative and cover every loss")
    denominator = sum(selected.values())
    if denominator == 0:
        raise MetaLearningError("task weights cannot all be zero")
    return sum(losses[name] * selected[name] for name in losses) / denominator
