from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from surgeonadaptrl.checkpointing import (
    CheckpointError,
    Progress,
    capture_random_state,
    checkpoint_matches_model,
    file_sha256,
    latest_checkpoint,
    load_checkpoint,
    parameter_fingerprint,
    prune_checkpoints,
    restore_random_state,
    save_checkpoint,
)


def model_and_optimizer() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Linear(8, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def train_once(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    inputs = torch.randn(5, 3)
    target = torch.randn(5, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def test_random_state_round_trip() -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_random_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_random_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_checkpoint_restores_exact_parameters_and_progress(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model, optimizer = model_and_optimizer()
    train_once(model, optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    optimizer.step()
    scheduler.step()
    expected = parameter_fingerprint(model)
    destination = tmp_path / "state.pt"
    digest = save_checkpoint(
        destination,
        model,
        optimizer,
        scheduler,
        None,
        Progress("meta", 2, 41, 128),
        29,
        {"variant": "full"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = load_checkpoint(destination, model, optimizer, scheduler)
    assert parameter_fingerprint(model) == expected
    assert result.progress == Progress("meta", 2, 41, 128)
    assert result.seed == 29
    assert result.metadata == {"variant": "full"}
    assert digest == file_sha256(destination)


def test_checkpoint_model_match_is_nonmutating(tmp_path: Path) -> None:
    model, optimizer = model_and_optimizer()
    train_once(model, optimizer)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, None, None, Progress("base", 1, 0, 5), 13)
    before = parameter_fingerprint(model)
    assert checkpoint_matches_model(path, model)
    assert parameter_fingerprint(model) == before


def test_latest_and_prune_checkpoints(tmp_path: Path) -> None:
    model, optimizer = model_and_optimizer()
    paths = []
    for index in range(4):
        path = tmp_path / f"epoch_{index}.pt"
        save_checkpoint(path, model, optimizer, None, None, Progress("base", index, 0, 0), 13)
        path.touch()
        paths.append(path)
    assert latest_checkpoint(tmp_path) == paths[-1]
    removed = prune_checkpoints(tmp_path, keep=2)
    assert len(removed) == 2
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    model, optimizer = model_and_optimizer()
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.pt", model, optimizer)


def test_incomplete_random_state_is_rejected() -> None:
    with pytest.raises(CheckpointError):
        restore_random_state({"python": random.getstate()})
