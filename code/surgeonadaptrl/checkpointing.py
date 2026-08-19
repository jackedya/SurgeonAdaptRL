"""Deterministic atomic training-state persistence for Section 3.9."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn


class CheckpointError(RuntimeError):
    """Checkpoint state is absent, malformed, or incompatible."""


@dataclass(frozen=True)
class Progress:
    stage: str
    epoch: int
    iteration: int
    samples: int


@dataclass(frozen=True)
class LoadResult:
    progress: Progress
    seed: int
    metadata: dict[str, str]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    if not required.issubset(state):
        raise CheckpointError("random-state payload is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    progress: Progress,
    seed: int,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": asdict(progress),
        "seed": int(seed),
        "random_state": capture_random_state(),
        "metadata": dict(metadata or {}),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def atomic_torch_save(payload: Mapping[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(dict(payload), temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    destination: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    progress: Progress,
    seed: int,
    metadata: Mapping[str, str] | None = None,
) -> str:
    payload = checkpoint_payload(model, optimizer, scheduler, scaler, progress, seed, metadata)
    atomic_torch_save(payload, destination)
    return file_sha256(destination)


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    required = {"format_version", "model", "optimizer", "progress", "seed", "random_state", "metadata"}
    missing = required.difference(payload)
    if missing:
        raise CheckpointError(f"checkpoint is missing fields: {sorted(missing)}")
    if payload["format_version"] != 1:
        raise CheckpointError("unsupported checkpoint format")
    progress = payload["progress"]
    if not isinstance(progress, dict) or not {"stage", "epoch", "iteration", "samples"}.issubset(progress):
        raise CheckpointError("checkpoint progress is malformed")
    return payload


def load_checkpoint(
    source: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> LoadResult:
    path = Path(source)
    if not path.is_file():
        raise CheckpointError("checkpoint does not exist")
    payload = _validate_payload(torch.load(path, map_location=device, weights_only=False))
    incompatibility = model.load_state_dict(payload["model"], strict=strict)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to_device(optimizer, torch.device(device))
    if scheduler is not None:
        if "scheduler" not in payload:
            raise CheckpointError("scheduler state was requested but is absent")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if "scaler" not in payload:
            raise CheckpointError("scaler state was requested but is absent")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_random_state(payload["random_state"])
    progress = Progress(**payload["progress"])
    return LoadResult(
        progress,
        int(payload["seed"]),
        {str(key): str(value) for key, value in payload["metadata"].items()},
        tuple(incompatibility.missing_keys),
        tuple(incompatibility.unexpected_keys),
    )


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    candidates = list(Path(directory).glob(pattern))
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)) if candidates else None


def prune_checkpoints(directory: str | Path, keep: int, pattern: str = "*.pt") -> list[Path]:
    if keep < 0:
        raise CheckpointError("keep count cannot be negative")
    candidates = sorted(Path(directory).glob(pattern), key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    removed: list[Path] = []
    for path in candidates[keep:]:
        path.unlink()
        removed.append(path)
    return removed


def parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_matches_model(source: str | Path, model: nn.Module) -> bool:
    payload = _validate_payload(torch.load(source, map_location="cpu", weights_only=False))
    temporary = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(payload["model"], strict=True)
        return parameter_fingerprint(model) == _state_dict_fingerprint(payload["model"])
    finally:
        model.load_state_dict(temporary, strict=True)


def _state_dict_fingerprint(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()
