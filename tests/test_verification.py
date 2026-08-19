from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgeonadaptrl.verification import (
    MinimalTrajectoryPolicy,
    SyntheticMotionDataset,
    hash_json_payload,
    independent_loss,
    run_data_check,
    run_forward_check,
    run_motion_formula_check,
    run_science_core_check,
    run_verification,
    write_summary,
)


def test_synthetic_dataset_contract() -> None:
    dataset = SyntheticMotionDataset(7, 8, 4)
    assert len(dataset) == 7
    record = dataset[0]
    assert record["features"].shape == (8, 6)
    assert record["positions"].shape == (8, 2)
    assert record["future"].shape == (4, 2)
    assert record["phase"].ndim == 0


def test_independent_loss_has_gradient() -> None:
    model = MinimalTrajectoryPolicy(6, 12, 4)
    batch = next(iter(DataLoader(SyntheticMotionDataset(4, 8, 4), batch_size=4)))
    predicted, logits = model(batch["features"], batch["positions"])
    loss = independent_loss(predicted, batch["future"], logits, batch["phase"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_data_and_forward_checks_pass() -> None:
    loader = DataLoader(SyntheticMotionDataset(4, 8, 4), batch_size=2)
    batch = next(iter(loader))
    assert run_data_check(loader).status == "PASS"
    assert run_forward_check(MinimalTrajectoryPolicy(6, 12, 4), batch, torch.device("cpu")).status == "PASS"


def test_motion_formula_check_is_analytic() -> None:
    check = run_motion_formula_check()
    assert check.status == "PASS"
    assert np.isclose(check.evidence["speed"], check.evidence["expected_speed"])


def test_science_core_path_runs_with_future_phase_targets() -> None:
    check = run_science_core_check()
    assert check.status == "PASS", check.detail
    assert check.evidence["gradient_norm"] > 0
    assert check.evidence["parameter_change"] > 0


def test_full_verification_and_json_round_trip(tmp_path: Path) -> None:
    summary = run_verification(tmp_path)
    assert summary.failed == 0
    assert summary.passed == 7
    assert summary.not_run == 3
    assert summary.status == "PARTIALLY_VERIFIED"
    destination = tmp_path / "report.json"
    write_summary(summary, destination)
    payload = json.loads(destination.read_text())
    assert payload["passed"] == 7
    assert len(payload["checks"]) == 10


def test_canonical_payload_hash_is_order_independent() -> None:
    assert hash_json_payload({"a": 1, "b": 2}) == hash_json_payload({"b": 2, "a": 1})
