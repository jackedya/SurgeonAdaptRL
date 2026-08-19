from __future__ import annotations

import pytest
import torch

from surgeonadaptrl.phase import (
    MultiStagePhaseClassifier,
    OnlinePhaseDecoder,
    PhaseError,
    TemporalPhaseClassifier,
    class_weights,
    collapse_phases,
    confusion_matrix,
    edit_score,
    levenshtein,
    macro_recall,
    phase_durations,
    phase_metrics,
    phase_runs,
    phase_similarity,
    supervised_phase_loss,
    temporal_smoothing_loss,
    transition_f1,
    transition_indices,
    transition_matrix,
)


def labels() -> torch.Tensor:
    return torch.tensor((0, 0, 0, 1, 1, 2, 2, 2, 3, 3))


def test_phase_runs_and_durations() -> None:
    runs = phase_runs(labels())
    assert [(run.phase, run.start, run.stop, run.duration) for run in runs] == [
        (0, 0, 3, 3),
        (1, 3, 5, 2),
        (2, 5, 8, 3),
        (3, 8, 10, 2),
    ]
    assert torch.equal(collapse_phases(labels()), torch.tensor((0, 1, 2, 3)))
    assert torch.equal(phase_durations(labels(), 5), torch.tensor((3.0, 2.0, 3.0, 2.0, 0.0)))


def test_transition_indices_and_matrix() -> None:
    assert torch.equal(transition_indices(labels()), torch.tensor((3, 5, 8)))
    matrix = transition_matrix(labels(), 4)
    assert matrix.shape == (4, 4)
    assert torch.allclose(matrix[:3].sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert matrix[0, 0] > matrix[0, 1]


def test_phase_similarity_matches_duration_definition() -> None:
    predicted = torch.tensor((0, 0, 1, 1, 1, 2, 2, 2, 3, 3))
    expected_difference = 2.0
    expected = 1.0 - expected_difference / labels().numel()
    assert torch.isclose(phase_similarity(predicted, labels(), 4), torch.tensor(expected))
    assert torch.isclose(phase_similarity(labels(), labels(), 4), torch.tensor(1.0))


def test_confusion_and_macro_recall() -> None:
    target = torch.tensor((0, 0, 1, 1))
    predicted = torch.tensor((0, 1, 1, 1))
    matrix = confusion_matrix(predicted, target, 2)
    assert torch.equal(matrix, torch.tensor(((1, 1), (0, 2))))
    assert torch.isclose(macro_recall(matrix), torch.tensor(0.75))


def test_transition_and_edit_metrics() -> None:
    predicted = torch.tensor((0, 0, 1, 1, 1, 2, 2, 3, 3, 3))
    assert torch.isclose(transition_f1(predicted, labels(), tolerance=1), torch.tensor(1.0))
    assert levenshtein(torch.tensor((0, 1, 3)), torch.tensor((0, 1, 2, 3))) == 1
    assert torch.isclose(edit_score(predicted, labels()), torch.tensor(1.0))


def test_phase_metrics_perfect_sequence() -> None:
    metrics = phase_metrics(labels(), labels(), 4)
    assert float(metrics.accuracy) == 1.0
    assert float(metrics.macro_recall) == 1.0
    assert float(metrics.transition_f1) == 1.0
    assert float(metrics.edit_score) == 1.0


def test_temporal_classifier_shape_and_causality() -> None:
    torch.manual_seed(2)
    model = TemporalPhaseClassifier(12, phases=4, channels=16, layers=3, kernel=5, dropout=0.0).eval()
    features = torch.randn(2, 20, 12)
    altered = features.clone()
    altered[:, 12:] = torch.randn_like(altered[:, 12:])
    first = model(features)
    second = model(altered)
    assert first.shape == (2, 20, 4)
    assert torch.allclose(first[:, :12], second[:, :12], atol=1e-6)


def test_multistage_loss_backward() -> None:
    model = MultiStagePhaseClassifier(8, 4, stages=2, channels=12, layers=2, kernel=3, dropout=0.0)
    features = torch.randn(3, 15, 8)
    target = torch.randint(0, 4, (3, 15))
    outputs = model(features)
    loss = supervised_phase_loss(outputs, target)
    loss.backward()
    assert len(outputs) == 2
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_temporal_smoothing_prefers_constant_logits() -> None:
    constant = torch.ones(2, 8, 4)
    varying = torch.randn(2, 8, 4)
    assert float(temporal_smoothing_loss(constant)) == 0.0
    assert temporal_smoothing_loss(varying) > 0


def test_class_weights_ignore_absent_classes() -> None:
    weights = class_weights(torch.tensor((0, 0, 0, 1)), 3)
    assert weights[0] < weights[1]
    assert weights[2] == 0


def test_online_decoder_uses_transition_prior() -> None:
    transition = torch.tensor(((0.1, 0.9), (0.0, 1.0)))
    decoder = OnlinePhaseDecoder(transition)
    assert decoder.update(torch.tensor((3.0, 0.0))) == 0
    assert decoder.update(torch.tensor((0.1, 0.0))) == 1
    decoder.reset()
    assert decoder.phase is None


def test_phase_validation() -> None:
    with pytest.raises(PhaseError):
        TemporalPhaseClassifier(0)
    with pytest.raises(PhaseError):
        phase_durations(torch.tensor((0, 4)), 4)
    with pytest.raises(PhaseError):
        confusion_matrix(torch.tensor((0,)), torch.tensor((0, 1)), 2)
