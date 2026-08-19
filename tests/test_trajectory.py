from __future__ import annotations

import numpy as np
import pytest
import torch

from surgeonadaptrl.trajectory import (
    BoundaryMode,
    ImageGeometry,
    TrajectoryError,
    acceleration,
    apply_boundary,
    bounding_box,
    cumulative_positions,
    curvature,
    finite_difference_matrix,
    interpolate_missing,
    linear_resample,
    motion_features,
    moving_average,
    path_length,
    phase_boundaries,
    procrustes_align,
    segment_by_phase,
    signed_area,
    sliding_indices,
    smooth_least_squares,
    straightness,
    validate_normalized,
    velocity,
)


def line(samples: int = 6) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float64)
    return torch.stack((0.1 + 0.01 * time, 0.2 + 0.02 * time), dim=-1)


def test_image_geometry_round_trip() -> None:
    geometry = ImageGeometry(1920, 1080)
    pixels = torch.tensor(((0.0, 0.0), (1919.0, 1079.0), (810.5, 522.25)))
    normalized = geometry.normalize(pixels)
    assert torch.allclose(geometry.denormalize(normalized), pixels)
    validate_normalized(normalized)


@pytest.mark.parametrize("width,height", [(0, 5), (5, 0), (-1, 2)])
def test_image_geometry_rejects_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(TrajectoryError):
        ImageGeometry(width, height)


def test_constant_velocity_motion_has_zero_higher_derivatives() -> None:
    positions = line()
    expected_speed = torch.tensor(np.sqrt(0.01**2 + 0.02**2), dtype=torch.float64)
    assert torch.allclose(velocity(positions), expected_speed.expand(5))
    assert torch.allclose(acceleration(positions), torch.zeros(4, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(curvature(positions), torch.zeros(4, dtype=torch.float64), atol=1e-14)
    features = motion_features(positions)
    assert features.stack().shape == (4, 3)


def test_path_geometry() -> None:
    positions = line()
    assert torch.isclose(straightness(positions), torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(path_length(positions), torch.linalg.vector_norm(positions[-1] - positions[0]))
    assert torch.allclose(bounding_box(positions), torch.tensor((0.1, 0.2, 0.15, 0.3), dtype=torch.float64))


def test_signed_area_respects_orientation() -> None:
    clockwise = torch.tensor(((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))
    assert signed_area(clockwise) < 0
    assert torch.isclose(signed_area(clockwise.flip(0)), -signed_area(clockwise))


def test_boundary_modes() -> None:
    values = torch.tensor(((-0.2, 1.3), (2.4, -1.1)))
    assert torch.equal(apply_boundary(values, BoundaryMode.CLAMP), torch.tensor(((0.0, 1.0), (1.0, 0.0))))
    reflected = apply_boundary(values, BoundaryMode.REFLECT)
    validate_normalized(reflected)
    with pytest.raises(TrajectoryError):
        apply_boundary(values, BoundaryMode.REJECT)


def test_cumulative_positions() -> None:
    start = torch.tensor((0.1, 0.2))
    changes = torch.tensor(((0.1, 0.0), (0.0, 0.2), (-0.1, 0.0)))
    expected = torch.tensor(((0.2, 0.2), (0.2, 0.4), (0.1, 0.4)))
    assert torch.allclose(cumulative_positions(start, changes), expected)


def test_resampling_preserves_endpoints_and_duration() -> None:
    result = linear_resample(line(7), 30.0, 15.0)
    assert result.positions.shape == (4, 2)
    assert torch.allclose(result.positions[0], line(7)[0])
    assert torch.allclose(result.positions[-1], line(7)[-1])
    assert torch.isclose(result.source_times[-1], result.target_times[-1])


def test_phase_segmentation() -> None:
    labels = torch.tensor((0, 0, 0, 1, 1, 2, 2, 2, 2))
    assert torch.equal(phase_boundaries(labels), torch.tensor((0, 3, 5, 9)))
    segments = segment_by_phase(line(9), labels, minimum_length=3)
    assert [phase for phase, _ in segments] == [0, 2]
    assert [values.size(0) for _, values in segments] == [3, 4]


def test_sliding_indices_cover_valid_windows() -> None:
    windows = list(sliding_indices(20, 6, 4, 3))
    assert [(left.start, left.stop, right.start, right.stop) for left, right in windows] == [
        (0, 6, 6, 10),
        (3, 9, 9, 13),
        (6, 12, 12, 16),
        (9, 15, 15, 19),
    ]


def test_missing_interpolation_uses_neighbors_and_edges() -> None:
    positions = torch.tensor(((9.0, 9.0), (0.0, 0.0), (9.0, 9.0), (2.0, 4.0), (9.0, 9.0)))
    valid = torch.tensor((False, True, False, True, False))
    expected = torch.tensor(((0.0, 0.0), (0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (2.0, 4.0)))
    assert torch.allclose(interpolate_missing(positions, valid), expected)


def test_smoothing_reduces_second_difference_energy() -> None:
    noisy = line(12).float()
    noisy[5, 1] += 0.2
    difference = finite_difference_matrix(noisy.size(0), 2)
    smoothed = smooth_least_squares(noisy, 10.0)
    assert torch.linalg.vector_norm(difference @ smoothed) < torch.linalg.vector_norm(difference @ noisy)
    assert moving_average(noisy, 1).shape == noisy.shape


def test_procrustes_alignment_recovers_rigid_scaled_path() -> None:
    source = line().float()
    rotation = torch.tensor(((0.0, -1.0), (1.0, 0.0)))
    target = source @ rotation * 2.5 + torch.tensor((0.3, 0.4))
    aligned = procrustes_align(source, target)
    assert torch.allclose(aligned, target, atol=1e-5)
