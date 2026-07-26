from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


@dataclass(frozen=True)
class Detection:
    box: Tensor
    score: float
    label: int
    mask: Tensor
    tip: Tensor
    embedding: Tensor


def build_detector(classes: int = 20) -> MaskRCNN:
    backbone = resnet_fpn_backbone("resnet50", weights="IMAGENET1K_V1", trainable_layers=5)
    return MaskRCNN(backbone, classes)


def mask_tip(mask: Tensor, box: Tensor) -> Tensor:
    points = torch.nonzero(mask > 0.5, as_tuple=False)
    if points.numel() == 0:
        return torch.stack(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    center = points.float().mean(dim=0)
    distances = torch.linalg.vector_norm(points.float() - center, dim=-1)
    endpoint = points[distances.argmax()].flip(0).float()
    return endpoint


def intersection_over_union(first: Tensor, second: Tensor) -> Tensor:
    top_left = torch.maximum(first[:2], second[:2])
    bottom_right = torch.minimum(first[2:], second[2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod()
    first_area = (first[2:] - first[:2]).clamp_min(0).prod()
    second_area = (second[2:] - second[:2]).clamp_min(0).prod()
    return intersection / (first_area + second_area - intersection).clamp_min(1e-8)


class AppearanceEncoder(nn.Module):
    def __init__(self, embedding: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding),
        )

    def forward(self, crops: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.layers(crops), dim=-1)


class KalmanFilter:
    def __init__(self) -> None:
        self.motion = np.eye(8)
        self.motion[:4, 4:] = np.eye(4)
        self.observation = np.zeros((4, 8))
        self.observation[:4, :4] = np.eye(4)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros(4)]
        covariance = np.diag([10.0, 10.0, 1e-2, 10.0, 100.0, 100.0, 1e-5, 100.0])
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = max(mean[3], 1.0)
        deviation = np.array([scale / 20, scale / 20, 1e-2, scale / 20, scale / 160, scale / 160, 1e-5, scale / 160])
        noise = np.diag(deviation**2)
        return self.motion @ mean, self.motion @ covariance @ self.motion.T + noise

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = max(mean[3], 1.0)
        deviation = np.array([scale / 20, scale / 20, 1e-1, scale / 20])
        noise = np.diag(deviation**2)
        return self.observation @ mean, self.observation @ covariance @ self.observation.T + noise

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_covariance = self.project(mean, covariance)
        gain = np.linalg.solve(projected_covariance, self.observation @ covariance).T
        innovation = measurement - projected_mean
        return mean + gain @ innovation, covariance - gain @ projected_covariance @ gain.T


@dataclass
class Track:
    identifier: int
    mean: np.ndarray
    covariance: np.ndarray
    embedding: Tensor
    label: int
    age: int = 1
    missed: int = 0
    hits: int = 1

    def box(self) -> Tensor:
        x, y, aspect, height = self.mean[:4]
        width = aspect * height
        return torch.tensor((x - width / 2, y - height / 2, x + width / 2, y + height / 2))


def box_measurement(box: Tensor) -> np.ndarray:
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    return np.array(
        [
            float((box[0] + box[2]) / 2),
            float((box[1] + box[3]) / 2),
            width / max(height, 1e-6),
            height,
        ]
    )


class DeepSort:
    def __init__(self, maximum_missed: int = 30, appearance_weight: float = 0.7, threshold: float = 0.7) -> None:
        self.maximum_missed = maximum_missed
        self.appearance_weight = appearance_weight
        self.threshold = threshold
        self.filter = KalmanFilter()
        self.tracks: list[Track] = []
        self.next_identifier = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.mean, track.covariance = self.filter.predict(track.mean, track.covariance)
            track.age += 1
            track.missed += 1

    def cost(self, track: Track, detection: Detection) -> float:
        appearance = 1.0 - float(torch.dot(track.embedding, detection.embedding))
        geometry = 1.0 - float(intersection_over_union(track.box(), detection.box))
        label_penalty = 0.0 if track.label == detection.label else 1.0
        return self.appearance_weight * appearance + (1.0 - self.appearance_weight) * geometry + label_penalty

    def associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        costs = np.array([[self.cost(track, detection) for detection in detections] for track in self.tracks])
        rows, columns = linear_sum_assignment(costs)
        matches = [(int(row), int(column)) for row, column in zip(rows, columns) if costs[row, column] <= self.threshold]
        matched_tracks = {row for row, _ in matches}
        matched_detections = {column for _, column in matches}
        return (
            matches,
            [index for index in range(len(self.tracks)) if index not in matched_tracks],
            [index for index in range(len(detections)) if index not in matched_detections],
        )

    def update(self, detections: list[Detection]) -> list[Track]:
        self.predict()
        matches, unmatched_tracks, unmatched_detections = self.associate(detections)
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.mean, track.covariance = self.filter.update(
                track.mean,
                track.covariance,
                box_measurement(detection.box),
            )
            track.embedding = torch.nn.functional.normalize(0.9 * track.embedding + 0.1 * detection.embedding, dim=0)
            track.missed = 0
            track.hits += 1
        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            mean, covariance = self.filter.initiate(box_measurement(detection.box))
            self.tracks.append(
                Track(
                    self.next_identifier,
                    mean,
                    covariance,
                    detection.embedding,
                    detection.label,
                )
            )
            self.next_identifier += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.maximum_missed]
        return self.tracks
