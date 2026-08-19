from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameRecord:
    image: Path
    video: str
    surgeon: int
    frame: int
    phase: int
    x: float
    y: float


@dataclass(frozen=True)
class SequenceRecord:
    frames: tuple[FrameRecord, ...]
    future: tuple[FrameRecord, ...]


def load_manifest(path: str | Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            records.append(
                FrameRecord(
                    image=Path(row["image"]),
                    video=row["video"],
                    surgeon=int(row["surgeon"]),
                    frame=int(row["frame"]),
                    phase=int(row["phase"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                )
            )
    return records


def group_videos(records: Sequence[FrameRecord]) -> dict[str, list[FrameRecord]]:
    videos: dict[str, list[FrameRecord]] = {}
    for record in records:
        videos.setdefault(record.video, []).append(record)
    for sequence in videos.values():
        sequence.sort(key=lambda item: item.frame)
    return videos


def build_sequences(
    records: Sequence[FrameRecord],
    window: int,
    horizon: int,
    stride: int,
) -> list[SequenceRecord]:
    output: list[SequenceRecord] = []
    for frames in group_videos(records).values():
        stop = len(frames) - window - horizon + 1
        for start in range(0, max(stop, 0), stride):
            observed = tuple(frames[start : start + window])
            future = tuple(frames[start + window : start + window + horizon])
            if len({item.surgeon for item in observed + future}) == 1:
                output.append(SequenceRecord(observed, future))
    return output


def image_tensor(path: Path, height: int, width: int) -> Tensor:
    image = Image.open(path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class SurgicalSequenceDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        manifest: str | Path,
        height: int,
        width: int,
        window: int,
        horizon: int,
        stride: int,
    ) -> None:
        self.height = height
        self.width = width
        self.sequences = build_sequences(load_manifest(manifest), window, horizon, stride)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sequence = self.sequences[index]
        frames = torch.stack([image_tensor(item.image, self.height, self.width) for item in sequence.frames])
        positions = torch.tensor([[item.x, item.y] for item in sequence.frames], dtype=torch.float32)
        future = torch.tensor([[item.x, item.y] for item in sequence.future], dtype=torch.float32)
        phases = torch.tensor([item.phase for item in sequence.frames], dtype=torch.long)
        future_phases = torch.tensor([item.phase for item in sequence.future], dtype=torch.long)
        surgeon = torch.tensor(sequence.frames[0].surgeon, dtype=torch.long)
        return {
            "frames": frames,
            "positions": positions,
            "future": future,
            "phases": phases,
            "future_phases": future_phases,
            "surgeon": surgeon,
        }


def read_tracking_json(path: str | Path) -> Iterator[FrameRecord]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    for item in payload["frames"]:
        yield FrameRecord(
            image=Path(item["image"]),
            video=str(item["video"]),
            surgeon=int(item["surgeon"]),
            frame=int(item["frame"]),
            phase=int(item["phase"]),
            x=float(item["tip"][0]),
            y=float(item["tip"][1]),
        )


def normalized_displacements(positions: Tensor) -> Tensor:
    return positions[..., 1:, :] - positions[..., :-1, :]


def quantize_displacements(values: Tensor, bins: int, low: float, high: float) -> Tensor:
    scaled = (values.clamp(low, high) - low) / (high - low)
    return torch.round(scaled * (bins - 1)).long()


def dequantize_displacements(indices: Tensor, bins: int, low: float, high: float) -> Tensor:
    return low + indices.float() * (high - low) / (bins - 1)
