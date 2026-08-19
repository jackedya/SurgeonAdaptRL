"""Dataset manifest validation and leakage-safe video-level partitions for Section 3.7."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


class ManifestError(ValueError):
    """A dataset manifest violates the trajectory data contract."""


@dataclass(frozen=True)
class FrameRecord:
    image: str
    video: str
    surgeon: str
    frame: int
    phase: int
    x: float
    y: float
    instrument: int | None = None

    def validate(self, phases: int = 10, instruments: int = 19) -> None:
        if not self.image or not self.video or not self.surgeon:
            raise ManifestError("image, video, and surgeon fields cannot be empty")
        if self.frame < 0 or not 0 <= self.phase < phases:
            raise ManifestError("frame or phase index is outside its valid range")
        if not 0 <= self.x <= 1 or not 0 <= self.y <= 1:
            raise ManifestError("trajectory coordinates must be normalized to [0, 1]")
        if self.instrument is not None and not 0 <= self.instrument < instruments:
            raise ManifestError("instrument index is outside its valid range")


@dataclass(frozen=True)
class VideoSummary:
    video: str
    surgeon: str
    frames: int
    first_frame: int
    last_frame: int
    phases: tuple[int, ...]
    instruments: tuple[int, ...]


@dataclass(frozen=True)
class ManifestSummary:
    records: int
    videos: int
    surgeons: int
    phase_counts: dict[int, int]
    instrument_counts: dict[int, int]
    coordinate_minimum: tuple[float, float]
    coordinate_maximum: tuple[float, float]


@dataclass(frozen=True)
class VideoPartition:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = (set(self.train), set(self.validation), set(self.test))
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ManifestError("video partitions overlap")


REQUIRED_COLUMNS = ("image", "video", "surgeon", "frame", "phase", "x", "y")


def parse_record(row: Mapping[str, str]) -> FrameRecord:
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise ManifestError(f"manifest columns are missing: {missing}")
    try:
        instrument_text = row.get("instrument", "")
        record = FrameRecord(
            image=row["image"].strip(),
            video=row["video"].strip(),
            surgeon=row["surgeon"].strip(),
            frame=int(row["frame"]),
            phase=int(row["phase"]),
            x=float(row["x"]),
            y=float(row["y"]),
            instrument=int(instrument_text) if instrument_text.strip() else None,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest contains an invalid numeric field") from error
    record.validate()
    return record


def read_manifest(path: str | Path) -> list[FrameRecord]:
    source = Path(path)
    if not source.is_file():
        raise ManifestError("manifest file does not exist")
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        records = [parse_record(row) for row in reader]
    if not records:
        raise ManifestError("manifest contains no records")
    validate_temporal_order(records)
    return records


def validate_temporal_order(records: Sequence[FrameRecord]) -> None:
    previous: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.video, record.frame)
        if key in seen:
            raise ManifestError(f"duplicate frame identifier in video {record.video}")
        seen.add(key)
        if record.video in previous and record.frame <= previous[record.video]:
            raise ManifestError(f"frames are not strictly ordered in video {record.video}")
        previous[record.video] = record.frame


def group_videos(records: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    grouped: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        grouped[record.video].append(record)
    for values in grouped.values():
        values.sort(key=lambda record: record.frame)
    return dict(grouped)


def video_summaries(records: Sequence[FrameRecord]) -> list[VideoSummary]:
    summaries = []
    for video, values in sorted(group_videos(records).items()):
        surgeons = {record.surgeon for record in values}
        if len(surgeons) != 1:
            raise ManifestError(f"video {video} has multiple surgeon labels")
        summaries.append(
            VideoSummary(
                video,
                next(iter(surgeons)),
                len(values),
                values[0].frame,
                values[-1].frame,
                tuple(sorted({record.phase for record in values})),
                tuple(sorted({record.instrument for record in values if record.instrument is not None})),
            )
        )
    return summaries


def summarize(records: Sequence[FrameRecord]) -> ManifestSummary:
    if not records:
        raise ManifestError("cannot summarize an empty manifest")
    phase_counts = Counter(record.phase for record in records)
    instrument_counts = Counter(record.instrument for record in records if record.instrument is not None)
    return ManifestSummary(
        len(records),
        len({record.video for record in records}),
        len({record.surgeon for record in records}),
        dict(sorted(phase_counts.items())),
        dict(sorted(instrument_counts.items())),
        (min(record.x for record in records), min(record.y for record in records)),
        (max(record.x for record in records), max(record.y for record in records)),
    )


def select_partition(records: Sequence[FrameRecord], partition: VideoPartition, split: str) -> list[FrameRecord]:
    partition.validate()
    if split not in {"train", "validation", "test"}:
        raise ManifestError("split must be train, validation, or test")
    selected_videos = set(getattr(partition, split))
    available = {record.video for record in records}
    missing = selected_videos.difference(available)
    if missing:
        raise ManifestError(f"partition refers to absent videos: {sorted(missing)}")
    return [record for record in records if record.video in selected_videos]


def official_cataracts_partition(video_names: Sequence[str]) -> VideoPartition:
    if len(video_names) != 50 or len(set(video_names)) != 50:
        raise ManifestError("CATARACTS official partition requires exactly 50 distinct videos")
    ordered = tuple(video_names)
    return VideoPartition(ordered[:25], tuple(), ordered[25:])


def surgeon_disjoint_partition(
    records: Sequence[FrameRecord],
    train_surgeons: Sequence[str],
    test_surgeons: Sequence[str],
    validation_surgeons: Sequence[str] = (),
) -> VideoPartition:
    surgeon_sets = tuple(map(set, (train_surgeons, validation_surgeons, test_surgeons)))
    if surgeon_sets[0] & surgeon_sets[1] or surgeon_sets[0] & surgeon_sets[2] or surgeon_sets[1] & surgeon_sets[2]:
        raise ManifestError("surgeon partitions overlap")
    grouped: dict[str, list[str]] = defaultdict(list)
    for summary in video_summaries(records):
        grouped[summary.surgeon].append(summary.video)
    requested = set().union(*surgeon_sets)
    absent = requested.difference(grouped)
    if absent:
        raise ManifestError(f"partition refers to absent surgeons: {sorted(absent)}")
    def collect(names: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(video for surgeon in names for video in grouped[surgeon]))

    return VideoPartition(collect(train_surgeons), collect(validation_surgeons), collect(test_surgeons))


def detect_video_leakage(partitions: Mapping[str, Sequence[FrameRecord]]) -> dict[tuple[str, str], tuple[str, ...]]:
    names = sorted(partitions)
    videos = {name: {record.video for record in partitions[name]} for name in names}
    overlap: dict[tuple[str, str], tuple[str, ...]] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = tuple(sorted(videos[first].intersection(videos[second])))
            if shared:
                overlap[(first, second)] = shared
    return overlap


def detect_frame_path_leakage(partitions: Mapping[str, Sequence[FrameRecord]]) -> dict[str, tuple[str, ...]]:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, records in partitions.items():
        for record in records:
            owners[record.image].add(split)
    return {path: tuple(sorted(splits)) for path, splits in owners.items() if len(splits) > 1}


def manifest_digest(records: Sequence[FrameRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_image_files(records: Sequence[FrameRecord], root: str | Path) -> list[str]:
    base = Path(root)
    missing = []
    for record in records:
        path = Path(record.image)
        resolved = path if path.is_absolute() else base / path
        if not resolved.is_file():
            missing.append(record.image)
    return sorted(set(missing))


def contiguous_segments(records: Sequence[FrameRecord], maximum_gap: int = 1) -> Iterator[list[FrameRecord]]:
    if maximum_gap <= 0:
        raise ManifestError("maximum gap must be positive")
    for values in group_videos(records).values():
        segment = [values[0]]
        for record in values[1:]:
            previous = segment[-1]
            if record.frame - previous.frame > maximum_gap or record.phase != previous.phase:
                yield segment
                segment = [record]
            else:
                segment.append(record)
        yield segment


def eligible_windows(records: Sequence[FrameRecord], observation: int, horizon: int, stride: int) -> int:
    if min(observation, horizon, stride) <= 0:
        raise ManifestError("window dimensions must be positive")
    total = 0
    for segment in contiguous_segments(records):
        available = len(segment) - observation - horizon
        if available >= 0:
            total += available // stride + 1
    return total


def surgeon_counts(records: Sequence[FrameRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for surgeon in sorted({record.surgeon for record in records}):
        selected = [record for record in records if record.surgeon == surgeon]
        result[surgeon] = {"frames": len(selected), "videos": len({record.video for record in selected})}
    return result


def assert_cadis_not_independent(
    cataracts_training_videos: Sequence[str],
    cadis_source_videos: Sequence[str],
) -> None:
    training = set(cataracts_training_videos)
    sources = set(cadis_source_videos)
    if not sources:
        raise ManifestError("CaDIS source-video set cannot be empty")
    if not sources.issubset(training):
        raise ManifestError("CaDIS source videos are not all present in CATARACTS training data")


def write_summary(records: Sequence[FrameRecord], destination: str | Path) -> None:
    payload = asdict(summarize(records))
    payload["sha256"] = manifest_digest(records)
    Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
