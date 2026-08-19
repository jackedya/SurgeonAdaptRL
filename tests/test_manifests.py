from __future__ import annotations

from pathlib import Path

import pytest

from surgeonadaptrl.manifests import (
    FrameRecord,
    ManifestError,
    VideoPartition,
    assert_cadis_not_independent,
    contiguous_segments,
    detect_frame_path_leakage,
    detect_video_leakage,
    eligible_windows,
    manifest_digest,
    official_cataracts_partition,
    parse_record,
    select_partition,
    summarize,
    surgeon_counts,
    surgeon_disjoint_partition,
    validate_temporal_order,
    verify_image_files,
    video_summaries,
    write_summary,
)


def records() -> list[FrameRecord]:
    values = []
    for video_index, surgeon in enumerate(("s0", "s1")):
        for frame in range(12):
            values.append(
                FrameRecord(
                    f"v{video_index}/{frame}.png",
                    f"v{video_index}",
                    surgeon,
                    frame,
                    frame // 6,
                    0.1 + frame * 0.01,
                    0.2 + frame * 0.01,
                    frame % 3,
                )
            )
    return values


def test_record_parsing_and_validation() -> None:
    row = {"image": "a.png", "video": "v", "surgeon": "s", "frame": "2", "phase": "1", "x": "0.2", "y": "0.3"}
    record = parse_record(row)
    assert record.frame == 2
    assert record.instrument is None
    record.validate()
    with pytest.raises(ManifestError):
        FrameRecord("a", "v", "s", 0, 0, -0.1, 0.2).validate()


def test_temporal_order_rejects_duplicates_and_reversal() -> None:
    validate_temporal_order(records())
    duplicate = records() + [records()[0]]
    with pytest.raises(ManifestError):
        validate_temporal_order(duplicate)
    reversed_records = records()
    reversed_records[0], reversed_records[1] = reversed_records[1], reversed_records[0]
    with pytest.raises(ManifestError):
        validate_temporal_order(reversed_records)


def test_summaries_count_manifest_content() -> None:
    summary = summarize(records())
    assert summary.records == 24
    assert summary.videos == 2
    assert summary.surgeons == 2
    assert summary.phase_counts == {0: 12, 1: 12}
    videos = video_summaries(records())
    assert [(video.video, video.frames, video.phases) for video in videos] == [("v0", 12, (0, 1)), ("v1", 12, (0, 1))]


def test_partitions_are_video_and_surgeon_disjoint() -> None:
    partition = surgeon_disjoint_partition(records(), ("s0",), ("s1",))
    assert partition.train == ("v0",)
    assert partition.test == ("v1",)
    assert len(select_partition(records(), partition, "train")) == 12
    with pytest.raises(ManifestError):
        VideoPartition(("v0",), (), ("v0",)).validate()


def test_official_partition_requires_fifty_distinct_videos() -> None:
    names = tuple(f"video_{index:02d}" for index in range(50))
    partition = official_cataracts_partition(names)
    assert len(partition.train) == 25
    assert len(partition.test) == 25
    with pytest.raises(ManifestError):
        official_cataracts_partition(names[:-1])


def test_leakage_detectors() -> None:
    first = records()[:12]
    second = records()[6:]
    video_overlap = detect_video_leakage({"train": first, "test": second})
    assert video_overlap == {("test", "train"): ("v0",)}
    frame_overlap = detect_frame_path_leakage({"train": first, "test": second})
    assert len(frame_overlap) == 6


def test_segments_and_window_count_respect_phase_boundaries() -> None:
    segments = list(contiguous_segments(records()))
    assert [len(segment) for segment in segments] == [6, 6, 6, 6]
    assert eligible_windows(records(), observation=3, horizon=2, stride=1) == 8
    assert eligible_windows(records(), observation=5, horizon=2, stride=1) == 0


def test_digest_is_order_sensitive_and_summary_writes(tmp_path: Path) -> None:
    values = records()
    assert manifest_digest(values) != manifest_digest(list(reversed(values)))
    destination = tmp_path / "summary.json"
    write_summary(values, destination)
    assert '"records": 24' in destination.read_text()
    assert '"sha256"' in destination.read_text()


def test_image_verification_and_counts(tmp_path: Path) -> None:
    values = records()
    assert len(verify_image_files(values, tmp_path)) == 24
    first_path = tmp_path / values[0].image
    first_path.parent.mkdir(parents=True)
    first_path.write_bytes(b"image")
    assert len(verify_image_files(values, tmp_path)) == 23
    assert surgeon_counts(values) == {"s0": {"frames": 12, "videos": 1}, "s1": {"frames": 12, "videos": 1}}


def test_cadis_relationship_guard() -> None:
    assert_cadis_not_independent(("v0", "v1", "v2"), ("v0", "v2"))
    with pytest.raises(ManifestError):
        assert_cadis_not_independent(("v0",), ("v1",))
