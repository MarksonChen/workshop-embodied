from __future__ import annotations

import numpy as np

from demo_j.dataset import (
    ReferenceSet,
    ROUNDED_SOURCE_OFFSET,
    TARGET_CROP_START,
    TIME_SCALE,
    exact_source_frames,
    take_references,
)


def test_exact_source_mapping_undoes_rounded_provenance() -> None:
    raw_start = np.asarray([100, 400], np.int32)
    stored = raw_start + ROUNDED_SOURCE_OFFSET
    mapped = exact_source_frames(stored)
    np.testing.assert_allclose(
        mapped[:, 0], raw_start + TARGET_CROP_START / TIME_SCALE, atol=1e-6
    )
    np.testing.assert_allclose(np.diff(mapped, axis=1), 1.0 / TIME_SCALE, atol=1e-6)


def test_reference_subset_slices_only_clip_arrays() -> None:
    arrays = np.arange(12, dtype=np.float32).reshape(3, 4)
    reference = ReferenceSet(
        qpos=arrays,
        qvel=arrays,
        features=arrays,
        contacts=arrays,
        root_position=arrays,
        root_quaternion=arrays,
        joint_angles=arrays,
        command=arrays,
        teacher_action=arrays,
        session_index=np.arange(3),
        parent_clip_id=np.arange(3),
        source_start=np.arange(3),
        raw_source_start=np.arange(3),
        source_frame=arrays,
        sessions=("one", "two"),
        split="train",
        manifest_sha256="hash",
    )
    subset = take_references(reference, [2, 0])
    np.testing.assert_array_equal(subset.qpos, arrays[[2, 0]])
    assert subset.sessions == reference.sessions
    assert subset.split == "train"
