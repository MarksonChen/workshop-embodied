"""Frozen public contract for ``MarksonChen/aldarondo2024-retargeted``."""

from __future__ import annotations

import os
from pathlib import Path


SCHEMA_VERSION = "1.0.0"
REPOSITORY_ID = "MarksonChen/aldarondo2024-retargeted"
UPSTREAM_DOI = "https://doi.org/10.7910/DVN/FB0MZT"
UPSTREAM_LICENSE = "ODC-By-1.0"
DEFAULT_ROOT = Path(
    os.environ.get(
        "DEMO_F_DATASET_ROOT",
        Path(__file__).resolve().parent / "release",
    )
)

CLIP_FRAMES = 64
FPS = 50
COMMAND_FRAME = 32
COMMAND_FUTURE_FRAME = 63

# Interleaved recording dates prevent validation/test from being a single late
# experimental period. These are embedded in every release manifest.
VAL_SESSIONS = (
    "2021_07_30_1", "2021_08_05_1", "2021_08_11_1",
    "2021_08_17_1", "2021_08_23_2", "2021_09_04_1",
)
TEST_SESSIONS = (
    "2021_08_01_1", "2021_08_07_1", "2021_08_13_1",
    "2021_08_19_1", "2021_08_27_1", "2021_09_15_1",
)
TRAIN_SESSIONS = (
    "2021_07_28_1", "2021_07_29_1", "2021_07_31_1", "2021_08_02_1",
    "2021_08_03_1", "2021_08_04_1", "2021_08_06_1", "2021_08_08_1",
    "2021_08_09_1", "2021_08_10_1", "2021_08_12_1", "2021_08_14_1",
    "2021_08_15_1", "2021_08_16_1", "2021_08_18_1", "2021_08_20_1",
    "2021_08_21_1", "2021_08_23_1", "2021_08_25_1", "2021_08_30_1",
    "2021_08_31_1", "2021_09_03_1", "2021_09_14_1", "2021_09_16_1",
    "2021_09_17_1", "2021_09_18_1",
)
SPLIT_SESSIONS = {
    "train": TRAIN_SESSIONS,
    "validation": VAL_SESSIONS,
    "test": TEST_SESSIONS,
}
SESSION_TO_SPLIT = {
    session: split
    for split, sessions in SPLIT_SESSIONS.items()
    for session in sessions
}

FIELDS = {
    "joint_angles": ["clips", CLIP_FRAMES, 10],
    "root_position": ["clips", CLIP_FRAMES, 3],
    "root_quaternion": ["clips", CLIP_FRAMES, 4],
    "feet_local": ["clips", CLIP_FRAMES, 4, 3],
    "contacts": ["clips", CLIP_FRAMES, 4],
    "command": ["clips", 3],
    "source_start": ["clips"],
    "source_speed_mps": ["clips"],
    "source_path_speed_mps": ["clips"],
    "ik_foot_rmse": ["clips"],
    "contact_speed_mean": ["clips"],
    "minimum_foot_height": ["clips"],
    "joint_limit_fraction": ["clips"],
}
DTYPES = {
    **{name: "float32" for name in FIELDS},
    "contacts": "uint8",
    "source_start": "int32",
}


def validate_split_contract() -> None:
    groups = [set(sessions) for sessions in SPLIT_SESSIONS.values()]
    assert tuple(map(len, groups)) == (26, 6, 6)
    assert len(set.union(*groups)) == 38
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
