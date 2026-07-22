from __future__ import annotations

import os
from pathlib import Path

from workshop.part3.config import ACTION_DIM, CLIP_FRAMES, STATE_DIM, TRANSITIONS


SCHEMA_VERSION = "0.2.0"
PARENT_VARIANT = "temporal-dilation-1p75-v1"
DATASET_VARIANT = "exact-fetch-feedback-projection-retime-1p75-v1"
PARENT_ROOT = Path(
    os.environ.get(
        "WORKSHOP_PART3_REFERENCE_DATA",
        Path(__file__).resolve().parents[2] / "data" / "part3_reference",
    )
)
DEFAULT_ROOT = Path(
    os.environ.get(
        "WORKSHOP_PART3_DATA",
        Path(__file__).resolve().parents[2] / "data" / "part3",
    )
)


FIELDS = {
    "reference_root_position": ("clips", CLIP_FRAMES, 3),
    "reference_root_quaternion": ("clips", CLIP_FRAMES, 4),
    "reference_joint_angles": ("clips", CLIP_FRAMES, 10),
    "realized_root_position": ("clips", CLIP_FRAMES, 3),
    "realized_root_quaternion": ("clips", CLIP_FRAMES, 4),
    "realized_joint_angles": ("clips", CLIP_FRAMES, 10),
    "realized_features": ("clips", CLIP_FRAMES, STATE_DIM),
    "realized_contacts": ("clips", CLIP_FRAMES, 4),
    "normalized_control": ("clips", TRANSITIONS, ACTION_DIM),
    "requested_actuator_torque": ("clips", TRANSITIONS, ACTION_DIM),
    "valid_transition_mask": ("clips", TRANSITIONS),
    "command": ("clips", 3),
    "initial_qp_pos": ("clips", 13, 3),
    "initial_qp_rot": ("clips", 13, 4),
    "initial_qp_vel": ("clips", 13, 3),
    "initial_qp_ang": ("clips", 13, 3),
    "parent_clip_id": ("clips",),
    "source_start": ("clips",),
    "source_speed_mps": ("clips",),
    "source_path_speed_mps": ("clips",),
    "joint_tracking_rmse": ("clips",),
    "root_tracking_rmse": ("clips",),
    "control_saturation_fraction": ("clips",),
    "minimum_torso_height": ("clips",),
    "minimum_upright": ("clips",),
    "maximum_planar_speed": ("clips",),
    "maximum_yaw_rate": ("clips",),
    "realized_command_speed": ("clips",),
    "solver_status": ("clips",),
}

DTYPES = {
    **{name: "float32" for name in FIELDS},
    "realized_contacts": "uint8",
    "valid_transition_mask": "uint8",
    "parent_clip_id": "int32",
    "source_start": "int32",
    "solver_status": "uint8",
}


def expected_shape(name: str, clips: int) -> tuple[int, ...]:
    shape = FIELDS[name]
    if shape[0] != "clips":
        raise AssertionError(shape)
    return (clips, *shape[1:])
