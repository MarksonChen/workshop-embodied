"""The 60-D Fetch feature contract shared by offline Demo F and online Demo G."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .dataset.contract import FPS


SL = {
    "root_velocity": (0, 2),
    "root_height": (2, 3),
    "rotation_delta_6d": (3, 9),
    "root_angular_velocity": (9, 12),
    "joint_angles": (12, 22),
    "joint_velocity": (22, 32),
    "feet_local": (32, 44),
    "feet_velocity": (44, 56),
    "contacts": (56, 60),
}
FEATURE_DIM = 60


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def trajectory_features(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_angles: np.ndarray,
    feet_local: np.ndarray,
    contacts: np.ndarray,
) -> np.ndarray:
    """Convert ``(..., time, *)`` release arrays into 60-D causal features."""

    root_position = np.asarray(root_position, np.float32)
    quaternion = np.asarray(root_quaternion, np.float32)
    angles = np.asarray(joint_angles, np.float32)
    feet = np.asarray(feet_local, np.float32)
    contacts = np.asarray(contacts, np.float32)
    leading = root_position.shape[:-2]
    time = root_position.shape[-2]
    if root_position.shape[-1] != 3 or quaternion.shape[-2:] != (time, 4):
        raise ValueError("root trajectory shapes disagree")

    # scipy uses xyzw; the public release uses wxyz.
    xyzw = np.concatenate((quaternion[..., 1:], quaternion[..., :1]), axis=-1)
    rotation = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(
        leading + (time, 3, 3)
    )
    world_velocity = np.zeros_like(root_position)
    world_velocity[..., 1:, :] = np.diff(root_position, axis=-2) * FPS
    world_velocity[..., 0, :] = world_velocity[..., 1, :]
    local_velocity = np.einsum("...tji,...tj->...ti", rotation, world_velocity)

    relative = np.broadcast_to(np.eye(3), leading + (time, 3, 3)).copy()
    relative[..., 1:, :, :] = np.einsum(
        "...tji,...tjk->...tik", rotation[..., :-1, :, :], rotation[..., 1:, :, :]
    )
    rotvec = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(
        leading + (time, 3)
    ) * FPS
    # ``relative = R(t-1)^T R(t)`` is already expressed in the previous root
    # frame, so its rotation vector is the desired root-local angular velocity.
    angular_local = rotvec

    joint_velocity = np.zeros_like(angles)
    joint_velocity[..., 1:, :] = np.diff(angles, axis=-2) * FPS
    joint_velocity[..., 0, :] = joint_velocity[..., 1, :]
    foot_velocity = np.zeros_like(feet)
    foot_velocity[..., 1:, :, :] = np.diff(feet, axis=-3) * FPS
    foot_velocity[..., 0, :, :] = foot_velocity[..., 1, :, :]

    output = np.concatenate(
        (
            local_velocity[..., :2],
            root_position[..., 2:3],
            _rotation_6d(relative),
            angular_local,
            angles,
            joint_velocity,
            feet.reshape(leading + (time, 12)),
            foot_velocity.reshape(leading + (time, 12)),
            contacts,
        ),
        axis=-1,
    ).astype(np.float32)
    if output.shape[-1] != FEATURE_DIM:
        raise AssertionError(output.shape)
    return output
