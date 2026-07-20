"""The shared recorded-motion side of the Demo B/E feature contract."""

from __future__ import annotations

import numpy as np

try:
    from .constants import ACTIVE_JOINTS, COMMAND_HORIZON_FRAMES, FM, FPS, FULL_FM, NKP
except ImportError:  # pragma: no cover
    from constants import ACTIVE_JOINTS, COMMAND_HORIZON_FRAMES, FM, FPS, FULL_FM, NKP


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw from MuJoCo's wxyz quaternion convention."""
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    rows = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return rows.reshape(q.shape[:-1] + (3, 3))


def mat_to_sixd(matrix: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns, concatenated like Demo B."""
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def motion_features(qpos: np.ndarray) -> np.ndarray:
    """Convert contiguous ``(..., time, 74)`` qpos into 85-D motion frames.

    The first frame has zero finite differences and an identity orientation
    increment.  Compute a complete clip before slicing crops so a crop boundary
    never fabricates a discontinuity.
    """
    qpos = np.asarray(qpos, np.float64)
    if qpos.shape[-1] != 74 or qpos.ndim < 2:
        raise ValueError(f"expected (..., time, 74) qpos, got {qpos.shape}")
    xy, height, quat = qpos[..., :2], qpos[..., 2:3], qpos[..., 3:7]
    joints = qpos[..., 7:][..., ACTIVE_JOINTS]
    yaw = quat_to_yaw(quat)
    world_velocity = np.zeros_like(xy)
    world_velocity[..., 1:, :] = np.diff(xy, axis=-2) * FPS
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    local_x = cosine * world_velocity[..., 0] - sine * world_velocity[..., 1]
    local_y = sine * world_velocity[..., 0] + cosine * world_velocity[..., 1]

    rotation = quat_to_mat(quat)
    delta_sixd = np.zeros(qpos.shape[:-1] + (6,), np.float64)
    delta_sixd[..., 0, :] = np.asarray([1, 0, 0, 0, 1, 0], np.float64)
    delta_rotation = np.einsum(
        "...tji,...tjk->...tik", rotation[..., :-1, :, :], rotation[..., 1:, :, :]
    )
    delta_sixd[..., 1:, :] = mat_to_sixd(delta_rotation)

    joint_velocity = np.zeros_like(joints)
    joint_velocity[..., 1:, :] = np.diff(joints, axis=-2) * FPS
    features = np.concatenate(
        [local_x[..., None], local_y[..., None], height, delta_sixd, joints, joint_velocity],
        axis=-1,
    ).astype(np.float32)
    if features.shape[-1] != FM:
        raise AssertionError(f"feature contract produced {features.shape[-1]}, expected {FM}")
    return features


def full_motion_features(qpos: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Original 281-D Demo B representation used by the known-good model.

    ``keypoints`` follows the Aldarondo HDF5 layout ``(time, 3, 23)`` in
    millimetres.  Marker positions and velocities are expressed in the root
    frame, so global arena position is not leaked into the motion tokenizer.
    """
    qpos = np.asarray(qpos, np.float64)
    keypoints = np.asarray(keypoints, np.float64)
    if qpos.ndim != 2 or qpos.shape[-1] != 74:
        raise ValueError(f"expected (time, 74) qpos, got {qpos.shape}")
    if keypoints.shape != (len(qpos), 3, NKP):
        raise ValueError(f"expected ({len(qpos)}, 3, {NKP}) keypoints, got {keypoints.shape}")

    xy, height, quaternion = qpos[:, :2], qpos[:, 2:3], qpos[:, 3:7]
    joints = qpos[:, 7:]
    yaw = quat_to_yaw(quaternion)
    world_velocity = np.zeros_like(xy)
    world_velocity[1:] = np.diff(xy, axis=0) * FPS
    cosine, sine = np.cos(-yaw), np.sin(-yaw)
    local_velocity = np.stack(
        [
            cosine * world_velocity[:, 0] - sine * world_velocity[:, 1],
            sine * world_velocity[:, 0] + cosine * world_velocity[:, 1],
        ],
        axis=-1,
    )

    rotation = quat_to_mat(quaternion)
    delta_sixd = np.zeros((len(qpos), 6), np.float64)
    delta_sixd[0] = np.asarray([1, 0, 0, 0, 1, 0], np.float64)
    delta_rotation = np.einsum(
        "tji,tjk->tik", rotation[:-1], rotation[1:]
    )
    delta_sixd[1:] = mat_to_sixd(delta_rotation)

    joint_velocity = np.zeros_like(joints)
    joint_velocity[1:] = np.diff(joints, axis=0) * FPS
    keypoint_world = keypoints.transpose(0, 2, 1) / 1000.0
    root_relative = keypoint_world - qpos[:, None, :3]
    keypoint_local = np.einsum(
        "tji,tkj->tki", rotation, root_relative
    ).reshape(len(qpos), -1)
    keypoint_velocity = np.zeros_like(keypoint_local)
    keypoint_velocity[1:] = np.diff(keypoint_local, axis=0) * FPS

    features = np.concatenate(
        [
            local_velocity,
            height,
            delta_sixd,
            joints,
            joint_velocity,
            keypoint_local,
            keypoint_velocity,
        ],
        axis=-1,
    ).astype(np.float32)
    if features.shape[-1] != FULL_FM:
        raise AssertionError(features.shape)
    return features


def hindsight_command(qpos: np.ndarray, frame: int) -> np.ndarray:
    """0.62 s egocentric displacement command beginning at ``frame``."""
    future = frame + COMMAND_HORIZON_FRAMES
    if frame < 0 or future >= len(qpos):
        raise IndexError((frame, future, len(qpos)))
    yaw = quat_to_yaw(qpos[:, 3:7])
    delta = qpos[future, :2] - qpos[frame, :2]
    cosine, sine = np.cos(-yaw[frame]), np.sin(-yaw[frame])
    return np.asarray(
        [
            cosine * delta[0] - sine * delta[1],
            sine * delta[0] + cosine * delta[1],
            (yaw[future] - yaw[frame] + np.pi) % (2 * np.pi) - np.pi,
        ],
        np.float32,
    )
