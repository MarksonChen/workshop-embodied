"""MJX-state side of Demo B's accepted 281-D motion feature contract."""

from __future__ import annotations

import jax.numpy as jp

from demo_b.constants import FULL_FM


FPS = 50.0


def joystick_to_hindsight_command(command, horizon_seconds: float):
    """Convert ``[forward velocity, yaw rate]`` to Demo B's command.

    Demo B conditions on the egocentric displacement and yaw change over a
    future horizon.  Constant forward velocity plus yaw rate traces an arc, so
    a turning command generally has nonzero lateral displacement.
    """
    velocity, yaw_rate = command
    angle = yaw_rate * horizon_seconds
    turning = jp.abs(yaw_rate) > 1e-6
    safe_yaw_rate = jp.where(turning, yaw_rate, 1.0)
    forward_scale = jp.where(
        turning, jp.sin(angle) / safe_yaw_rate, horizon_seconds
    )
    lateral_scale = jp.where(
        turning, (1.0 - jp.cos(angle)) / safe_yaw_rate, 0.0
    )
    return jp.asarray(
        [velocity * forward_scale, velocity * lateral_scale, angle]
    )


def quaternion_to_yaw(quaternion):
    w, x, y, z = jp.moveaxis(quaternion, -1, 0)
    return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quaternion_to_matrix(quaternion):
    quaternion = quaternion / jp.maximum(jp.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = jp.moveaxis(quaternion, -1, 0)
    entries = jp.stack(
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
    return entries.reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_sixd(matrix):
    return jp.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def keypoint_local(data, body_ids, body_offsets, root_position, root_quaternion):
    """Return the 23 calibrated skeleton sites in the root coordinate frame."""
    body_position = data.xpos[body_ids]
    body_rotation = data.xmat[body_ids].reshape((-1, 3, 3))
    world = body_position + jp.einsum("kij,kj->ki", body_rotation, body_offsets)
    root_rotation = quaternion_to_matrix(root_quaternion)
    return jp.einsum("ji,kj->ki", root_rotation, world - root_position)


def full_motion_feature(
    previous_qpos,
    current_qpos,
    previous_keypoints,
    current_keypoints,
):
    """281-D feature at the current 50-Hz physics sample.

    ``qpos`` uses world root translation, even when the MuJoCo walker is
    attached beneath a fixed spawn frame.  Keypoints are already root-local.
    """
    world_velocity = (current_qpos[:2] - previous_qpos[:2]) * FPS
    yaw = quaternion_to_yaw(current_qpos[3:7])
    cosine, sine = jp.cos(-yaw), jp.sin(-yaw)
    local_velocity = jp.stack(
        [
            cosine * world_velocity[0] - sine * world_velocity[1],
            sine * world_velocity[0] + cosine * world_velocity[1],
        ]
    )
    previous_rotation = quaternion_to_matrix(previous_qpos[3:7])
    current_rotation = quaternion_to_matrix(current_qpos[3:7])
    delta_sixd = matrix_to_sixd(previous_rotation.T @ current_rotation)
    joints = current_qpos[7:]
    joint_velocity = (current_qpos[7:] - previous_qpos[7:]) * FPS
    keypoint_velocity = (current_keypoints - previous_keypoints) * FPS
    feature = jp.concatenate(
        [
            local_velocity,
            current_qpos[2:3],
            delta_sixd,
            joints,
            joint_velocity,
            current_keypoints.reshape(-1),
            keypoint_velocity.reshape(-1),
        ]
    )
    if feature.shape != (FULL_FM,):
        raise ValueError(f"full motion feature has shape {feature.shape}, expected {(FULL_FM,)}")
    return feature
