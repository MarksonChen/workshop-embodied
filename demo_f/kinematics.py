"""Differentiable kinematics for the unmodified Brax v1 Fetch body.

Only the four lower-limb endpoints are needed by the retargeter.  The equations
below are a direct transcription of ``System.default_qp`` and Fetch's joint
offsets.  Keeping this tiny function local lets retarget optimization run in the
main JAX environment while rendering still uses the original Brax model.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def world_feet(root_position, yaw, local_feet):
    """Transform torso-local foot points by a yaw-only root trajectory."""

    root_position = np.asarray(root_position, np.float32)
    yaw = np.asarray(yaw, np.float32)
    local_feet = np.asarray(local_feet, np.float32)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = -sine
    rotation[..., 1, 0] = sine
    rotation[..., 1, 1] = cosine
    rotation[..., 2, 2] = 1.0
    return root_position[..., None, :] + np.einsum(
        "...tij,...tfj->...tfi", rotation, local_feet
    )


def _rot_x(angle, xp):
    zero = xp.zeros_like(angle)
    one = xp.ones_like(angle)
    cosine, sine = xp.cos(angle), xp.sin(angle)
    return xp.stack(
        (
            one, zero, zero,
            zero, cosine, -sine,
            zero, sine, cosine,
        ),
        axis=-1,
    ).reshape(angle.shape + (3, 3))


def _rot_y(angle, xp):
    zero = xp.zeros_like(angle)
    one = xp.ones_like(angle)
    cosine, sine = xp.cos(angle), xp.sin(angle)
    return xp.stack(
        (
            cosine, zero, sine,
            zero, one, zero,
            -sine, zero, cosine,
        ),
        axis=-1,
    ).reshape(angle.shape + (3, 3))


def _mv(rotation, vector, xp):
    return xp.einsum("...ij,...j->...i", rotation, vector)


def _leg_foot(bar_position, bar_rotation, hip_angle, knee_angle, side, xp):
    """One lower-leg distal endpoint relative to the torso."""

    hip_rotation = _rot_y(hip_angle, xp)
    upper_rotation = xp.einsum("...ij,...jk->...ik", bar_rotation, hip_rotation)
    upper_parent_offset = xp.asarray((0.0, 0.875 * side, 0.0))
    upper_child_offset = xp.asarray((0.0, 0.0, 0.375))
    upper_position = bar_position + _mv(
        bar_rotation,
        upper_parent_offset - _mv(hip_rotation, upper_child_offset, xp),
        xp,
    )
    knee_rotation = _rot_y(knee_angle, xp)
    lower_rotation = xp.einsum("...ij,...jk->...ik", upper_rotation, knee_rotation)
    # Fetch's left and right lower-joint lateral offsets have opposite signs.
    lower_parent_offset = xp.asarray((0.0, -0.25 * side, -0.25))
    lower_child_offset = xp.asarray((0.0, 0.0, 0.25))
    lower_position = upper_position + _mv(
        upper_rotation,
        lower_parent_offset - _mv(knee_rotation, lower_child_offset, xp),
        xp,
    )
    return lower_position + _mv(
        lower_rotation, xp.asarray((0.0, 0.0, -0.5)), xp
    )


def _fetch_feet(joint_angles, xp):
    joint_angles = xp.asarray(joint_angles)
    if joint_angles.shape[-1] != 10:
        raise ValueError(f"expected (..., 10) joint angles, got {joint_angles.shape}")
    leading = joint_angles.shape[:-1]
    front_position = xp.broadcast_to(xp.asarray((1.0, 0.0, 0.0)), leading + (3,))
    back_position = xp.broadcast_to(xp.asarray((-1.0, 0.0, 0.0)), leading + (3,))
    front_rotation = _rot_x(joint_angles[..., 0], xp)
    back_rotation = _rot_x(joint_angles[..., 1], xp)
    return xp.stack(
        (
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 2],
                joint_angles[..., 3],
                -1.0,
                xp,
            ),
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 4],
                joint_angles[..., 5],
                1.0,
                xp,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 6],
                joint_angles[..., 7],
                -1.0,
                xp,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 8],
                joint_angles[..., 9],
                1.0,
                xp,
            ),
        ),
        axis=-2,
    )


def fetch_feet(joint_angles):
    """Return differentiable JAX Fetch foot endpoints for ``(..., 10)`` angles.

    Output order is front-right, front-left, back-right, back-left.  Positions
    are in the torso frame, matching the semantic target representation.
    """

    return _fetch_feet(joint_angles, jnp)


def fetch_feet_numpy(joint_angles: np.ndarray) -> np.ndarray:
    """Return the same kinematics in NumPy for lightweight validation."""

    return np.asarray(_fetch_feet(np.asarray(joint_angles, np.float32), np), np.float32)
