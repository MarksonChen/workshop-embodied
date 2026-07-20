"""Differentiable kinematics for the unmodified Brax v1 Fetch body.

Only the four lower-limb endpoints are needed by the retargeter.  The equations
below are a direct transcription of ``System.default_qp`` and Fetch's joint
offsets.  Keeping this tiny function local lets retarget optimization run in the
main JAX environment while rendering still uses the original Brax model.
"""

from __future__ import annotations

import jax.numpy as jnp


def _rot_x(angle):
    zero = jnp.zeros_like(angle)
    one = jnp.ones_like(angle)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack(
        (
            one, zero, zero,
            zero, cosine, -sine,
            zero, sine, cosine,
        ),
        axis=-1,
    ).reshape(angle.shape + (3, 3))


def _rot_y(angle):
    zero = jnp.zeros_like(angle)
    one = jnp.ones_like(angle)
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack(
        (
            cosine, zero, sine,
            zero, one, zero,
            -sine, zero, cosine,
        ),
        axis=-1,
    ).reshape(angle.shape + (3, 3))


def _mv(rotation, vector):
    return jnp.einsum("...ij,...j->...i", rotation, vector)


def _leg_foot(bar_position, bar_rotation, hip_angle, knee_angle, side):
    """One lower-leg distal endpoint relative to the torso."""

    hip_rotation = _rot_y(hip_angle)
    upper_rotation = jnp.einsum("...ij,...jk->...ik", bar_rotation, hip_rotation)
    upper_parent_offset = jnp.asarray((0.0, 0.875 * side, 0.0))
    upper_child_offset = jnp.asarray((0.0, 0.0, 0.375))
    upper_position = bar_position + _mv(
        bar_rotation,
        upper_parent_offset - _mv(hip_rotation, upper_child_offset),
    )
    knee_rotation = _rot_y(knee_angle)
    lower_rotation = jnp.einsum("...ij,...jk->...ik", upper_rotation, knee_rotation)
    # Fetch's left and right lower-joint lateral offsets have opposite signs.
    lower_parent_offset = jnp.asarray((0.0, -0.25 * side, -0.25))
    lower_child_offset = jnp.asarray((0.0, 0.0, 0.25))
    lower_position = upper_position + _mv(
        upper_rotation,
        lower_parent_offset - _mv(knee_rotation, lower_child_offset),
    )
    return lower_position + _mv(lower_rotation, jnp.asarray((0.0, 0.0, -0.5)))


def fetch_feet(joint_angles):
    """Returns four Fetch foot endpoints for ``(..., 10)`` joint angles.

    Output order is front-right, front-left, back-right, back-left.  Positions
    are in the torso frame, matching the semantic target representation.
    """

    joint_angles = jnp.asarray(joint_angles)
    if joint_angles.shape[-1] != 10:
        raise ValueError(f"expected (..., 10) joint angles, got {joint_angles.shape}")
    leading = joint_angles.shape[:-1]
    front_position = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), leading + (3,))
    back_position = jnp.broadcast_to(jnp.asarray((-1.0, 0.0, 0.0)), leading + (3,))
    front_rotation = _rot_x(joint_angles[..., 0])
    back_rotation = _rot_x(joint_angles[..., 1])
    return jnp.stack(
        (
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 2],
                joint_angles[..., 3],
                -1.0,
            ),
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 4],
                joint_angles[..., 5],
                1.0,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 6],
                joint_angles[..., 7],
                -1.0,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 8],
                joint_angles[..., 9],
                1.0,
            ),
        ),
        axis=-2,
    )
