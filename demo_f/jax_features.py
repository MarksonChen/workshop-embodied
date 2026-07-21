"""Pure-JAX online implementation of Demo F's 60-D feature contract."""

from __future__ import annotations

import jax.numpy as jnp

from .config import FPS
from .config import FEATURE_DIM


CONTACT_VELOCITY_EPS = 1e-7


def contact_flags(contact_velocity, body_indices):
    """Apply Demo H's frozen raw-physics contact convention."""

    selected = jnp.asarray(contact_velocity)[jnp.asarray(body_indices)]
    return jnp.any(jnp.abs(selected) > CONTACT_VELOCITY_EPS, axis=-1)


def quaternion_matrix(quaternion):
    """Convert a normalized ``wxyz`` quaternion to a body-to-world matrix."""

    quaternion = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1e-8)
    w, x, y, z = quaternion
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def matrix_rotvec(matrix):
    """Convert a relative rotation matrix to its local rotation vector."""

    cosine = jnp.clip((jnp.trace(matrix) - 1) / 2, -1.0, 1.0)
    angle = jnp.arccos(cosine)
    skew = jnp.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        )
    )
    sine = jnp.sin(angle)
    scale = jnp.where(jnp.abs(sine) > 1e-5, angle / (2 * sine), 0.5)
    return skew * scale


def transition_feature(
    previous_root_position,
    root_position,
    previous_root_quaternion,
    root_quaternion,
    previous_joint_angles,
    joint_angles,
    previous_feet_local,
    feet_local,
    contacts,
):
    """Construct one causal feature from ``x[t-1]``, ``x[t]``, and contacts."""

    previous_rotation = quaternion_matrix(previous_root_quaternion)
    rotation = quaternion_matrix(root_quaternion)
    relative = previous_rotation.T @ rotation
    world_velocity = (root_position - previous_root_position) * FPS
    local_velocity = rotation.T @ world_velocity
    feature = jnp.concatenate(
        (
            local_velocity[:2],
            root_position[2:3],
            relative[:, :2].reshape(-1),
            matrix_rotvec(relative) * FPS,
            joint_angles,
            (joint_angles - previous_joint_angles) * FPS,
            feet_local.reshape(-1),
            ((feet_local - previous_feet_local) * FPS).reshape(-1),
            contacts.astype(jnp.float32),
        )
    )
    if feature.shape != (FEATURE_DIM,):
        raise ValueError(f"feature contract produced {feature.shape}, expected {(FEATURE_DIM,)}")
    return feature
