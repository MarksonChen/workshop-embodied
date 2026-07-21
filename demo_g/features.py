"""Compatibility imports for the shared Demo F feature contract."""

from demo_f.jax_features import (
    CONTACT_VELOCITY_EPS,
    contact_flags,
    matrix_rotvec,
    quaternion_matrix,
    transition_feature,
)

__all__ = [
    "CONTACT_VELOCITY_EPS",
    "contact_flags",
    "matrix_rotvec",
    "quaternion_matrix",
    "transition_feature",
]
