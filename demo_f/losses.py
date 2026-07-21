"""Small data-prior losses with no training/evaluation side effects."""

from __future__ import annotations

import torch

from .config import JOINT_LIMIT
from .features import SL


def joint_limit_loss(
    normalized_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    *,
    margin: float = 0.95,
) -> torch.Tensor:
    """Penalize decoded joint angles before Fetch's hard limits."""

    joint_slice = slice(*SL["joint_angles"])
    angles = (
        normalized_features[..., joint_slice] * feature_std[joint_slice]
        + feature_mean[joint_slice]
    )
    excess = torch.relu(angles.abs() - margin * JOINT_LIMIT)
    return excess.square().mean()
