"""The one new network block in Demo H: a short-plan feedback controller."""

from __future__ import annotations

import torch
import torch.nn as nn

from demo_h.config import (
    ACTION_DIM,
    COMMAND_DIM,
    FEATURE_DIM,
    PHASE_DIM,
    PLAN_DIM,
    PRIOR_CONTROL_LIMIT,
)


class FeedbackActionDecoder(nn.Module):
    """Gaussian control head conditioned on state, plan, phase, and goal."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = PLAN_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 192,
    ):
        super().__init__()
        input_dim = feature_dim + latent_dim + action_dim + PHASE_DIM + COMMAND_DIM
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        # Begin exactly at the strongest causal null: repeat the previous
        # control. Learning therefore has to justify every plan-conditioned
        # correction rather than first rediscovering action continuity.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def forward(self, feature, plan, previous_control, phase, command):
        previous_control = previous_control.clamp(
            -PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT
        )
        correction = self.network(
            torch.cat((feature, plan, previous_control, phase, command), dim=-1)
        )
        previous_mean = torch.atanh(previous_control)
        return previous_mean + correction

    def distribution(self, feature, plan, previous_control, phase, command):
        mean = self(feature, plan, previous_control, phase, command)
        log_std = self.log_std.clamp(-5.0, 1.0)
        return mean, log_std

def pre_tanh(control: torch.Tensor) -> torch.Tensor:
    return torch.atanh(control.clamp(-PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT))


def tanh_gaussian_nll(
    mean: torch.Tensor, log_std: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    """Exact elementwise density after transforming a Gaussian through tanh."""

    bounded = control.clamp(-PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT)
    residual = (torch.atanh(bounded) - mean) / log_std.exp()
    gaussian = 0.5 * (
        residual.square()
        + 2.0 * log_std
        + torch.log(control.new_tensor(2.0 * torch.pi))
    )
    return gaussian + torch.log1p(-bounded.square())
