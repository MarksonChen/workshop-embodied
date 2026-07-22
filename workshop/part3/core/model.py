from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from workshop.part2.core.motion import hindsight_command
from workshop.part3.config import (
    ACTION_DIM,
    ACTION_PHASES,
    CLIP_FRAMES,
    COMMAND_DIM,
    COMMAND_HORIZON_FRAMES,
    FEATURE_DIM,
    PHASE_DIM,
    PLAN_DIM,
    PRIOR_CONTROL_LIMIT,
    PriorConfig,
)


class FeedbackActionDecoder(nn.Module):
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
        return (mean, log_std)


def pre_tanh(control: torch.Tensor) -> torch.Tensor:
    return torch.atanh(control.clamp(-PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT))


def tanh_gaussian_nll(
    mean: torch.Tensor, log_std: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    bounded = control.clamp(-PRIOR_CONTROL_LIMIT, PRIOR_CONTROL_LIMIT)
    residual = (torch.atanh(bounded) - mean) / log_std.exp()
    gaussian = 0.5 * (
        residual.square()
        + 2.0 * log_std
        + torch.log(control.new_tensor(2.0 * torch.pi))
    )
    return gaussian + torch.log1p(-bounded.square())


@dataclass
class StateActionWindows:
    history: torch.Tensor
    future: torch.Tensor
    command: torch.Tensor
    current_feature: torch.Tensor
    true_plan: torch.Tensor
    previous_control: torch.Tensor
    phase: torch.Tensor
    action_command: torch.Tensor
    target_control: torch.Tensor
    anchors: np.ndarray


def command_frames(
    anchors: np.ndarray, config: PriorConfig
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(anchors, np.int64) * config.downsample - 1
    return (start, start + COMMAND_HORIZON_FRAMES)


def state_action_windows(
    tokens: torch.Tensor,
    normalized_features: torch.Tensor,
    controls: torch.Tensor,
    dataset,
    config: PriorConfig,
) -> StateActionWindows:
    last_command_anchor = (
        CLIP_FRAMES - 1 - COMMAND_HORIZON_FRAMES + 1
    ) // config.downsample
    last_future_anchor = tokens.shape[1] - ACTION_PHASES
    anchors = np.arange(
        config.history_tokens,
        min(last_command_anchor, last_future_anchor) + 1,
        dtype=np.int64,
    )
    if not len(anchors):
        raise ValueError("history/future/command contract leaves no windows")
    histories = torch.stack(
        [tokens[:, a - config.history_tokens : a] for a in anchors], dim=1
    )
    futures = torch.stack([tokens[:, a : a + ACTION_PHASES] for a in anchors], dim=1)
    command_start, command_future = command_frames(anchors, config)
    commands = np.stack(
        [
            hindsight_command(
                dataset.root_position,
                dataset.root_quaternion,
                start=int(start),
                future=int(future),
            )
            for start, future in zip(command_start, command_future, strict=True)
        ],
        axis=1,
    )
    current = []
    previous = []
    target = []
    for anchor in anchors:
        start = int(anchor * config.downsample - 1)
        current.append(normalized_features[:, start : start + ACTION_PHASES])
        previous.append(controls[:, start - 1 : start - 1 + ACTION_PHASES])
        target.append(controls[:, start : start + ACTION_PHASES])
    current = torch.stack(current, dim=1)
    previous = torch.stack(previous, dim=1)
    target = torch.stack(target, dim=1)
    true_plan = futures[:, :, :1].expand(-1, -1, ACTION_PHASES, -1)
    phase = (
        torch.eye(ACTION_PHASES, device=tokens.device)
        .view(1, 1, ACTION_PHASES, ACTION_PHASES)
        .expand(len(tokens), len(anchors), -1, -1)
    )
    action_command = (
        torch.as_tensor(commands, device=tokens.device)
        .unsqueeze(2)
        .expand(-1, -1, ACTION_PHASES, -1)
    )
    return StateActionWindows(
        history=histories.flatten(0, 1),
        future=futures.flatten(0, 1),
        command=torch.as_tensor(commands, device=tokens.device).flatten(0, 1),
        current_feature=current.flatten(0, 2),
        true_plan=true_plan.flatten(0, 2),
        previous_control=previous.flatten(0, 2),
        phase=phase.flatten(0, 2),
        action_command=action_command.flatten(0, 2),
        target_control=target.flatten(0, 2),
        anchors=anchors,
    )
