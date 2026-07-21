"""Permanent temporal alignment between 12.5 Hz plans and 50 Hz controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from demo_h.config import CLIP_FRAMES, PriorConfig
from demo_h.dataset.commands import hindsight_command


COMMAND_HORIZON = 31
ACTION_PHASES = 4


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


def state_action_windows(
    tokens: torch.Tensor,
    normalized_features: torch.Tensor,
    controls: torch.Tensor,
    dataset,
    config: PriorConfig,
) -> StateActionWindows:
    """Build leak-free state and action targets.

    If target token ``anchor`` covers frames ``4a..4a+3``, history ends at
    frame ``4a-1``. Controls ``4a-1..4a+2`` produce those four target frames.
    The command begins at that same causal history endpoint.
    """

    last_command_anchor = (CLIP_FRAMES - 1 - COMMAND_HORIZON + 1) // config.downsample
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
    commands = np.stack(
        [
            hindsight_command(
                dataset.root_position,
                dataset.root_quaternion,
                start=int(a * config.downsample - 1),
                future=int(a * config.downsample - 1 + COMMAND_HORIZON),
            )
            for a in anchors
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
    phase = torch.eye(ACTION_PHASES, device=tokens.device).view(
        1, 1, ACTION_PHASES, ACTION_PHASES
    ).expand(len(tokens), len(anchors), -1, -1)
    action_command = torch.as_tensor(commands, device=tokens.device).unsqueeze(2).expand(
        -1, -1, ACTION_PHASES, -1
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
