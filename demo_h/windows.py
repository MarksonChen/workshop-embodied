"""Permanent temporal alignment between 12.5 Hz plans and 50 Hz controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from demo_f.commands import hindsight_command
from demo_h.config import ACTION_PHASES, COMMAND_HORIZON_FRAMES, PriorConfig


@dataclass
class StateActionWindows:
    history: torch.Tensor
    future: torch.Tensor
    command: torch.Tensor
    action_history: torch.Tensor
    action_anchor_command: torch.Tensor
    current_feature: torch.Tensor
    true_plan: torch.Tensor
    previous_control: torch.Tensor
    phase: torch.Tensor
    action_command: torch.Tensor
    target_control: torch.Tensor
    anchors: np.ndarray
    action_anchors: np.ndarray


def command_frames(
    anchors: np.ndarray, config: PriorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed episode command window used by every causal anchor."""

    anchors = np.asarray(anchors, np.int64)
    start = np.full_like(
        anchors, config.history_tokens * config.downsample - 1, dtype=np.int64
    )
    return start, start + COMMAND_HORIZON_FRAMES


def state_action_windows(
    tokens: torch.Tensor,
    normalized_features: torch.Tensor,
    controls: torch.Tensor,
    dataset,
    config: PriorConfig,
    *,
    command_mode: str = "episode",
) -> StateActionWindows:
    """Build leak-free state and action targets.

    If target token ``anchor`` covers frames ``4a..4a+3``, history ends at
    frame ``4a-1``. Controls ``4a-1..4a+2`` produce those four target frames.
    The command begins at that same causal history endpoint.
    """

    if command_mode not in {"episode", "local"}:
        raise ValueError(f"unsupported command mode {command_mode!r}")
    last_future_anchor = tokens.shape[1] - ACTION_PHASES
    last_command_anchor = (
        normalized_features.shape[1] - COMMAND_HORIZON_FRAMES
    ) // config.downsample
    if command_mode == "local":
        last_future_anchor = min(last_future_anchor, last_command_anchor)
    anchors = np.arange(
        config.history_tokens,
        last_future_anchor + 1,
        dtype=np.int64,
    )
    action_anchors = np.arange(
        config.history_tokens,
        (last_command_anchor + 1) if command_mode == "local" else tokens.shape[1],
        dtype=np.int64,
    )
    if not len(anchors) or not len(action_anchors):
        raise ValueError("history/future/action contract leaves no windows")
    histories = torch.stack(
        [tokens[:, a - config.history_tokens : a] for a in anchors], dim=1
    )
    futures = torch.stack([tokens[:, a : a + ACTION_PHASES] for a in anchors], dim=1)
    action_histories = torch.stack(
        [tokens[:, a - config.history_tokens : a] for a in action_anchors], dim=1
    )
    # A rollout receives one command and holds it fixed.  Use the exact same
    # convention in every training window: displacement from the causal reset
    # frame over the standard 31-frame horizon.  The former per-anchor command
    # silently shortened training to the first 20 of the 48 deployed actions.
    if command_mode == "episode":
        command_start = int(command_frames(anchors[:1], config)[0][0])
        episode_command = hindsight_command(
            dataset.root_position,
            dataset.root_quaternion,
            start=command_start,
            future=command_start + COMMAND_HORIZON_FRAMES,
        )
        commands = np.broadcast_to(
            episode_command[:, None], (len(tokens), len(anchors), 3)
        ).copy()
        action_anchor_commands = np.broadcast_to(
            episode_command[:, None], (len(tokens), len(action_anchors), 3)
        ).copy()
    else:
        def local_commands(local_anchors: np.ndarray) -> np.ndarray:
            return np.stack(
                [
                    hindsight_command(
                        dataset.root_position,
                        dataset.root_quaternion,
                        start=int(anchor * config.downsample - 1),
                        future=int(
                            anchor * config.downsample
                            - 1
                            + COMMAND_HORIZON_FRAMES
                        ),
                    )
                    for anchor in local_anchors
                ],
                axis=1,
            )

        commands = local_commands(anchors)
        action_anchor_commands = local_commands(action_anchors)
    current = []
    previous = []
    target = []
    for anchor in action_anchors:
        start = int(anchor * config.downsample - 1)
        current.append(normalized_features[:, start : start + ACTION_PHASES])
        previous.append(controls[:, start - 1 : start - 1 + ACTION_PHASES])
        target.append(controls[:, start : start + ACTION_PHASES])
    current = torch.stack(current, dim=1)
    previous = torch.stack(previous, dim=1)
    target = torch.stack(target, dim=1)
    true_plan = torch.stack(
        [tokens[:, anchor : anchor + 1] for anchor in action_anchors], dim=1
    ).expand(-1, -1, ACTION_PHASES, -1)
    phase = torch.eye(ACTION_PHASES, device=tokens.device).view(
        1, 1, ACTION_PHASES, ACTION_PHASES
    ).expand(len(tokens), len(action_anchors), -1, -1)
    action_command = (
        torch.as_tensor(action_anchor_commands, device=tokens.device)
        .unsqueeze(2)
        .expand(-1, -1, ACTION_PHASES, -1)
    )
    return StateActionWindows(
        history=histories.flatten(0, 1),
        future=futures.flatten(0, 1),
        command=torch.as_tensor(commands, device=tokens.device).flatten(0, 1),
        action_history=action_histories.flatten(0, 1),
        action_anchor_command=torch.as_tensor(
            action_anchor_commands, device=tokens.device
        ).flatten(0, 1),
        current_feature=current.flatten(0, 2),
        true_plan=true_plan.flatten(0, 2),
        previous_control=previous.flatten(0, 2),
        phase=phase.flatten(0, 2),
        action_command=action_command.flatten(0, 2),
        target_control=target.flatten(0, 2),
        anchors=anchors,
        action_anchors=action_anchors,
    )
