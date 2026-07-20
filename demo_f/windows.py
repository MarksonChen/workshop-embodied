"""Causal latent windows shared by Demo F training and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from .config import PriorConfig
from .dataset import hindsight_commands
from .dataset.contract import COMMAND_FRAME, COMMAND_FUTURE_FRAME


@torch.inference_mode()
def encode_in_batches(model, values: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Encode a complete split without requiring it to fit in one GPU batch."""

    return torch.cat(
        [
            model.encode(values[offset : offset + batch_size])
            for offset in range(0, len(values), batch_size)
        ]
    )


def predictor_windows(
    tokens,
    dataset,
    config: PriorConfig,
    *,
    target_tokens: int | None = None,
):
    """Extract honest history, target-token, and hindsight-command windows.

    ``target_tokens`` may exceed the predictor's one-token output when training
    a short autoregressive rollout.  Evaluation omits it and retains the frozen
    next-token likelihood contract.
    """

    target_tokens = config.future_tokens if target_tokens is None else target_tokens
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    command_horizon = COMMAND_FUTURE_FRAME - COMMAND_FRAME
    last_command_anchor = (config.clip_frames - 1 - command_horizon) // config.downsample
    last_target_anchor = tokens.shape[1] - target_tokens
    anchors = np.arange(
        config.history_tokens,
        min(last_command_anchor, last_target_anchor) + 1,
        dtype=np.int64,
    )
    if not len(anchors):
        raise ValueError("model history/future leaves no valid command window")
    history = torch.stack(
        [tokens[:, anchor - config.history_tokens : anchor] for anchor in anchors],
        dim=1,
    ).flatten(0, 1)
    future = torch.stack(
        [tokens[:, anchor : anchor + target_tokens] for anchor in anchors],
        dim=1,
    ).flatten(0, 1)
    raw_command = hindsight_commands(
        dataset.root_position,
        dataset.root_quaternion,
        anchors * config.downsample,
        command_horizon,
    ).reshape(-1, 3)
    return history, future, torch.from_numpy(raw_command).to(tokens.device), anchors
