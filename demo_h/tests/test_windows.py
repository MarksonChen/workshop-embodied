from types import SimpleNamespace

import numpy as np
import torch

from demo_h.config import PriorConfig
from demo_h.windows import state_action_windows


def test_token_to_action_alignment_has_no_future_leakage():
    config = PriorConfig(latent_dim=2)
    tokens = torch.arange(16, dtype=torch.float32).view(1, 16, 1).expand(-1, -1, 2)
    features = torch.arange(64, dtype=torch.float32).view(1, 64, 1).expand(-1, -1, 60)
    controls = torch.arange(63, dtype=torch.float32).view(1, 63, 1).expand(-1, -1, 10)
    root = np.zeros((1, 64, 3), np.float32)
    root[0, :, 0] = np.arange(64)
    quaternion = np.zeros((1, 64, 4), np.float32)
    quaternion[..., 0] = 1.0
    data = SimpleNamespace(root_position=root, root_quaternion=quaternion)

    windows = state_action_windows(tokens, features, controls, data, config)

    assert windows.anchors.tolist() == list(range(4, 13))
    assert windows.action_anchors.tolist() == list(range(4, 16))
    # anchor 4: history ends at token 3/frame 15, and controls 15..18
    # produce the four frames summarized by target token 4.
    assert windows.history[0, -1, 0].item() == 3
    assert windows.future[0, 0, 0].item() == 4
    assert windows.current_feature[:4, 0].tolist() == [15, 16, 17, 18]
    assert windows.target_control[:4, 0].tolist() == [15, 16, 17, 18]
    assert windows.previous_control[:4, 0].tolist() == [14, 15, 16, 17]
    assert windows.command[0, 0].item() == 31
    assert windows.command[-1, 0].item() == 31
    assert windows.action_anchor_command[-1, 0].item() == 31
    assert windows.target_control[-4:, 0].tolist() == [59, 60, 61, 62]
