import torch

from demo_f.config import JOINT_LIMIT
from demo_f.features import FEATURE_DIM, SL
from demo_f.train import joint_limit_loss
from demo_f.windows import predictor_windows


def test_joint_limit_loss_is_zero_inside_margin():
    features = torch.zeros((2, 4, FEATURE_DIM))
    mean = torch.zeros(FEATURE_DIM)
    std = torch.ones(FEATURE_DIM)
    features[..., slice(*SL["joint_angles"])] = 0.5 * JOINT_LIMIT

    assert joint_limit_loss(features, mean, std).item() == 0.0


def test_joint_limit_loss_penalizes_physical_denormalized_angles():
    features = torch.zeros((1, 1, FEATURE_DIM))
    mean = torch.zeros(FEATURE_DIM)
    std = torch.ones(FEATURE_DIM)
    joint_slice = slice(*SL["joint_angles"])
    mean[joint_slice] = 0.2
    std[joint_slice] = 2.0
    features[..., joint_slice] = JOINT_LIMIT

    expected_excess = 2.0 * JOINT_LIMIT + 0.2 - 0.95 * JOINT_LIMIT
    expected = expected_excess**2
    assert torch.isclose(joint_limit_loss(features, mean, std), torch.tensor(expected))


class _Dataset:
    root_position = torch.zeros((2, 64, 3)).numpy()
    root_quaternion = torch.zeros((2, 64, 4)).numpy()
    root_quaternion[..., 0] = 1.0


def test_predictor_windows_can_return_longer_training_rollout():
    from demo_f.config import PriorConfig

    tokens = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    history, future, command, anchors = predictor_windows(
        tokens, _Dataset(), PriorConfig(), target_tokens=4
    )

    assert anchors.tolist() == [4, 5, 6, 7, 8]
    assert history.shape == (10, 4, 3)
    assert future.shape == (10, 4, 3)
    assert command.shape == (10, 3)
    assert torch.equal(future[0], tokens[0, 4:8])
