import inspect
from types import SimpleNamespace

import numpy as np
import torch

from demo_f.config import PriorConfig
from demo_f.dataset import hindsight_commands, loader
from demo_f.dataset.contract import validate_split_contract
from demo_f.features import FEATURE_DIM, trajectory_features
from demo_f.windows import predictor_windows


def test_session_splits_are_disjoint():
    validate_split_contract()


def test_training_loader_has_no_raw_hdf5_dependency():
    source = inspect.getsource(loader)
    assert "h5py" not in source
    assert "demo_b" not in source
    assert "Aldarondo2024" not in source


def test_fetch_feature_contract_is_60d():
    frames = 8
    root = np.zeros((2, frames, 3), np.float32)
    root[..., 2] = 1.375
    quaternion = np.zeros((2, frames, 4), np.float32)
    quaternion[..., 0] = 1
    angles = np.zeros((2, frames, 10), np.float32)
    feet = np.zeros((2, frames, 4, 3), np.float32)
    contacts = np.zeros((2, frames, 4), np.uint8)
    features = trajectory_features(root, quaternion, angles, feet, contacts)
    assert features.shape == (2, frames, FEATURE_DIM)
    assert np.isfinite(features).all()


def test_hindsight_commands_use_each_requested_anchor():
    root = np.zeros((1, 8, 3), np.float32)
    root[0, :, 0] = np.arange(8) * 0.1
    quaternion = np.zeros((1, 8, 4), np.float32)
    quaternion[..., 0] = 1
    command = hindsight_commands(root, quaternion, np.asarray([1, 3]), 3)
    np.testing.assert_allclose(command[0, :, 0], 0.3, atol=1e-6)
    np.testing.assert_allclose(command[0, :, 1:], 0.0, atol=1e-6)


def test_predictor_windows_align_next_tokens_and_hindsight_commands():
    config = PriorConfig()
    tokens = torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
    root = np.zeros((2, 64, 3), np.float32)
    root[:, :, 0] = np.arange(64) * 0.1
    quaternion = np.zeros((2, 64, 4), np.float32)
    quaternion[..., 0] = 1
    dataset = SimpleNamespace(root_position=root, root_quaternion=quaternion)

    history, future, command, anchors = predictor_windows(tokens, dataset, config)

    np.testing.assert_array_equal(anchors, [4, 5, 6, 7, 8])
    assert history.shape == (10, 4, 3)
    assert future.shape == (10, 1, 3)
    torch.testing.assert_close(history[0], tokens[0, :4])
    torch.testing.assert_close(future[4], tokens[0, 8:9])
    torch.testing.assert_close(history[5], tokens[1, :4])
    np.testing.assert_allclose(command[:, 0].numpy(), 3.1, atol=1e-5)
    np.testing.assert_allclose(command[:, 1:].numpy(), 0.0, atol=1e-6)
