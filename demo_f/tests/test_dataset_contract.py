import inspect

import numpy as np

from demo_f.dataset import loader
from demo_f.dataset.contract import validate_split_contract
from demo_f.features import FEATURE_DIM, trajectory_features


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
