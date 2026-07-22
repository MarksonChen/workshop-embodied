from pathlib import Path

import numpy as np
import torch

from workshop.part2.config import PriorConfig as MotionConfig
from workshop.part2.core.artifacts import sha256
from workshop.part2.core.model import predictor_windows
from workshop.part2.data import load_manifest as load_motion_manifest
from workshop.part2.data import load_split as load_motion_split
from workshop.part3.config import COMMAND_SLICE, OBS_DIM
from workshop.part3.core.artifacts import load_policy_checkpoint, save_policy_checkpoint
from workshop.part3.data import load_manifest as load_action_manifest
from workshop.part3.data import load_split as load_action_split


def test_prepared_data_contracts():
    motion_manifest = load_motion_manifest()
    action_manifest = load_action_manifest()
    motion = load_motion_split("validation")
    action = load_action_split("validation")
    assert motion_manifest["variant"] == "dynamic-similarity-v2"
    assert (
        action_manifest["variant"] == "exact-fetch-feedback-projection-retime-1p75-v1"
    )
    assert motion.features.shape == (1166, 64, 60)
    assert action.features.shape == (278, 64, 60)
    assert action.normalized_control.shape == (278, 63, 10)


def test_motion_windows_are_causal():
    config = MotionConfig()
    data = load_motion_split("validation")
    tokens = torch.zeros((len(data.features), 16, config.latent_dim))
    history, future, command, anchors = predictor_windows(tokens, data, config)
    assert anchors.tolist() == [4, 5, 6, 7, 8]
    assert history.shape[1:] == (4, 16)
    assert future.shape[1:] == (1, 16)
    assert command.shape[-1] == 3


def test_part3_observation_layout():
    assert COMMAND_SLICE.stop == OBS_DIM == 1094


def test_policy_checkpoint_contract(tmp_path: Path):
    prior = tmp_path / "prior.npz"
    prior.write_bytes(b"prior")
    checkpoint = tmp_path / "policy.pkl"
    save_policy_checkpoint(
        checkpoint,
        (np.asarray([1.0]),),
        beta=0.1,
        prior_path=prior,
        training={"seed": 0},
    )
    params, metadata = load_policy_checkpoint(
        checkpoint,
        expected_prior_sha256=sha256(prior),
    )
    assert params[0].tolist() == [1.0]
    assert metadata["beta"] == 0.1
