import pickle

import pytest

from demo_h.artifacts import load_policy_checkpoint, save_policy_checkpoint
from demo_h.config import OBSERVATION_CONTRACT_VERSION, OBS_DIM, PriorConfig


def test_checkpoint_envelope_binds_arm_prior_and_layout(tmp_path):
    prior = tmp_path / "prior.npz"
    prior.write_bytes(b"prior")
    checkpoint = tmp_path / "policy.pkl"
    metadata = save_policy_checkpoint(
        checkpoint, {"weights": [1, 2]}, arm="h2", prior_path=prior
    )
    params, loaded = load_policy_checkpoint(
        checkpoint,
        expected_arm="h2",
        expected_prior_sha256=metadata["prior_sha256"],
    )
    assert params == {"weights": [1, 2]}
    assert loaded["observation_dim"] == OBS_DIM
    assert loaded["observation_contract_version"] == OBSERVATION_CONTRACT_VERSION
    with pytest.raises(ValueError, match="prior mismatch"):
        load_policy_checkpoint(
            checkpoint, expected_arm="h2", expected_prior_sha256="wrong"
        )
    payload = pickle.loads(checkpoint.read_bytes())
    payload["observation_contract_version"] = "wrong"
    checkpoint.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="observation contract"):
        load_policy_checkpoint(
            checkpoint,
            expected_arm="h2",
            expected_prior_sha256=metadata["prior_sha256"],
        )


def test_checkpoint_round_trips_training_metadata(tmp_path):
    prior = tmp_path / "prior.npz"
    prior.write_bytes(b"prior")
    checkpoint = tmp_path / "policy.pkl"
    training = {
        "sweep_id": "comparison-v1",
        "beta": 0.1,
        "seed": 2,
        "num_timesteps": 30_000_000,
    }
    metadata = save_policy_checkpoint(
        checkpoint,
        {"weights": [1, 2]},
        arm="h2",
        prior_path=prior,
        run_metadata=training,
    )
    _, loaded = load_policy_checkpoint(
        checkpoint,
        expected_arm="h2",
        expected_prior_sha256=metadata["prior_sha256"],
    )
    assert loaded["training"] == training


def test_online_prior_layout_fails_closed():
    PriorConfig().validate_online_contract()
    with pytest.raises(ValueError, match="online contract"):
        PriorConfig(latent_dim=8).validate_online_contract()
