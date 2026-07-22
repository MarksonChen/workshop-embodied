from __future__ import annotations

import json

import pytest

from demo_h.artifacts import save_policy_checkpoint
from demo_j.analysis.contracts import (
    SWEEP_ID,
    TRAINING_ENVS,
    TRAINING_SPEED_RANGE,
    TRAINING_TIMESTEPS,
    checkpoint_contract,
)


def test_locked_sweep_rejects_sidecar_checkpoint_mismatch(tmp_path) -> None:
    prior = tmp_path / "prior.npz"
    prior.write_bytes(b"prior")
    checkpoint = tmp_path / "policy.pkl"
    embedded = {
        "sweep_id": SWEEP_ID,
        "beta": 0.1,
        "seed": 2,
        "num_timesteps": TRAINING_TIMESTEPS,
        "num_envs": TRAINING_ENVS,
        "task_speed_training_range": list(TRAINING_SPEED_RANGE),
    }
    metadata = save_policy_checkpoint(
        checkpoint,
        {"weights": [1, 2]},
        arm="h2",
        prior_path=prior,
        run_metadata=embedded,
    )
    sidecar = {
        "arm": "h2",
        "beta": 0.1,
        "seed": 1,
        "sweep_id": SWEEP_ID,
        "num_timesteps": TRAINING_TIMESTEPS,
        "num_envs": TRAINING_ENVS,
        "task_speed_training_range": list(TRAINING_SPEED_RANGE),
        "prior": {"sha256": metadata["prior_sha256"]},
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="training envelope mismatch"):
        checkpoint_contract(checkpoint, require_sweep=True)
