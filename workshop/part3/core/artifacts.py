from __future__ import annotations

import pickle
from pathlib import Path

from workshop.part2.core.artifacts import sha256

from ..config import ACTION_DIM, OBSERVATION_CONTRACT_VERSION, OBS_DIM


CHECKPOINT_SCHEMA = "workshop-part3-policy-v1"


def save_policy_checkpoint(
    path: Path,
    params,
    *,
    beta: float,
    prior_path: Path,
    training: dict,
) -> dict:
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "beta": float(beta),
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "prior_sha256": sha256(prior_path),
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "training": dict(training),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump({**metadata, "params": params}, stream)
    return metadata


def load_policy_checkpoint(path: Path, *, expected_prior_sha256: str):
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported policy checkpoint {payload.get('schema')!r}")
    expected = {
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "prior_sha256": expected_prior_sha256,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
    }
    mismatch = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatch:
        raise ValueError(f"policy contract mismatch: {mismatch}")
    metadata = {name: value for name, value in payload.items() if name != "params"}
    return payload["params"], metadata
