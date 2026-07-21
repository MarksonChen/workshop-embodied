"""Fail-closed Demo H PPO checkpoint I/O, including legacy checkpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from demo_f.artifacts import sha256

from .config import (
    ACTION_DIM,
    LEGACY_OBSERVATION_CONTRACT_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    OBS_DIM,
)


PPO_CHECKPOINT_SCHEMA = "demo-h-ppo-checkpoint-v2"


def save_policy_checkpoint(path: Path, params, *, arm: str, prior_path: Path) -> dict:
    """Save parameters with the contract needed to load them safely."""

    if arm not in {"h1", "h2"} or prior_path is None:
        raise ValueError("Demo H checkpoints require arm h1/h2 and a frozen prior")
    prior_hash = sha256(prior_path)
    metadata = {
        "schema": PPO_CHECKPOINT_SCHEMA,
        "arm": arm,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "prior_sha256": prior_hash,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
    }
    envelope = {**metadata, "params": params}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(envelope, stream)
    return metadata


def _legacy_metadata(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(
            f"legacy checkpoint {path} has no sidecar; its arm and prior cannot be verified"
        )
    report = json.loads(sidecar.read_text())
    prior = report.get("prior")
    return {
        "schema": report.get("schema"),
        "arm": report.get("arm"),
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "prior_sha256": None if prior is None else prior.get("sha256"),
        "observation_contract_version": LEGACY_OBSERVATION_CONTRACT_VERSION,
    }


def _validate_metadata(metadata, expected_arm, expected_prior_sha256) -> None:
    if metadata["arm"] != expected_arm:
        raise ValueError(
            f"checkpoint arm {metadata['arm']!r} does not match requested {expected_arm!r}"
        )
    if metadata["prior_sha256"] != expected_prior_sha256:
        raise ValueError(
            "checkpoint/prior mismatch: "
            f"trained with {metadata['prior_sha256']}, loaded {expected_prior_sha256}"
        )
    if metadata["observation_dim"] != OBS_DIM:
        raise ValueError(
            f"checkpoint observation dimension {metadata['observation_dim']} "
            f"!= {OBS_DIM}"
        )
    if metadata["action_dim"] != ACTION_DIM:
        raise ValueError(
            f"checkpoint action dimension {metadata['action_dim']} != {ACTION_DIM}"
        )
    if metadata.get("observation_contract_version") != OBSERVATION_CONTRACT_VERSION:
        raise ValueError(
            "checkpoint observation contract "
            f"{metadata.get('observation_contract_version')!r} != "
            f"{OBSERVATION_CONTRACT_VERSION!r}"
        )


def load_policy_checkpoint(
    path: Path,
    *,
    expected_arm: str,
    expected_prior_sha256: str,
):
    """Load a new envelope or verify the JSON sidecar of a legacy tuple."""

    if expected_arm not in {"h1", "h2"}:
        raise ValueError(f"unsupported Demo H arm {expected_arm!r}")
    path = Path(path)
    sidecar_metadata = _legacy_metadata(path) if path.with_suffix(".json").is_file() else None
    if sidecar_metadata is not None:
        # Reject a wrong prior before unpickling version-specific JAX arrays.
        _validate_metadata(sidecar_metadata, expected_arm, expected_prior_sha256)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, dict) and payload.get("schema") == PPO_CHECKPOINT_SCHEMA:
        metadata = {name: value for name, value in payload.items() if name != "params"}
        params = payload["params"]
        _validate_metadata(metadata, expected_arm, expected_prior_sha256)
    else:
        if sidecar_metadata is None:
            raise ValueError(
                f"legacy checkpoint {path} has no sidecar; compatibility is unknown"
            )
        metadata, params = sidecar_metadata, payload
    return params, metadata
