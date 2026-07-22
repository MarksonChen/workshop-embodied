"""Memory-efficient loader for the versioned Demo H release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demo_f.config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)

from .contract import DATASET_VARIANT, DEFAULT_ROOT, DTYPES, FIELDS, SCHEMA_VERSION


@dataclass
class BodyActionSet:
    features: np.ndarray
    normalized_control: np.ndarray
    requested_actuator_torque: np.ndarray
    valid_transition_mask: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    contacts: np.ndarray
    command: np.ndarray
    session_index: np.ndarray
    parent_clip_id: np.ndarray
    source_speed_mps: np.ndarray
    source_path_speed_mps: np.ndarray
    sessions: tuple[str, ...]

def load_manifest(
    root: Path = DEFAULT_ROOT,
    *,
    expected_variant: str = DATASET_VARIANT,
) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing Demo H dataset at {path}; run `python -m demo_h.dataset.project`"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"dataset schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    if not manifest.get("complete_release", False):
        raise ValueError("prior training requires a complete Demo H release")
    if manifest.get("variant") != expected_variant:
        raise ValueError(
            f"unexpected Demo H dataset variant {manifest.get('variant')!r}; "
            f"expected {expected_variant!r}"
        )
    if manifest.get("fields") != {name: list(shape) for name, shape in FIELDS.items()}:
        raise ValueError("Demo H manifest fields differ from the frozen contract")
    if manifest.get("dtypes") != DTYPES:
        raise ValueError("Demo H manifest dtypes differ from the frozen contract")
    feature_contract = manifest.get(
        "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
    )
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"dataset feature contract {feature_contract!r}; "
            f"expected {FEATURE_CONTRACT_VERSION!r}"
        )
    return manifest


def load_split(
    split: str,
    root: Path = DEFAULT_ROOT,
    *,
    expected_variant: str = DATASET_VARIANT,
) -> BodyActionSet:
    root = Path(root)
    manifest = load_manifest(root, expected_variant=expected_variant)
    rows = [row for row in manifest["sessions"] if row["split"] == split]
    if not rows:
        raise ValueError(f"release has no {split!r} split")
    values = {
        name: []
        for name in (
            "features",
            "normalized_control",
            "requested_actuator_torque",
            "valid_transition_mask",
            "root_position",
            "root_quaternion",
            "contacts",
            "command",
            "session_index",
            "parent_clip_id",
            "source_speed_mps",
            "source_path_speed_mps",
        )
    }
    sessions = []
    for session_index, row in enumerate(rows):
        with np.load(root / row["shard"]) as shard:
            count = len(shard["parent_clip_id"])
            values["features"].append(shard["realized_features"].astype(np.float32))
            values["normalized_control"].append(
                shard["normalized_control"].astype(np.float32)
            )
            values["requested_actuator_torque"].append(
                shard["requested_actuator_torque"].astype(np.float32)
            )
            values["valid_transition_mask"].append(
                shard["valid_transition_mask"].astype(bool)
            )
            values["root_position"].append(
                shard["realized_root_position"].astype(np.float32)
            )
            values["root_quaternion"].append(
                shard["realized_root_quaternion"].astype(np.float32)
            )
            values["contacts"].append(shard["realized_contacts"].astype(bool))
            values["command"].append(shard["command"].astype(np.float32))
            values["session_index"].append(
                np.full(count, session_index, dtype=np.int16)
            )
            values["parent_clip_id"].append(shard["parent_clip_id"].astype(np.int32))
            values["source_speed_mps"].append(
                shard["source_speed_mps"].astype(np.float32)
            )
            values["source_path_speed_mps"].append(
                shard["source_path_speed_mps"].astype(np.float32)
            )
            sessions.append(row["session"])
    return BodyActionSet(
        **{name: np.concatenate(parts) for name, parts in values.items()},
        sessions=tuple(sessions),
    )
