"""Dataset-only loader for Demo F; this module never reads raw rodent files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)
from ..features import trajectory_features
from .contract import DEFAULT_ROOT, REPOSITORY_ID, SCHEMA_VERSION


@dataclass
class FetchMotionSet:
    features: np.ndarray
    command: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    session_index: np.ndarray
    source_start: np.ndarray
    source_speed_mps: np.ndarray
    source_path_speed_mps: np.ndarray
    sessions: tuple[str, ...]


def download_dataset(root: Path = DEFAULT_ROOT) -> Path:
    """Download the public release without importing any raw-data dependency."""

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPOSITORY_ID, repo_type="dataset", local_dir=root)
    return root


def load_manifest(root: Path = DEFAULT_ROOT, *, download: bool = False) -> dict:
    root = Path(root)
    path = root / "manifest.json"
    if not path.exists() and download:
        download_dataset(root)
    if not path.exists():
        raise FileNotFoundError(
            f"missing standalone Demo F dataset at {path}; run "
            "`python -m demo_f.dataset.build` or pass download=True after publication"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"dataset schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
        )
    if not manifest.get("complete_release", False):
        raise ValueError("Demo F canonical training requires a complete 38-session release")
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
    download: bool = False,
) -> FetchMotionSet:
    root = Path(root)
    manifest = load_manifest(root, download=download)
    rows = [row for row in manifest["sessions"] if row["split"] == split]
    if not rows:
        raise ValueError(f"release has no {split!r} rows")
    features, commands, roots, quaternions, starts, speeds, path_speeds, indices, sessions = (
        [], [], [], [], [], [], [], [], []
    )
    for session_index, row in enumerate(rows):
        with np.load(root / row["shard"]) as shard:
            feature = trajectory_features(
                shard["root_position"],
                shard["root_quaternion"],
                shard["joint_angles"],
                shard["feet_local"],
                shard["contacts"],
            )
            count = len(feature)
            features.append(feature)
            commands.append(shard["command"].astype(np.float32))
            roots.append(shard["root_position"].astype(np.float32))
            quaternions.append(shard["root_quaternion"].astype(np.float32))
            starts.append(shard["source_start"].astype(np.int32))
            speeds.append(shard["source_speed_mps"].astype(np.float32))
            path_speeds.append(shard["source_path_speed_mps"].astype(np.float32))
            indices.append(np.full(count, session_index, np.int16))
            sessions.append(row["session"])
    return FetchMotionSet(
        features=np.concatenate(features),
        command=np.concatenate(commands),
        root_position=np.concatenate(roots),
        root_quaternion=np.concatenate(quaternions),
        session_index=np.concatenate(indices),
        source_start=np.concatenate(starts),
        source_speed_mps=np.concatenate(speeds),
        source_path_speed_mps=np.concatenate(path_speeds),
        sessions=tuple(sessions),
    )
