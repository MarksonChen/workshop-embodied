"""Memory-efficient loader for the versioned Demo H release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import DEFAULT_ROOT, SCHEMA_VERSION


@dataclass
class BodyActionSet:
    features: np.ndarray
    controls: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    command: np.ndarray
    session_index: np.ndarray
    parent_clip_id: np.ndarray
    source_speed_mps: np.ndarray
    sessions: tuple[str, ...]


def load_manifest(root: Path = DEFAULT_ROOT) -> dict:
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
    return manifest


def load_split(split: str, root: Path = DEFAULT_ROOT) -> BodyActionSet:
    root = Path(root)
    manifest = load_manifest(root)
    rows = [row for row in manifest["sessions"] if row["split"] == split]
    if not rows:
        raise ValueError(f"release has no {split!r} split")
    values = {
        name: []
        for name in (
            "features",
            "controls",
            "root_position",
            "root_quaternion",
            "command",
            "session_index",
            "parent_clip_id",
            "source_speed_mps",
        )
    }
    sessions = []
    for session_index, row in enumerate(rows):
        with np.load(root / row["shard"]) as shard:
            count = len(shard["parent_clip_id"])
            values["features"].append(shard["realized_features"].astype(np.float32))
            values["controls"].append(shard["normalized_control"].astype(np.float32))
            values["root_position"].append(
                shard["realized_root_position"].astype(np.float32)
            )
            values["root_quaternion"].append(
                shard["realized_root_quaternion"].astype(np.float32)
            )
            values["command"].append(shard["command"].astype(np.float32))
            values["session_index"].append(
                np.full(count, session_index, dtype=np.int16)
            )
            values["parent_clip_id"].append(shard["parent_clip_id"].astype(np.int32))
            values["source_speed_mps"].append(
                shard["source_speed_mps"].astype(np.float32)
            )
            sessions.append(row["session"])
    return BodyActionSet(
        **{name: np.concatenate(parts) for name, parts in values.items()},
        sessions=tuple(sessions),
    )
