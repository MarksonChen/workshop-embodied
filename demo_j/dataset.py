"""Reference states and exact source-time provenance for Demo J."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from demo_f.config import FPS
from demo_f.dataset.contract import DEFAULT_ROOT as ORIGINAL_DEMO_F_ROOT
from demo_f.dataset.retime import crop_starts
from demo_h.dataset.contract import DEFAULT_ROOT, DATASET_VARIANT, PARENT_ROOT

from demo_j.fetch_mjx import host_model, joint_qpos_addresses, joint_qvel_addresses


TIME_SCALE = 1.75
TARGET_CROP_START = int(crop_starts(time_scale=TIME_SCALE, crops_per_parent=1)[0])
ROUNDED_SOURCE_OFFSET = int(round(TARGET_CROP_START / TIME_SCALE))


@dataclass(frozen=True)
class ReferenceSet:
    """Dense arrays small enough to place directly on one accelerator."""

    qpos: np.ndarray
    qvel: np.ndarray
    features: np.ndarray
    contacts: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    joint_angles: np.ndarray
    command: np.ndarray
    teacher_action: np.ndarray
    session_index: np.ndarray
    parent_clip_id: np.ndarray
    source_start: np.ndarray
    raw_source_start: np.ndarray
    source_frame: np.ndarray
    sessions: tuple[str, ...]
    split: str
    manifest_sha256: str

    @property
    def clips(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def frames(self) -> int:
        return int(self.qpos.shape[1])


def take_references(reference: ReferenceSet, indices: np.ndarray) -> ReferenceSet:
    """Return a clip subset while preserving immutable provenance metadata."""

    indices = np.asarray(indices, np.int32)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty vector")
    if np.any(indices < 0) or np.any(indices >= reference.clips):
        raise IndexError(indices)
    updates = {}
    for field in fields(reference):
        value = getattr(reference, field.name)
        if isinstance(value, np.ndarray) and value.shape[:1] == (reference.clips,):
            updates[field.name] = value[indices]
    return replace(reference, **updates)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_source_frames(source_start: np.ndarray, frames: int = 64) -> np.ndarray:
    """Map every retimed output frame back to its fractional raw-data frame."""

    source_start = np.asarray(source_start, np.int64)
    raw_parent_start = source_start - ROUNDED_SOURCE_OFFSET
    within_parent = (TARGET_CROP_START + np.arange(frames)) / TIME_SCALE
    return raw_parent_start[:, None] + within_parent[None]


def _verified_raw_source_starts(
    rows: list[dict],
    parent_ids: list[np.ndarray],
    stored_starts: list[np.ndarray],
) -> np.ndarray:
    """Resolve retimed clips to the immutable unretimed Demo F release.

    Demo F v1 stored only a rounded retimed ``source_start``.  We recover the
    exact unretimed clip start, but accept it only when both the selected
    retimed parent row and the original session shard independently confirm
    the mapping.  This turns an old lossy field into checked provenance rather
    than silently treating the rounded value as an exact neural timestamp.
    """

    original_manifest = json.loads(
        (ORIGINAL_DEMO_F_ROOT / "manifest.json").read_text()
    )
    original_rows = {
        (row["split"], row["session"]): row
        for row in original_manifest["sessions"]
    }
    resolved: list[np.ndarray] = []
    for row, ids, starts in zip(rows, parent_ids, stored_starts, strict=True):
        with np.load(PARENT_ROOT / row["parent_shard"]) as parent:
            if np.any(ids < 0) or np.any(ids >= len(parent["source_start"])):
                raise ValueError(f"parent clip index out of bounds for {row['session']}")
            selected = np.asarray(parent["source_start"][ids], np.int32)
        if not np.array_equal(selected, starts):
            raise ValueError(f"Demo H/retimed-parent mismatch for {row['session']}")

        original_row = original_rows[(row["split"], row["session"])]
        with np.load(ORIGINAL_DEMO_F_ROOT / original_row["shard"]) as original:
            original_starts = np.asarray(original["source_start"], np.int32)
        raw = starts - ROUNDED_SOURCE_OFFSET
        if not np.all(np.isin(raw, original_starts)):
            raise ValueError(f"unverified raw source start for {row['session']}")
        resolved.append(raw)
    return np.concatenate(resolved)


def _reference_qvel(
    roots: np.ndarray,
    quaternions: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Finite-difference targets in MuJoCo free-joint velocity coordinates."""

    clips, frames = roots.shape[:2]
    model = host_model()
    qvel = np.zeros((clips, frames, model.nv), np.float32)
    linear = np.zeros_like(roots)
    linear[:, 1:] = np.diff(roots, axis=1) * FPS
    linear[:, 0] = linear[:, 1]
    qvel[..., :3] = linear

    xyzw = np.concatenate((quaternions[..., 1:], quaternions[..., :1]), axis=-1)
    rotation_matrices = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix()
    rotation_matrices = rotation_matrices.reshape(clips, frames, 3, 3)
    relative_matrices = rotation_matrices[:, 1:] @ np.swapaxes(
        rotation_matrices[:, :-1], -1, -2
    )
    relative_world = Rotation.from_matrix(relative_matrices.reshape(-1, 3, 3))
    angular = np.zeros_like(roots)
    angular[:, 1:] = relative_world.as_rotvec().reshape(clips, frames - 1, 3) * FPS
    angular[:, 0] = angular[:, 1]
    qvel[..., 3:6] = angular

    rates = np.zeros_like(angles)
    rates[:, 1:] = np.diff(angles, axis=1) * FPS
    rates[:, 0] = rates[:, 1]
    qvel[..., joint_qvel_addresses()] = rates
    return qvel


def load_reference_set(split: str, root: Path = DEFAULT_ROOT) -> ReferenceSet:
    """Load one immutable session split from Demo H's projected Demo F motion."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("variant") != DATASET_VARIANT:
        raise ValueError(f"unexpected reference variant {manifest.get('variant')!r}")
    if not manifest.get("complete_release", False):
        raise ValueError("Demo J requires the complete projected release")
    rows = [row for row in manifest["sessions"] if row["split"] == split]
    if not rows:
        raise ValueError(f"no {split!r} sessions")

    names = (
        "realized_root_position",
        "realized_root_quaternion",
        "realized_joint_angles",
        "realized_features",
        "realized_contacts",
        "command",
        "normalized_control",
        "parent_clip_id",
        "source_start",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    indices: list[np.ndarray] = []
    parent_ids: list[np.ndarray] = []
    stored_starts: list[np.ndarray] = []
    sessions: list[str] = []
    for session_index, row in enumerate(rows):
        with np.load(root / row["shard"]) as shard:
            count = len(shard["parent_clip_id"])
            for name in names:
                parts[name].append(np.asarray(shard[name]))
            indices.append(np.full(count, session_index, np.int16))
            parent_ids.append(np.asarray(shard["parent_clip_id"], np.int32))
            stored_starts.append(np.asarray(shard["source_start"], np.int32))
        sessions.append(row["session"])
    values = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    roots = values["realized_root_position"].astype(np.float32)
    quaternions = values["realized_root_quaternion"].astype(np.float32)
    angles = values["realized_joint_angles"].astype(np.float32)

    model = host_model()
    qpos = np.broadcast_to(model.qpos0, roots.shape[:2] + (model.nq,)).copy()
    qpos[..., :3] = roots
    qpos[..., 3:7] = quaternions
    qpos[..., joint_qpos_addresses()] = angles
    qvel = _reference_qvel(roots, quaternions, angles)
    source_start = values["source_start"].astype(np.int32)
    raw_source_start = _verified_raw_source_starts(
        rows, parent_ids, stored_starts
    )
    expected_source_start = raw_source_start + ROUNDED_SOURCE_OFFSET
    if not np.array_equal(source_start, expected_source_start):
        raise ValueError("retimed source-start offset is inconsistent")
    return ReferenceSet(
        qpos=qpos.astype(np.float32),
        qvel=qvel,
        features=values["realized_features"].astype(np.float32),
        contacts=values["realized_contacts"].astype(np.uint8),
        root_position=roots,
        root_quaternion=quaternions,
        joint_angles=angles,
        command=values["command"].astype(np.float32),
        teacher_action=values["normalized_control"].astype(np.float32),
        session_index=np.concatenate(indices),
        parent_clip_id=values["parent_clip_id"].astype(np.int32),
        source_start=source_start,
        raw_source_start=raw_source_start,
        source_frame=(
            raw_source_start[:, None]
            + (TARGET_CROP_START + np.arange(roots.shape[1]))[None] / TIME_SCALE
        ),
        sessions=tuple(sessions),
        split=split,
        manifest_sha256=_sha256(manifest_path),
    )


def validate_source_alignment(reference: ReferenceSet) -> dict[str, object]:
    """Check exact clock mappings without loading or exposing neural values."""

    source = reference.source_frame
    if source.shape != (reference.clips, reference.frames):
        raise ValueError(source.shape)
    increments = np.diff(source, axis=1)
    if not np.allclose(increments, 1.0 / TIME_SCALE, atol=1e-6):
        raise ValueError("retimed source coordinates are not uniformly monotonic")
    raw_root = source[:, 0] - TARGET_CROP_START / TIME_SCALE
    if not np.allclose(raw_root, reference.raw_source_start, atol=1e-6):
        raise ValueError("source mapping does not recover the verified parent starts")
    return {
        "split": reference.split,
        "clips": reference.clips,
        "sessions": len(reference.sessions),
        "time_scale": TIME_SCALE,
        "target_crop_start": TARGET_CROP_START,
        "first_source_offset": TARGET_CROP_START / TIME_SCALE,
        "source_step": 1.0 / TIME_SCALE,
        "minimum_source_frame": float(source.min()),
        "maximum_source_frame": float(source.max()),
    }
