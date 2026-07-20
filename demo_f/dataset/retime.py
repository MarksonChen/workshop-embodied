"""Build a dynamically scaled Demo F release from the accepted kinematic one.

The original retargeter scales rat distances to Fetch size but deliberately
preserves the source 50 Hz clock.  That is useful for inspecting kinematic
correspondence, but it does not preserve gravity-relative locomotion dynamics.
This derived release applies the Froude-similar time scale ``sqrt(L_fetch /
L_source)`` while keeping the output sampled at 50 Hz.

No discontinuous source clips are joined.  Each 64-frame parent clip is
interpolated independently and yields four non-overlapping 64-frame target-time
crops.  Session-level train/validation/test splits remain unchanged.

Run from the repository root:

    uv run --extra workshop python -m demo_f.dataset.retime
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..config import FETCH_TRUNK_LENGTH, FPS, JOINT_LIMIT
from ..kinematics import fetch_feet
from ..retarget import _world_feet
from .contract import (
    CLIP_FRAMES,
    COMMAND_FRAME,
    COMMAND_FUTURE_FRAME,
    DEFAULT_ROOT,
    DYNAMIC_ROOT,
)
from .loader import hindsight_commands, load_manifest


# Median source trunk length across the four frozen inspection clips is about
# 9.35 cm.  The accepted retargeter maps that trunk to Fetch's 2.0-unit trunk.
# Freeze a rounded global scale rather than deriving it from output/test data.
REFERENCE_SOURCE_TRUNK_M = 0.09355
LENGTH_SCALE = FETCH_TRUNK_LENGTH / REFERENCE_SOURCE_TRUNK_M
TIME_SCALE = math.sqrt(LENGTH_SCALE)
VELOCITY_SCALE = LENGTH_SCALE / TIME_SCALE
SOURCE_SPEED_MPS = 0.20
TARGET_SPEED_FETCH = SOURCE_SPEED_MPS * VELOCITY_SCALE
COMMAND_HORIZON_SECONDS = (COMMAND_FUTURE_FRAME - COMMAND_FRAME) / FPS
TARGET_COMMAND_X = TARGET_SPEED_FETCH * COMMAND_HORIZON_SECONDS
CROPS_PER_PARENT = 4

# The parent release's 5% saturation threshold is appropriate for inspecting a
# kinematic retarget, but not for fitting a generative prior whose rollout gate
# is 1%.  Keep a small physical margin in the *training data* instead of asking
# the network to learn a stricter constraint than its targets obey.
DYNAMIC_JOINT_LIMIT_FRACTION_MAX = 0.01
DYNAMIC_MINIMUM_GLOBAL_PASS_RATE = 0.80

# Every derived crop has exactly the same shape. Jitting this once avoids
# dispatching the dozens of primitive operations in ``fetch_feet`` separately
# for each of the ~12k crops in the complete release.
_fetch_feet_sequence = jax.jit(fetch_feet)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retimed_length(frames: int = CLIP_FRAMES) -> int:
    """Number of target-time samples spanning the complete parent clip."""

    return int(math.floor((frames - 1) * TIME_SCALE)) + 1


def crop_starts(frames: int = CLIP_FRAMES) -> np.ndarray:
    """Four disjoint crops spread across one independently retimed clip."""

    available = retimed_length(frames) - CLIP_FRAMES
    starts = np.rint(np.linspace(0, available, CROPS_PER_PARENT)).astype(np.int32)
    if np.any(np.diff(starts) < CLIP_FRAMES):
        raise AssertionError("dynamic crops unexpectedly overlap")
    return starts


def _linear(values: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    lower = np.floor(source_time).astype(np.int32)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (source_time - lower).astype(np.float32)
    shape = (len(weight),) + (1,) * (values.ndim - 1)
    return values[lower] * (1.0 - weight.reshape(shape)) + values[upper] * weight.reshape(shape)


def _yaw(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion, np.float32), -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quaternion(yaw: np.ndarray) -> np.ndarray:
    output = np.zeros((len(yaw), 4), np.float32)
    output[:, 0] = np.cos(yaw / 2)
    output[:, 3] = np.sin(yaw / 2)
    return output


def retime_clip(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_angles: np.ndarray,
    contacts: np.ndarray,
    target_start: int,
) -> dict[str, np.ndarray]:
    """Interpolate one physical-time crop without joining parent clips."""

    target_frame = target_start + np.arange(CLIP_FRAMES, dtype=np.float64)
    source_time = target_frame / TIME_SCALE
    if source_time[-1] > len(root_position) - 1 + 1e-9:
        raise ValueError("retimed crop exceeds its parent clip")
    root = _linear(root_position, source_time).astype(np.float32)
    yaw = _linear(_yaw(root_quaternion)[:, None], source_time)[:, 0]
    quaternion = _yaw_quaternion(yaw)
    angles = _linear(joint_angles, source_time).astype(np.float32)
    angles = np.clip(angles, -JOINT_LIMIT, JOINT_LIMIT)
    contact_index = np.minimum(
        np.rint(source_time).astype(np.int32), len(contacts) - 1
    )
    contact = np.asarray(contacts[contact_index], np.uint8)
    feet = np.asarray(_fetch_feet_sequence(jnp.asarray(angles)), np.float32)
    return {
        "root_position": root,
        "root_quaternion": quaternion,
        "joint_angles": angles,
        "feet_local": feet,
        "contacts": contact,
        "source_time": source_time.astype(np.float32),
    }


def _quality(root, quaternion, feet, contacts, angles) -> tuple[float, float, float]:
    yaw = _yaw(quaternion)
    world = _world_feet(root, yaw, feet)
    speed = np.zeros(world.shape[:2], np.float32)
    speed[1:] = np.linalg.norm(np.diff(world, axis=0), axis=-1) * FPS
    stance = contacts[1:].astype(bool) & contacts[:-1].astype(bool)
    stance_speed = speed[1:][stance]
    return (
        float(stance_speed.mean()) if len(stance_speed) else math.nan,
        float(world[..., 2].min()),
        float((np.abs(angles) > 0.99 * JOINT_LIMIT).mean()),
    )


def retime_shard(parent: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive four physical-time crops for every parent trajectory."""

    rows = {name: [] for name in parent}
    starts = crop_starts()
    for parent_index in range(len(parent["joint_angles"])):
        for target_start in starts:
            clip = retime_clip(
                parent["root_position"][parent_index],
                parent["root_quaternion"][parent_index],
                parent["joint_angles"][parent_index],
                parent["contacts"][parent_index],
                int(target_start),
            )
            command = hindsight_commands(
                clip["root_position"][None],
                clip["root_quaternion"][None],
                np.asarray((COMMAND_FRAME,)),
                COMMAND_FUTURE_FRAME - COMMAND_FRAME,
            )[0, 0]
            stance_speed, minimum_height, limit_fraction = _quality(
                clip["root_position"],
                clip["root_quaternion"],
                clip["feet_local"],
                clip["contacts"],
                clip["joint_angles"],
            )
            source_offset = int(round(float(clip["source_time"][0])))
            rows["joint_angles"].append(clip["joint_angles"])
            rows["root_position"].append(clip["root_position"])
            rows["root_quaternion"].append(clip["root_quaternion"])
            rows["feet_local"].append(clip["feet_local"])
            rows["contacts"].append(clip["contacts"])
            rows["command"].append(command)
            rows["source_start"].append(
                int(parent["source_start"][parent_index]) + source_offset
            )
            rows["source_speed_mps"].append(parent["source_speed_mps"][parent_index])
            rows["source_path_speed_mps"].append(
                parent["source_path_speed_mps"][parent_index]
            )
            # The spatial IK target is unchanged; this remains provenance for
            # the parent pose rather than a newly optimized quantity.
            rows["ik_foot_rmse"].append(parent["ik_foot_rmse"][parent_index])
            rows["contact_speed_mean"].append(stance_speed)
            rows["minimum_foot_height"].append(minimum_height)
            rows["joint_limit_fraction"].append(limit_fraction)
    dtypes = {
        "contacts": np.uint8,
        "source_start": np.int32,
    }
    return {
        name: np.asarray(values, dtype=dtypes.get(name, np.float32))
        for name, values in rows.items()
    }


def _read_parent(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name] for name in source.files}


def build(parent_root: Path, output_root: Path) -> dict:
    parent_root, output_root = Path(parent_root), Path(output_root)
    parent_manifest_path = parent_root / "manifest.json"
    parent_manifest = load_manifest(parent_root)
    sessions, counts = [], {split: 0 for split in parent_manifest["splits"]}
    candidate_total = 0
    gates = {
        **parent_manifest["quality_gates"],
        "joint_limit_fraction_max": DYNAMIC_JOINT_LIMIT_FRACTION_MAX,
    }
    for parent_row in parent_manifest["sessions"]:
        parent_shard = parent_root / parent_row["shard"]
        arrays = retime_shard(_read_parent(parent_shard))
        candidate_count = len(arrays["joint_angles"])
        keep = (
            np.isfinite(arrays["contact_speed_mean"])
            & (arrays["ik_foot_rmse"] <= gates["ik_foot_rmse_max"])
            & (arrays["minimum_foot_height"] >= gates["minimum_foot_height_min"])
            & (
                arrays["joint_limit_fraction"]
                <= gates["joint_limit_fraction_max"]
            )
        )
        rejected_starts = arrays["source_start"][~keep].astype(int).tolist()
        arrays = {name: values[keep] for name, values in arrays.items()}
        shard = output_root / parent_row["shard"]
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, **arrays)
        count = len(arrays["joint_angles"])
        candidate_total += candidate_count
        split = parent_row["split"]
        counts[split] += count
        sessions.append(
            {
                **parent_row,
                "source_file": str(parent_shard.relative_to(parent_root)),
                "source_sha256": parent_row["shard_sha256"],
                "strict_blocks": candidate_count,
                "released_clips": count,
                "rejected_clips": candidate_count - count,
                "pass_rate": count / candidate_count,
                "mean_source_speed_mps": float(arrays["source_speed_mps"].mean()),
                "mean_source_path_speed_mps": float(
                    arrays["source_path_speed_mps"].mean()
                ),
                "mean_ik_foot_rmse": float(arrays["ik_foot_rmse"].mean()),
                "rejected_source_starts": rejected_starts,
                "shard_bytes": shard.stat().st_size,
                "shard_sha256": sha256(shard),
                "build_seconds": None,
            }
        )

    dynamic = {
        "method": "Froude-similar temporal dilation",
        "reference_source_trunk_m": REFERENCE_SOURCE_TRUNK_M,
        "fetch_trunk_length": FETCH_TRUNK_LENGTH,
        "length_scale": LENGTH_SCALE,
        "time_scale": TIME_SCALE,
        "velocity_scale": VELOCITY_SCALE,
        "source_fps": FPS,
        "target_fps": FPS,
        "retimed_parent_frames": retimed_length(),
        "crops_per_parent": CROPS_PER_PARENT,
        "crop_starts_target_frames": crop_starts().tolist(),
        "interpolation": {
            "root_and_joints": "linear",
            "yaw": "unwrapped-linear",
            "contacts": "nearest",
            "feet": "recomputed Fetch forward kinematics",
        },
        "reference_source_speed_mps": SOURCE_SPEED_MPS,
        "recommended_fetch_speed": TARGET_SPEED_FETCH,
        "command_horizon_seconds": COMMAND_HORIZON_SECONDS,
        "recommended_command_x": TARGET_COMMAND_X,
    }
    manifest = {
        **parent_manifest,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "dynamic-similarity-v2",
        "counts": counts,
        "sessions": sessions,
        "global_pass_rate": sum(counts.values()) / candidate_total,
        "minimum_global_pass_rate": DYNAMIC_MINIMUM_GLOBAL_PASS_RATE,
        "quality_gates": gates,
        "dynamic_scaling": dynamic,
        "derivation": {
            "parent_manifest_sha256": sha256(parent_manifest_path),
            "parent_repository_id": parent_manifest["repository_id"],
            "script": "demo_f/dataset/retime.py",
            "script_sha256": sha256(Path(__file__)),
            "continuity": "each output crop comes from exactly one contiguous parent clip",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "README.md").write_text(
        "# Dynamically scaled Demo F release\n\n"
        "This local derived release time-dilates every accepted retargeted clip "
        f"by {TIME_SCALE:.4f}x while retaining 50 Hz output. It preserves session "
        "splits and never joins parent clips. See `manifest.json` for complete "
        "provenance and scaling constants.\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DYNAMIC_ROOT)
    args = parser.parse_args()
    manifest = build(args.parent_root, args.output_root)
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "counts": manifest["counts"],
                "dynamic_scaling": manifest["dynamic_scaling"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
