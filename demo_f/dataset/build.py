"""Build the standalone Hugging Face-ready retargeted Coltrane dataset.

This is the only Demo F path allowed to read raw Aldarondo HDF5. Training uses
``demo_f.dataset``, which consumes the resulting release exclusively.

    uv run --extra workshop python -m demo_f.dataset.build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from demo_b.strict_locomotion import (
    GAIT_PAIRS,
    MAX_NECK_DRIFT_MM,
    MAX_TURN_DEGREES,
    MIN_GAIT_COORDINATION,
    MIN_SPEED,
    strict_block_starts,
)

from ..config import ANIMAL, DATA_ROOT, JOINT_LIMIT, JOINT_NAMES, RetargetConfig
from .contract import (
    CLIP_FRAMES,
    COMMAND_FRAME,
    COMMAND_FUTURE_FRAME,
    DEFAULT_ROOT,
    FIELDS,
    FPS,
    REPOSITORY_ID,
    SCHEMA_VERSION,
    SESSION_TO_SPLIT,
    SPLIT_SESSIONS,
    UPSTREAM_DOI,
    UPSTREAM_LICENSE,
    validate_split_contract,
)
from ..kinematics import fetch_feet
from ..retarget import _world_feet, optimize_batch, prepare_source


MIN_GLOBAL_PASS_RATE = 0.75
QUALITY_GATES = {
    "ik_foot_rmse_max": 0.20,
    "minimum_foot_height_min": -0.20,
    "joint_limit_fraction_max": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    """Hash every source file that can change a released trajectory or split."""

    repository = Path(__file__).resolve().parents[2]
    paths = {
        "demo_b/strict_locomotion.py": repository / "demo_b" / "strict_locomotion.py",
        "demo_f/config.py": repository / "demo_f" / "config.py",
        "demo_f/dataset/build.py": Path(__file__).resolve(),
        "demo_f/dataset/contract.py": Path(__file__).with_name("contract.py"),
        "demo_f/kinematics.py": repository / "demo_f" / "kinematics.py",
        "demo_f/retarget.py": repository / "demo_f" / "retarget.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def aggregate_code_hash(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def git_state() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--", "demo_b/strict_locomotion.py", "demo_f"),
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "head": head.stdout.strip() or None,
        "working_tree_dirty": bool(status.stdout.strip()),
    }


def hindsight_command(root_position: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    delta = root_position[:, COMMAND_FUTURE_FRAME, :2] - root_position[:, COMMAND_FRAME, :2]
    heading = yaw[:, COMMAND_FRAME]
    cosine, sine = np.cos(-heading), np.sin(-heading)
    turn = (
        yaw[:, COMMAND_FUTURE_FRAME] - yaw[:, COMMAND_FRAME] + np.pi
    ) % (2 * np.pi) - np.pi
    return np.stack(
        (
            cosine * delta[:, 0] - sine * delta[:, 1],
            sine * delta[:, 0] + cosine * delta[:, 1],
            turn,
        ),
        axis=-1,
    ).astype(np.float32)


def batch_quality(sources: list[dict], angles: np.ndarray) -> dict[str, np.ndarray]:
    actual = np.asarray(fetch_feet(jnp.asarray(angles)), dtype=np.float32)
    target = np.stack([source["target_feet_local"] for source in sources])
    root = np.stack([source["root_position"] for source in sources])
    yaw = np.stack([source["yaw"] for source in sources])
    contact = np.stack([source["contacts"] for source in sources])
    error = actual - target
    world = np.stack(
        [_world_feet(root[index], yaw[index], actual[index]) for index in range(len(sources))]
    )
    speed = np.zeros(world.shape[:3], np.float32)
    speed[:, 1:] = np.linalg.norm(np.diff(world, axis=1), axis=-1) * FPS
    stance_pair = contact[:, 1:] & contact[:, :-1]
    contact_speed = np.full(len(sources), np.nan, np.float32)
    for index in range(len(sources)):
        values = speed[index, 1:][stance_pair[index]]
        if len(values):
            contact_speed[index] = values.mean()
    rmse = np.sqrt(np.mean(np.square(error), axis=(1, 2, 3))).astype(np.float32)
    minimum_height = world[..., 2].min(axis=(1, 2)).astype(np.float32)
    limit_fraction = np.mean(
        np.abs(angles) > 0.99 * JOINT_LIMIT, axis=(1, 2)
    ).astype(np.float32)
    quality_pass = (
        np.isfinite(rmse)
        & np.isfinite(minimum_height)
        & np.isfinite(contact_speed)
        & (rmse <= QUALITY_GATES["ik_foot_rmse_max"])
        & (minimum_height >= QUALITY_GATES["minimum_foot_height_min"])
        & (limit_fraction <= QUALITY_GATES["joint_limit_fraction_max"])
    )
    return {
        "feet_local": actual,
        "ik_foot_rmse": rmse,
        "contact_speed_mean": contact_speed,
        "minimum_foot_height": minimum_height,
        "joint_limit_fraction": limit_fraction,
        "quality_pass": quality_pass,
    }


def read_session_clips(path: Path, config: RetargetConfig) -> tuple[np.ndarray, list[dict], str]:
    """Read one raw session once and return strict starts plus semantic targets."""

    with h5py.File(path, "r") as source:
        qpos = source["/pose/qpos"][:].astype(np.float32)
        keypoint_data = source["/pose/keypoints"]
        names = [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in keypoint_data.attrs["names"]
        ]
        neck = keypoint_data[:, 2, names.index("SpineF")].astype(np.float64)
        starts = strict_block_starts(qpos, neck)
        clips = []
        for start in starts:
            stop = int(start) + CLIP_FRAMES
            keypoints = np.transpose(keypoint_data[int(start):stop], (0, 2, 1)) / 1_000.0
            clips.append(prepare_source(keypoints, names, qpos[int(start):stop, :7], config))
    return starts.astype(np.int32), clips, sha256(path)


def strict_starts_and_hash(path: Path) -> tuple[np.ndarray, str]:
    """Recover provenance for a completed shard without repeating IK."""

    with h5py.File(path, "r") as source:
        qpos = source["/pose/qpos"][:].astype(np.float32)
        keypoint_data = source["/pose/keypoints"]
        names = [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in keypoint_data.attrs["names"]
        ]
        neck = keypoint_data[:, 2, names.index("SpineF")].astype(np.float64)
    return strict_block_starts(qpos, neck).astype(np.int32), sha256(path)


def recover_session(session: str, output_root: Path) -> dict:
    """Reconstruct a session manifest row from an already validated shard."""

    split = SESSION_TO_SPLIT[session]
    source_path = DATA_ROOT / ANIMAL / f"{session}.h5"
    shard = output_root / "data" / split / f"{session}.npz"
    starts, source_digest = strict_starts_and_hash(source_path)
    with np.load(shard) as data:
        arrays = {key: data[key] for key in data.files}
    retained = arrays["source_start"]
    rejected = np.setdiff1d(starts, retained, assume_unique=True)
    return {
        "session": session,
        "split": split,
        "source_file": f"coltrane/{session}.h5",
        "source_sha256": source_digest,
        "strict_blocks": int(len(starts)),
        "released_clips": int(len(retained)),
        "rejected_clips": int(len(rejected)),
        "pass_rate": float(len(retained) / len(starts)),
        "mean_source_speed_mps": float(arrays["source_speed_mps"].mean()),
        "mean_source_path_speed_mps": float(arrays["source_path_speed_mps"].mean()),
        "mean_ik_foot_rmse": float(arrays["ik_foot_rmse"].mean()),
        "rejected_source_starts": rejected.astype(int).tolist(),
        "shard": str(shard.relative_to(output_root)),
        "shard_bytes": shard.stat().st_size,
        "shard_sha256": sha256(shard),
        "build_seconds": None,
    }


def build_session(
    session: str,
    output_root: Path,
    config: RetargetConfig,
    batch_size: int,
) -> dict:
    split = SESSION_TO_SPLIT[session]
    source_path = DATA_ROOT / ANIMAL / f"{session}.h5"
    if not source_path.exists():
        raise FileNotFoundError(f"missing {source_path}; set ALDARONDO_ROOT")
    started = time.perf_counter()
    starts, sources, source_digest = read_session_clips(source_path, config)
    if not sources:
        raise ValueError(f"{session}: no blocks pass the strict locomotion screen")

    angle_batches, quality_batches = [], []
    for begin in range(0, len(sources), batch_size):
        group = sources[begin:begin + batch_size]
        print(f"[{session}] IK {begin + 1}-{begin + len(group)}/{len(sources)}", flush=True)
        angles, _ = optimize_batch(
            np.stack([source["target_feet_local"] for source in group]),
            np.stack([source["contacts"] for source in group]),
            np.stack([source["root_position"] for source in group]),
            np.stack([source["yaw"] for source in group]),
            config,
        )
        angle_batches.append(angles)
        quality_batches.append(batch_quality(group, angles))

    angles = np.concatenate(angle_batches)
    quality = {
        key: np.concatenate([batch[key] for batch in quality_batches])
        for key in quality_batches[0]
    }
    keep = quality.pop("quality_pass")
    pass_rate = float(keep.mean())

    roots = np.stack([source["root_position"] for source in sources])
    quaternions = np.stack([source["root_quaternion"] for source in sources])
    yaws = np.stack([source["yaw"] for source in sources])
    contacts = np.stack([source["contacts"] for source in sources]).astype(np.uint8)
    source_speed = np.asarray([source["source_speed"] for source in sources], np.float32)
    source_path_speed = np.asarray(
        [source["source_path_speed"] for source in sources], np.float32
    )
    command = hindsight_command(roots, yaws)
    feet = quality.pop("feet_local")
    arrays = {
        "joint_angles": angles[keep],
        "root_position": roots[keep],
        "root_quaternion": quaternions[keep],
        "feet_local": feet[keep],
        "contacts": contacts[keep],
        "command": command[keep],
        "source_start": starts[keep],
        "source_speed_mps": source_speed[keep],
        "source_path_speed_mps": source_path_speed[keep],
        **{key: value[keep] for key, value in quality.items()},
    }
    shard = output_root / "data" / split / f"{session}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard, **arrays)
    elapsed = time.perf_counter() - started
    row = {
        "session": session,
        "split": split,
        "source_file": f"coltrane/{session}.h5",
        "source_sha256": source_digest,
        "strict_blocks": int(len(starts)),
        "released_clips": int(keep.sum()),
        "rejected_clips": int((~keep).sum()),
        "pass_rate": pass_rate,
        "mean_source_speed_mps": float(source_speed[keep].mean()),
        "mean_source_path_speed_mps": float(source_path_speed[keep].mean()),
        "mean_ik_foot_rmse": float(arrays["ik_foot_rmse"].mean()),
        "rejected_source_starts": starts[~keep].astype(int).tolist(),
        "shard": str(shard.relative_to(output_root)),
        "shard_bytes": shard.stat().st_size,
        "shard_sha256": sha256(shard),
        "build_seconds": elapsed,
    }
    print(
        f"[{session}] {row['released_clips']}/{row['strict_blocks']} clips -> {shard} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return row


def render_card(template: Path, output: Path, rows: list[dict]) -> None:
    counts = {
        split: sum(row["released_clips"] for row in rows if row["split"] == split)
        for split in SPLIT_SESSIONS
    }
    appendix = (
        "\n## Release statistics\n\n"
        f"Schema `{SCHEMA_VERSION}` contains **{sum(counts.values()):,} clips**: "
        f"{counts['train']:,} train, {counts['validation']:,} validation, and "
        f"{counts['test']:,} test. Generated statistics are authoritative in "
        "`manifest.json`.\n"
    )
    output.write_text(template.read_text() + appendix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=RetargetConfig.optimizer_steps)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_split_contract()
    output = args.output.resolve()
    if output.exists():
        if args.overwrite and args.resume:
            raise SystemExit("choose only one of --overwrite and --resume")
        if args.overwrite:
            shutil.rmtree(output)
        elif not args.resume:
            raise SystemExit(
                f"{output} exists; pass --resume or --overwrite for this generated release"
            )
    output.mkdir(parents=True, exist_ok=True)
    selected = args.sessions or [
        session for split in ("train", "validation", "test") for session in SPLIT_SESSIONS[split]
    ]
    unknown = sorted(set(selected) - set(SESSION_TO_SPLIT))
    if unknown:
        raise SystemExit(f"unknown Coltrane sessions: {unknown}")
    config = RetargetConfig(optimizer_steps=args.steps)
    complete_release = set(selected) == set(SESSION_TO_SPLIT)
    started = time.perf_counter()
    rows = []
    for session in selected:
        shard = output / "data" / SESSION_TO_SPLIT[session] / f"{session}.npz"
        if args.resume and shard.exists():
            print(f"[{session}] recovering completed shard", flush=True)
            rows.append(recover_session(session, output))
        else:
            rows.append(build_session(session, output, config, args.batch_size))
    strict_total = sum(row["strict_blocks"] for row in rows)
    released_total = sum(row["released_clips"] for row in rows)
    global_pass_rate = released_total / strict_total
    if complete_release and global_pass_rate < MIN_GLOBAL_PASS_RATE:
        raise RuntimeError(
            f"only {global_pass_rate:.1%} of strict blocks pass target quality gates; "
            f"minimum is {MIN_GLOBAL_PASS_RATE:.1%}"
        )
    counts = {
        split: sum(row["released_clips"] for row in rows if row["split"] == split)
        for split in SPLIT_SESSIONS
    }
    transformation_hashes = code_hashes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository_id": REPOSITORY_ID,
        "complete_release": complete_release,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "animal": ANIMAL,
        "fps": FPS,
        "clip_frames": CLIP_FRAMES,
        "coordinate_system": "Brax Fetch model units; scalar-first quaternions",
        "foot_order": ["front_right", "front_left", "back_right", "back_left"],
        "joint_names": list(JOINT_NAMES),
        "fields": FIELDS,
        "splits": {key: list(value) for key, value in SPLIT_SESSIONS.items()},
        "counts": counts,
        "upstream": {
            "doi": UPSTREAM_DOI,
            "license": UPSTREAM_LICENSE,
            "attribution": "Aldarondo et al., Nature 632, 594-602 (2024)",
        },
        "selection": {
            "block_stride": CLIP_FRAMES,
            "minimum_speed_mps": MIN_SPEED,
            "minimum_gait_coordination": MIN_GAIT_COORDINATION,
            "maximum_turn_degrees": MAX_TURN_DEGREES,
            "maximum_neck_drift_mm": MAX_NECK_DRIFT_MM,
            "gait_pairs": [list(pair) for pair in GAIT_PAIRS],
        },
        "retarget_config": asdict(config),
        "quality_gates": QUALITY_GATES,
        "global_pass_rate": global_pass_rate,
        "minimum_global_pass_rate": MIN_GLOBAL_PASS_RATE,
        "transformation_code_sha256": transformation_hashes,
        "retarget_code_sha256": aggregate_code_hash(transformation_hashes),
        "git": git_state(),
        "sessions": rows,
        "build_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    render_card(Path(__file__).with_name("DATASET_CARD.md"), output / "README.md", rows)
    print(f"wrote release {output} ({sum(counts.values()):,} clips)", flush=True)


if __name__ == "__main__":
    main()
