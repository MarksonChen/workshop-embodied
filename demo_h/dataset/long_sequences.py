"""Build true five-second Demo H trajectories without looping short clips.

The two stages intentionally use different environments::

    uv run --extra workshop python -m demo_h.dataset.long_sequences retarget

    env -u LD_LIBRARY_PATH uv run --no-project --isolated \
      --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
      --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
      python -m demo_h.dataset.long_sequences project

Every example comes from one continuous strict-locomotion source segment.  No
released 64-frame crops are concatenated, looped, or interpolated across a
boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from demo_f.artifacts import sha256
from demo_f.config import ANIMAL, DATA_ROOT, FPS, JOINT_LIMIT, RetargetConfig
from demo_f.dataset.contract import SPLIT_SESSIONS


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = ROOT / "release_long_reference"
DEFAULT_ROOT = ROOT / "release_long"
SCHEMA = "demo-h-long-sequences-v1"
TIME_SCALE = 1.75
FRAMES = 256
SOURCE_FRAMES = int(math.ceil((FRAMES - 1) / TIME_SCALE)) + 1
SOURCE_STRIDE = SOURCE_FRAMES


@dataclass(frozen=True)
class LongBodyActionSet:
    features: np.ndarray
    normalized_control: np.ndarray
    root_position: np.ndarray
    root_quaternion: np.ndarray
    contacts: np.ndarray
    source_start: np.ndarray
    source_speed_mps: np.ndarray
    source_path_speed_mps: np.ndarray
    session_index: np.ndarray
    sessions: tuple[str, ...]


def _linear(values: np.ndarray, source_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    lower = np.floor(source_time).astype(np.int32)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = source_time - lower
    shape = (len(weight),) + (1,) * (values.ndim - 1)
    return values[lower] * (1.0 - weight.reshape(shape)) + values[upper] * weight.reshape(shape)


def _yaw_quaternion(yaw: np.ndarray) -> np.ndarray:
    quaternion = np.zeros((len(yaw), 4), np.float32)
    quaternion[:, 0] = np.cos(yaw / 2)
    quaternion[:, 3] = np.sin(yaw / 2)
    return quaternion


def _retime(source: dict, angles: np.ndarray) -> dict[str, np.ndarray]:
    source_time = np.arange(FRAMES, dtype=np.float64) / TIME_SCALE
    if source_time[-1] > len(angles) - 1:
        raise AssertionError("source window is too short for requested retiming")
    retimed_angles = np.clip(
        _linear(angles, source_time), -JOINT_LIMIT, JOINT_LIMIT
    ).astype(np.float32)
    contact_index = np.rint(source_time).astype(np.int32)
    return {
        "root_position": _linear(source["root_position"], source_time).astype(
            np.float32
        ),
        "root_quaternion": _yaw_quaternion(
            _linear(source["yaw"][:, None], source_time)[:, 0]
        ),
        "joint_angles": retimed_angles,
        "contacts": np.asarray(source["contacts"][contact_index], np.uint8),
    }


def _segment_window_starts(begin: int, end: int) -> np.ndarray:
    count = (end - begin) // SOURCE_STRIDE
    if count < 1:
        return np.empty(0, np.int32)
    # Spread non-overlapping windows across the complete eligible segment so a
    # long run contributes its beginning and end, not only its first seconds.
    return np.rint(np.linspace(begin, end - SOURCE_FRAMES, count)).astype(np.int32)


def _read_split(split: str, config: RetargetConfig):
    import h5py

    from demo_b.strict_locomotion import merge_blocks, strict_block_starts
    from demo_f.retarget import prepare_source

    sources, provenance, sessions = [], [], []
    for session_index, session in enumerate(SPLIT_SESSIONS[split]):
        path = DATA_ROOT / ANIMAL / f"{session}.h5"
        with h5py.File(path, "r") as archive:
            qpos = archive["/pose/qpos"][:].astype(np.float32)
            keypoints = archive["/pose/keypoints"]
            names = [
                value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
                for value in keypoints.attrs["names"]
            ]
            neck = keypoints[:, 2, names.index("SpineF")].astype(np.float64)
            bounds = merge_blocks(strict_block_starts(qpos, neck))
            starts = np.concatenate(
                [_segment_window_starts(begin, end) for begin, end in bounds]
            ) if bounds else np.empty(0, np.int32)
            for start in starts:
                stop = int(start) + SOURCE_FRAMES
                points = np.transpose(keypoints[int(start):stop], (0, 2, 1)) / 1_000.0
                source = prepare_source(points, names, qpos[int(start):stop, :7], config)
                sources.append(source)
                provenance.append((session_index, int(start)))
        sessions.append(session)
        print(f"[{split}] {session}: {len(starts)} long windows", flush=True)
    return sources, provenance, tuple(sessions)


def _optimize_sources(
    sources: list[dict], config: RetargetConfig, batch_size: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from demo_f.dataset.build import batch_quality
    from demo_f.retarget import optimize_batch

    angle_parts = []
    quality_parts: dict[str, list[np.ndarray]] = {}
    for offset in range(0, len(sources), batch_size):
        group = sources[offset : offset + batch_size]
        pad = batch_size - len(group)
        padded = group + [group[-1]] * pad
        angles, _ = optimize_batch(
            np.stack([source["target_feet_local"] for source in padded]),
            np.stack([source["contacts"] for source in padded]),
            np.stack([source["root_position"] for source in padded]),
            np.stack([source["yaw"] for source in padded]),
            config,
        )
        quality = batch_quality(padded, angles)
        angle_parts.append(angles[: len(group)])
        for name, values in quality.items():
            quality_parts.setdefault(name, []).append(values[: len(group)])
        print(f"retargeted {offset + len(group):,}/{len(sources):,}", flush=True)
    return np.concatenate(angle_parts), {
        name: np.concatenate(parts) for name, parts in quality_parts.items()
    }


def retarget_release(
    output_root: Path,
    splits: tuple[str, ...],
    *,
    batch_size: int,
    optimizer_steps: int,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = RetargetConfig(optimizer_steps=optimizer_steps)
    rows, counts = [], {}
    for split in splits:
        sources, provenance, sessions = _read_split(split, config)
        if not sources:
            raise ValueError(f"{split} contains no {SOURCE_FRAMES}-frame strict runs")
        angles, quality = _optimize_sources(sources, config, batch_size)
        keep = quality.pop("quality_pass")
        retimed = [
            _retime(source, angle)
            for source, angle, accepted in zip(sources, angles, keep, strict=True)
            if accepted
        ]
        kept_provenance = [
            row for row, accepted in zip(provenance, keep, strict=True) if accepted
        ]
        kept_sources = [
            source for source, accepted in zip(sources, keep, strict=True) if accepted
        ]
        arrays = {
            name: np.stack([clip[name] for clip in retimed])
            for name in ("root_position", "root_quaternion", "joint_angles", "contacts")
        }
        arrays.update(
            session_index=np.asarray([row[0] for row in kept_provenance], np.int16),
            source_start=np.asarray([row[1] for row in kept_provenance], np.int32),
            source_speed_mps=np.asarray(
                [source["source_speed"] for source in kept_sources], np.float32
            ),
            source_path_speed_mps=np.asarray(
                [source["source_path_speed"] for source in kept_sources], np.float32
            ),
        )
        shard = output_root / f"{split}.npz"
        np.savez_compressed(shard, **arrays)
        counts[split] = len(retimed)
        rows.append(
            {
                "split": split,
                "sessions": list(sessions),
                "candidates": len(sources),
                "released": len(retimed),
                "shard": shard.name,
                "shard_sha256": sha256(shard),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "stage": "continuous-kinematic-reference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "animal": ANIMAL,
        "fps": FPS,
        "frames": FRAMES,
        "source_frames": SOURCE_FRAMES,
        "time_scale": TIME_SCALE,
        "continuity": "one raw strict-locomotion segment per example; never looped or joined",
        "retarget_config": asdict(config),
        "counts": counts,
        "splits": rows,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output_root), "counts": counts}, indent=2))
    return manifest


def project_release(
    reference_root: Path,
    output_root: Path,
    splits: tuple[str, ...],
    *,
    batch_size: int,
) -> dict:
    # Lazy import keeps the retarget stage independent of the legacy Brax pin.
    from demo_h.dataset.project import ExactFetchProjector

    reference_root, output_root = Path(reference_root), Path(output_root)
    reference_manifest = json.loads((reference_root / "manifest.json").read_text())
    if reference_manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected long-reference schema")
    output_root.mkdir(parents=True, exist_ok=True)
    projector = ExactFetchProjector(batch_size=batch_size, clip_frames=FRAMES)
    split_rows, counts = [], {}
    for split in splits:
        with np.load(reference_root / f"{split}.npz") as archive:
            reference = {name: archive[name] for name in archive.files}
        parts: dict[str, list[np.ndarray]] = {}
        kept = []
        for offset in range(0, len(reference["joint_angles"]), batch_size):
            stop = min(offset + batch_size, len(reference["joint_angles"]))
            projected = projector.project(
                {
                    name: reference[name][offset:stop]
                    for name in ("joint_angles", "root_position", "root_quaternion")
                }
            )
            accepted = projected.pop("accepted")
            kept.append(np.flatnonzero(accepted) + offset)
            for name, values in projected.items():
                parts.setdefault(name, []).append(values[accepted])
            print(f"[{split}] projected {stop:,}/{len(reference['joint_angles']):,}", flush=True)
        keep = np.concatenate(kept)
        arrays = {name: np.concatenate(values) for name, values in parts.items()}
        for name in (
            "session_index",
            "source_start",
            "source_speed_mps",
            "source_path_speed_mps",
        ):
            arrays[name] = reference[name][keep]
        shard = output_root / f"{split}.npz"
        np.savez_compressed(shard, **arrays)
        counts[split] = len(keep)
        split_rows.append(
            {
                "split": split,
                "candidates": len(reference["joint_angles"]),
                "released": len(keep),
                "sessions": next(
                    row["sessions"]
                    for row in reference_manifest["splits"]
                    if row["split"] == split
                ),
                "shard": shard.name,
                "shard_sha256": sha256(shard),
            }
        )
    manifest = {
        **reference_manifest,
        "stage": "continuous-exact-physics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_manifest_sha256": sha256(reference_root / "manifest.json"),
        "counts": counts,
        "splits": split_rows,
        "physics": projector.physics_contract(),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output_root), "counts": counts}, indent=2))
    return manifest


def load_split(split: str, root: Path = DEFAULT_ROOT) -> LongBodyActionSet:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("stage") != "continuous-exact-physics":
        raise ValueError("unexpected long-sequence release")
    row = next(row for row in manifest["splits"] if row["split"] == split)
    with np.load(root / row["shard"]) as archive:
        return LongBodyActionSet(
            features=archive["realized_features"].astype(np.float32),
            normalized_control=archive["normalized_control"].astype(np.float32),
            root_position=archive["realized_root_position"].astype(np.float32),
            root_quaternion=archive["realized_root_quaternion"].astype(np.float32),
            contacts=archive["realized_contacts"].astype(np.uint8),
            source_start=archive["source_start"].astype(np.int32),
            source_speed_mps=archive["source_speed_mps"].astype(np.float32),
            source_path_speed_mps=archive["source_path_speed_mps"].astype(np.float32),
            session_index=archive["session_index"].astype(np.int16),
            sessions=tuple(row["sessions"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    retarget = subparsers.add_parser("retarget")
    retarget.add_argument("--output-root", type=Path, default=REFERENCE_ROOT)
    retarget.add_argument(
        "--splits", nargs="+", choices=tuple(SPLIT_SESSIONS), default=tuple(SPLIT_SESSIONS)
    )
    retarget.add_argument("--batch-size", type=int, default=32)
    retarget.add_argument("--optimizer-steps", type=int, default=RetargetConfig.optimizer_steps)
    project = subparsers.add_parser("project")
    project.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    project.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    project.add_argument(
        "--splits", nargs="+", choices=tuple(SPLIT_SESSIONS), default=tuple(SPLIT_SESSIONS)
    )
    project.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.stage == "retarget":
        retarget_release(
            args.output_root,
            tuple(args.splits),
            batch_size=args.batch_size,
            optimizer_steps=args.optimizer_steps,
        )
    else:
        project_release(
            args.reference_root,
            args.output_root,
            tuple(args.splits),
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
