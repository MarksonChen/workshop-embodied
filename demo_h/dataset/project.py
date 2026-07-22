"""Project Demo F references into the exact Brax v1 Fetch physics.

Run this module in the historical Demo A environment, for example::

    env -u LD_LIBRARY_PATH uv run --no-project --isolated \
      --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
      --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
      python -m demo_h.dataset.project --splits train validation test

The source pose is a soft reference.  A bounded, deterministic feedback
controller produces controls, and the released state is always the state that
the unchanged simulator actually realizes.  This is deliberately described as
physics-derived pseudo-labelling, not animal inverse dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from brax.v1 import math
from brax.v1.envs import fetch

from demo_f.artifacts import guard_derived_release_output, sha256
from demo_f.commands import hindsight_command, yaw_from_quaternion
from demo_f.config import FEATURE_CONTRACT_VERSION
from demo_f.features import trajectory_features
from demo_f.jax_features import contact_flags
from demo_f.kinematics import fetch_feet
from demo_h.config import (
    CLIP_FRAMES,
    DT,
    FPS,
    FETCH_FOOT_NAMES,
    MAX_COMMAND_SPEED,
    MAX_CONTROL_SATURATION,
    MAX_JOINT_TRACKING_RMSE,
    MAX_PLANAR_SPEED,
    MAX_YAW_RATE,
    MIN_TORSO_HEIGHT,
    MIN_UPRIGHT,
    PD_POSITION_GAIN,
    PD_VELOCITY_GAIN,
    TORQUE_STRENGTH,
)

from .contract import (
    DATASET_VARIANT,
    DEFAULT_ROOT,
    DTYPES,
    FIELDS,
    PARENT_ROOT,
    PARENT_VARIANT,
    SCHEMA_VERSION,
)


def _reference_rates(angles: np.ndarray, roots: np.ndarray, quaternions: np.ndarray):
    angle_rate = np.empty_like(angles)
    angle_rate[:, 1:] = np.diff(angles, axis=1) / DT
    angle_rate[:, 0] = angle_rate[:, 1]
    root_velocity = (roots[:, 1] - roots[:, 0]) / DT
    yaw = yaw_from_quaternion(quaternions)
    root_angular = np.zeros((len(angles), 3), np.float32)
    root_angular[:, 2] = (yaw[:, 1] - yaw[:, 0]) / DT
    return angle_rate, root_velocity, root_angular


class ExactFetchProjector:
    """Batched deterministic pseudo-label generator in exact deployment physics."""

    def __init__(self, batch_size: int = 256, clip_frames: int = CLIP_FRAMES):
        self.batch_size = int(batch_size)
        self.clip_frames = int(clip_frames)
        if self.clip_frames < 2:
            raise ValueError("clip_frames must include at least one transition")
        self.env = fetch.Fetch()
        self.sys = self.env.sys
        self.connected = jnp.arange(11)
        self.target_index = self.sys.body.index["Target"]
        self.foot_indices = jnp.asarray(
            tuple(self.sys.body.index[name] for name in FETCH_FOOT_NAMES)
        )
        self._rollout = jax.jit(self._rollout_impl)

    def _initial_qp(
        self,
        angles: np.ndarray,
        roots: np.ndarray,
        quaternions: np.ndarray,
        angle_rate: np.ndarray,
        root_velocity: np.ndarray,
        root_angular: np.ndarray,
    ):
        # Brax's angle_vel velocity coordinate has the opposite sign from the
        # finite-difference angle convention for this legacy Fetch model.
        qp = jax.vmap(
            lambda angle, velocity: self.sys.default_qp(
                joint_angle=angle, joint_velocity=velocity
            )
        )(jnp.asarray(angles[:, 0]), -jnp.asarray(angle_rate[:, 0]))
        qp = jax.tree_util.tree_map(jnp.asarray, qp)
        root_quaternion = jnp.asarray(quaternions[:, 0])
        local_position = qp.pos[:, self.connected] - qp.pos[:, :1]
        world_offset = jax.vmap(
            jax.vmap(math.rotate, in_axes=(0, None))
        )(local_position, root_quaternion)
        # All clips begin at a common horizontal origin.  This changes no
        # velocity, contact, or egocentric command.
        origin = jnp.asarray(roots[:, 0]).at[:, :2].set(0.0)
        position = qp.pos.at[:, self.connected].set(origin[:, None] + world_offset)
        position = position.at[:, self.target_index].set(
            jnp.tile(jnp.asarray((1_000.0, 1_000.0, 2.0)), (len(angles), 1))
        )
        rotation = qp.rot.at[:, self.connected].set(
            jax.vmap(jax.vmap(math.quat_mul, in_axes=(None, 0)))(
                root_quaternion, qp.rot[:, self.connected]
            )
        )
        base_velocity = jax.vmap(
            jax.vmap(math.rotate, in_axes=(0, None))
        )(qp.vel[:, self.connected], root_quaternion)
        base_angular = jax.vmap(
            jax.vmap(math.rotate, in_axes=(0, None))
        )(qp.ang[:, self.connected], root_quaternion)
        root_velocity = jnp.asarray(root_velocity)
        root_angular = jnp.asarray(root_angular)
        orbital_velocity = jax.vmap(
            jax.vmap(jnp.cross, in_axes=(None, 0))
        )(root_angular, world_offset)
        velocity = qp.vel.at[:, self.connected].set(
            base_velocity + root_velocity[:, None] + orbital_velocity
        )
        angular = qp.ang.at[:, self.connected].set(
            base_angular + root_angular[:, None]
        )
        return qp.replace(pos=position, rot=rotation, vel=velocity, ang=angular)

    def _rollout_impl(self, initial_qp, reference_angles, reference_rates):
        def step(qp, frame):
            angle, velocity = jax.vmap(self.sys.joints[0].angle_vel)(qp)
            control = jnp.clip(
                (
                    PD_POSITION_GAIN * (reference_angles[:, frame] - angle)
                    + PD_VELOCITY_GAIN
                    * (reference_rates[:, frame] - velocity)
                )
                / TORQUE_STRENGTH,
                -1.0,
                1.0,
            )
            next_qp, info = jax.vmap(self.sys.step)(qp, control)
            next_angle, _ = jax.vmap(self.sys.joints[0].angle_vel)(next_qp)
            contact = jax.vmap(contact_flags, in_axes=(0, None))(
                info.contact.vel, self.foot_indices
            )
            return next_qp, (
                next_qp.pos[:, 0],
                next_qp.rot[:, 0],
                next_angle,
                control,
                contact,
            )

        return jax.lax.scan(step, initial_qp, jnp.arange(1, self.clip_frames))

    def project(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        count = len(arrays["joint_angles"])
        if arrays["joint_angles"].shape[1] != self.clip_frames:
            raise ValueError(
                f"expected {self.clip_frames} frames, got "
                f"{arrays['joint_angles'].shape[1]}"
            )
        if count > self.batch_size:
            raise ValueError(f"batch {count} exceeds compiled size {self.batch_size}")
        pad = self.batch_size - count
        padded = {
            name: np.concatenate((value, np.repeat(value[-1:], pad, axis=0)))
            if pad
            else value
            for name, value in arrays.items()
        }
        angle_rate, root_velocity, root_angular = _reference_rates(
            padded["joint_angles"],
            padded["root_position"],
            padded["root_quaternion"],
        )
        initial_qp = self._initial_qp(
            padded["joint_angles"],
            padded["root_position"],
            padded["root_quaternion"],
            angle_rate,
            root_velocity,
            root_angular,
        )
        _, stream = self._rollout(
            initial_qp,
            jnp.asarray(padded["joint_angles"]),
            jnp.asarray(angle_rate),
        )
        stream = [np.swapaxes(np.asarray(value), 0, 1)[:count] for value in stream]
        root, quaternion, angle, control, contact = stream
        initial_angle, _ = jax.vmap(self.sys.joints[0].angle_vel)(initial_qp)
        realized_root = np.concatenate(
            (np.asarray(initial_qp.pos[:count, 0])[:, None], root), axis=1
        ).astype(np.float32)
        realized_quaternion = np.concatenate(
            (np.asarray(initial_qp.rot[:count, 0])[:, None], quaternion), axis=1
        ).astype(np.float32)
        realized_angle = np.concatenate(
            (np.asarray(initial_angle[:count])[:, None], angle), axis=1
        ).astype(np.float32)
        # Feature-contract v1 forward-fills frame zero from the first simulated
        # transition, matching the accepted Demo F/H checkpoints.
        realized_contact = np.concatenate(
            (contact[:, :1], contact), axis=1
        ).astype(np.uint8)
        feet = np.asarray(fetch_feet(jnp.asarray(realized_angle)), np.float32)
        features = trajectory_features(
            realized_root,
            realized_quaternion,
            realized_angle,
            feet,
            realized_contact,
        )

        reference_root = padded["root_position"][:count].copy()
        reference_root[..., :2] -= reference_root[:, :1, :2]
        realized_delta = realized_root - realized_root[:, :1]
        reference_delta = reference_root - reference_root[:, :1]
        root_rmse = np.sqrt(
            np.mean(np.square(realized_delta - reference_delta), axis=(1, 2))
        ).astype(np.float32)
        joint_rmse = np.sqrt(
            np.mean(
                np.square(realized_angle - padded["joint_angles"][:count]),
                axis=(1, 2),
            )
        ).astype(np.float32)
        saturation = np.mean(np.abs(control) >= 0.999, axis=(1, 2)).astype(np.float32)
        upright = 1.0 - 2.0 * (
            np.square(realized_quaternion[..., 1])
            + np.square(realized_quaternion[..., 2])
        )
        minimum_height = realized_root[..., 2].min(axis=1).astype(np.float32)
        minimum_upright = upright.min(axis=1).astype(np.float32)
        maximum_planar_speed = np.linalg.norm(
            np.diff(realized_root[..., :2], axis=1) * FPS, axis=-1
        ).max(axis=1).astype(np.float32)
        realized_yaw = yaw_from_quaternion(realized_quaternion)
        maximum_yaw_rate = np.abs(np.diff(realized_yaw, axis=1) * FPS).max(
            axis=1
        ).astype(np.float32)
        command = hindsight_command(realized_root, realized_quaternion)
        realized_command_speed = (
            np.linalg.norm(command[:, :2], axis=1) / ((63 - 32) / FPS)
        ).astype(np.float32)
        finite = np.isfinite(realized_root).all(axis=(1, 2))
        finite &= np.isfinite(realized_angle).all(axis=(1, 2))
        finite &= np.isfinite(control).all(axis=(1, 2))
        accepted = (
            finite
            & (minimum_height >= MIN_TORSO_HEIGHT)
            & (minimum_upright >= MIN_UPRIGHT)
            & (saturation <= MAX_CONTROL_SATURATION)
            & (joint_rmse <= MAX_JOINT_TRACKING_RMSE)
            & (maximum_planar_speed <= MAX_PLANAR_SPEED)
            & (maximum_yaw_rate <= MAX_YAW_RATE)
            & (realized_command_speed <= MAX_COMMAND_SPEED)
        )
        return {
            "accepted": accepted,
            "reference_root_position": reference_root.astype(np.float32),
            "reference_root_quaternion": padded["root_quaternion"][:count].astype(np.float32),
            "reference_joint_angles": padded["joint_angles"][:count].astype(np.float32),
            "realized_root_position": realized_root,
            "realized_root_quaternion": realized_quaternion,
            "realized_joint_angles": realized_angle,
            "realized_features": features.astype(np.float32),
            "realized_contacts": realized_contact,
            "normalized_control": control.astype(np.float32),
            "requested_actuator_torque": (-TORQUE_STRENGTH * control).astype(np.float32),
            "valid_transition_mask": np.ones(
                (count, self.clip_frames - 1), np.uint8
            ),
            "command": command,
            "initial_qp_pos": np.asarray(initial_qp.pos[:count], np.float32),
            "initial_qp_rot": np.asarray(initial_qp.rot[:count], np.float32),
            "initial_qp_vel": np.asarray(initial_qp.vel[:count], np.float32),
            "initial_qp_ang": np.asarray(initial_qp.ang[:count], np.float32),
            "joint_tracking_rmse": joint_rmse,
            "root_tracking_rmse": root_rmse,
            "control_saturation_fraction": saturation,
            "minimum_torso_height": minimum_height,
            "minimum_upright": minimum_upright,
            "maximum_planar_speed": maximum_planar_speed,
            "maximum_yaw_rate": maximum_yaw_rate,
            "realized_command_speed": realized_command_speed,
            "solver_status": accepted.astype(np.uint8),
        }

    def physics_contract(self) -> dict:
        serialized = self.sys.config.SerializeToString(deterministic=True)
        return {
            "brax_version": __import__("brax").__version__,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "jax_backend": jax.default_backend(),
            "jax_device_kinds": sorted({device.device_kind for device in jax.devices()}),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "python_version": platform.python_version(),
            "fetch_config_sha256": hashlib.sha256(serialized).hexdigest(),
            "dt": float(self.sys.config.dt),
            "substeps": int(self.sys.config.substeps),
            "gravity": [
                float(self.sys.config.gravity.x),
                float(self.sys.config.gravity.y),
                float(self.sys.config.gravity.z),
            ],
            "friction": float(self.sys.config.friction),
            "actuator_order": list(self.sys.joints[0].index),
            "control_range": [-1.0, 1.0],
            "actuator_axis_torque": "requested_actuator_torque = -300 * normalized_control",
        }


def _load_parent(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _empty_rows() -> dict[str, list[np.ndarray]]:
    return {name: [] for name in FIELDS}


def project_release(
    parent_root: Path,
    output_root: Path,
    splits: tuple[str, ...],
    *,
    batch_size: int = 256,
    max_clips_per_shard: int | None = None,
    parent_variant: str | None = PARENT_VARIANT,
    output_variant: str = DATASET_VARIANT,
    overwrite: bool = False,
) -> dict:
    parent_root, output_root = Path(parent_root), Path(output_root)
    complete_splits = set(splits) == {"train", "validation", "test"}
    if output_root.resolve() == DEFAULT_ROOT.resolve() and (
        not complete_splits
        or max_clips_per_shard is not None
        or output_variant != DATASET_VARIANT
    ):
        raise ValueError("partial or experimental projection requires a distinct output root")
    parent_root, output_root = guard_derived_release_output(
        parent_root,
        output_root,
        overwrite=overwrite,
        expected_manifest={
            "schema_version": SCHEMA_VERSION,
            "variant": output_variant,
        },
    )
    parent_manifest_path = parent_root / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    actual_parent_variant = parent_manifest.get("variant")
    if actual_parent_variant != parent_variant:
        raise ValueError(
            f"expected Demo F {parent_variant!r}, got {actual_parent_variant!r}"
        )
    projector = ExactFetchProjector(batch_size=batch_size)
    started = time.perf_counter()
    sessions = []
    counts = {split: 0 for split in splits}
    rejected_total = 0
    for parent_row in parent_manifest["sessions"]:
        split = parent_row["split"]
        if split not in splits:
            continue
        source = _load_parent(parent_root / parent_row["shard"])
        parent_count = len(source["joint_angles"])
        if max_clips_per_shard is not None:
            parent_count = min(parent_count, int(max_clips_per_shard))
            source = {name: values[:parent_count] for name, values in source.items()}
        rows = _empty_rows()
        rejected = []
        for offset in range(0, parent_count, batch_size):
            stop = min(offset + batch_size, parent_count)
            batch = {name: values[offset:stop] for name, values in source.items()}
            projected = projector.project(batch)
            keep = projected.pop("accepted")
            parent_ids = np.arange(offset, stop, dtype=np.int32)
            projected.update(
                parent_clip_id=parent_ids,
                source_start=batch["source_start"].astype(np.int32),
                source_speed_mps=batch["source_speed_mps"].astype(np.float32),
                source_path_speed_mps=batch["source_path_speed_mps"].astype(np.float32),
            )
            rejected.extend(parent_ids[~keep].astype(int).tolist())
            for name in FIELDS:
                rows[name].append(projected[name][keep])
        arrays = {
            name: np.concatenate(parts).astype(DTYPES[name])
            for name, parts in rows.items()
        }
        shard = output_root / parent_row["shard"]
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, **arrays)
        accepted_count = len(arrays["parent_clip_id"])
        counts[split] += accepted_count
        rejected_total += len(rejected)
        sessions.append(
            {
                "session": parent_row["session"],
                "split": split,
                "parent_shard": parent_row["shard"],
                "parent_shard_sha256": parent_row["shard_sha256"],
                "candidate_clips": parent_count,
                "released_clips": accepted_count,
                "rejected_clips": len(rejected),
                "rejected_parent_clip_ids": rejected,
                "shard": parent_row["shard"],
                "shard_bytes": shard.stat().st_size,
                "shard_sha256": sha256(shard),
            }
        )
        print(
            f"{split:10s} {parent_row['session']} "
            f"accepted={accepted_count}/{parent_count}",
            flush=True,
        )
    total = sum(row["candidate_clips"] for row in sessions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete_release": complete_splits and max_clips_per_shard is None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": output_variant,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "fps": FPS,
        "clip_frames": CLIP_FRAMES,
        "temporal_contract": "normalized_control[t] acts over [t,t+1) and produces realized state[t+1]",
        "fields": {name: list(shape) for name, shape in FIELDS.items()},
        "dtypes": DTYPES,
        "splits": list(splits),
        "counts": counts,
        "candidate_clips": total,
        "rejected_clips": rejected_total,
        "global_pass_rate": (total - rejected_total) / max(total, 1),
        "derivation": {
            "parent_manifest": str(parent_manifest_path),
            "parent_manifest_sha256": sha256(parent_manifest_path),
            "parent_variant": actual_parent_variant,
            "method": "bounded joint-reference feedback in exact Fetch physics",
            "state_semantics": "simulator-realized rollout, never the kinematic reference",
            "control_is_pseudo_label": True,
            "position_gain": PD_POSITION_GAIN,
            "velocity_gain": PD_VELOCITY_GAIN,
            "requested_axis_torque_sign": -TORQUE_STRENGTH,
            "code_sha256": {
                "demo_f/commands.py": sha256(
                    Path(__file__).resolve().parents[2] / "demo_f" / "commands.py"
                ),
                "demo_f/features.py": sha256(
                    Path(__file__).resolve().parents[2] / "demo_f" / "features.py"
                ),
                "demo_f/jax_features.py": sha256(
                    Path(__file__).resolve().parents[2] / "demo_f" / "jax_features.py"
                ),
                "demo_h/config.py": sha256(Path(__file__).resolve().parents[1] / "config.py"),
                "demo_h/dataset/project.py": sha256(Path(__file__).resolve()),
            },
        },
        "gates": {
            "minimum_torso_height": MIN_TORSO_HEIGHT,
            "minimum_upright": MIN_UPRIGHT,
            "maximum_control_saturation_fraction": MAX_CONTROL_SATURATION,
            "maximum_joint_tracking_rmse": MAX_JOINT_TRACKING_RMSE,
            "maximum_planar_speed": MAX_PLANAR_SPEED,
            "maximum_yaw_rate": MAX_YAW_RATE,
            "maximum_realized_command_speed": MAX_COMMAND_SPEED,
        },
        "physics": projector.physics_contract(),
        "sessions": sessions,
        "build_seconds": time.perf_counter() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, default=PARENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-clips-per-shard", type=int)
    parser.add_argument("--parent-variant", default=PARENT_VARIANT)
    parser.add_argument(
        "--output-variant",
        default=DATASET_VARIANT,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = project_release(
        args.parent_root,
        args.output_root,
        tuple(args.splits),
        batch_size=args.batch_size,
        max_clips_per_shard=args.max_clips_per_shard,
        parent_variant=args.parent_variant,
        output_variant=args.output_variant,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: manifest[key] for key in ("counts", "global_pass_rate", "build_seconds")}, indent=2))


if __name__ == "__main__":
    main()
