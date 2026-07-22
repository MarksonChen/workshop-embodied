"""Build a positive-control Demo H release from Demo J SNN rollouts.

This is deliberately a separate dataset arm.  It asks whether Demo H's prior
and KL machinery become easier to interpret when the state/action sequences
come from a controller that already closes the loop in Fetch physics.  SNN
seeds 0 and 2 generate train/validation data; seed 1 is test-only so the neural
benchmark used by Demo J never enters prior fitting or model selection.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from demo_f.artifacts import guard_derived_release_output, sha256
from demo_f.commands import hindsight_command, yaw_from_quaternion
from demo_f.config import FEATURE_CONTRACT_VERSION
from demo_f.features import trajectory_features
from demo_f.kinematics import fetch_feet_numpy
from demo_h.config import (
    CLIP_FRAMES,
    COMMAND_HORIZON_SECONDS,
    FPS,
    TORQUE_STRENGTH,
    TRANSITIONS,
)
from demo_h.dataset.loader import load_split
from demo_j.control.config import SNNConfig
from demo_j.control.tracking import FetchTracking
from demo_j.data.physics import (
    XML_PATH,
    foot_site_indices,
    host_model,
    joint_qpos_addresses,
)
from demo_j.control.policy import policy_sequence
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.control.snn import initial_state

from .contract import (
    CONTROLLER_DATASET_VARIANT,
    CONTROLLER_ROOT,
    DTYPES,
    FIELDS,
    SCHEMA_VERSION,
)


SUPPORTED_CHECKPOINT_SCHEMAS = {
    "demo-j-snn-distillation-v1",
    "demo-j-snn-ppo-v1",
}


def _load_controller(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if saved.get("schema") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError(
            f"unsupported controller checkpoint {saved.get('schema')!r}: {path}"
        )
    required = {
        "params",
        "observation_mean",
        "observation_std",
        "config",
        "seed",
    }
    missing = required - set(saved)
    if missing:
        raise ValueError(f"controller checkpoint {path} lacks {sorted(missing)}")
    return {
        "path": path,
        "sha256": sha256(path),
        "schema": saved["schema"],
        "seed": int(saved["seed"]),
        "params": jax.tree.map(jnp.asarray, saved["params"]),
        "mean": jnp.asarray(saved["observation_mean"]),
        "std": jnp.asarray(saved["observation_std"]),
        "config": SNNConfig(**saved["config"]),
        "training_reference_manifest_sha256": saved.get(
            "training_reference_manifest_sha256"
        ),
    }


def _pad_body_rows(values: np.ndarray, *, quaternion: bool = False) -> np.ndarray:
    """Map modern MJX's 11 links to the legacy 13-row compatibility field."""

    if values.ndim != 3 or values.shape[1] > 13:
        raise ValueError(values.shape)
    output = np.zeros((len(values), 13, values.shape[-1]), np.float32)
    if quaternion:
        output[..., 0] = 1.0
    output[:, : values.shape[1]] = values
    return output


def _rollout_controller(reference, controller: dict, batch_size: int) -> dict:
    environment = FetchTracking(
        reference,
        random_start=False,
        track_frames=TRANSITIONS,
    )
    reset = jax.vmap(environment.reset_to)
    physics_step = jax.vmap(environment.step)
    sites = jnp.asarray(foot_site_indices())
    config = controller["config"]

    @jax.jit
    def rollout(indices, params, mean, std):
        state = reset(indices)
        neuronal = initial_state((batch_size,), config)
        alive = jnp.ones((batch_size,), bool)
        initial = (
            state.pipeline_state.qpos,
            state.pipeline_state.qvel,
            state.pipeline_state.x.pos,
            state.pipeline_state.x.rot,
            state.pipeline_state.xd.vel,
            state.pipeline_state.xd.ang,
            state.pipeline_state.site_xpos[:, sites, 2] <= 0.025,
        )

        def advance(carry, _):
            state, neuronal, alive = carry
            observation = jnp.clip((state.obs - mean) / std, -10.0, 10.0)
            neuronal, (action, _) = policy_sequence(
                params, neuronal, observation[None], config
            )
            action = action[0]
            state = physics_step(state, action)
            qpos = state.pipeline_state.qpos
            qvel = state.pipeline_state.qvel
            finite = jnp.all(jnp.isfinite(qpos), axis=-1) & jnp.all(
                jnp.isfinite(qvel), axis=-1
            )
            failed = (
                (state.metrics["root_error"] > 1.0)
                | (state.metrics["root_angle_error_deg"] > 120.0)
                | (state.metrics["joint_error"] > 4.5)
                | (qpos[:, 2] < 0.5)
                | ~finite
            )
            alive = alive & ~failed
            contacts = state.pipeline_state.site_xpos[:, sites, 2] <= 0.025
            return (state, neuronal, alive), (qpos, qvel, action, contacts, alive)

        (_, _, alive), stream = jax.lax.scan(
            advance,
            (state, neuronal, alive),
            xs=None,
            length=TRANSITIONS,
        )
        return initial, stream, alive

    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "qpos",
            "qvel",
            "action",
            "contacts",
            "alive_history",
            "initial_pos",
            "initial_rot",
            "initial_vel",
            "initial_ang",
            "accepted",
        )
    }
    for offset in range(0, reference.clips, batch_size):
        stop = min(offset + batch_size, reference.clips)
        count = stop - offset
        indices = np.arange(offset, stop, dtype=np.int32)
        if count < batch_size:
            indices = np.pad(indices, (0, batch_size - count), mode="edge")
        initial, stream, accepted = jax.device_get(
            rollout(
                jnp.asarray(indices),
                controller["params"],
                controller["mean"],
                controller["std"],
            )
        )
        initial_qpos, initial_qvel, pos, rot, vel, ang, initial_contact = initial
        qpos, qvel, action, contacts, alive_history = stream
        qpos = np.concatenate(
            (initial_qpos[:count, None], np.swapaxes(qpos, 0, 1)[:count]), axis=1
        )
        qvel = np.concatenate(
            (initial_qvel[:count, None], np.swapaxes(qvel, 0, 1)[:count]), axis=1
        )
        contacts = np.concatenate(
            (
                initial_contact[:count, None],
                np.swapaxes(contacts, 0, 1)[:count],
            ),
            axis=1,
        )
        parts["qpos"].append(qpos)
        parts["qvel"].append(qvel)
        parts["action"].append(np.swapaxes(action, 0, 1)[:count])
        parts["contacts"].append(contacts)
        parts["alive_history"].append(np.swapaxes(alive_history, 0, 1)[:count])
        parts["initial_pos"].append(_pad_body_rows(pos[:count]))
        parts["initial_rot"].append(_pad_body_rows(rot[:count], quaternion=True))
        parts["initial_vel"].append(_pad_body_rows(vel[:count]))
        parts["initial_ang"].append(_pad_body_rows(ang[:count]))
        parts["accepted"].append(np.asarray(accepted[:count]))
        print(
            f"  controller seed={controller['seed']} clips={stop}/{reference.clips}",
            flush=True,
        )
    return {name: np.concatenate(values) for name, values in parts.items()}


def _dataset_arrays(
    reference, base, rollout: dict
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    qpos = np.asarray(rollout["qpos"], np.float32)
    action = np.asarray(rollout["action"], np.float32)
    contacts = np.asarray(rollout["contacts"], np.uint8)
    finite = np.isfinite(qpos).all(axis=(1, 2)) & np.isfinite(action).all(axis=(1, 2))
    # Rejected non-finite rows must not reach scipy's quaternion conversion.
    safe_qpos = np.where(finite[:, None, None], qpos, reference.qpos)
    angles = safe_qpos[..., joint_qpos_addresses()]
    feet = fetch_feet_numpy(angles)
    features = trajectory_features(
        safe_qpos[..., :3], safe_qpos[..., 3:7], angles, feet, contacts
    )
    root = safe_qpos[..., :3]
    quaternion = safe_qpos[..., 3:7]
    command = hindsight_command(root, quaternion)
    planar_step = np.linalg.norm(np.diff(root[..., :2], axis=1), axis=-1)
    yaw = yaw_from_quaternion(quaternion)
    upright = 1.0 - 2.0 * (
        np.square(quaternion[..., 1]) + np.square(quaternion[..., 2])
    )
    reference_delta = reference.root_position - reference.root_position[:, :1]
    realized_delta = root - root[:, :1]
    finite &= np.isfinite(features).all(axis=(1, 2))
    accepted = np.asarray(rollout["accepted"], bool) & finite
    arrays = {
        "reference_root_position": reference.root_position.astype(np.float32),
        "reference_root_quaternion": reference.root_quaternion.astype(np.float32),
        "reference_joint_angles": reference.joint_angles.astype(np.float32),
        "realized_root_position": root.astype(np.float32),
        "realized_root_quaternion": quaternion.astype(np.float32),
        "realized_joint_angles": angles.astype(np.float32),
        "realized_features": features.astype(np.float32),
        "realized_contacts": contacts,
        "normalized_control": action,
        "requested_actuator_torque": (-TORQUE_STRENGTH * action).astype(np.float32),
        "valid_transition_mask": np.ones((reference.clips, TRANSITIONS), np.uint8),
        "command": command.astype(np.float32),
        "initial_qp_pos": rollout["initial_pos"].astype(np.float32),
        "initial_qp_rot": rollout["initial_rot"].astype(np.float32),
        "initial_qp_vel": rollout["initial_vel"].astype(np.float32),
        "initial_qp_ang": rollout["initial_ang"].astype(np.float32),
        "parent_clip_id": reference.parent_clip_id.astype(np.int32),
        "source_start": reference.source_start.astype(np.int32),
        "source_speed_mps": base.source_speed_mps.astype(np.float32),
        "source_path_speed_mps": base.source_path_speed_mps.astype(np.float32),
        "joint_tracking_rmse": np.sqrt(
            np.mean(np.square(angles - reference.joint_angles), axis=(1, 2))
        ).astype(np.float32),
        "root_tracking_rmse": np.sqrt(
            np.mean(np.square(realized_delta - reference_delta), axis=(1, 2))
        ).astype(np.float32),
        "control_saturation_fraction": np.mean(
            np.abs(action) >= 0.999, axis=(1, 2)
        ).astype(np.float32),
        "minimum_torso_height": root[..., 2].min(axis=1).astype(np.float32),
        "minimum_upright": upright.min(axis=1).astype(np.float32),
        "maximum_planar_speed": (planar_step * FPS).max(axis=1).astype(np.float32),
        "maximum_yaw_rate": (np.abs(np.diff(yaw, axis=1)) * FPS)
        .max(axis=1)
        .astype(np.float32),
        "realized_command_speed": (
            np.linalg.norm(command[:, :2], axis=1) / COMMAND_HORIZON_SECONDS
        ).astype(np.float32),
        "solver_status": accepted.astype(np.uint8),
    }
    for name, values in arrays.items():
        expected = (reference.clips, *FIELDS[name][1:])
        if values.shape != expected:
            raise ValueError(f"{name} has {values.shape}, expected {expected}")
    return arrays, accepted


def _validate_assignments(training: list[dict], test: dict, held_out_seed: int) -> None:
    training_seeds = [row["seed"] for row in training]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError(f"training controller seeds are not unique: {training_seeds}")
    if held_out_seed in training_seeds:
        raise ValueError(f"held-out SNN seed {held_out_seed} entered train/validation")
    if test["seed"] != held_out_seed:
        raise ValueError(
            f"test controller seed {test['seed']} is not held-out seed {held_out_seed}"
        )


def build_release(
    training_checkpoints: tuple[Path, ...],
    test_checkpoint: Path,
    *,
    projected_root: Path = PROJECTED_ROOT,
    output_root: Path = CONTROLLER_ROOT,
    batch_size: int = 256,
    held_out_seed: int = 1,
    overwrite: bool = False,
) -> dict:
    started = time.perf_counter()
    training = [_load_controller(path) for path in training_checkpoints]
    test = _load_controller(test_checkpoint)
    _validate_assignments(training, test, held_out_seed)
    projected_root, output_root = guard_derived_release_output(
        projected_root,
        output_root,
        overwrite=overwrite,
        expected_manifest={
            "schema_version": SCHEMA_VERSION,
            "variant": CONTROLLER_DATASET_VARIANT,
        },
    )
    rows = []
    counts = {split: 0 for split in ("train", "validation", "test")}
    candidate_counts = {split: 0 for split in counts}
    for split in counts:
        reference = load_projected_reference(split, projected_root)
        base = load_split(split)
        if reference.sessions != base.sessions:
            raise ValueError(f"session order mismatch for {split}")
        if not np.array_equal(reference.parent_clip_id, base.parent_clip_id):
            raise ValueError(f"clip provenance mismatch for {split}")
        controllers = training if split != "test" else [test]
        for controller in controllers:
            expected_reference = controller["training_reference_manifest_sha256"]
            if (
                expected_reference is not None
                and expected_reference != reference.manifest_sha256
            ):
                raise ValueError(
                    f"controller seed {controller['seed']} was trained against a "
                    "different projected-reference contract"
                )
            print(
                f"{split}: rolling {reference.clips} clips with SNN seed "
                f"{controller['seed']}",
                flush=True,
            )
            rollout = _rollout_controller(reference, controller, batch_size)
            arrays, accepted = _dataset_arrays(reference, base, rollout)
            candidate_counts[split] += reference.clips
            for session_index, session in enumerate(reference.sessions):
                candidate = reference.session_index == session_index
                keep = candidate & accepted
                session_arrays = {
                    name: values[keep].astype(DTYPES[name], copy=False)
                    for name, values in arrays.items()
                }
                shard_relative = (
                    Path("shards")
                    / split
                    / (f"{session}__snn_seed{controller['seed']}.npz")
                )
                shard = output_root / shard_relative
                shard.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(shard, **session_arrays)
                released = int(keep.sum())
                candidates = int(candidate.sum())
                rejected_ids = reference.parent_clip_id[candidate & ~accepted]
                counts[split] += released
                rows.append(
                    {
                        "session": f"{session}__snn_seed{controller['seed']}",
                        "source_session": session,
                        "controller_seed": controller["seed"],
                        "controller_checkpoint_sha256": controller["sha256"],
                        "split": split,
                        "candidate_clips": candidates,
                        "released_clips": released,
                        "rejected_clips": candidates - released,
                        "rejected_parent_clip_ids": rejected_ids.astype(int).tolist(),
                        "shard": str(shard_relative),
                        "shard_bytes": shard.stat().st_size,
                        "shard_sha256": sha256(shard),
                    }
                )
            print(
                f"{split}: seed={controller['seed']} accepted="
                f"{int(accepted.sum())}/{reference.clips}",
                flush=True,
            )

    model = host_model()
    controller_rows = [
        {
            "path": str(row["path"]),
            "sha256": row["sha256"],
            "schema": row["schema"],
            "seed": row["seed"],
            "role": role,
        }
        for row, role in [
            *((row, "train+validation") for row in training),
            (test, "test-only"),
        ]
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete_release": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "variant": CONTROLLER_DATASET_VARIANT,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "fps": FPS,
        "clip_frames": CLIP_FRAMES,
        "temporal_contract": (
            "normalized_control[t] is the SNN output executed during [t,t+1); "
            "the saved modern-MJX state[t+1] is its direct consequence"
        ),
        "fields": {name: list(shape) for name, shape in FIELDS.items()},
        "dtypes": DTYPES,
        "splits": ["train", "validation", "test"],
        "counts": counts,
        "candidate_counts": candidate_counts,
        "global_pass_rate": sum(counts.values()) / sum(candidate_counts.values()),
        "derivation": {
            "method": "closed-loop deterministic Demo J SNN policy rollout",
            "state_semantics": "simulator-realized modern-MJX rollout",
            "action_semantics": "bounded action emitted by the recurrent SNN policy",
            "reference_semantics": "independent modern-MJX replay used only as tracking intention",
            "controllers": controller_rows,
            "held_out_neural_benchmark_seed": held_out_seed,
            "leakage_contract": (
                "SNN seed 1 is absent from train and validation and appears only in test"
            ),
            "initial_qp_compatibility_fields": (
                "first 11 rows are modern-MJX link transforms; remaining two legacy "
                "rows are zero/identity sentinels and are never consumed by prior training"
            ),
            "requested_axis_torque_sign": -TORQUE_STRENGTH,
        },
        "acceptance": {
            "rule": (
                "retain complete finite 63-transition rollouts that never cross the "
                "Demo J tracking failure boundary"
            ),
            "minimum_torso_height": 0.5,
            "maximum_root_tracking_error": 1.0,
            "maximum_root_angle_error_deg": 120.0,
            "maximum_joint_error_l2": 4.5,
        },
        "physics": {
            "runtime": "modern MJX",
            "xml": str(XML_PATH),
            "xml_sha256": sha256(XML_PATH),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "brax": __import__("brax").__version__,
            "backend": jax.default_backend(),
            "devices": sorted({device.device_kind for device in jax.devices()}),
            "simulation_dt": float(model.opt.timestep),
            "control_dt": float(model.opt.timestep) * 4,
            "actuator_gear": model.actuator_gear[:, 0].astype(float).tolist(),
        },
        "sessions": rows,
        "build_seconds": time.perf_counter() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--test-checkpoint", type=Path, required=True)
    parser.add_argument("--projected-root", type=Path, default=PROJECTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=CONTROLLER_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--held-out-seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_release(
        tuple(args.training_checkpoints),
        args.test_checkpoint,
        projected_root=args.projected_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        held_out_seed=args.held_out_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "counts",
                    "candidate_counts",
                    "global_pass_rate",
                    "build_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
