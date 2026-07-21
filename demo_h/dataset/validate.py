"""Schema, physical-quality, and independent replay checks for Demo H."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_f.artifacts import sha256
from demo_f.commands import hindsight_command, yaw_from_quaternion
from demo_f.config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)
from demo_f.dataset.contract import SPLIT_SESSIONS
from demo_f.features import trajectory_features
from demo_f.kinematics import fetch_feet_numpy

from demo_h.config import (
    CLIP_FRAMES,
    COMMAND_HORIZON_FRAMES,
    FPS,
    MAX_COMMAND_SPEED,
    MAX_CONTROL_SATURATION,
    MAX_JOINT_TRACKING_RMSE,
    MAX_PLANAR_SPEED,
    MAX_YAW_RATE,
    MIN_TORSO_HEIGHT,
    MIN_UPRIGHT,
    TORQUE_STRENGTH,
    TRANSITIONS,
)

from .contract import (
    DATASET_VARIANT,
    DEFAULT_ROOT,
    DTYPES,
    FIELDS,
    SCHEMA_VERSION,
    expected_shape,
)


def _require_replay_backend(expected_backend: str | None) -> str:
    import jax

    actual_backend = jax.default_backend()
    if expected_backend is not None and actual_backend != expected_backend:
        raise RuntimeError(
            f"release was projected with JAX backend {expected_backend!r}, but replay "
            f"is using {actual_backend!r}; use the pinned jax[cuda12] command from "
            "demo_h/README.md"
        )
    return actual_backend


def replay_report(
    path: Path,
    clips: int = 16,
    *,
    expected_backend: str | None = None,
) -> dict:
    """Independently replay stored controls in exact Brax v1 Fetch physics.

    Brax is imported lazily so schema-only validation remains available in the
    main workshop environment. Run replay validation in the same isolated
    Brax 0.12.3 environment used by the projector.
    """

    import jax
    import jax.numpy as jnp
    from brax.v1.envs import fetch

    actual_backend = _require_replay_backend(expected_backend)

    with np.load(path) as archive:
        count = min(int(clips), len(archive["parent_clip_id"]))
        if count < 2:
            raise ValueError("replay/shuffle gate needs at least two clips")
        arrays = {
            name: archive[name][:count]
            for name in (
                "initial_qp_pos",
                "initial_qp_rot",
                "initial_qp_vel",
                "initial_qp_ang",
                "normalized_control",
                "realized_root_position",
                "realized_root_quaternion",
                "realized_joint_angles",
            )
        }
    env = fetch.Fetch()
    base = env.sys.default_qp()
    qp = jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value)[None], count, axis=0), base
    ).replace(
        pos=jnp.asarray(arrays["initial_qp_pos"]),
        rot=jnp.asarray(arrays["initial_qp_rot"]),
        vel=jnp.asarray(arrays["initial_qp_vel"]),
        ang=jnp.asarray(arrays["initial_qp_ang"]),
    )

    def execute(initial, controls):
        def step(state, control):
            state, _ = jax.vmap(env.sys.step)(state, control)
            angle, _ = jax.vmap(env.sys.joints[0].angle_vel)(state)
            return state, (state.pos[:, 0], state.rot[:, 0], angle)

        _, output = jax.lax.scan(step, initial, jnp.swapaxes(controls, 0, 1))
        root, quaternion, angle = [jnp.swapaxes(value, 0, 1) for value in output]
        initial_angle, _ = jax.vmap(env.sys.joints[0].angle_vel)(initial)
        return (
            jnp.concatenate((initial.pos[:, 0, None], root), axis=1),
            jnp.concatenate((initial.rot[:, 0, None], quaternion), axis=1),
            jnp.concatenate((initial_angle[:, None], angle), axis=1),
        )

    execute = jax.jit(execute)
    paired = execute(qp, jnp.asarray(arrays["normalized_control"]))
    shuffled_controls = np.roll(arrays["normalized_control"], 1, axis=0)
    shuffled = execute(qp, jnp.asarray(shuffled_controls))
    paired = [np.asarray(value) for value in paired]
    shuffled = [np.asarray(value) for value in shuffled]
    target = [
        arrays["realized_root_position"],
        arrays["realized_root_quaternion"],
        arrays["realized_joint_angles"],
    ]
    paired_errors = [
        float(np.sqrt(np.mean(np.square(value - truth))))
        for value, truth in zip(paired, target, strict=True)
    ]
    shuffled_errors = [
        float(np.sqrt(np.mean(np.square(value - truth))))
        for value, truth in zip(shuffled, target, strict=True)
    ]
    # Same-CUDA-backend replay agrees to about 1e-5. CPU and GPU executions of
    # the legacy PBD solver diverge after contact, so a backend mismatch is
    # rejected above rather than hidden by loosening this integrity threshold.
    if max(paired_errors) > 1e-3:
        raise ValueError(f"independent replay mismatch: {paired_errors}")
    if shuffled_errors[2] <= paired_errors[2] + 1e-3:
        raise ValueError("shuffled controls were not materially worse")
    return {
        "clips": count,
        "jax_backend": actual_backend,
        "paired_root_rmse": paired_errors[0],
        "paired_quaternion_rmse": paired_errors[1],
        "paired_joint_rmse": paired_errors[2],
        "shuffled_root_rmse": shuffled_errors[0],
        "shuffled_quaternion_rmse": shuffled_errors[1],
        "shuffled_joint_rmse": shuffled_errors[2],
    }


def validate_archive(path: Path) -> dict:
    with np.load(path) as archive:
        missing = sorted(set(FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        clips = len(archive["parent_clip_id"])
        for name in FIELDS:
            value = archive[name]
            if value.shape != expected_shape(name, clips):
                raise ValueError(f"{path}:{name} has {value.shape}")
            if str(value.dtype) != DTYPES[name]:
                raise ValueError(f"{path}:{name} has dtype {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}:{name} contains non-finite values")
        controls = archive["normalized_control"]
        if np.abs(controls).max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError(f"{path} contains out-of-range controls")
        expected_torque = -TORQUE_STRENGTH * controls
        torque_error = float(
            np.max(np.abs(expected_torque - archive["requested_actuator_torque"]), initial=0.0)
        )
        if torque_error > 1e-4:
            raise ValueError(f"{path} torque/control mismatch {torque_error}")
        if not archive["valid_transition_mask"].all():
            raise ValueError(f"{path} contains invalid retained transitions")
        if not archive["solver_status"].all():
            raise ValueError(f"{path} contains failed retained projections")

        realized_root = archive["realized_root_position"]
        realized_quaternion = archive["realized_root_quaternion"]
        realized_angles = archive["realized_joint_angles"]
        realized_contacts = archive["realized_contacts"]
        if not np.array_equal(realized_contacts[:, 0], realized_contacts[:, 1]):
            raise ValueError(f"{path} violates feature-contract-v1 frame-zero contacts")
        feet = fetch_feet_numpy(realized_angles)
        recomputed_features = trajectory_features(
            realized_root,
            realized_quaternion,
            realized_angles,
            feet,
            realized_contacts,
        )
        if not np.allclose(
            recomputed_features,
            archive["realized_features"],
            rtol=1e-5,
            atol=5e-5,
        ):
            error = float(
                np.max(np.abs(recomputed_features - archive["realized_features"]))
            )
            raise ValueError(f"{path} feature-contract mismatch {error}")
        recomputed_command = hindsight_command(realized_root, realized_quaternion)
        if not np.allclose(recomputed_command, archive["command"], atol=2e-5):
            raise ValueError(f"{path} hindsight-command mismatch")

        reference_delta = (
            archive["reference_root_position"]
            - archive["reference_root_position"][:, :1]
        )
        realized_delta = realized_root - realized_root[:, :1]
        derived = {
            "joint_tracking_rmse": np.sqrt(
                np.mean(
                    np.square(realized_angles - archive["reference_joint_angles"]),
                    axis=(1, 2),
                )
            ),
            "root_tracking_rmse": np.sqrt(
                np.mean(np.square(realized_delta - reference_delta), axis=(1, 2))
            ),
            "control_saturation_fraction": np.mean(
                np.abs(controls) >= 0.999, axis=(1, 2)
            ),
            "minimum_torso_height": realized_root[..., 2].min(axis=1),
            "minimum_upright": (
                1.0
                - 2.0
                * (
                    np.square(realized_quaternion[..., 1])
                    + np.square(realized_quaternion[..., 2])
                )
            ).min(axis=1),
            "maximum_planar_speed": np.linalg.norm(
                np.diff(realized_root[..., :2], axis=1) * FPS, axis=-1
            ).max(axis=1),
            "maximum_yaw_rate": np.abs(
                np.diff(yaw_from_quaternion(realized_quaternion), axis=1) * FPS
            ).max(axis=1),
            "realized_command_speed": np.linalg.norm(
                recomputed_command[:, :2], axis=1
            )
            / (COMMAND_HORIZON_FRAMES / FPS),
        }
        for name, expected in derived.items():
            if not np.allclose(expected, archive[name], rtol=1e-5, atol=2e-5):
                raise ValueError(f"{path}:{name} does not match realized trajectory")
        gates = {
            "control_saturation_fraction": (None, MAX_CONTROL_SATURATION),
            "joint_tracking_rmse": (None, MAX_JOINT_TRACKING_RMSE),
            "minimum_torso_height": (MIN_TORSO_HEIGHT, None),
            "minimum_upright": (MIN_UPRIGHT, None),
            "maximum_planar_speed": (None, MAX_PLANAR_SPEED),
            "maximum_yaw_rate": (None, MAX_YAW_RATE),
            "realized_command_speed": (None, MAX_COMMAND_SPEED),
        }
        for name, (minimum, maximum) in gates.items():
            values = archive[name]
            if minimum is not None and values.min(initial=np.inf) < minimum - 1e-6:
                raise ValueError(f"{path}:{name} falls below its release gate")
            if maximum is not None and values.max(initial=-np.inf) > maximum + 1e-6:
                raise ValueError(f"{path}:{name} exceeds its release gate")
        return {
            "clips": clips,
            "joint_tracking_rmse_median": float(np.median(archive["joint_tracking_rmse"])),
            "root_tracking_rmse_median": float(np.median(archive["root_tracking_rmse"])),
            "saturation_fraction": float(np.mean(np.abs(controls) >= 0.999)),
            "minimum_height": float(archive["minimum_torso_height"].min(initial=np.inf)),
            "minimum_upright": float(archive["minimum_upright"].min(initial=np.inf)),
        }


def validate_release(root: Path = DEFAULT_ROOT) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema {manifest.get('schema_version')!r}")
    if not manifest.get("complete_release", False):
        raise ValueError("Demo H training requires a complete release")
    if manifest.get("variant") != DATASET_VARIANT:
        raise ValueError(f"unexpected dataset variant {manifest.get('variant')!r}")
    feature_contract = manifest.get(
        "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
    )
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"dataset feature contract {feature_contract!r}; "
            f"expected {FEATURE_CONTRACT_VERSION!r}"
        )
    if manifest.get("clip_frames") != CLIP_FRAMES:
        raise ValueError("manifest clip length differs from the frozen contract")
    if manifest.get("fields") != {name: list(shape) for name, shape in FIELDS.items()}:
        raise ValueError("manifest fields differ from the frozen schema")
    if manifest.get("dtypes") != DTYPES:
        raise ValueError("manifest dtypes differ from the frozen schema")
    if manifest.get("splits") != list(SPLIT_SESSIONS):
        raise ValueError("manifest split names differ from Demo F's frozen splits")
    rows = []
    counts = {split: 0 for split in manifest["splits"]}
    sessions = {split: set() for split in manifest["splits"]}
    for session in manifest["sessions"]:
        shard = root / session["shard"]
        if sha256(shard) != session.get("shard_sha256"):
            raise ValueError(f"checksum mismatch: {shard}")
        report = validate_archive(shard)
        if report["clips"] != session["released_clips"]:
            raise ValueError(f"manifest count mismatch for {session['session']}")
        counts[session["split"]] += report["clips"]
        sessions[session["split"]].add(session["session"])
        rows.append(report)
    if counts != manifest["counts"]:
        raise ValueError(f"split count mismatch: {counts} != {manifest['counts']}")
    expected_sessions = {name: set(values) for name, values in SPLIT_SESSIONS.items()}
    if sessions != expected_sessions:
        raise ValueError("manifest sessions differ from Demo F's frozen splits")
    split_names = list(sessions)
    for left, name in enumerate(split_names):
        for other in split_names[left + 1 :]:
            if sessions[name] & sessions[other]:
                raise ValueError(f"session leakage between {name} and {other}")
    total = sum(row["clips"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "complete_release": manifest["complete_release"],
        "counts": counts,
        "clips": total,
        "transitions": total * TRANSITIONS,
        "joint_tracking_rmse_median_across_shards": float(
            np.median([row["joint_tracking_rmse_median"] for row in rows])
        ),
        "root_tracking_rmse_median_across_shards": float(
            np.median([row["root_tracking_rmse_median"] for row in rows])
        ),
        "saturation_fraction_mean_across_shards": float(
            np.mean([row["saturation_fraction"] for row in rows])
        ),
        "minimum_height": float(min(row["minimum_height"] for row in rows)),
        "minimum_upright": float(min(row["minimum_upright"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--replay-clips",
        type=int,
        default=0,
        help="also replay this many clips from the first non-empty shard",
    )
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text())
    expected_backend = manifest.get("physics", {}).get("jax_backend", "gpu")
    if args.replay_clips:
        _require_replay_backend(expected_backend)
    report = validate_release(args.dataset_root)
    if args.replay_clips:
        shard = next(
            args.dataset_root / row["shard"]
            for row in manifest["sessions"]
            if row["released_clips"] >= 2
        )
        report["independent_replay"] = replay_report(
            shard,
            args.replay_clips,
            expected_backend=expected_backend,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
