"""Schema, physical-quality, and independent replay checks for Demo H."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import DEFAULT_ROOT, DTYPES, FIELDS, SCHEMA_VERSION, expected_shape


def replay_report(path: Path, clips: int = 16) -> dict:
    """Independently replay stored controls in exact Brax v1 Fetch physics.

    Brax is imported lazily so schema-only validation remains available in the
    main workshop environment. Run replay validation in the same isolated
    Brax 0.12.3 environment used by the projector.
    """

    import jax
    import jax.numpy as jnp
    from brax.v1.envs import fetch

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
    # Recompiling the legacy PBD solver at a different batch size changes GPU
    # reduction order slightly; sub-milliradian/sub-millimetre agreement is
    # the appropriate independent-replay threshold.
    if max(paired_errors) > 1e-3:
        raise ValueError(f"independent replay mismatch: {paired_errors}")
    if shuffled_errors[2] <= paired_errors[2] + 1e-3:
        raise ValueError("shuffled controls were not materially worse")
    return {
        "clips": count,
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
        expected_torque = -300.0 * controls
        torque_error = float(
            np.max(np.abs(expected_torque - archive["requested_actuator_torque"]), initial=0.0)
        )
        if torque_error > 1e-4:
            raise ValueError(f"{path} torque/control mismatch {torque_error}")
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
    rows = []
    counts = {split: 0 for split in manifest["splits"]}
    sessions = {split: set() for split in manifest["splits"]}
    for session in manifest["sessions"]:
        report = validate_archive(root / session["shard"])
        if report["clips"] != session["released_clips"]:
            raise ValueError(f"manifest count mismatch for {session['session']}")
        counts[session["split"]] += report["clips"]
        sessions[session["split"]].add(session["session"])
        rows.append(report)
    if counts != manifest["counts"]:
        raise ValueError(f"split count mismatch: {counts} != {manifest['counts']}")
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
        "transitions": total * 63,
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
    report = validate_release(args.dataset_root)
    if args.replay_clips:
        manifest = json.loads((args.dataset_root / "manifest.json").read_text())
        shard = next(
            args.dataset_root / row["shard"]
            for row in manifest["sessions"]
            if row["released_clips"] >= 2
        )
        report["independent_replay"] = replay_report(shard, args.replay_clips)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
