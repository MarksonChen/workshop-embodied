"""Retarget continuous Coltrane locomotion clips onto Brax Fetch.

The source-to-target correspondence is deliberately sparse: trunk frame plus
four distal feet.  Source bone rotations are never copied.  Instead, bounded
sequence-level IK matches semantic foot trajectories while regularizing pose,
velocity, and acceleration.  Detected stance runs are pinned in world space to
remove marker jitter and foot skate before IK.

Run from the repository root:

    uv run --extra workshop python -m demo_f.retarget
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.signal import savgol_filter

from .config import (
    ANIMAL,
    DATA_ROOT,
    FETCH_NOMINAL_FEET,
    FETCH_STAND_HEIGHT,
    FETCH_TRUNK_LENGTH,
    FPS,
    INSPECTION_CLIPS,
    JOINT_LIMIT,
    OUT,
    SOURCE_FEET,
    ClipSpec,
    RetargetConfig,
)
from .kinematics import fetch_feet


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-8)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Vectorized rotation matrix to scalar-first quaternion."""

    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.concatenate((xyzw[:, 3:4], xyzw[:, :3]), axis=-1).astype(np.float32)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(begin), int(end)) for begin, end in edges.reshape(-1, 2)]


def prepare_source(
    keypoints: np.ndarray,
    names: list[str],
    root_qpos: np.ndarray,
    config: RetargetConfig,
) -> dict:
    """Build semantic Fetch targets from one in-memory rodent clip.

    ``keypoints`` must be ``(frames, 23, 3)`` in metres.  Keeping raw HDF5 I/O
    outside this function lets the standalone dataset builder process each
    session once and retarget many clips efficiently.
    """

    keypoints = np.asarray(keypoints, dtype=np.float64)
    root_qpos = np.asarray(root_qpos, dtype=np.float64)
    frames = len(keypoints)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 3:
        raise ValueError(f"expected (frames, keypoints, xyz), got {keypoints.shape}")
    if root_qpos.shape != (frames, 7):
        raise ValueError(f"expected root qpos {(frames, 7)}, got {root_qpos.shape}")
    if frames < config.smoothing_window:
        raise ValueError("clip is shorter than the smoothing window")
    keypoints = savgol_filter(
        keypoints,
        config.smoothing_window,
        2,
        axis=0,
        mode="interp",
    )
    index = {name: names.index(name) for name in names}
    origin = keypoints[:, index["SpineM"]]
    forward = _normalize(keypoints[:, index["SpineF"]] - keypoints[:, index["SpineL"]])
    lateral = (
        keypoints[:, index["ShoulderL"]]
        - keypoints[:, index["ShoulderR"]]
        + keypoints[:, index["HipL"]]
        - keypoints[:, index["HipR"]]
    )
    lateral = _normalize(lateral - (lateral * forward).sum(-1, keepdims=True) * forward)
    up = _normalize(np.cross(forward, lateral))
    flip = up[:, 2] < 0
    lateral[flip] *= -1
    up[flip] *= -1
    lateral = _normalize(np.cross(up, forward))
    body_rotation = np.stack((forward, lateral, up), axis=-1)
    body_local = np.einsum("tji,tkj->tki", body_rotation, keypoints - origin[:, None])

    foot_index = np.asarray([index[name] for name in SOURCE_FEET])
    feet_world = keypoints[:, foot_index]
    feet_local = body_local[:, foot_index]
    trunk_length = float(
        np.median(
            np.linalg.norm(
                keypoints[:, index["SpineF"]] - keypoints[:, index["SpineL"]],
                axis=-1,
            )
        )
    )
    ground = float(np.quantile(feet_world[..., 2], 0.05))
    root_height = float(np.median(origin[:, 2] - ground))
    longitudinal_scale = FETCH_TRUNK_LENGTH / trunk_length
    vertical_scale = FETCH_STAND_HEIGHT / root_height

    # Root trajectory is uniformly body-length scaled and initially faces +x.
    initial_yaw = math.atan2(forward[0, 1], forward[0, 0])
    cosine, sine = math.cos(-initial_yaw), math.sin(-initial_yaw)
    align2 = np.asarray(((cosine, -sine), (sine, cosine)))
    root_xy = (origin[:, :2] - origin[:1, :2]) @ align2.T * longitudinal_scale
    root_z = FETCH_STAND_HEIGHT + (
        origin[:, 2] - np.median(origin[:, 2])
    ) * vertical_scale
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0])) - initial_yaw
    root_rotation = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    root_rotation[:, 0, 0] = np.cos(yaw)
    root_rotation[:, 0, 1] = -np.sin(yaw)
    root_rotation[:, 1, 0] = np.sin(yaw)
    root_rotation[:, 1, 1] = np.cos(yaw)
    root_rotation[:, 2, 2] = 1.0
    root_position = np.column_stack((root_xy, root_z))

    nominal = np.asarray(FETCH_NOMINAL_FEET, dtype=np.float64)
    target_local = nominal[None].repeat(frames, axis=0)
    # Match gait excursion around each foot's median rather than forcing rat and
    # Fetch proportions to be identical.  The static target morphology remains
    # exactly Fetch; stride phase and amplitude come from the rat.
    target_local[..., 0] += longitudinal_scale * (
        feet_local[..., 0] - np.median(feet_local[..., 0], axis=0, keepdims=True)
    )
    target_local[..., 1] += (
        config.lateral_residual_scale
        * longitudinal_scale
        * (feet_local[..., 1] - np.median(feet_local[..., 1], axis=0, keepdims=True))
    )
    target_foot_height = np.maximum(feet_world[..., 2] - ground, 0.0) * vertical_scale
    target_local[..., 2] = target_foot_height - root_z[:, None]

    source_velocity = np.zeros((frames, 4), dtype=np.float64)
    source_velocity[1:] = np.linalg.norm(np.diff(feet_world[..., :2], axis=0), axis=-1) * FPS
    contacts = (
        (feet_world[..., 2] <= ground + config.contact_height_m)
        & (source_velocity <= config.contact_speed_mps)
    )

    # Pin every multi-frame stance run in target world space, then transform the
    # corrected trajectory back into the instantaneous Fetch torso frame.
    target_world = root_position[:, None] + np.einsum(
        "tij,tfj->tfi", root_rotation, target_local
    )
    for foot in range(4):
        for begin, end in _runs(contacts[:, foot]):
            if end - begin < 2:
                contacts[begin:end, foot] = False
                continue
            target_world[begin:end, foot, :2] = np.median(
                target_world[begin:end, foot, :2], axis=0
            )
            target_world[begin:end, foot, 2] = 0.0
    target_local = np.einsum(
        "tji,tfj->tfi", root_rotation, target_world - root_position[:, None]
    )

    duration = (frames - 1) / FPS
    measured_speed = float(np.linalg.norm(root_qpos[-1, :2] - root_qpos[0, :2]) / duration)
    path_speed = float(np.linalg.norm(np.diff(root_qpos[:, :2], axis=0), axis=-1).mean() * FPS)
    return {
        "keypoints": keypoints.astype(np.float32),
        "keypoint_names": np.asarray(names),
        "root_position": root_position.astype(np.float32),
        "root_quaternion": _rotation_to_quaternion(root_rotation),
        "yaw": yaw.astype(np.float32),
        "target_feet_local": target_local.astype(np.float32),
        "contacts": contacts,
        "source_speed": measured_speed,
        "source_path_speed": path_speed,
        "longitudinal_scale": longitudinal_scale,
        "vertical_scale": vertical_scale,
        "ground_m": ground,
        "trunk_length_m": trunk_length,
    }


def load_source(spec: ClipSpec, config: RetargetConfig) -> dict:
    """Load one provenance-pinned inspection clip from raw Aldarondo data."""

    path = DATA_ROOT / ANIMAL / f"{spec.session}.h5"
    if not path.exists():
        raise FileNotFoundError(f"missing source session {path}; set ALDARONDO_ROOT")
    stop = spec.start + spec.frames
    with h5py.File(path, "r") as source:
        dataset = source["/pose/keypoints"]
        names = [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in dataset.attrs["names"]
        ]
        keypoints = np.transpose(dataset[spec.start:stop], (0, 2, 1)) / 1_000.0
        root_qpos = source["/pose/qpos"][spec.start:stop, :7]
    if len(keypoints) != spec.frames:
        raise ValueError(f"{spec}: requested frames exceed source session")
    return prepare_source(keypoints, names, root_qpos, config)


def optimize_batch(
    targets: np.ndarray,
    contacts: np.ndarray,
    root_positions: np.ndarray,
    yaw: np.ndarray,
    config: RetargetConfig,
) -> tuple[np.ndarray, list[float]]:
    """Bounded sequence IK for ``(clips, frames, four feet, xyz)`` targets."""

    target = jnp.asarray(targets)
    contact = jnp.asarray(contacts)
    root_position = jnp.asarray(root_positions)
    yaw = jnp.asarray(yaw)
    coordinate_weights = jnp.asarray(config.foot_weights)
    raw = jnp.zeros(target.shape[:2] + (10,), dtype=jnp.float32)
    optimizer = optax.adam(config.learning_rate)
    state = optimizer.init(raw)

    def objective(unbounded):
        angles = JOINT_LIMIT * jnp.tanh(unbounded)
        actual = fetch_feet(angles)
        error = actual - target
        target_weight = 1.0 + config.contact_target_weight * contact[..., None]
        foot_loss = jnp.mean(jnp.square(error) * coordinate_weights * target_weight)

        cosine, sine = jnp.cos(yaw), jnp.sin(yaw)
        world_x = root_position[..., None, 0] + cosine[..., None] * actual[..., 0] - sine[..., None] * actual[..., 1]
        world_y = root_position[..., None, 1] + sine[..., None] * actual[..., 0] + cosine[..., None] * actual[..., 1]
        world_z = root_position[..., None, 2] + actual[..., 2]
        world = jnp.stack((world_x, world_y, world_z), axis=-1)
        stance_pair = contact[:, 1:] & contact[:, :-1]
        stance_count = jnp.maximum(jnp.sum(stance_pair), 1)
        stance_delta = world[:, 1:] - world[:, :-1]
        stance_velocity_loss = jnp.sum(
            jnp.square(stance_delta) * stance_pair[..., None]
        ) / (3.0 * stance_count)
        contact_count = jnp.maximum(jnp.sum(contact), 1)
        contact_height_loss = jnp.sum(
            jnp.square(world_z) * contact
        ) / contact_count
        pose_loss = jnp.mean(jnp.square(angles))
        velocity_loss = jnp.mean(jnp.square(jnp.diff(angles, axis=1)))
        acceleration_loss = jnp.mean(jnp.square(jnp.diff(angles, n=2, axis=1)))
        return (
            foot_loss
            + config.stance_velocity_weight * stance_velocity_loss
            + config.contact_height_weight * contact_height_loss
            + config.pose_weight * pose_loss
            + config.velocity_weight * velocity_loss
            + config.acceleration_weight * acceleration_loss
        )

    @jax.jit
    def update(unbounded, optimizer_state):
        loss, gradient = jax.value_and_grad(objective)(unbounded)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, unbounded)
        return optax.apply_updates(unbounded, updates), optimizer_state, loss

    losses = []
    for step in range(config.optimizer_steps):
        raw, state, loss = update(raw, state)
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == config.optimizer_steps:
            value = float(loss)
            losses.append(value)
            print(f"[sequence IK] {step + 1:4d}/{config.optimizer_steps} loss={value:.7f}", flush=True)
    return np.asarray(JOINT_LIMIT * jnp.tanh(raw), dtype=np.float32), losses


def _world_feet(root_position, yaw, local_feet):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.zeros((len(yaw), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = -sine
    rotation[:, 1, 0] = sine
    rotation[:, 1, 1] = cosine
    rotation[:, 2, 2] = 1.0
    return root_position[:, None] + np.einsum("tij,tfj->tfi", rotation, local_feet)


def diagnostics(spec: ClipSpec, source: dict, angles: np.ndarray, loss_trace: list[float]) -> dict:
    actual_local = np.asarray(fetch_feet(jnp.asarray(angles)))
    target_local = source["target_feet_local"]
    error = actual_local - target_local
    world = _world_feet(source["root_position"], source["yaw"], actual_local)
    speed = np.zeros(world.shape[:2], np.float32)
    speed[1:] = np.linalg.norm(np.diff(world, axis=0), axis=-1) * FPS
    contacts = source["contacts"]
    stance_pair = contacts[1:] & contacts[:-1]
    contact_speed = speed[1:][stance_pair]
    contact_height = np.abs(world[..., 2][contacts])
    duration = (spec.frames - 1) / FPS
    root_speed = float(
        np.linalg.norm(source["root_position"][-1, :2] - source["root_position"][0, :2])
        / duration
    )
    return {
        "label": spec.label,
        "source": f"{ANIMAL}/{spec.session}.h5",
        "start": spec.start,
        "frames": spec.frames,
        "duration_s": duration,
        "source_speed_mps": source["source_speed"],
        "retarget_root_speed_fetch_units_per_s": root_speed,
        "longitudinal_scale": source["longitudinal_scale"],
        "vertical_scale": source["vertical_scale"],
        "source_trunk_length_m": source["trunk_length_m"],
        "source_ground_m": source["ground_m"],
        "ik_foot_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "ik_foot_rmse_x": float(np.sqrt(np.mean(np.square(error[..., 0])))),
        "ik_foot_rmse_y": float(np.sqrt(np.mean(np.square(error[..., 1])))),
        "ik_foot_rmse_z": float(np.sqrt(np.mean(np.square(error[..., 2])))),
        "contact_fraction": float(contacts.mean()),
        "contact_speed_mean": float(contact_speed.mean()) if len(contact_speed) else math.nan,
        "contact_speed_p95": float(np.quantile(contact_speed, 0.95)) if len(contact_speed) else math.nan,
        "contact_height_mean": float(contact_height.mean()) if len(contact_height) else math.nan,
        "minimum_foot_height": float(world[..., 2].min()),
        "maximum_abs_joint_degrees": float(np.degrees(np.abs(angles)).max()),
        "joint_limit_fraction": float((np.abs(angles) > 0.99 * JOINT_LIMIT).mean()),
        "loss_trace": loss_trace,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=RetargetConfig.optimizer_steps)
    parser.add_argument("--learning-rate", type=float, default=RetargetConfig.learning_rate)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="subset of v0100 v0150 v0200 v0217 (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = RetargetConfig(optimizer_steps=args.steps, learning_rate=args.learning_rate)
    selected = [spec for spec in INSPECTION_CLIPS if args.labels is None or spec.label in args.labels]
    if not selected:
        raise SystemExit("no inspection clips selected")
    sources = []
    for spec in selected:
        source = load_source(spec, config)
        sources.append(source)
        print(
            f"[{spec.label}] {spec.session}@{spec.start} | "
            f"measured={source['source_speed']:.3f} m/s | "
            f"scale={source['longitudinal_scale']:.2f}x | "
            f"contact={source['contacts'].mean():.1%}",
            flush=True,
        )
    targets = np.stack([source["target_feet_local"] for source in sources])
    contacts = np.stack([source["contacts"] for source in sources])
    root_positions = np.stack([source["root_position"] for source in sources])
    yaws = np.stack([source["yaw"] for source in sources])
    angles, loss_trace = optimize_batch(
        targets,
        contacts,
        root_positions,
        yaws,
        config,
    )

    output_dir = OUT / "retarget"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch, (spec, source) in enumerate(zip(selected, sources, strict=True)):
        actual_local = np.asarray(fetch_feet(jnp.asarray(angles[batch])), dtype=np.float32)
        row = diagnostics(spec, source, angles[batch], loss_trace)
        rows.append(row)
        artifact = output_dir / f"{spec.label}.npz"
        np.savez_compressed(
            artifact,
            angles=angles[batch],
            root_position=source["root_position"],
            root_quaternion=source["root_quaternion"],
            yaw=source["yaw"],
            target_feet_local=source["target_feet_local"],
            actual_feet_local=actual_local,
            contacts=source["contacts"],
            source_keypoints=source["keypoints"],
            source_keypoint_names=source["keypoint_names"],
            source_speed_mps=np.float32(source["source_speed"]),
            source_session=np.asarray(spec.session),
            source_start=np.int64(spec.start),
            fps=np.int32(FPS),
        )
        print(
            f"[{spec.label}] wrote {artifact} | foot RMSE={row['ik_foot_rmse']:.4f} | "
            f"limit={row['joint_limit_fraction']:.1%}",
            flush=True,
        )
    report = {
        "status": "inspection_required",
        "animal": ANIMAL,
        "retarget_config": asdict(config),
        "clips": rows,
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
