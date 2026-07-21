"""Frozen P0 rollout gates in the exact Demo A Fetch physics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.envs import fetch

from demo_f.commands import hindsight_command
from demo_f.jax_features import contact_flags, transition_feature
from demo_f.kinematics import fetch_feet
from demo_h.config import (
    ACTION_DIM,
    BUFFER_FRAMES,
    COMMAND_HORIZON_FRAMES,
    COMMAND_HORIZON_SECONDS,
    DT,
    FETCH_FOOT_NAMES,
    HISTORY_TOKENS,
    OUT,
    PHASE_DIM,
    TARGET_SPEED_FETCH,
)
from demo_h.dataset.contract import DEFAULT_ROOT
from demo_h.prior import DEFAULT_PRIOR, load_prior


def _feature(sys, previous_qp, qp, info):
    previous_angles, _ = sys.joints[0].angle_vel(previous_qp)
    angles, _ = sys.joints[0].angle_vel(qp)
    previous_feet = fetch_feet(previous_angles)
    feet = fetch_feet(angles)
    foot_indices = jnp.asarray(
        tuple(sys.body.index[name] for name in FETCH_FOOT_NAMES)
    )
    contacts = contact_flags(info.contact.vel, foot_indices)
    return transition_feature(
        previous_qp.pos[0],
        qp.pos[0],
        previous_qp.rot[0],
        qp.rot[0],
        previous_angles,
        angles,
        previous_feet,
        feet,
        contacts,
    ), contacts


def make_rollout(sys, prior, steps: int):
    def rollout(initial_qp, feature_buffer, previous_control, command):
        initial_plan = jnp.zeros(
            (prior.metadata["config"]["latent_dim"],), dtype=jnp.float32
        )

        def step(carry, _):
            qp, buffer, previous, plan, phase = carry

            def refresh(_):
                tokens = prior.encode(buffer)
                return prior.predict_plan(tokens[-HISTORY_TOKENS:], command)

            plan = jax.lax.cond(phase == 0, refresh, lambda _: plan, operand=None)
            phase_vector = jax.nn.one_hot(phase, PHASE_DIM)
            mean = prior.action_mean(buffer[-1], plan, previous, phase_vector, command)
            control = jnp.tanh(mean)
            next_qp, info = sys.step(qp, control)
            feature, contacts = _feature(sys, qp, next_qp, info)
            buffer = jnp.roll(buffer, -1, axis=0).at[-1].set(feature)
            next_phase = (phase + 1) % PHASE_DIM
            return (
                next_qp,
                buffer,
                control,
                plan,
                next_phase,
            ), (next_qp, control, contacts, plan)

        return jax.lax.scan(
            step,
            (
                initial_qp,
                feature_buffer,
                previous_control,
                initial_plan,
                jnp.int32(0),
            ),
            xs=None,
            length=steps,
        )[1]

    return jax.jit(rollout)


def _qp_from_archive(env, archive, index: int, frame: int):
    qp = env.sys.default_qp().replace(
        pos=jnp.asarray(archive["initial_qp_pos"][index]),
        rot=jnp.asarray(archive["initial_qp_rot"][index]),
        vel=jnp.asarray(archive["initial_qp_vel"][index]),
        ang=jnp.asarray(archive["initial_qp_ang"][index]),
    )
    for control in archive["normalized_control"][index, :frame]:
        qp, _ = env.sys.step(qp, jnp.asarray(control))
    return qp


def in_support_reset(env, dataset_root: Path, command_target: np.ndarray):
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    frame = BUFFER_FRAMES - 1
    candidates = []
    for row in manifest["sessions"]:
        if row["split"] != "test" or not row["released_clips"]:
            continue
        path = dataset_root / row["shard"]
        with np.load(path) as archive:
            command = hindsight_command(
                archive["realized_root_position"],
                archive["realized_root_quaternion"],
                start=frame,
                future=frame + COMMAND_HORIZON_FRAMES,
            )
            distance = np.linalg.norm(command - command_target, axis=1)
            index = int(np.argmin(distance))
            candidates.append((float(distance[index]), row, path, index))
    _, row, path, index = min(candidates, key=lambda item: item[0])
    with np.load(path) as source:
        archive = {name: source[name] for name in source.files}
    qp = _qp_from_archive(env, archive, index, frame)
    buffer = jnp.asarray(archive["realized_features"][index, : frame + 1])
    previous = jnp.asarray(archive["normalized_control"][index, frame - 1])
    command = hindsight_command(
        archive["realized_root_position"][index : index + 1],
        archive["realized_root_quaternion"][index : index + 1],
        start=frame,
        future=frame + COMMAND_HORIZON_FRAMES,
    )[0]
    provenance = {
        "session": row["session"],
        "parent_clip_id": int(archive["parent_clip_id"][index]),
        "frame": frame,
        "command": command.tolist(),
        "target_command_error": float(np.linalg.norm(command - command_target)),
    }
    return qp, buffer, previous, jnp.asarray(command), provenance


def standing_reset(env, command: np.ndarray):
    state = env.reset(jax.random.PRNGKey(0))
    qp = state.qp
    angles, _ = env.sys.joints[0].angle_vel(qp)
    feet = fetch_feet(angles)
    foot_indices = jnp.asarray(
        tuple(env.sys.body.index[name] for name in FETCH_FOOT_NAMES)
    )
    contacts = state.obs[-qp.pos.shape[0] :][foot_indices]
    feature = transition_feature(
        qp.pos[0],
        qp.pos[0],
        qp.rot[0],
        qp.rot[0],
        angles,
        angles,
        feet,
        feet,
        contacts,
    )
    return (
        qp,
        jnp.repeat(feature[None], BUFFER_FRAMES, axis=0),
        jnp.zeros(ACTION_DIM),
        jnp.asarray(command),
        {"command": np.asarray(command).tolist()},
    )


def summarize(initial_qp, stream) -> tuple[dict, dict[str, np.ndarray]]:
    qps, controls, contacts, plans = stream
    qps = jax.tree_util.tree_map(np.asarray, qps)
    controls, contacts, plans = map(np.asarray, (controls, contacts, plans))
    root = np.concatenate((np.asarray(initial_qp.pos[0])[None], qps.pos[:, 0]))
    quaternion = np.concatenate((np.asarray(initial_qp.rot[0])[None], qps.rot[:, 0]))
    speed = np.diff(root[:, 0]) / DT
    upright = 1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2)
    alive = np.cumprod(
        ((root[:, 2] >= 0.6875) & (upright >= 0.0)).astype(np.float32)
    )
    alive_steps = max(int(alive[1:].sum()), 1)
    switches = np.abs(np.diff(contacts.astype(np.float32), axis=0)).sum(axis=0)
    report = {
        "steps": len(controls),
        "survival_fraction": float(alive[1:].mean()),
        "mean_speed_alive": float((speed * alive[1:]).sum() / alive_steps),
        "forward_displacement": float(root[-1, 0] - root[0, 0]),
        "minimum_height": float(root[:, 2].min()),
        "minimum_upright": float(upright.min()),
        "action_rms": float(np.sqrt(np.mean(np.square(controls)))),
        "action_saturation_fraction": float(np.mean(np.abs(controls) >= 0.999)),
        "contact_switches_per_foot": switches.tolist(),
        "passes_survival": bool(alive[-1] > 0),
        "passes_locomotion": bool(
            alive[-1] > 0 and root[-1, 0] - root[0, 0] > 0.25
        ),
    }
    trace = {
        "qp_pos": qps.pos,
        "qp_rot": qps.rot,
        "qp_vel": qps.vel,
        "qp_ang": qps.ang,
        "initial_qp_pos": np.asarray(initial_qp.pos),
        "initial_qp_rot": np.asarray(initial_qp.rot),
        "initial_qp_vel": np.asarray(initial_qp.vel),
        "initial_qp_ang": np.asarray(initial_qp.ang),
        "controls": controls,
        "contacts": contacts.astype(np.uint8),
        "plans": plans,
        "speed": speed,
        "height": root[:, 2],
        "upright": upright,
    }
    return report, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED_FETCH)
    parser.add_argument("--output-dir", type=Path, default=OUT / "p0")
    args = parser.parse_args()
    prior = load_prior(args.prior)
    env = fetch.Fetch()
    target = np.asarray(
        (args.target_speed * COMMAND_HORIZON_SECONDS, 0.0, 0.0), np.float32
    )
    resets = {
        "in_support": in_support_reset(env, args.dataset_root, target),
        "standing": standing_reset(env, target),
    }
    rollout = make_rollout(env.sys, prior, args.steps)
    report = {
        "schema": "demo-h-p0-evaluation-v1",
        "steps": args.steps,
        "prior": str(args.prior),
        "target_speed": args.target_speed,
        "command": target.tolist(),
        "resets": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (qp, buffer, previous, command, provenance) in resets.items():
        started = time.perf_counter()
        stream = rollout(qp, buffer, previous, command)
        jax.block_until_ready(stream[1])
        metrics, trace = summarize(qp, stream)
        metrics["compile_and_rollout_seconds"] = time.perf_counter() - started
        metrics["provenance"] = provenance
        report["resets"][name] = metrics
        np.savez_compressed(args.output_dir / f"{name}.npz", **trace)
        print(f"{name}: {json.dumps(metrics)}", flush=True)
    report["passes_in_support_gate"] = report["resets"]["in_support"][
        "passes_locomotion"
    ]
    report["passes_workshop_gate"] = report["resets"]["standing"][
        "passes_locomotion"
    ]
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
