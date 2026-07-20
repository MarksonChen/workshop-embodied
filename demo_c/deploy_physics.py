"""Zero-shot reality check: drive MIMIC-MJX with a dream-trained navigator.

The Demo C policy emits a 0.64-s displacement command. The existing frozen joystick
stack converts it to velocity commands and then to 16-D intentions/torques in real
MJX physics. No policy parameter is updated here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from demo_c.config import TASK
from demo_c.motor import FrozenMotor
from demo_c.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out" / "physics"
BRIDGE_CALIBRATION = OUT / "bridge_calibration.json"
DEFAULT_JOYSTICK = ROOT / "rl" / "runs" / "RodentJoystick-highlvl-20260718-012844-963586" / "52428800"


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def make_base_observation(xy, yaw, goal, body_velocity, previous_action):
    dxy = goal - xy
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = np.array([c * dxy[0] - s * dxy[1], s * dxy[0] + c * dxy[1]], np.float32)
    velocity = np.clip(body_velocity / np.array([0.35, 0.20, 1.2], np.float32), -2, 2)
    return np.concatenate(
        (local / TASK.goal_radius_max, [np.linalg.norm(dxy) / TASK.goal_radius_max], velocity, previous_action)
    ).astype(np.float32)


def calibrated_forward_velocity(displacement, calibration):
    """Invert the measured settled, zero-turn joystick response curve.

    The checkpoint has a pronounced low-speed dead zone, so a single scalar gain is
    not an adequate bridge. Monotonic accumulation removes measurement wiggles; for a
    flat response we retain the largest tested velocity before motion begins.
    """
    rows = sorted(
        (
            row for row in calibration["rows"]
            if not row["fell"] and abs(row["vyaw"]) < 1e-8
        ),
        key=lambda row: row["vx"],
    )
    if len(rows) < 2:
        raise ValueError("bridge calibration needs at least two zero-turn measurements")
    actual = np.maximum.accumulate(np.maximum([row["forward"] for row in rows], 0.0))
    # Dictionary assignment intentionally keeps the last/largest velocity on a
    # dead-zone plateau.
    inverse = {float(response): float(row["vx"]) for response, row in zip(actual, rows)}
    items = sorted(inverse.items())
    responses = np.asarray([item[0] for item in items], np.float32)
    velocities = np.asarray([item[1] for item in items], np.float32)
    return float(np.interp(float(displacement), responses, velocities))


class PhysicsRuntime:
    """Existing MIMIC decoder + high-level joystick checkpoint, loaded lazily."""

    def __init__(self, checkpoint: Path):
        import jax
        jax.random.key = jax.random.PRNGKey
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from brax.training.acme import running_statistics, specs as acme_specs
        from brax.training.agents.ppo import networks as ppo_networks
        from rl.render_joystick import build_env

        self.jax = jax; self.jnp = jnp
        self.env = build_env(16); self.inner = self.env.env
        self.reset_jit = jax.jit(self.env.reset); self.step_jit = jax.jit(self.env.step)
        initial = self.reset_jit(jax.random.key(0))
        obs_size = jax.tree.map(lambda x: x.shape[-1], initial.obs)
        network = ppo_networks.make_ppo_networks(
            obs_size,
            self.env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=(1024, 512, 256),
            value_hidden_layer_sizes=(1024, 512, 256),
            policy_obs_key="state",
            value_obs_key="state",
        )
        norm_target = running_statistics.init_state(
            {"state": acme_specs.Array((obs_size["state"],), jnp.float32)}
        )
        target = (
            norm_target,
            network.policy_network.init(jax.random.key(0)),
            network.value_network.init(jax.random.key(0)),
        )
        restored = ocp.PyTreeCheckpointer().restore(str(checkpoint.resolve()), item=target)
        inference = ppo_networks.make_inference_fn(network)
        policy_raw = inference((restored[0], restored[1]), deterministic=True)
        self.policy = jax.jit(policy_raw)
        steps_per_command = round(TASK.step_seconds / float(self.inner.dt))

        def command_rollout(state, rng, command):
            state = state.replace(info={**state.info, "command": command})

            def body(carry, _):
                current, key = carry
                key, action_key = jax.random.split(key)
                intention, _ = policy_raw(current.obs, action_key)
                current = self.env.step(current, intention)
                current = current.replace(info={**current.info, "command": command})
                return (current, key), (current.data.qpos, current.done)

            (state, rng), trace = jax.lax.scan(body, (state, rng), None, length=steps_per_command)
            return state, rng, trace

        self.command_jit = jax.jit(command_rollout)

    def reset(self, seed):
        return self.reset_jit(self.jax.random.key(seed))

    def pin(self, state, command):
        return state.replace(info={**state.info, "command": self.jnp.asarray(command)})

    def run_command(self, state, command, rng, frames=None):
        state, rng, (qpos, _done) = self.command_jit(state, rng, self.jnp.asarray(command))
        if frames is not None:
            frames.extend(np.asarray(qpos))
        return state, rng


@torch.inference_mode()
def run_episode(runtime, motor, policy, payload, goal_offset, seed, capture=False):
    state = runtime.reset(seed)
    q0 = np.asarray(state.data.qpos); xy0 = q0[:2].copy(); yaw0 = quat_yaw(q0[3:7])
    goal = xy0 + np.asarray(goal_offset, np.float32)
    history = motor.sample_histories(1, torch.Generator(device="cpu").manual_seed(seed))
    previous_action = np.zeros(2, np.float32); body_velocity = np.zeros(3, np.float32)
    rng = runtime.jax.random.key(seed + 1000)
    frames = [q0.copy()] if capture else None
    actions, requested_commands, velocity_commands, distances = [], [], [], [float(np.linalg.norm(goal - xy0))]
    fell = False

    for _ in range(TASK.horizon):
        q_before = np.asarray(state.data.qpos); xy_before = q_before[:2].copy(); yaw_before = quat_yaw(q_before[3:7])
        context = motor.context(history)
        base = make_base_observation(xy_before, yaw_before, goal, body_velocity, previous_action)
        observation = torch.as_tensor(base[None])
        if payload["variant"] == "wam":
            observation = torch.cat((observation, context), -1)
        action = policy.act(observation, deterministic=True)[0]
        command_raw = motor.action_to_command(action)[0].cpu().numpy()
        bridge_mode = payload.get("bridge_mode", "calibrated")
        gains = payload.get("bridge_gains", {"forward_gain": 1.0, "turn_gain": 1.0})
        if bridge_mode == "decoded":
            history, dream_delta_t, dream_invalid = motor.advance(history, action, context, decode=True)
            dream_delta = dream_delta_t[0].cpu().numpy()
            if bool(dream_invalid[0]):
                break
            velocity_command = np.array(
                [max(0.0, dream_delta[0]) / (TASK.step_seconds * gains["forward_gain"]),
                 dream_delta[2] / (TASK.step_seconds * gains["turn_gain"])], np.float32
            )
        else:
            history = motor.advance(history, action, context, decode=False)[0]
            if bridge_mode == "calibrated":
                forward_velocity = calibrated_forward_velocity(
                    command_raw[0], payload["bridge_calibration"]
                )
                turn_gain = gains["turn_gain"]
            else:
                forward_gain = gains["forward_gain"] if bridge_mode == "inverse_gain" else 1.0
                turn_gain = gains["turn_gain"] if bridge_mode == "inverse_gain" else 1.0
                forward_velocity = command_raw[0] / (TASK.step_seconds * forward_gain)
            velocity_command = np.array(
                [forward_velocity, command_raw[2] / (TASK.step_seconds * turn_gain)],
                np.float32,
            )
        velocity_command[0] = np.clip(velocity_command[0], 0.0, 0.5)
        velocity_command[1] = np.clip(velocity_command[1], -1.0, 1.0)
        state, rng = runtime.run_command(state, velocity_command, rng, frames)

        q_after = np.asarray(state.data.qpos); world_delta = q_after[:2] - xy_before
        c, s = math.cos(-yaw_before), math.sin(-yaw_before)
        local_delta = np.array(
            [c * world_delta[0] - s * world_delta[1], s * world_delta[0] + c * world_delta[1]], np.float32
        )
        body_velocity = np.array(
            [local_delta[0], local_delta[1], wrap(quat_yaw(q_after[3:7]) - yaw_before)], np.float32
        ) / TASK.step_seconds
        previous_action = action[0].cpu().numpy(); actions.append(previous_action.tolist())
        requested_commands.append(command_raw.tolist())
        velocity_commands.append(velocity_command.tolist())
        distance = float(np.linalg.norm(goal - q_after[:2])); distances.append(distance)
        fell = bool(np.asarray(state.done))
        if fell or distance < TASK.reach_radius:
            break

    q_final = np.asarray(state.data.qpos)
    path = np.asarray(frames)[:, :2] if frames is not None else np.stack((xy0, q_final[:2]))
    return {
        "variant": payload["variant"],
        "seed": seed,
        "goal": goal.tolist(),
        "goal_offset": list(map(float, goal_offset)),
        "success": distances[-1] < TASK.reach_radius and not fell,
        "fell": fell,
        "steps": len(actions),
        "initial_distance": distances[0],
        "final_distance": distances[-1],
        "path_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "actions": actions,
        "requested_displacements": requested_commands,
        "velocity_commands": velocity_commands,
        "bridge_mode": payload.get("bridge_mode", "calibrated"),
        "bridge_gains": payload.get("bridge_gains", {"forward_gain": 1.0, "turn_gain": 1.0}),
        "distances": distances,
        "qpos": frames,
    }


def render_episode(runtime, result, path):
    import imageio.v2 as imageio
    import mujoco
    qpos = result["qpos"]
    if qpos is None:
        return
    model = runtime.inner.mj_model
    cameras = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    camera = next((name for name in cameras if name and "close_profile" in name), -1)
    renderer = mujoco.Renderer(model, height=480, width=640); data = mujoco.MjData(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=round(1 / float(runtime.inner.dt))) as writer:
        for q in qpos:
            data.qpos[:] = q; mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera); writer.append_data(renderer.render())
    renderer.close()


def plot_paths(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for result in results:
        qpos = result["qpos"]
        if qpos is None:
            continue
        xy = np.asarray(qpos)[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], label=f"{result['variant']} ({result['final_distance']:.2f} m)")
        ax.scatter(*result["goal"], marker="*", s=100)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set(xlabel="x (m)", ylabel="y (m)", title="zero-shot MJX navigation")
    ax.legend(frameon=False); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", type=Path, default=[
        ROOT / "demo_c/out/checkpoints/goal_only_seed0.pt",
        ROOT / "demo_c/out/checkpoints/wam_seed0.pt",
    ])
    parser.add_argument("--joystick", type=Path, default=DEFAULT_JOYSTICK)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--goal-index",
        type=int,
        choices=range(8),
        help="run one goal from the frozen eight-episode suite (useful for a reproducible render)",
    )
    parser.add_argument(
        "--bridge",
        choices=("raw", "inverse_gain", "calibrated", "decoded"),
        default="calibrated",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    runtime = PhysicsRuntime(args.joystick); motor = FrozenMotor("cpu")
    policies = [load_policy(path, "cpu") for path in args.policies]
    for _, payload in policies:
        expected_world = payload.get("metrics", {}).get("world_checkpoint")
        if expected_world and Path(expected_world).resolve() != motor.checkpoint:
            raise SystemExit(
                f"policy expects world checkpoint {expected_world}, but deployment loaded {motor.checkpoint}"
            )
        payload["bridge_mode"] = args.bridge
    if args.bridge != "raw" and not BRIDGE_CALIBRATION.exists():
        raise SystemExit(
            f"{args.bridge} bridge requires {BRIDGE_CALIBRATION}; "
            "run python -m demo_c.calibrate_bridge first"
        )
    if BRIDGE_CALIBRATION.exists():
        calibration = json.loads(BRIDGE_CALIBRATION.read_text())
        for _, payload in policies:
            payload["bridge_gains"] = {key: calibration[key] for key in ("forward_gain", "turn_gain")}
            payload["bridge_calibration"] = calibration
    n = 1 if args.smoke or args.goal_index is not None else args.episodes
    rng = np.random.default_rng(20260719)
    generated = 8 if args.goal_index is not None else n
    radii = rng.uniform(TASK.goal_radius_min, TASK.goal_radius_max, generated)
    bearings = rng.uniform(-TASK.goal_bearing_max, TASK.goal_bearing_max, generated)
    goals = np.stack((radii * np.cos(bearings), radii * np.sin(bearings)), -1)
    indexed_goals = (
        [(args.goal_index, goals[args.goal_index])]
        if args.goal_index is not None
        else list(enumerate(goals))
    )
    all_results, captured = [], []
    for model, payload in policies:
        for position, (episode, goal) in enumerate(indexed_goals):
            capture = args.render and position == 0
            result = run_episode(runtime, motor, model, payload, goal, episode, capture)
            if capture:
                render_episode(runtime, result, OUT / f"{payload['variant']}_episode{episode}.mp4")
                captured.append({**result, "qpos": result["qpos"]})
            result["qpos"] = None
            all_results.append(result)
            print(
                f"{payload['variant']:>9} ep{episode}: success={result['success']} fell={result['fell']} "
                f"distance={result['final_distance']:.3f} m",
                flush=True,
            )
    summary = {}
    for _, payload in policies:
        rows = [r for r in all_results if r["variant"] == payload["variant"]]
        summary[payload["variant"]] = {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "fall_rate": float(np.mean([r["fell"] for r in rows])),
            "final_distance_mean": float(np.mean([r["final_distance"] for r in rows])),
            "episodes": len(rows),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep the frozen eight-episode benchmark stable. Smoke tests and workshop
    # renders are useful diagnostics, but must not silently overwrite it.
    if (
        not args.smoke
        and not args.render
        and args.goal_index is None
        and args.episodes == 8
        and args.bridge == "calibrated"
    ):
        metrics_name = "metrics.json"
    else:
        mode = "smoke" if args.smoke else "render" if args.render else "evaluation"
        metrics_name = f"metrics_{mode}_{n}ep_{args.bridge}.json"
    (OUT / metrics_name).write_text(
        json.dumps({"summary": summary, "episodes": all_results}, indent=2) + "\n"
    )
    if captured:
        # Restore captured arrays only for plotting; JSON remains compact.
        plot_paths(captured, OUT / "paths.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
