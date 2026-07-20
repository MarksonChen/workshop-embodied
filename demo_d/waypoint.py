"""Drive the learned torque policy to waypoints with Demo B-style commands.

The waypoint controller is deliberately not learned.  It only converts the
current goal bearing into ``[dx_ego, dy_ego, dyaw]``; the Demo D policy must
turn that command into all 38 joint torques while remaining upright.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from demo_d.config import OUT
from demo_d.env import replace_command
from demo_d.runtime import load_policy, load_validation_environment, resolve_run, resolve_step


# Compact shapes fit a rodent moving near 0.1--0.2 m/s and keep the workshop
# rollout short.  Coordinates are offsets in the initial root frame.
SHAPES = {
    "square": ((0.24, 0.00), (0.24, 0.24), (0.00, 0.24), (0.00, 0.00)),
    "zigzag": ((0.20, 0.12), (0.40, -0.12), (0.60, 0.12), (0.78, 0.00)),
    "triangle": ((0.28, 0.00), (0.14, 0.24), (0.00, 0.00)),
}


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    """Return yaw from a MuJoCo ``[w, x, y, z]`` quaternion."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def steer(
    current_xy: np.ndarray,
    current_yaw: float,
    goal_xy: np.ndarray,
    forward_max: float = 0.08,
    turn_max: float = 0.30,
    stop_distance: float = 0.16,
) -> tuple[np.ndarray, float]:
    """Convert a world-frame waypoint into one bounded hindsight command."""
    delta = np.asarray(goal_xy) - np.asarray(current_xy)
    distance = float(np.linalg.norm(delta))
    bearing = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = (bearing - current_yaw + math.pi) % (2.0 * math.pi) - math.pi
    turn = float(np.clip(heading_error, -turn_max, turn_max))
    alignment = max(0.0, math.cos(heading_error))
    forward = forward_max * (0.25 + 0.75 * alignment) * min(1.0, distance / stop_distance)
    return np.asarray([forward, 0.0, turn], np.float32), distance


def world_goals(initial_xy: np.ndarray, initial_yaw: float, offsets) -> np.ndarray:
    """Rotate ego-frame shape offsets into the rollout's world frame."""
    rotation = np.asarray(
        [[math.cos(initial_yaw), -math.sin(initial_yaw)],
         [math.sin(initial_yaw), math.cos(initial_yaw)]],
        dtype=np.float32,
    )
    return np.asarray(initial_xy) + np.asarray(offsets, np.float32) @ rotation.T


def _plot(path: np.ndarray, goals: np.ndarray, reached: int, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=130)
    ax.plot(path[:, 0], path[:, 1], color="#35d0bf", linewidth=2.0, label="physical path")
    ax.scatter(*path[0], color="#35d0bf", edgecolor="white", s=48, zorder=4, label="start")
    for index, goal in enumerate(goals):
        color = "#79d38f" if index < reached else "#e0655e"
        ax.scatter(*goal, color=color, edgecolor="white", marker="*", s=150, zorder=5)
        ax.annotate(str(index + 1), goal, xytext=(7, 6), textcoords="offset points")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Demo D physical waypoint control ({reached}/{len(goals)})")
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="square")
    parser.add_argument("--max-steps", type=int, default=2000, help="100 Hz control steps")
    parser.add_argument("--reach", type=float, default=0.07)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    jax.random.key = jax.random.PRNGKey
    run = resolve_run(args.run)
    step_path = resolve_step(run, args.step)
    env = load_validation_environment()
    policy, _ = load_policy(env, step_path)
    reset = jax.jit(env.reset)
    command_step = jax.jit(env.command_step)
    state = reset(jax.random.PRNGKey(0), clip_idx=0, start_frame=0)

    initial_qpos = np.asarray(state.data.qpos)
    initial_yaw = yaw_from_quaternion(initial_qpos[3:7])
    goals = world_goals(initial_qpos[:2], initial_yaw, SHAPES[args.shape])
    goal_index = 0
    reached_log = []
    trajectory = [state]
    path = [initial_qpos[:2].copy()]
    command = np.asarray([0.08, 0.0, 0.0], np.float32)
    state = replace_command(state, jnp.asarray(command))
    rng = jax.random.PRNGKey(1)

    for control_step in range(args.max_steps):
        qpos = np.asarray(state.data.qpos)
        current_yaw = yaw_from_quaternion(qpos[3:7])
        command, distance = steer(qpos[:2], current_yaw, goals[goal_index])
        if distance < args.reach:
            reached_log.append(
                {"goal": goal_index + 1, "control_step": control_step, "distance_m": distance}
            )
            goal_index += 1
            if goal_index == len(goals):
                break
            command, _ = steer(qpos[:2], current_yaw, goals[goal_index])

        state = replace_command(state, jnp.asarray(command))
        rng, action_key = jax.random.split(rng)
        action, _ = policy(state.obs, action_key)
        state = command_step(state, action, jnp.asarray(command))
        trajectory.append(state)
        path.append(np.asarray(state.data.qpos[:2]))
        if float(state.done) > 0.5:
            break

    path_array = np.asarray(path)
    stem = f"waypoint-{args.shape}-{run.name}-{step_path.name}"
    plot_path = OUT / "waypoints" / f"{stem}.png"
    report_path = OUT / "waypoints" / f"{stem}.json"
    _plot(path_array, goals, goal_index, plot_path)
    report = {
        "run": str(run),
        "step": int(step_path.name),
        "shape": args.shape,
        "reached": goal_index,
        "total_goals": len(goals),
        "fell": bool(float(state.done) > 0.5),
        "control_steps": len(path) - 1,
        "reached_log": reached_log,
        "final_xy": path_array[-1].tolist(),
        "goals_xy": goals.tolist(),
        "minimum_distance_m": [
            float(np.linalg.norm(path_array - goal, axis=1).min()) for goal in goals
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"reached={goal_index}/{len(goals)} fell={report['fell']} "
        f"steps={report['control_steps']} plot={plot_path}"
    )
    print(f"Wrote {report_path}")

    if args.render:
        video_path = OUT / "videos" / f"{stem}.mp4"
        # The controller runs at 100 Hz; render every other state at 50 fps.
        frames = env.render_commands(trajectory[::2], width=640, height=480)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(video_path, fps=50) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Wrote {video_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
