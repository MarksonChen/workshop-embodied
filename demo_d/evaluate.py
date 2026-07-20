"""Evaluate natural imitation and direct command control on fixed held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_d.config import EVAL, OUT, VAL_CLIPS
from demo_d.env import COMMAND_SIZE, local_planar_velocity, quaternion_yaw
from demo_d.metrics import IMITATION_REWARD_CEILING, command_tracking_score, reportability
from demo_d.runtime import load_policy, load_validation_environment, resolve_run, resolve_step


def _batched_functions(env):
    reset = jax.jit(
        jax.vmap(lambda key, clip, start: env.reset(key, clip_idx=clip, start_frame=start))
    )
    imitation_step = jax.jit(jax.vmap(env.step))
    command_step = jax.jit(jax.vmap(env.command_step))
    return reset, imitation_step, command_step


def _reset_batch(reset, clip_ids, start_frames):
    keys = jax.random.split(jax.random.PRNGKey(123), len(clip_ids))
    return reset(keys, jnp.asarray(clip_ids), jnp.asarray(start_frames))


def evaluate_imitation(env, policy, runtime=None, steps: int = EVAL.imitation_steps) -> dict:
    n = len(VAL_CLIPS)
    reset_fn, step_fn, _ = runtime or _batched_functions(env)
    state = _reset_batch(reset_fn, np.arange(n), np.zeros(n, np.int32))
    alive = jnp.ones(n, dtype=bool)
    returns = jnp.zeros(n)
    alive_steps = jnp.zeros(n)
    joint_error = jnp.zeros(n)
    root_error = jnp.zeros(n)
    rng = jax.random.PRNGKey(456)

    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step_fn(state, action)
        alive = jnp.logical_and(alive, state.done < 0.5)
        valid = alive.astype(jnp.float32)
        # Keep the natural-imitation score comparable to pipeline v3 by
        # removing v4's explicit command-grounding reward.
        imitation_reward = state.reward - state.metrics["rewards/command_tracking"]
        returns += jnp.clip(imitation_reward, 0.0, IMITATION_REWARD_CEILING) * valid
        joint_error += jnp.where(
            alive, jnp.nan_to_num(state.metrics["joint_l2_error"]), 0.0
        )
        root_error += jnp.where(
            alive, jnp.nan_to_num(state.metrics["root_pos_distance"]), 0.0
        )
        alive_steps += alive.astype(jnp.float32)

    denom = jnp.maximum(alive_steps, 1.0)
    per_clip = {
        "clip_ids": list(VAL_CLIPS),
        "score": np.asarray(returns / (steps * IMITATION_REWARD_CEILING)).tolist(),
        "survival": np.asarray(alive_steps / steps).tolist(),
        "joint_l2": np.asarray(joint_error / denom).tolist(),
        "root_distance_m": np.asarray(root_error / denom).tolist(),
    }
    return {
        "imitation_score": float(jnp.mean(returns / (steps * IMITATION_REWARD_CEILING))),
        "imitation_survival": float(jnp.mean(alive_steps / steps)),
        "imitation_full_horizon": float(jnp.mean(alive.astype(jnp.float32))),
        "imitation_joint_l2": float(jnp.mean(joint_error / denom)),
        "imitation_root_distance_m": float(jnp.mean(root_error / denom)),
        "imitation_per_clip": per_clip,
    }


def _pin_batch(state, commands):
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, :COMMAND_SIZE].set(commands)
    return state.replace(obs=obs)


def command_trials():
    """Return fixed commands and paired reset indices for direct evaluation."""
    commands = np.repeat(np.asarray(EVAL.commands, np.float32), len(EVAL.seeds), axis=0)
    clip_ids = np.tile(np.arange(len(EVAL.seeds), dtype=np.int32), len(EVAL.commands))
    starts = np.tile(np.asarray(EVAL.seeds, np.int32) * 5, len(EVAL.commands))
    return commands, clip_ids, starts


def evaluate_commands(env, policy, runtime=None, steps: int = EVAL.command_steps) -> dict:
    commands, clip_ids, starts = command_trials()
    n = len(commands)
    # Every command sees the same three initial states, so condition differences
    # cannot be attributed to different validation clips.
    reset_fn, _, step_fn = runtime or _batched_functions(env)
    state = _reset_batch(reset_fn, clip_ids, starts)
    commands_jax = jnp.asarray(commands)
    state = _pin_batch(state, commands_jax)

    alive = jnp.ones(n, dtype=bool)
    rng = jax.random.PRNGKey(789)
    velocity_rows = []
    yaw_rows = []
    alive_rows = []
    previous_xy = state.data.qpos[:, :2]
    previous_yaw = jax.vmap(quaternion_yaw)(state.data.qpos[:, 3:7])

    for index in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        state = step_fn(state, action, commands_jax)

        current_xy = state.data.qpos[:, :2]
        local_velocity = local_planar_velocity(
            previous_xy, current_xy, previous_yaw, float(env.dt)
        )
        yaw = jax.vmap(quaternion_yaw)(state.data.qpos[:, 3:7])
        dyaw = (yaw - previous_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
        yaw_rate = dyaw / float(env.dt)
        previous_xy = current_xy
        previous_yaw = yaw
        alive = jnp.logical_and(alive, state.done < 0.5)

        if index >= EVAL.warmup_steps:
            velocity_rows.append(local_velocity)
            yaw_rows.append(yaw_rate)
            alive_rows.append(alive)

    velocity = np.asarray(jnp.stack(velocity_rows))
    yaw_rate = np.asarray(jnp.stack(yaw_rows))
    alive_time = np.asarray(jnp.stack(alive_rows), dtype=np.float32)
    velocity = np.where(alive_time[..., None] > 0, velocity, 0.0)
    yaw_rate = np.where(alive_time > 0, yaw_rate, 0.0)
    command_time = np.broadcast_to(commands[None], (velocity.shape[0], *commands.shape))
    score_time = command_tracking_score(velocity, yaw_rate, command_time) * alive_time
    denom = np.maximum(alive_time.sum(axis=0), 1.0)
    mean_velocity = (velocity * alive_time[..., None]).sum(axis=0) / denom[:, None]
    mean_yaw = (yaw_rate * alive_time).sum(axis=0) / denom
    per_episode_score = score_time.mean(axis=0)
    per_episode_survival = alive_time.mean(axis=0)

    rows = []
    for i, command in enumerate(commands):
        rows.append(
            {
                "command": command.tolist(),
                "seed": int(EVAL.seeds[i % len(EVAL.seeds)]),
                "score": float(per_episode_score[i]),
                "survival": float(per_episode_survival[i]),
                "velocity_xy": mean_velocity[i].tolist(),
                "yaw_rate": float(mean_yaw[i]),
            }
        )
    return {
        "command_score": float(np.mean(per_episode_score)),
        "command_survival": float(np.mean(per_episode_survival)),
        "command_rows": rows,
    }


def evaluate_checkpoint(env, runtime, run: Path, step: int) -> dict:
    step_path = resolve_step(run, step)
    policy, _ = load_policy(env, step_path)
    result = {"step": step}
    result.update(evaluate_imitation(env, policy, runtime))
    result.update(evaluate_commands(env, policy, runtime))
    return result


def plot_report(report: dict, path: Path) -> None:
    """Make the compact workshop comparison paired with the full JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline, trained = report["baseline"], report["trained"]
    labels = ("natural imitation", "direct command")
    scores = np.asarray(
        [
            [baseline["imitation_score"], baseline["command_score"]],
            [trained["imitation_score"], trained["command_score"]],
        ]
    )
    survival = np.asarray(
        [
            [baseline["imitation_survival"], baseline["command_survival"]],
            [trained["imitation_survival"], trained["command_survival"]],
        ]
    )
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=150)
    for axis, values, title in zip(axes, (scores, survival), ("bounded score", "survival")):
        axis.bar(x - width / 2, values[0], width, label="random checkpoint 0", color="#9a9a9a")
        axis.bar(x + width / 2, values[1], width, label="trained", color="#35a7a0")
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].axhline(EVAL.command_survival_min, color="#d2674b", linestyle="--", linewidth=1)
    axes[0].set_ylabel("higher is better")
    axes[0].legend(frameon=False, fontsize=8)
    verdict = "PASS" if report["verdict"]["reportable"] else "NOT YET REPORTABLE"
    fig.suptitle(f"Demo D held-out checkpoint comparison — {verdict}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--step", type=int, default=None, help="trained step (default: latest)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    jax.random.key = jax.random.PRNGKey
    run = resolve_run(args.run)
    trained_path = resolve_step(run, args.step)
    trained_step = int(trained_path.name)
    env = load_validation_environment()
    runtime = _batched_functions(env)
    baseline = evaluate_checkpoint(env, runtime, run, 0)
    trained = evaluate_checkpoint(env, runtime, run, trained_step)
    verdict = reportability(baseline, trained)
    report = {"run": str(run), "baseline": baseline, "trained": trained, "verdict": verdict}

    out = Path(args.out) if args.out else OUT / "evaluation" / f"{run.name}-{trained_step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    figure = out.with_suffix(".png")
    plot_report(report, figure)
    print(
        f"baseline imitation={baseline['imitation_score']:.3f} command={baseline['command_score']:.3f}; "
        f"trained imitation={trained['imitation_score']:.3f} command={trained['command_score']:.3f}"
    )
    print(f"survival imitation={trained['imitation_survival']:.3f} command={trained['command_survival']:.3f}")
    print(f"reportable={verdict['reportable']} gates={verdict['gates']}")
    print(f"Wrote {out}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
