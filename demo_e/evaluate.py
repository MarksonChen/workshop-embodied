"""Fixed-command functional and motion-likelihood evaluation for Demo E."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

from .config import ENV, EVAL, OUT
from .env import COMMAND_HORIZON_SECONDS, build_env, replace_command
from .features import joystick_to_hindsight_command
from .runtime import (
    checkpoint_steps,
    load_policy,
    resolve_run,
    resolve_step,
)


def make_counterfactual_scorer(prior):
    """Score one realized token/history under every fixed command."""
    raw_commands = jax.vmap(
        lambda command: joystick_to_hindsight_command(
            command, COMMAND_HORIZON_SECONDS
        )
    )(
        jnp.asarray(EVAL.commands)
    )

    def one(history, realized, index):
        predicted = jax.vmap(
            lambda command: prior.predict(
                history, prior.normalize_command(command)
            )[index]
        )(raw_commands)
        return jax.vmap(lambda mean: prior.log_prob(realized, mean))(predicted)

    return jax.jit(jax.vmap(one))


def _tracking_score(vx, yaw, command):
    vx_error = (np.asarray(vx) - command[0]) / 0.06
    yaw_error = (np.asarray(yaw) - command[1]) / 0.35
    return np.exp(-(vx_error**2 + yaw_error**2))


def make_rollout(env, network):
    """Compile all commands/seeds into one deterministic physical rollout."""
    inference = ppo_networks.make_inference_fn(network)

    def rollout(params, key, command):
        policy = inference(params, deterministic=True)
        state = replace_command(env.reset(key), command)

        def one_step(carry, _):
            state, key, stopped = carry
            key, action_key = jax.random.split(key)
            score_history = state.info["latent_history"]
            score_index = state.info["prior_tokens"] % 8

            def advance(current):
                action, _ = policy(current.obs, action_key)
                return replace_command(env.step(current, action), command)

            next_state = jax.lax.cond(stopped, lambda current: current, advance, state)
            stopped = stopped | (next_state.done > 0.5)
            output = {
                "qpos": next_state.data.qpos,
                "alive": (~stopped).astype(jnp.float32),
                "vx": next_state.metrics["local_vel_x"],
                "yaw": next_state.metrics["yaw_rate"],
                "energy": next_state.metrics["energy"],
                "lateral": next_state.metrics["local_vel_y"],
                "action_rate": next_state.metrics["action_rate"],
                "task_reward": next_state.metrics["r_task"],
                "prior_raw": next_state.metrics["r_prior_raw"],
                "prior_score": next_state.metrics["r_prior_score"],
                "prior_update": next_state.metrics["prior_update"],
                "prior_history": score_history,
                "prior_index": score_index,
                "prior_realized": next_state.info["last_prior_realized"],
            }
            return (next_state, key, stopped), output

        _, trajectory = jax.lax.scan(
            one_step,
            (state, key, jnp.asarray(False)),
            xs=None,
            length=EVAL.rollout_steps,
        )
        return trajectory

    return jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0)))


def summarize(trajectory: dict, command: np.ndarray, logp_clip: np.ndarray) -> dict:
    warmup = round(EVAL.warmup_seconds / ENV.control_dt)
    alive = trajectory["alive"][warmup:] > 0.5
    updates = (trajectory["prior_update"][warmup:] > 0.5) & alive
    tracking = _tracking_score(
        trajectory["vx"][warmup:], trajectory["yaw"][warmup:], command
    )
    lower, upper = np.asarray(logp_clip)
    raw = trajectory["prior_raw"][warmup:][updates]
    likelihood = (
        np.clip(raw, lower, upper) - lower
    ) / max(float(upper - lower), 1e-8)
    reward_score = trajectory["prior_score"][warmup:][updates]
    return {
        "functional": float(np.mean(tracking * alive)),
        "survival": float(np.mean(alive)),
        "vx_mean": float(np.mean(trajectory["vx"][warmup:])),
        "vx_mae": float(np.mean(np.abs(trajectory["vx"][warmup:] - command[0]))),
        "yaw_mae": float(np.mean(np.abs(trajectory["yaw"][warmup:] - command[1]))),
        "lateral_abs": float(np.mean(np.abs(trajectory["lateral"][warmup:]))),
        "energy": float(np.mean(trajectory["energy"][warmup:])),
        "action_rate": float(np.mean(trajectory["action_rate"][warmup:])),
        "task_reward": float(np.mean(trajectory["task_reward"][warmup:])),
        "motion_logp": float(np.mean(raw)) if len(raw) else float("nan"),
        "motion_likelihood_score": (
            float(np.mean(likelihood)) if len(likelihood) else float("nan")
        ),
        "reward_prior_score": (
            float(np.mean(reward_score)) if len(reward_score) else float("nan")
        ),
        "likelihood_tokens": int(np.sum(updates)),
    }


def conditional_summary(scores: np.ndarray, matching_index: int) -> dict:
    if not len(scores):
        return {
            "conditional_top1": float("nan"),
            "conditional_rank": float("nan"),
            "conditional_margin": float("nan"),
        }
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == matching_index, axis=1) + 1
    matched = scores[:, matching_index]
    counterfactual = np.max(
        np.where(
            np.arange(scores.shape[1])[None] == matching_index,
            -np.inf,
            scores,
        ),
        axis=1,
    )
    return {
        "conditional_top1": float(np.mean(rank == 1)),
        "conditional_rank": float(np.mean(rank)),
        "conditional_margin": float(np.mean(matched - counterfactual)),
    }


def _aggregate(rows: list[dict]) -> dict:
    fields = (
        "functional",
        "survival",
        "vx_mae",
        "yaw_mae",
        "task_reward",
        "motion_logp",
        "motion_likelihood_score",
        "reward_prior_score",
        "conditional_top1",
        "conditional_rank",
        "conditional_margin",
    )
    return {field: float(np.mean([row[field] for row in rows])) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("e0", "e1", "both"), default="both")
    parser.add_argument("--e0-run", type=Path)
    parser.add_argument("--e1-run", type=Path)
    parser.add_argument("--e0-step", type=int)
    parser.add_argument("--e1-step", type=int)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument(
        "--trial-batch-size", type=int, default=EVAL.trial_batch_size
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arms = ("e0", "e1") if args.arm == "both" else (args.arm,)
    explicit_runs = {"e0": args.e0_run, "e1": args.e1_run}
    explicit_steps = {"e0": args.e0_step, "e1": args.e1_step}
    runs = {arm: resolve_run(arm, explicit_runs[arm]) for arm in arms}
    selected: dict[str, list[Path]] = {}
    for arm, run in runs.items():
        if args.all_checkpoints:
            selected[arm] = [run / str(step) for step in checkpoint_steps(run)]
        else:
            selected[arm] = [resolve_step(run, explicit_steps[arm])]

    # beta=0 makes evaluation observational: the frozen score is measured for
    # both arms but cannot alter either trajectory.
    env = build_env(beta=0.0, score_motion=True)
    initial_checkpoint = next(iter(selected.values()))[0]
    _, _, network = load_policy(env, initial_checkpoint)
    rollout = make_rollout(env, network)
    counterfactual_scorer = make_counterfactual_scorer(env.unwrapped.prior)
    trials = [
        (command_index, np.asarray(command, np.float32), seed)
        for command_index, command in enumerate(EVAL.commands)
        for seed in EVAL.seeds
    ]
    keys = jnp.stack(
        [
            jax.random.PRNGKey(10_000 * command_index + seed)
            for command_index, _, seed in trials
        ]
    )
    commands = jnp.asarray(np.stack([command for _, command, _ in trials]))
    rows: list[dict] = []
    saved: dict[str, np.ndarray] = {}
    started = time.perf_counter()

    for arm in arms:
        for checkpoint in selected[arm]:
            _, params, _ = load_policy(env, checkpoint)
            if args.trial_batch_size <= 0:
                raise ValueError("--trial-batch-size must be positive")
            pieces = []
            for offset in range(0, len(trials), args.trial_batch_size):
                stop = offset + args.trial_batch_size
                pieces.append(
                    jax.tree.map(
                        np.asarray,
                        rollout(params, keys[offset:stop], commands[offset:stop]),
                    )
                )
            batch = jax.tree.map(lambda *values: np.concatenate(values), *pieces)
            step = int(checkpoint.name)
            token_inputs = []
            token_counts = []
            for trial_index in range(len(trials)):
                alive = batch["alive"][trial_index] > 0.5
                updates = batch["prior_update"][trial_index] > 0.5
                mask = alive & updates & (
                    np.arange(EVAL.rollout_steps)
                    >= round(EVAL.warmup_seconds / ENV.control_dt)
                )
                token_inputs.append(
                    (
                        batch["prior_history"][trial_index][mask],
                        batch["prior_realized"][trial_index][mask],
                        batch["prior_index"][trial_index][mask],
                    )
                )
                token_counts.append(int(np.sum(mask)))
            histories = np.concatenate([value[0] for value in token_inputs], axis=0)
            realized = np.concatenate([value[1] for value in token_inputs], axis=0)
            indices = np.concatenate([value[2] for value in token_inputs], axis=0)
            all_counterfactual = np.asarray(
                counterfactual_scorer(
                    jnp.asarray(histories), jnp.asarray(realized), jnp.asarray(indices)
                )
            )
            split_points = np.cumsum(token_counts)[:-1]
            per_trial_counterfactual = np.split(all_counterfactual, split_points)
            for trial_index, (command_index, command, seed) in enumerate(trials):
                trajectory = jax.tree.map(
                    lambda value: value[trial_index], batch
                )
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "command_index": command_index,
                        "command_vx": float(command[0]),
                        "command_yaw": float(command[1]),
                        "seed": seed,
                        **summarize(
                            trajectory,
                            command,
                            np.asarray(env.unwrapped.prior.norm["logp_clip"]),
                        ),
                        **conditional_summary(
                            per_trial_counterfactual[trial_index], command_index
                        ),
                    }
                )
                # Save renderable trajectories only for the final selected
                # checkpoint of each arm, keeping the report artifact compact.
                if checkpoint == selected[arm][-1]:
                    prefix = f"{arm}_c{command_index}_s{seed}"
                    saved[f"{prefix}_qpos"] = trajectory["qpos"]
                    saved[f"{prefix}_alive"] = trajectory["alive"]

    learning_curve = []
    for arm in arms:
        for step in sorted({row["step"] for row in rows if row["arm"] == arm}):
            group = [
                row for row in rows if row["arm"] == arm and row["step"] == step
            ]
            learning_curve.append({"arm": arm, "step": step, **_aggregate(group)})

    comparison = None
    if set(arms) == {"e0", "e1"}:
        final = {
            arm: next(
                row
                for row in reversed(learning_curve)
                if row["arm"] == arm
            )
            for arm in arms
        }
        comparison = {
            "functional_retention": final["e1"]["functional"]
            / max(final["e0"]["functional"], 1e-8),
            "raw_logp_delta": final["e1"]["motion_logp"]
            - final["e0"]["motion_logp"],
            "reward_score_delta": final["e1"]["reward_prior_score"]
            - final["e0"]["reward_prior_score"],
            "e1_improves_raw_logp": final["e1"]["motion_logp"]
            > final["e0"]["motion_logp"],
        }
        gates = {
            "e1_survival": final["e1"]["survival"] >= 0.90,
            "function_retained": comparison["functional_retention"] >= 0.90,
            "raw_logp_improved": comparison["e1_improves_raw_logp"],
            "matching_command_top1": final["e1"]["conditional_top1"] >= 0.50,
            "matching_command_margin": final["e1"]["conditional_margin"] > 0.0,
            "finite": all(
                np.isfinite(value)
                for arm in final.values()
                for value in arm.values()
                if isinstance(value, (int, float))
            ),
        }
        comparison["verdict"] = {"reportable": all(gates.values()), "gates": gates}

    report = {
        "runs": {arm: str(path) for arm, path in runs.items()},
        "checkpoints": {
            arm: [int(path.name) for path in paths]
            for arm, paths in selected.items()
        },
        "evaluation_seconds": time.perf_counter() - started,
        "likelihood_definition": (
            "Demo B conditional Gaussian log likelihood, linearly normalized "
            "between the source real-motion 1st and 99th percentiles"
        ),
        "reward_score_definition": {
            "formula": "clip((raw_logp - floor) / (ceiling - floor), 0, 1)",
            "floor": ENV.prior_logp_floor,
            "ceiling": ENV.prior_logp_ceiling,
            "note": "reward units only; raw_logp is the scientific measurement",
        },
        "learning_curve": learning_curve,
        "comparison": comparison,
        "rows": rows,
    }
    output = args.output or OUT / (
        "evaluation-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rollout_path = output.with_suffix(".npz")
    np.savez_compressed(rollout_path, **saved)
    print(json.dumps({"learning_curve": learning_curve, "comparison": comparison}, indent=2))
    print(output)
    print(rollout_path)


if __name__ == "__main__":
    main()
