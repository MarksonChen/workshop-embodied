"""Train the one-stage Demo D torque policy from random initialization.

Usage:
    uv run python -m demo_d.train --smoke
    uv run python -m demo_d.train --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from functools import partial

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import orbax.checkpoint as ocp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from mujoco_playground import wrapper as playground_wrapper

from demo_d.config import OUT, PIPELINE_VERSION, REFERENCE_SHA256, TRAINING, resolved_config
from demo_d.provenance import write_pointer
from demo_d.runtime import load_split_environments


def training_arguments(smoke: bool) -> dict:
    if not smoke:
        return {
            "num_timesteps": TRAINING.num_timesteps,
            "eval_every": TRAINING.eval_every,
            "num_envs": TRAINING.num_envs,
            "batch_size": TRAINING.batch_size,
        }
    return {"num_timesteps": 300_000, "eval_every": 150_000, "num_envs": 256, "batch_size": 64}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    jax.random.key = jax.random.PRNGKey
    from track_mjx.device_utils import enable_jit_cache, replicate_for_pmap

    enable_jit_cache()
    jax.device_put_replicated = replicate_for_pmap

    train_env, val_env = load_split_environments()
    budget = training_arguments(args.smoke)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = "smoke" if args.smoke else "report"
    run = OUT / "policy" / f"{label}-seed{args.seed}-{stamp}"
    run.mkdir(parents=True, exist_ok=False)

    config = resolved_config()
    config.update(
        {
            "demo_d": {
                "from_scratch": True,
                "parent_checkpoint": None,
                "initialization": "random",
                "pipeline_version": PIPELINE_VERSION,
                "reference_sha256": REFERENCE_SHA256,
            },
            "seed": args.seed,
            "smoke": args.smoke,
            "realized_budget_request": budget,
        }
    )
    (run / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    progress_path = run / f"progress_{stamp}.jsonl"

    def progress(step, metrics):
        row = {"step": int(step)}
        for key, value in metrics.items():
            try:
                row[key] = value.item()
            except Exception:
                row[key] = float(value) if isinstance(value, (int, float)) else str(value)
        with progress_path.open("a") as fid:
            fid.write(json.dumps(row, sort_keys=True) + "\n")
        print("DEMO_D_METRICS " + json.dumps(row, sort_keys=True), flush=True)

    def save_policy(current_step, make_policy, params):
        del make_policy
        path = run / str(int(current_step))
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        checkpointer.save(path, params, force=True, save_args=save_args)
        print(f"[checkpoint] {path}", flush=True)

    num_evals = max(2, int(budget["num_timesteps"] / budget["eval_every"]) + 1)
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=TRAINING.policy_layers,
        value_hidden_layer_sizes=TRAINING.value_layers,
        policy_obs_key="state",
        value_obs_key="state",
    )
    episode_length = (
        250 - TRAINING.start_frame_max - TRAINING.command_horizon_frames - 1
    ) * 2
    ppo.train(
        environment=train_env,
        eval_env=val_env,
        num_timesteps=budget["num_timesteps"],
        num_evals=num_evals,
        num_envs=budget["num_envs"],
        num_eval_envs=64 if not args.smoke else 16,
        episode_length=episode_length,
        action_repeat=1,
        learning_rate=TRAINING.learning_rate,
        entropy_cost=TRAINING.entropy_cost,
        discounting=TRAINING.discounting,
        unroll_length=TRAINING.unroll_length,
        batch_size=budget["batch_size"],
        num_minibatches=TRAINING.num_minibatches,
        num_updates_per_batch=TRAINING.num_updates_per_batch,
        normalize_observations=True,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        deterministic_eval=True,
        network_factory=network_factory,
        seed=args.seed,
        progress_fn=progress,
        policy_params_fn=save_policy,
        restore_checkpoint_path=None,
        wrap_env_fn=partial(playground_wrapper.wrap_for_brax_training),
    )
    pointer_name = "smoke_policy" if args.smoke else "policy"
    pointer = write_pointer(
        pointer_name,
        run,
        seed=args.seed,
        smoke=args.smoke,
        progress=str(progress_path.resolve()),
    )
    print(f"Demo D run: {run}")
    print(f"Pointer: {pointer}")


if __name__ == "__main__":
    main()
