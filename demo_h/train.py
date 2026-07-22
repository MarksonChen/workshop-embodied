"""Post-train the frozen Demo H prior with optional KL regularization."""

from __future__ import annotations

import argparse
import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

from brax.training.agents.ppo import train as ppo

from demo_a.train_fetch import FetchV2
from demo_h.artifacts import save_policy_checkpoint
from demo_h.config import (
    ACTION_DIM,
    OUT,
    TASK_SPEED_MAX,
    TASK_SPEED_MIN,
)
from demo_h.env import DemoHFetchRun
from demo_h.policy import make_residual_ppo_networks
from demo_h.prior import DEFAULT_PRIOR, load_prior
from demo_h.wrappers import wrap_demo_h_for_training


DEFAULT_TIMESTEPS = 30_000_000
DEFAULT_ENVS = 2_048
DEFAULT_BETA = 0.10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("h1", "h2"), required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--num-timesteps", type=float, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    parser.add_argument("--num-evals", type=int, default=3)
    parser.add_argument("--speed-min", type=float, default=TASK_SPEED_MIN)
    parser.add_argument("--speed-max", type=float, default=TASK_SPEED_MAX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sweep-id",
        help="Optional immutable experiment label embedded in the checkpoint",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    timesteps = 2_000_000 if args.smoke else int(args.num_timesteps)
    num_envs = min(args.num_envs, 512) if args.smoke else args.num_envs
    started = time.time()
    progress_rows = []

    beta = 0.0 if args.arm == "h1" else args.beta
    prior = load_prior(args.prior)
    inner = DemoHFetchRun(
        task_speed_min=args.speed_min,
        task_speed_max=args.speed_max,
    )
    environment = FetchV2(inner)
    network_factory = functools.partial(make_residual_ppo_networks, prior=prior)
    wrap_env_fn = functools.partial(wrap_demo_h_for_training, prior=prior, beta=beta)
    # For H2, reference cross-entropy plus this entropy is exactly -mean KL.
    entropy_cost = 1e-2 if args.arm == "h1" else beta / ACTION_DIM

    print(
        f"devices={jax.devices()} arm={args.arm} beta={beta:g} "
        f"steps={timesteps:,} envs={num_envs} obs={environment.observation_size} "
        f"entropy={entropy_cost:g}",
        flush=True,
    )

    def progress(step, metrics):
        row = {
            "step": int(step),
            "seconds": time.time() - started,
            "shaped_return_excluding_ppo_entropy": float(
                metrics.get("eval/episode_reward", float("nan"))
            ),
            "task": float(metrics.get("eval/episode_task_reward", float("nan"))),
            "speed_reward": float(
                metrics.get("eval/episode_speed_reward", float("nan"))
            ),
            "reference_logp": float(
                metrics.get("eval/episode_reference_logp", float("nan"))
            ),
        }
        progress_rows.append(row)
        print(json.dumps(row), flush=True)

    _, params, metrics = ppo.train(
        environment=environment,
        num_timesteps=timesteps,
        num_evals=2 if args.smoke else args.num_evals,
        episode_length=1000,
        num_envs=num_envs,
        batch_size=256,
        num_minibatches=8,
        num_updates_per_batch=4,
        unroll_length=20,
        learning_rate=3e-4,
        entropy_cost=entropy_cost,
        discounting=0.97,
        reward_scaling=1.0,
        normalize_observations=False,
        deterministic_eval=True,
        seed=args.seed,
        network_factory=network_factory,
        wrap_env_fn=wrap_env_fn,
        progress_fn=progress,
    )
    elapsed = time.time() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    beta_label = f"{beta:g}".replace(".", "p")
    prefix = (
        f"{args.sweep_id}_{args.arm}_beta{beta_label}" if args.sweep_id else args.arm
    )
    output = OUT / f"{prefix}_seed{args.seed}_{stamp}.pkl"
    run_metadata = {
        "sweep_id": args.sweep_id,
        "beta": beta,
        "seed": args.seed,
        "num_timesteps": timesteps,
        "num_envs": num_envs,
        "task_speed_training_range": [args.speed_min, args.speed_max],
    }
    checkpoint_metadata = save_policy_checkpoint(
        output,
        params,
        arm=args.arm,
        prior_path=args.prior,
        run_metadata=run_metadata,
    )
    report = {
        "schema": "demo-h-ppo-training-v2",
        "arm": args.arm,
        "beta": beta,
        "reference_kl_implementation": (
            None
            if args.arm == "h1"
            else "beta/dim * reference log-prob reward + beta/dim * PPO entropy"
        ),
        "progress_return_note": (
            "environment reward excludes PPO entropy; use shaping-disabled task and "
            "analytic KL metrics for interpretation"
        ),
        "seed": args.seed,
        "sweep_id": args.sweep_id,
        "num_timesteps": timesteps,
        "num_envs": num_envs,
        "training_seconds": elapsed,
        "transitions_per_second": timesteps / elapsed,
        "task_speed_training_range": [args.speed_min, args.speed_max],
        "prior": {
            "path": str(args.prior),
            "sha256": checkpoint_metadata["prior_sha256"],
        },
        "progress": progress_rows,
        "final_metrics": {key: float(value) for key, value in metrics.items()},
        "checkpoint": str(output),
        "checkpoint_contract": checkpoint_metadata,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"training complete in {elapsed:.1f}s | wrote {output}", flush=True)


if __name__ == "__main__":
    main()
