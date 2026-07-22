from __future__ import annotations

import argparse
import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
from brax.training.agents.ppo import train as ppo

from workshop.part1.environment import FetchV2

from .config import ACTION_DIM, OUT, TASK_SPEED_MAX, TASK_SPEED_MIN
from .core.artifacts import save_policy_checkpoint
from .core.control import PriorFetchRun
from .core.control import make_residual_ppo_networks
from .core.prior import DEFAULT_PRIOR, load_prior
from .core.control import wrap_training


def train(
    prior_path: Path = DEFAULT_PRIOR,
    beta: float = 0.10,
    timesteps: int = 30_000_000,
    num_envs: int = 2_048,
    num_evals: int = 3,
    speed_min: float = TASK_SPEED_MIN,
    speed_max: float = TASK_SPEED_MAX,
    seed: int = 0,
) -> Path:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    started = time.time()
    progress_rows = []
    prior_path = Path(prior_path)
    prior = load_prior(prior_path)
    environment = FetchV2(
        PriorFetchRun(task_speed_min=speed_min, task_speed_max=speed_max)
    )
    network_factory = functools.partial(make_residual_ppo_networks, prior=prior)
    wrapper = functools.partial(wrap_training, prior=prior, beta=beta)
    entropy_cost = beta / ACTION_DIM if beta else 1e-2

    def progress(step, metrics):
        row = {
            "step": int(step),
            "seconds": time.time() - started,
            "return": float(metrics.get("eval/episode_reward", float("nan"))),
            "task": float(metrics.get("eval/episode_task_reward", float("nan"))),
            "reference_logp": float(
                metrics.get("eval/episode_reference_logp", float("nan"))
            ),
        }
        progress_rows.append(row)
        print(json.dumps(row), flush=True)

    _, params, metrics = ppo.train(
        environment=environment,
        num_timesteps=timesteps,
        num_evals=num_evals,
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
        seed=seed,
        network_factory=network_factory,
        wrap_env_fn=wrapper,
        progress_fn=progress,
    )
    elapsed = time.time() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    beta_label = f"{beta:g}".replace(".", "p")
    output = OUT / f"policy_beta{beta_label}_seed{seed}_{stamp}.pkl"
    training = {
        "beta": beta,
        "seed": seed,
        "timesteps": timesteps,
        "num_envs": num_envs,
        "speed_range": [speed_min, speed_max],
    }
    contract = save_policy_checkpoint(
        output,
        params,
        beta=beta,
        prior_path=prior_path,
        training=training,
    )
    report = {
        "schema": "workshop-part3-training-v1",
        **training,
        "training_seconds": elapsed,
        "transitions_per_second": timesteps / elapsed,
        "prior": {"path": str(prior_path), "sha256": contract["prior_sha256"]},
        "progress": progress_rows,
        "final_metrics": {name: float(value) for name, value in metrics.items()},
        "checkpoint": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {output} after {elapsed:.1f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--timesteps", type=float, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-evals", type=int, default=3)
    parser.add_argument("--speed-min", type=float, default=TASK_SPEED_MIN)
    parser.add_argument("--speed-max", type=float, default=TASK_SPEED_MAX)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    train(
        prior_path=args.prior,
        beta=args.beta,
        timesteps=2_000_000 if args.smoke else int(args.timesteps),
        num_envs=min(args.num_envs, 512) if args.smoke else args.num_envs,
        num_evals=2 if args.smoke else args.num_evals,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
