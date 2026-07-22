from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from .environment import FetchRun, FetchV2, wrap_training


OUT = Path(__file__).resolve().parent / "out"


def train(
    timesteps: int = 30_000_000,
    num_envs: int = 2_048,
    num_evals: int = 3,
    seed: int = 0,
    save_progress: bool = False,
) -> Path:
    started = time.time()
    environment = FetchV2(FetchRun())
    progress_dir = OUT / "progress"

    def progress(step, metrics):
        reward = float(metrics.get("eval/episode_reward", float("nan")))
        print(
            f"{time.time() - started:6.1f}s | {int(step):>10,}/{timesteps:,} | "
            f"return {reward:8.2f}",
            flush=True,
        )

    def save(step, _, params):
        if not save_progress:
            return
        progress_dir.mkdir(parents=True, exist_ok=True)
        with (progress_dir / f"step_{int(step):010d}.pkl").open("wb") as handle:
            pickle.dump(params, handle)

    _, params, _ = ppo.train(
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
        entropy_cost=1e-2,
        discounting=0.97,
        reward_scaling=1.0,
        normalize_observations=True,
        seed=seed,
        network_factory=ppo_networks.make_ppo_networks,
        wrap_env_fn=wrap_training,
        progress_fn=progress,
        policy_params_fn=save,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = OUT / f"policy_seed{seed}_{stamp}.pkl"
    with output.open("wb") as handle:
        pickle.dump(params, handle)
    print(f"saved {output} after {time.time() - started:.1f}s", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=float, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=2_048)
    parser.add_argument("--num-evals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    train(
        timesteps=2_000_000 if args.smoke else int(args.timesteps),
        num_envs=min(args.num_envs, 512) if args.smoke else args.num_envs,
        num_evals=2 if args.smoke else args.num_evals,
        seed=args.seed,
        save_progress=args.save_progress,
    )


if __name__ == "__main__":
    main()
