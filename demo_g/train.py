"""Train the paired Demo G task-only and task-plus-prior PPO arms.

The PPO call is intentionally the same one used in Demo A. The only scientific
difference between G0 and G1 is ``beta`` in the batch-level prior wrapper.

    env -u LD_LIBRARY_PATH uv run --no-project --isolated \
      --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
      --with 'jaxlib==0.4.30' python -m demo_g.train --arm g0 --smoke
"""

from __future__ import annotations

import argparse
import functools
import json
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from demo_a.train_fetch import FetchV2
from demo_f.artifacts import sha256

from .config import (
    DEFAULT_BETA,
    DEFAULT_ENVS,
    DEFAULT_EVALS,
    DEFAULT_SCORE_STRIDE,
    DEFAULT_TIMESTEPS,
    PRIOR_LOGP_CENTER,
    PRIOR_LOGP_SCALE,
)
from .env import DemoGFetchRun
from .prior import DEFAULT_PRIOR, load_prior
from .wrappers import wrap_demo_g_for_training


OUT = Path(__file__).resolve().parent / "out"


def git_provenance() -> dict:
    return {
        "commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
        ).stdout.strip(),
        "dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("g0", "g1"), required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--score-stride", type=int, default=DEFAULT_SCORE_STRIDE)
    parser.add_argument("--num-timesteps", type=float, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    parser.add_argument("--num-evals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    beta = 0.0 if args.arm == "g0" else args.beta
    timesteps = 2_000_000 if args.smoke else int(args.num_timesteps)
    num_envs = min(args.num_envs, 512) if args.smoke else args.num_envs
    num_evals = args.num_evals or (2 if args.smoke else DEFAULT_EVALS)
    prior = load_prior(args.prior)
    inner = DemoGFetchRun(prior)
    environment = FetchV2(inner)
    print(
        f"devices={jax.devices()} | arm={args.arm} beta={beta:g} | "
        f"score_stride={args.score_stride} "
        f"logp=({PRIOR_LOGP_CENTER:g},{PRIOR_LOGP_SCALE:g}) | "
        f"source_speed={inner.source_speed_mps:.3f}m/s "
        f"target_fetch={inner.v_target:.3f} sigma={inner.sigma:.3f} | "
        f"obs={environment.observation_size} act={environment.action_size} | "
        f"steps={timesteps:,} envs={num_envs}",
        flush=True,
    )
    started = time.time()
    progress_rows = []

    def progress(step, metrics):
        values = {
            "return": metrics.get("eval/episode_reward", float("nan")),
            "task": metrics.get("eval/episode_task_reward", float("nan")),
            "prior": metrics.get("eval/episode_prior_reward", float("nan")),
            "logp": metrics.get("eval/episode_prior_logp", float("nan")),
        }
        print(
            f"[{time.time() - started:5.0f}s] step={int(step):>11,} | "
            + " ".join(f"{name}={float(value):8.3f}" for name, value in values.items()),
            flush=True,
        )
        progress_rows.append(
            {
                "step": int(step),
                "seconds": time.time() - started,
                **{name: float(value) for name, value in values.items()},
            }
        )

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
        seed=args.seed,
        network_factory=ppo_networks.make_ppo_networks,
        wrap_env_fn=functools.partial(
            wrap_demo_g_for_training,
            prior=prior,
            beta=beta,
            score_stride=args.score_stride,
        ),
        progress_fn=progress,
    )
    elapsed = time.time() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = OUT / f"{args.arm}_seed{args.seed}_{stamp}.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(params, stream)
    report = {
        "arm": args.arm,
        "beta": beta,
        "score_stride": args.score_stride,
        "prior_logp_center": PRIOR_LOGP_CENTER,
        "prior_logp_scale": PRIOR_LOGP_SCALE,
        "seed": args.seed,
        "num_timesteps": timesteps,
        "num_envs": num_envs,
        "num_evals": num_evals,
        "source_speed_mps": inner.source_speed_mps,
        "target_speed_fetch": inner.v_target,
        "tracking_sigma_fetch": inner.sigma,
        "prior_command": [float(value) for value in inner.prior_command],
        "training_seconds": elapsed,
        "transitions_per_second": timesteps / elapsed,
        "git": git_provenance(),
        "prior_archive": str(args.prior),
        "prior_archive_sha256": sha256(args.prior),
        "prior": prior.metadata,
        "progress": progress_rows,
        "checkpoint": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"training complete in {elapsed:.1f}s | wrote {output}", flush=True)


if __name__ == "__main__":
    main()
