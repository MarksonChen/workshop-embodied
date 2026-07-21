"""Train a fast feed-forward PPO probe on Demo J reference tracking.

This is deliberately an environment/learnability gate, not the final matched
recurrent ANN baseline.  If a conventional actor cannot solve the task, making
the spiking actor larger would obscure a task or physics problem.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from track_mjx.device_utils import replicate_for_pmap

from demo_j.dataset import load_reference_set, take_references
from demo_j.env import FetchTracking


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1_024)
    parser.add_argument("--num-evals", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overfit-clip",
        type=int,
        help="train and evaluate on one train clip from frame zero",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Brax 0.14 still calls the helper removed by JAX 0.10.  TRACK-MJX carries
    # the official NamedSharding replacement; install it before entering the
    # otherwise unchanged Brax trainer.
    jax.device_put_replicated = replicate_for_pmap

    timesteps = 2_000_000 if args.smoke else args.num_timesteps
    num_envs = min(args.num_envs, 512) if args.smoke else args.num_envs
    training_reference = load_reference_set("train")
    validation_reference = load_reference_set("validation")
    random_start = True
    if args.overfit_clip is not None:
        training_reference = take_references(
            training_reference, [args.overfit_clip]
        )
        validation_reference = training_reference
        random_start = False
    environment = FetchTracking(training_reference, random_start=random_start)
    evaluation_environment = FetchTracking(
        validation_reference, random_start=False
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
    )
    started = time.perf_counter()
    progress: list[dict[str, float | int]] = []

    print(
        f"device={jax.devices()[0]} steps={timesteps:,} envs={num_envs} "
        f"obs={environment.observation_size} action={environment.action_size}",
        flush=True,
    )

    def report(step: int, metrics) -> None:
        row = {"step": int(step), "seconds": time.perf_counter() - started}
        for key in (
            "eval/episode_reward",
            "eval/episode_tracking_reward",
            "eval/episode_root_error",
            "eval/episode_joint_error",
            "eval/episode_foot_error",
            "eval/episode_completed",
        ):
            if key in metrics:
                row[key.removeprefix("eval/episode_")] = float(metrics[key])
        progress.append(row)
        print(json.dumps(row), flush=True)

    make_inference_fn, params, final_metrics = ppo.train(
        environment=environment,
        eval_env=evaluation_environment,
        num_timesteps=timesteps,
        num_evals=2 if args.smoke else args.num_evals,
        episode_length=59,
        num_envs=num_envs,
        num_eval_envs=128,
        batch_size=256,
        num_minibatches=8,
        num_updates_per_batch=4,
        unroll_length=20,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        discounting=0.97,
        reward_scaling=0.2,
        normalize_observations=True,
        max_grad_norm=1.0,
        deterministic_eval=True,
        seed=args.seed,
        network_factory=network_factory,
        progress_fn=report,
    )
    del make_inference_fn
    elapsed = time.perf_counter() - started
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / f"ann_probe_seed{args.seed}_{stamp}.pkl"
    with checkpoint.open("wb") as stream:
        pickle.dump(params, stream)
    report_path = checkpoint.with_suffix(".json")
    payload = {
        "schema": "demo-j-ann-probe-v1",
        "role": "feed-forward environment learnability probe",
        "seed": args.seed,
        "num_timesteps": timesteps,
        "num_envs": num_envs,
        "training_seconds": elapsed,
        "transitions_per_second": timesteps / elapsed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_reference_manifest_sha256": training_reference.manifest_sha256,
        "validation_reference_manifest_sha256": validation_reference.manifest_sha256,
        "progress": progress,
        "final_metrics": {key: float(value) for key, value in final_metrics.items()},
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"complete seconds={elapsed:.1f} checkpoint={checkpoint} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
