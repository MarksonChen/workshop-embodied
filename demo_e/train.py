"""Train one controlled Demo E arm.

The only learned component is a fresh high-level PPO policy that emits a
16-dimensional intention.  The low-level imitation decoder and Demo B scorer
are frozen in both arms; E0 uses beta=0 and E1 uses the configured beta.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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

from .config import ENV, OUT, PRIOR_READY_FOR_RL, TRAINING, resolved_config
from .env import build_env, wrap_for_training
from .provenance import run_metadata, write_pointer


def _budget(profile: str) -> dict[str, int]:
    if profile == "smoke":
        return {
            "num_timesteps": TRAINING.smoke_timesteps,
            "num_envs": TRAINING.smoke_envs,
            "batch_size": TRAINING.smoke_batch_size,
        }
    if profile == "workshop":
        return {
            "num_timesteps": TRAINING.workshop_timesteps,
            "num_envs": TRAINING.workshop_envs,
            "batch_size": TRAINING.workshop_batch_size,
        }
    return {
        "num_timesteps": TRAINING.num_timesteps,
        "num_envs": TRAINING.num_envs,
        "batch_size": TRAINING.batch_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("e0", "e1"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--smoke", action="store_true")
    profile.add_argument("--workshop", action="store_true")
    parser.add_argument("--num-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    run_profile = "smoke" if args.smoke else "workshop" if args.workshop else "report"
    if args.arm == "e1" and run_profile != "smoke" and not PRIOR_READY_FOR_RL:
        raise RuntimeError(
            "E1 report/workshop training is disabled: no Demo B transition "
            "checkpoint has passed the replicated source + physical gate. "
            "Use --smoke only to test wiring."
        )

    # Compatibility shims for the pinned TRACK-MJX/Brax stack.
    jax.random.key = jax.random.PRNGKey
    from track_mjx.device_utils import enable_jit_cache, replicate_for_pmap

    enable_jit_cache()
    jax.device_put_replicated = replicate_for_pmap

    beta = 0.0 if args.arm == "e0" else ENV.beta
    train_env = build_env(beta=beta, score_motion=args.arm == "e1")
    eval_env = build_env(beta=beta, score_motion=args.arm == "e1")
    requested = _budget(run_profile)
    for key, value in (
        ("num_timesteps", args.num_timesteps),
        ("num_envs", args.num_envs),
        ("batch_size", args.batch_size),
    ):
        if value is not None:
            requested[key] = value

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    label = run_profile
    run = OUT / "policy" / f"{args.arm}-{label}-seed{args.seed}-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    configuration = resolved_config()
    configuration.update(run_metadata(args.arm, args.seed, beta))
    configuration.update(
        {
            "profile": run_profile,
            "requested_budget": requested,
            "jax_devices": [str(device) for device in jax.devices()],
        }
    )
    (run / "config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n"
    )
    progress_path = run / "progress.jsonl"
    started = time.perf_counter()

    def progress(step, metrics):
        row = {"step": int(step), "wall_seconds": time.perf_counter() - started}
        for key, value in metrics.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = str(value)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print("DEMO_E_METRICS " + json.dumps(row, sort_keys=True), flush=True)

    def save_policy(step, make_policy, params):
        del make_policy
        path = run / str(int(step))
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        checkpointer.save(path, params, force=True, save_args=save_args)
        print(f"[checkpoint] {path}", flush=True)

    network_factory = partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=TRAINING.policy_layers,
        value_hidden_layer_sizes=TRAINING.value_layers,
        policy_obs_key="state",
        value_obs_key="state",
    )
    ppo.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=requested["num_timesteps"],
        num_evals=TRAINING.num_checkpoints,
        num_envs=requested["num_envs"],
        num_eval_envs=TRAINING.num_eval_envs,
        episode_length=ENV.episode_length,
        action_repeat=1,
        learning_rate=TRAINING.learning_rate,
        entropy_cost=TRAINING.entropy_cost,
        discounting=TRAINING.discounting,
        unroll_length=TRAINING.unroll_length,
        batch_size=requested["batch_size"],
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
        run_evals=False,
        wrap_env_fn=wrap_for_training,
    )
    elapsed = time.perf_counter() - started
    pointer_suffix = "policy" if run_profile == "report" else run_profile
    pointer_name = f"{args.arm}_{pointer_suffix}"
    pointer = write_pointer(
        pointer_name,
        run,
        arm=args.arm,
        seed=args.seed,
        beta=beta,
        training_seconds=elapsed,
    )
    print(f"Demo E {args.arm} complete in {elapsed:.1f}s: {run}")
    print(f"Pointer: {pointer}")


if __name__ == "__main__":
    main()
