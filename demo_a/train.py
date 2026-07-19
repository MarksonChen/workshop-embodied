"""Train a policy directly on a task env (NO decoder) -- e.g. Demo A's RL-only
from-scratch walker on RodentMaintainVelocity (raw 38-D torque control).

Thin launcher around track-mjx scripts/train_task.py applying the fixes this setup
needs (see rl/README.md):
  - typed-key shim (jax.random.key -> PRNGKey)
  - device_put_replicated alias (brax 0.14 vs JAX 0.10)
  - checkpoint-only eval callback (the stock one renders+forks ffmpeg each eval and
    hangs; see train_joystick.py)
  - EGL rendering off by default (training doesn't render), no XLA preallocation,
    wandb offline by default

Usage:
    # True smoke test (fast): confirm the loop runs + saves a ckpt
    uv run python demo_a/train.py --smoke

    # Real run (Demo A convergence probe / full walker)
    uv run python demo_a/train.py --task RodentMaintainVelocity \\
        --num_envs 8192 --batch_size 2048 --num_timesteps 1e8 --eval_every 5000000
"""
import argparse
import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("WANDB_MODE", "offline")

import jax

# Typed-key shim: track-mjx branches on `key.ndim == 1` but jax.random.key(0) now
# yields a typed key (ndim 0). See rl/train_joystick.py.
jax.random.key = jax.random.PRNGKey

# brax 0.14 calls jax.device_put_replicated (removed in JAX 0.10); alias to the
# repo's equivalent. See rl/train_joystick.py.
from track_mjx.device_utils import replicate_for_pmap

jax.device_put_replicated = replicate_for_pmap

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "ref" / "repos" / "track-mjx" / "scripts"
RUNS_DIR = REPO_ROOT / "demo_a" / "runs"


def build_argv(args) -> list[str]:
    argv = [
        "train_task.py",
        "--task", args.task,
        "--checkpoint_dir", str(RUNS_DIR),
        "--num_envs", str(args.num_envs),
        "--num_timesteps", str(args.num_timesteps),
        "--batch_size", str(args.batch_size),
        "--eval_every", str(args.eval_every),
        "--seed", str(args.seed),
        "--wandb_project", args.wandb_project,
    ]
    if args.env:
        argv += ["--env", args.env]
    return argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to validate the loop end-to-end")
    ap.add_argument("--task", default="RodentMaintainVelocity")
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--num_timesteps", default="1e8")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="embodied-demoa")
    ap.add_argument("--env", default=None,
                    help='Env overrides, e.g. "reward_terms.alive.weight=0.0"')
    args = ap.parse_args()

    if args.smoke:
        # 256*20=5120, 64*16=1024, 5120/1024=5 -> valid (num_envs*unroll % batch*minibatches).
        args.num_envs = 256
        args.batch_size = 64
        args.num_timesteps = "3e5"
        args.eval_every = 150_000
        print(">>> SMOKE TEST: num_envs=256 batch=64 timesteps=3e5", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPTS))
    sys.argv = build_argv(args)

    import train_task

    # Replace the eval callback with a checkpoint-only version (the stock one
    # renders a video + forks ffmpeg each eval and hangs; see train_joystick.py).
    train_task.create_policy_params_fn = _checkpoint_only_policy_params_fn

    train_task.main()


def _checkpoint_only_policy_params_fn(
    ppo_params, ckpt_path, env, jit_reset, jit_step, jit_logging_inference_fn
):
    """Matches scripts/utils.create_policy_params_fn but only saves an Orbax
    checkpoint each eval -- no rollout, no render, no subprocess."""
    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        checkpointer.save(
            ckpt_path / f"{current_step}", params, force=True, save_args=save_args
        )
        print(f"[checkpoint] saved step {current_step}", flush=True)

    return policy_params_fn


if __name__ == "__main__":
    main()
