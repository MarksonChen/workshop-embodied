"""demo_a/train_go1.py -- train MuJoCo Playground's Go1 joystick locomotion.

Phase 1 of the rodent-retarget plan (see ref/docs/demo_a.md): use a PROVEN quadruped
locomotion env (position/PD actuators + velocity tracking + foot air-time/slip shaping,
MJX-fast) to confirm a real gait trains reliably in <10 min -- BEFORE retargeting the
Go1 body to a rodent. This sidesteps the reward-hacking twitch-slide of the spartan
RodentMaintainVelocity env.

Shim: Playground's mjx_env.make_data passes the old `nconmax` kwarg; the installed mjx
uses `naconmax`. Drop it (feet-only Go1 has minimal contacts -> default sizing is fine).

Usage:
    uv run python demo_a/train_go1.py --smoke      # ~30s wiring check
    env -u LD_LIBRARY_PATH uv run python demo_a/train_go1.py   # full proven run
"""
import argparse
import functools
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import mujoco.mjx as _mjx

_orig_make_data = _mjx.make_data


def _make_data_compat(*a, **k):
    k.pop("nconmax", None)  # Playground old kwarg -> installed mjx uses naconmax
    return _orig_make_data(*a, **k)


_mjx.make_data = _make_data_compat

import jax

jax.random.key = jax.random.PRNGKey  # typed-key shim (defensive)

# brax 0.14 calls jax.device_put_replicated (removed in JAX 0.10); alias to the
# repo's equivalent (see rl/train_joystick.py, PROJECT_STATE section 5).
from track_mjx.device_utils import replicate_for_pmap

jax.device_put_replicated = replicate_for_pmap

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

ENV = "Go1JoystickFlatTerrain"
RUNS = Path(__file__).resolve().parent / "runs_go1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num_timesteps", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None,
                    help="Self-contained XML to inject into the Go1 env (e.g. the "
                         "rodent model demo_a/models/rodent_go1.xml). Keeps Go1 physics.")
    args = ap.parse_args()

    if args.model:
        from mujoco_playground._src.locomotion.go1 import go1_constants as _C
        _xml = Path(args.model).resolve()
        _C.task_to_xml = lambda task: _xml  # inject custom model (same 12-DoF skeleton)
        print(f"injecting model: {_xml}", flush=True)

    env = registry.load(ENV)
    ppo_params = locomotion_params.brax_ppo_config(ENV)
    tp = dict(ppo_params)
    if args.smoke:
        tp["num_timesteps"] = 2_000_000
        tp["num_evals"] = 2
        print(">>> SMOKE: 2e6 steps", flush=True)
    elif args.num_timesteps:
        tp["num_timesteps"] = int(args.num_timesteps)

    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in tp:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **tp.pop("network_factory")
        )
    try:
        randomization_fn = registry.get_domain_randomizer(ENV)
    except Exception as e:
        print(f"(no domain randomizer: {e})", flush=True)
        randomization_fn = None

    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        ln = float(metrics.get("eval/avg_episode_length", float("nan")))
        print(f"[{time.time()-t0:6.0f}s] step {int(step):>12,} | reward {r:8.2f} | len {ln:6.0f}",
              flush=True)

    train_fn = functools.partial(
        ppo.train, **tp,
        network_factory=network_factory,
        randomization_fn=randomization_fn,
        progress_fn=progress,
        seed=args.seed,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"Training {ENV}: {tp['num_timesteps']:.0e} steps, {tp['num_envs']} envs", flush=True)
    make_inference_fn, params, _ = train_fn(environment=env)
    dt = time.time() - t0
    out = RUNS / f"go1_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(out, "wb") as f:
        pickle.dump(params, f)
    print(f"Training complete in {dt:.0f}s ({dt/60:.1f} min). Saved {out}", flush=True)


if __name__ == "__main__":
    main()
