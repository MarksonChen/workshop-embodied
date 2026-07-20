"""Train brax v1 'fetch' (a creature-like 10-DoF quadruped that locomotes to a
target -- less robotic than Go1) using brax 0.12.3's v2 PPO, via a thin v1->v2 Env
adapter (the v1 env is not a v2 PipelineEnv, and brax 0.12.3 ships no v1 trainer).

Runs in an ISOLATED brax 0.12.3 + jax 0.4.30 environment (our main venv is brax 0.14):
    env -u LD_LIBRARY_PATH uv run \
        --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' --with 'jaxlib==0.4.30' \
        python demo_a/train_fetch.py [--smoke]

The adapter carries the ENTIRE v1 State as the v2 `pipeline_state` (a pytree), so the
v2 auto-reset/episode wrappers tree-select it correctly on done.
"""
import argparse
import functools
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp

try:
    from .fetch_run import make_env, deciles_dir
except ImportError:  # direct script entry point used by the workshop commands
    from fetch_run import make_env, deciles_dir
from brax.envs.base import Env as V2Env
from brax.envs.base import State as V2State
from brax.envs.base import Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

OUT = Path(__file__).resolve().parent / "out"


class FetchV2(V2Env):
    """Adapt a brax v1 fetch-family env to the v2 Env interface. pipeline_state =
    full v1 State. The inner env ('fetch' reach-a-target, or 'run' constant-speed)
    is injected so the same PPO plumbing drives either task."""

    def __init__(self, inner):
        self._env = inner

    def reset(self, rng) -> V2State:
        s = self._env.reset(rng)
        return V2State(pipeline_state=s, obs=s.obs, reward=s.reward, done=s.done,
                       metrics=dict(s.metrics), info={})

    def step(self, state: V2State, action) -> V2State:
        s = self._env.step(state.pipeline_state, action)
        # Preserve keys that outer wrappers add to metrics (e.g. EvalWrapper injects
        # 'reward'); only UPDATE the native v1 fetch keys. Rebuilding the dict here
        # would drop those keys and break the PPO/eval scan's carry-structure check.
        metrics = dict(state.metrics)
        metrics.update(s.metrics)
        return state.replace(pipeline_state=s, obs=s.obs, reward=s.reward, done=s.done,
                             metrics=metrics)

    @property
    def observation_size(self):
        return self._env.observation_size

    @property
    def action_size(self):
        return self._env.action_size

    @property
    def backend(self):
        return "generalized"


class RawKeyVmapWrapper(Wrapper):
    """Like brax's VmapWrapper, but feeds the inner env RAW keys. The v1 fetch env's
    jumpy random ops predate jax's typed keys; brax's default VmapWrapper hands them
    typed scalar keys (ndim 0) which they reject. We convert to raw uint32 keys first."""

    def reset(self, rng) -> V2State:
        if rng.dtype != jp.uint32:  # typed key -> raw uint32 key data
            rng = jax.random.key_data(rng)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: V2State, action) -> V2State:
        return jax.vmap(self.env.step)(state, action)


def wrap_fetch_for_training(env, episode_length=1000, action_repeat=1, **_):
    """Raw-key-safe replacement for brax's default training wrap (Episode + Vmap +
    AutoReset). Passed to ppo.train via wrap_env_fn -- keeps the PPO algorithm intact."""
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = RawKeyVmapWrapper(env)
    env = AutoResetWrapper(env)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--env", choices=["fetch", "run"], default="fetch",
                    help="'fetch' = reach-a-target (scramble); 'run' = constant-speed run (gait probe)")
    ap.add_argument("--num_timesteps", type=float, default=5e7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_deciles", action="store_true",
                    help="snapshot params at each of 11 evals (0,10,...,100%% of training) "
                         "into out/deciles/ -- for watching the gait evolve over training")
    args = ap.parse_args()

    n = 2_000_000 if args.smoke else int(args.num_timesteps)
    n_envs = 512 if args.smoke else 2048
    n_evals = 11 if args.save_deciles else (2 if args.smoke else 10)
    print(f"jax devices: {jax.devices()}")
    env = FetchV2(make_env(args.env))
    print(f"FetchV2[{args.env}] | obs {env.observation_size} | act {env.action_size} "
          f"| steps {n:.0e} | evals {n_evals}")

    t0 = time.time()
    dec_dir = deciles_dir(OUT, args.env)

    def save_ckpt(step, make_policy, params):
        """policy_params_fn hook -- pickle the (normalizer, policy, value) tuple per eval."""
        if not args.save_deciles:
            return
        dec_dir.mkdir(parents=True, exist_ok=True)
        pct = round(100 * int(step) / n)
        with open(dec_dir / f"fetch_{pct:03d}pct_{int(step):010d}.pkl", "wb") as f:
            pickle.dump(params, f)
        print(f"       [ckpt] saved {pct}% @ step {int(step):,}", flush=True)

    def progress(step, metrics):
        r = float(metrics.get("eval/episode_reward", float("nan")))
        pct = 100 * int(step) / n
        print(f"[{time.time()-t0:5.0f}s] {pct:5.1f}% step {int(step):>11,} | reward {r:9.2f}",
              flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_timesteps=n,
        num_evals=n_evals,
        episode_length=1000,
        num_envs=n_envs,
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
        wrap_env_fn=wrap_fetch_for_training,
        progress_fn=progress,
        policy_params_fn=save_ckpt,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"fetch{'_run' if args.env == 'run' else ''}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
    with open(out, "wb") as f:
        pickle.dump(params, f)
    print(f"Training complete in {time.time()-t0:.0f}s. Saved {out}", flush=True)


if __name__ == "__main__":
    main()
