"""demo_a/render_go1.py -- render a trained Go1 policy walking under a pinned
forward command. Phase-1 visualization (the proven quadruped gait, before rodent
retargeting). Rebuilds the brax PPO policy from the pickled (normalizer, policy,
value) tuple saved by demo_a/train_go1.py.
"""
import glob
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco.mjx as _mjx

_orig_make_data = _mjx.make_data


def _make_data_compat(*a, **k):
    k.pop("nconmax", None)
    return _orig_make_data(*a, **k)


_mjx.make_data = _make_data_compat

import jax
import jax.numpy as jp
import numpy as np

jax.random.key = jax.random.PRNGKey

import imageio.v2 as imageio
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ENV = "Go1JoystickFlatTerrain"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("demo_a/runs_go1/*.pkl"))[-1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    print(f"ckpt: {ckpt}")

    env = registry.load(ENV)
    pp = locomotion_params.brax_ppo_config(ENV)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))

    obs_size = jax.tree.map(lambda x: x.shape[-1], state.obs)
    nf = dict(pp["network_factory"])
    net = ppo_networks.make_ppo_networks(
        obs_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **nf,
    )
    params = pickle.load(open(ckpt, "rb"))
    policy = jax.jit(
        ppo_networks.make_inference_fn(net)((params[0], params[1]), deterministic=True)
    )

    def pin(s):  # pin a forward command for a clean walk
        if isinstance(s.info, dict) and "command" in s.info:
            return s.replace(info={**s.info, "command": jp.array([1.0, 0.0, 0.0])})
        return s

    rng = jax.random.PRNGKey(1)
    state = pin(state)
    states = [state]
    for _ in range(steps):
        rng, k = jax.random.split(rng)
        act, _ = policy(state.obs, k)
        state = pin(step(state, act))
        states.append(state)

    xs = np.array([float(s.data.qpos[0]) for s in states])
    zs = np.array([float(s.data.qpos[2]) for s in states])
    dur = len(states) * float(env.dt)
    print(f"rolled {len(states)} steps ({dur:.2f}s) | fwd {xs[-1]-xs[0]:+.2f} m "
          f"-> {(xs[-1]-xs[0])/dur:+.2f} m/s | mean height {zs.mean():.2f} m")

    frames = env.render(states, height=480, width=640, camera="track")
    out = Path("demo_a/out") / f"go1_{Path(ckpt).stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=int(round(1.0 / float(env.dt))))
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
