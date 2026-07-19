"""Roll out a trained brax v1 'fetch' policy and render it.

Rebuilds the PPO inference fn from the pickled params (must match train_fetch.py:
normalize_observations=True -> running_statistics.normalize) and rolls the *raw*
v1 fetch env (single env, raw PRNG keys). Writes an interactive brax-viewer HTML
(always) and, if imageio is present, an MP4.

Isolated env (same as training):
    env -u LD_LIBRARY_PATH uv run --isolated \
        --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' --with 'imageio[ffmpeg]' \
        python demo_a/render_fetch.py [--params demo_a/out/fetch_*.pkl] [--steps 500]
"""
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jp

from brax.v1.envs import fetch as v1fetch
from brax.v1.io import html, image
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

OUT = Path(__file__).resolve().parent / "out"


def load_policy(params_path, env):
    with open(params_path, "rb") as f:
        params = pickle.load(f)  # (normalizer, policy, value) -- make_policy uses [0],[1]
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    return make_policy(params, deterministic=True)


def rollout(env, policy, steps, seed):
    reset = jax.jit(env.reset)
    @jax.jit
    def step(state, key):
        act, _ = policy(state.obs, key)
        return env.step(state, act)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    qps, rewards = [state.qp], []
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        state = step(state, key)
        qps.append(state.qp)
        rewards.append(float(state.reward))
    return qps, rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="pkl path (default: latest in out/)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params_path = args.params
    if params_path is None:
        cands = sorted(OUT.glob("fetch_*.pkl"))
        if not cands:
            raise SystemExit("no fetch_*.pkl in demo_a/out/ -- train first")
        params_path = cands[-1]
    print(f"params: {params_path}")

    env = v1fetch.Fetch()
    policy = load_policy(params_path, env)
    qps, rewards = rollout(env, policy, args.steps, args.seed)
    print(f"rolled out {len(qps)} frames | mean step reward {sum(rewards)/len(rewards):.3f} "
          f"| total {sum(rewards):.1f}")

    stem = Path(params_path).stem
    html_path = OUT / f"{stem}.html"
    html_path.write_text(html.render(env.sys, qps))
    print(f"wrote {html_path}")

    # MP4 is best-effort (needs imageio[ffmpeg]); HTML above always works.
    try:
        import imageio.v2 as imageio
        import numpy as np
        frames = [np.asarray(image.render_array(env.sys, qp, 480, 480)) for qp in qps]
        mp4_path = OUT / f"{stem}.mp4"
        imageio.mimwrite(mp4_path, frames, fps=int(1 / env.sys.config.dt) if env.sys.config.dt else 30)
        print(f"wrote {mp4_path}")
    except Exception as e:  # noqa: BLE001
        print(f"(mp4 skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
