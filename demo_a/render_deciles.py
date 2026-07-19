"""Render one short video per decile checkpoint (0,10,...,100% of training) so the
gait's evolution is visible, and print an objective per-decile table:

  - reward / hits over the rollout (is it reaching targets?)
  - net torso displacement + mean speed + path straightness (is it locomoting?)
  - foot-height cyclicity: peak of the normalized autocorrelation of a hind/front
    foot's height, and its period. A clear peak (strength ~>0.3) at a plausible
    period (~0.2-1.0 s) is the signature of a *periodic gait* rather than a scramble.

All deciles are rolled out from the SAME seed, so the target sequence is identical
across snapshots and any difference is due to the policy alone.

    env -u LD_LIBRARY_PATH uv run --isolated \
        --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' --with 'imageio[ffmpeg]' \
        python demo_a/render_deciles.py [--steps 300] [--seed 0]
"""
import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from brax.v1.envs import fetch as v1fetch
from brax.v1.io import image
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

OUT = Path(__file__).resolve().parent / "out"
DEC = OUT / "deciles"


def cyclicity(signal, dt, min_lag=5):
    """Return (strength in [-1,1], period_s) of the strongest self-repeat."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    if x.std() < 1e-6:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    hi = max(min_lag + 1, len(x) // 2)
    seg = ac[min_lag:hi]
    if len(seg) == 0:
        return 0.0, 0.0
    k = int(np.argmax(seg)) + min_lag
    return float(ac[k]), k * dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    args = ap.parse_args()

    ckpts = sorted(DEC.glob("fetch_*pct_*.pkl"))
    if not ckpts:
        raise SystemExit(f"no decile checkpoints in {DEC} -- run train_fetch.py --save_deciles")
    print(f"{len(ckpts)} checkpoints | {args.steps} steps/rollout | seed {args.seed}")

    env = v1fetch.Fetch()
    dt = env.sys.config.dt
    torso_i = env.torso_idx
    foot_i = env.sys.body.index["Front Left Lower"]

    # Build the network ONCE and jit a params-threaded step so it compiles once and
    # is reused for every checkpoint (params differ but pytree structure is identical).
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(network)
    reset = jax.jit(env.reset)

    @jax.jit
    def step(params, state, key):
        act, _ = make_policy(params, deterministic=True)(state.obs, key)
        return env.step(state, act)

    try:
        import imageio.v2 as imageio
        have_mp4 = True
    except Exception:  # noqa: BLE001
        have_mp4 = False
        print("(imageio missing -> HTML only, no mp4)")

    print(f"\n{'pct':>4} {'reward':>8} {'hits':>5} {'netXY':>7} {'speed':>6} "
          f"{'straight':>8} {'cyc':>6} {'period':>7}")
    rows = []
    for ck in ckpts:
        pct = int(ck.name.split("pct")[0].split("_")[-1])
        with open(ck, "rb") as f:
            params = pickle.load(f)

        rng = jax.random.PRNGKey(args.seed)
        state = reset(rng)
        qps = [state.qp]
        txy, fz, rew, hits = [], [], [], 0.0
        for _ in range(args.steps):
            rng, key = jax.random.split(rng)
            state = step(params, state, key)
            qps.append(state.qp)
            txy.append(np.asarray(state.qp.pos[torso_i, :2]))
            fz.append(float(state.qp.pos[foot_i, 2]))
            rew.append(float(state.reward))
            hits += float(state.metrics["hits"])

        txy = np.asarray(txy)
        steps_disp = np.linalg.norm(np.diff(txy, axis=0), axis=1)
        path_len = float(steps_disp.sum())
        net = float(np.linalg.norm(txy[-1] - txy[0]))
        speed = path_len / (args.steps * dt)
        straight = net / (path_len + 1e-6)
        cyc, period = cyclicity(fz, dt)
        rows.append((pct, sum(rew), hits, net, speed, straight, cyc, period))
        print(f"{pct:>4} {sum(rew):>8.1f} {hits:>5.0f} {net:>7.1f} {speed:>6.2f} "
              f"{straight:>8.2f} {cyc:>6.2f} {period:>6.2f}s", flush=True)

        if have_mp4:
            frames = [np.asarray(image.render_array(env.sys, qp, args.width, args.height, ssaa=1))
                      for qp in qps]
            mp4 = OUT / f"decile_{pct:03d}pct.mp4"
            imageio.mimwrite(mp4, frames, fps=int(round(1 / dt)))

    if have_mp4:
        print(f"\nwrote {len(rows)} videos: {OUT}/decile_XXXpct.mp4")


if __name__ == "__main__":
    main()
