"""Honest 'gait vs scramble' diagnostic across decile checkpoints.

The render_deciles table showed foot cyclicity pinned at the autocorrelation floor
(~0.1 s / 10 Hz) -- the signature of a high-frequency buzz, not a stride. This does
a proper spectral decomposition to separate the two:

  - dominant foot-height frequency (Hz)
  - power fraction in the GAIT band (1-6 Hz ~ a real quadruped stride)
  - power fraction in the BUZZ band (>8 Hz ~ paddling/chatter)
  - stride-band autocorrelation peak (lag 10-60 only), i.e. is there ANY
    stride-scale rhythm hiding under the buzz?

A locomotor gait => dominant freq in the gait band and high gait-fraction.
A scramble => energy dominated by the buzz band. This tells us whether MORE
training converts the scramble into a gait, or just speeds the scramble up.

    env -u LD_LIBRARY_PATH uv run --isolated \
        --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
        python demo_a/analyze_gait.py [--steps 600]
"""
import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from brax.v1.envs import fetch as v1fetch
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

OUT = Path(__file__).resolve().parent / "out"
DEC = OUT / "deciles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpts = sorted(DEC.glob("fetch_*pct_*.pkl"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {DEC}")

    env = v1fetch.Fetch()
    dt = env.sys.config.dt
    fs = 1.0 / dt
    feet = [env.sys.body.index[b] for b in
            ["Front Left Lower", "Front Right Lower", "Back Left Lower", "Back Right Lower"]]

    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(network)
    reset = jax.jit(env.reset)

    @jax.jit
    def step(params, state, key):
        act, _ = make_policy(params, deterministic=True)(state.obs, key)
        return env.step(state, act)

    def bands(sig):
        """dominant freq, gait-band frac (1-6Hz), buzz-band frac (>8Hz)."""
        x = np.asarray(sig) - np.mean(sig)
        if np.std(x) < 1e-6:
            return 0.0, 0.0, 0.0
        x = x * np.hanning(len(x))
        psd = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), d=dt)
        psd[0] = 0.0
        tot = psd.sum() + 1e-12
        dom = f[np.argmax(psd)]
        gait = psd[(f >= 1) & (f < 6)].sum() / tot
        buzz = psd[f >= 8].sum() / tot
        return dom, gait, buzz

    def stride_autocorr(sig):
        """max autocorr in the STRIDE band only (lag 10-60 = 0.2-1.2 s)."""
        x = np.asarray(sig, float) - np.mean(sig)
        if np.std(x) < 1e-6:
            return 0.0, 0.0
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac = ac / ac[0]
        lo, hi = 10, min(60, len(x) // 2)
        if hi <= lo:
            return 0.0, 0.0
        seg = ac[lo:hi]
        k = int(np.argmax(seg)) + lo
        return float(ac[k]), k * dt

    print(f"{len(ckpts)} checkpoints | {args.steps} steps | fs={fs:.0f}Hz\n")
    print(f"{'pct':>4} {'domHz':>6} {'gaitPow':>8} {'buzzPow':>8} "
          f"{'stride_ac':>10} {'stride_T':>9}  verdict")
    for ck in ckpts:
        pct = int(ck.name.split("pct")[0].split("_")[-1])
        with open(ck, "rb") as f:
            params = pickle.load(f)
        rng = jax.random.PRNGKey(args.seed)
        state = reset(rng)
        fz = []
        for _ in range(args.steps):
            rng, key = jax.random.split(rng)
            state = step(params, state, key)
            fz.append([float(state.qp.pos[i, 2]) for i in feet])
        fz = np.asarray(fz)  # (T, 4)
        # aggregate over the 4 feet
        doms, gaits, buzzes, acs, Ts = [], [], [], [], []
        for j in range(4):
            d, g, b = bands(fz[:, j])
            a, t = stride_autocorr(fz[:, j])
            doms.append(d); gaits.append(g); buzzes.append(b); acs.append(a); Ts.append(t)
        dom, gait, buzz = np.mean(doms), np.mean(gaits), np.mean(buzzes)
        ac, T = np.mean(acs), np.mean(Ts)
        verdict = ("GAIT" if (1 <= dom < 6 and gait > buzz)
                   else "buzz/scramble" if buzz > gait else "mixed")
        print(f"{pct:>4} {dom:>6.1f} {gait:>8.2f} {buzz:>8.2f} "
              f"{ac:>10.2f} {T:>8.2f}s  {verdict}", flush=True)


if __name__ == "__main__":
    main()
