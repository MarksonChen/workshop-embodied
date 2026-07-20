"""Plot time-varying diagnostics for one paired Demo G rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .env import BUFFER_FRAMES
from .evaluate import load_params, make_rollout_runtime, stack_params
from .prior import DEFAULT_PRIOR, load_prior


OUT = Path(__file__).resolve().parent / "out" / "videos"


def trailing_mean(values: np.ndarray, width: int = 10) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (width - 1, 0)), mode="edge")
    return np.stack(
        [np.convolve(row, np.ones(width) / width, mode="valid") for row in padded]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=OUT / "seed0_trace.png")
    args = parser.parse_args()

    prior = load_prior(args.prior)
    env, reset, paired_step = make_rollout_runtime(prior)
    params = stack_params(load_params(args.g0), load_params(args.g1))
    rng = jax.random.PRNGKey(args.seed)
    states = reset(jnp.stack((rng, rng)))
    trace = {name: [] for name in ("speed", "upright", "height", "contacts", "features")}
    for _ in range(args.steps):
        rng, action_key = jax.random.split(rng)
        states, _ = paired_step(states, jnp.stack((action_key, action_key)), params)
        features = np.asarray(states.info["prior_features"][:, -1])
        trace["features"].append(features)
        trace["speed"].append(np.asarray(states.metrics["speed"]))
        trace["upright"].append(np.asarray(states.metrics["upright"]))
        trace["height"].append(features[:, 2])
        trace["contacts"].append(features[:, 56:60])

    values = {name: np.asarray(rows).swapaxes(0, 1) for name, rows in trace.items()}
    contact_switch = np.zeros((2, args.steps), np.float32)
    contact_switch[:, 1:] = (
        np.abs(np.diff(values["contacts"], axis=1)).mean(axis=-1) * 50.0
    )
    logp = np.full((2, args.steps), np.nan, np.float32)
    windows, owners, frames = [], [], []
    for arm in range(2):
        for frame in range(BUFFER_FRAMES - 1, args.steps, 4):
            windows.append(values["features"][arm, frame - BUFFER_FRAMES + 1 : frame + 1])
            owners.append(arm)
            frames.append(frame)
    command = jnp.asarray(
        (prior.command_scale * prior.source_speed_mps, 0.0, 0.0), jnp.float32
    )
    scores = np.asarray(
        jax.jit(jax.vmap(lambda window: prior.log_prob(window, command)))(
            jnp.asarray(np.stack(windows))
        )
    )
    for arm, frame, score in zip(owners, frames, scores, strict=True):
        logp[arm, frame] = score

    time = np.arange(args.steps) / 50.0
    colors = ("#c85c5c", "#365f9d")
    labels = ("G0 task only", "G1 + frozen prior")
    figure, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True, dpi=150)
    for arm in range(2):
        axes[0].plot(time, values["speed"][arm], color=colors[arm], alpha=0.2)
        axes[0].plot(time, trailing_mean(values["speed"])[arm], color=colors[arm], label=labels[arm])
        axes[1].plot(time, values["upright"][arm], color=colors[arm])
        axes[2].plot(time, values["contacts"][arm].sum(axis=-1), color=colors[arm])
        axes[3].plot(time, trailing_mean(contact_switch, 10)[arm], color=colors[arm])
        mask = np.isfinite(logp[arm])
        axes[4].plot(time[mask], logp[arm, mask], color=colors[arm], marker=".", markersize=2)
    axes[0].axhline(
        env.v_target, color="black", linestyle="--", linewidth=1, label="target"
    )
    axes[0].set_ylabel("speed")
    axes[1].set_ylabel("upright")
    axes[2].set_ylabel("feet in contact")
    axes[3].set_ylabel("switches/s")
    axes[4].set_ylabel("raw log p")
    axes[4].set_xlabel("time (s)")
    axes[0].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Demo G paired rollout diagnostics (seed 101)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    np.savez_compressed(args.output.with_suffix(".npz"), time=time, logp=logp, **values)
    print(f"wrote {args.output} and {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
