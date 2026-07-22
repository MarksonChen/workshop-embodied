from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.v1.io import image

from .environment import FetchRun


OUT = Path(__file__).resolve().parent / "out"


def latest_checkpoint() -> Path:
    checkpoints = sorted(OUT.glob("policy_*.pkl"))
    if not checkpoints:
        raise FileNotFoundError("train Part 1 before rendering")
    return checkpoints[-1]


def render(
    checkpoint: Path,
    output: Path = OUT / "rollout.mp4",
    target_speed: float = 3.0,
    steps: int = 250,
    seed: int = 0,
) -> dict:
    with Path(checkpoint).open("rb") as handle:
        params = pickle.load(handle)
    environment = FetchRun(target_speed=target_speed)
    networks = ppo_networks.make_ppo_networks(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    policy = make_policy(params, deterministic=True)
    reset = jax.jit(environment.reset)

    @jax.jit
    def step(state, key):
        action, _ = policy(state.obs, key)
        return environment.step(state, action)

    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    states = [state.qp]
    speeds = []
    rewards = []
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        state = step(state, key)
        states.append(state.qp)
        speeds.append(float(state.metrics["speed"]))
        rewards.append(float(state.reward))
    frames = [
        np.asarray(image.render_array(environment.sys, qp, 480, 352, ssaa=1))
        for qp in states
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, frames, fps=round(1 / environment.sys.config.dt))
    report = {
        "checkpoint": str(checkpoint),
        "target_speed": target_speed,
        "mean_speed": float(np.mean(speeds)),
        "speed_rmse": float(np.sqrt(np.mean((np.asarray(speeds) - target_speed) ** 2))),
        "return": float(np.sum(rewards)),
        "video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=OUT / "rollout.mp4")
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = render(
        args.checkpoint or latest_checkpoint(),
        args.output,
        args.speed,
        args.steps,
        args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
