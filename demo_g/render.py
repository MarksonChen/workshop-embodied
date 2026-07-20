"""Render a predeclared Demo G checkpoint pair as individual and paired MP4s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from brax.v1.io import image as brax_image

from .config import DEFAULT_BETA
from .evaluate import load_params, make_rollout_runtime, sha256, stack_params
from .prior import DEFAULT_PRIOR, load_prior


OUT = Path(__file__).resolve().parent / "out" / "videos"


def checkpoint_beta(path: Path) -> float:
    metadata = path.with_suffix(".json")
    if metadata.is_file():
        return float(json.loads(metadata.read_text()).get("beta", DEFAULT_BETA))
    return DEFAULT_BETA


def label(frame: np.ndarray, title: str, speed: float, time_seconds: float) -> np.ndarray:
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 34), fill=(255, 255, 255))
    draw.text((8, 5), title, fill=(20, 20, 20))
    draw.text(
        (8, 19), f"t={time_seconds:4.1f}s   forward speed={speed:4.2f}", fill=(40, 40, 40)
    )
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    import imageio.v2 as imageio

    prior = load_prior(args.prior)
    runtime = make_rollout_runtime(prior)
    env, reset, paired_step = runtime
    params = stack_params(load_params(args.g0), load_params(args.g1))
    rng = jax.random.PRNGKey(args.seed)
    states = reset(jnp.stack((rng, rng)))
    beta = checkpoint_beta(args.g1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "g0": args.output_dir / "seed0_g0_task_only.mp4",
        "g1": args.output_dir / "seed0_g1_motion_prior.mp4",
        "pair": args.output_dir / "seed0_g0_vs_g1.mp4",
    }
    writers = {
        name: imageio.get_writer(path, fps=50, codec="libx264", quality=8)
        for name, path in paths.items()
    }
    try:
        for frame_index in range(args.steps + 1):
            speeds = np.asarray(states.metrics["speed"])
            rendered = []
            titles = ("G0 — task reward only", f"G1 — task + frozen prior (beta={beta:g})")
            for arm, title in enumerate(titles):
                qp = jax.tree_util.tree_map(lambda value: value[arm], states.qp)
                pixels = brax_image.render_array(
                    env.sys, qp, args.width, args.height, ssaa=1
                )
                rendered.append(
                    label(pixels, title, float(speeds[arm]), frame_index / 50.0)
                )
            writers["g0"].append_data(rendered[0])
            writers["g1"].append_data(rendered[1])
            writers["pair"].append_data(np.concatenate(rendered, axis=1))
            if frame_index == args.steps:
                break
            rng, action_key = jax.random.split(rng)
            states, _ = paired_step(states, jnp.stack((action_key, action_key)), params)
    finally:
        for writer in writers.values():
            writer.close()

    report = {
        "schema": "demo-g-render-v1",
        "seed": args.seed,
        "steps": args.steps,
        "fps": 50,
        "g0_checkpoint": str(args.g0),
        "g0_sha256": sha256(args.g0),
        "g1_checkpoint": str(args.g1),
        "g1_sha256": sha256(args.g1),
        "beta": beta,
        "videos": {name: str(path) for name, path in paths.items()},
    }
    metadata = args.output_dir / "seed0_render.json"
    metadata.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
