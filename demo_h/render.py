"""Render a saved exact-physics Demo H P0 or PPO trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as np
from brax.v1.envs import fetch
from brax.v1.io import html, image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    with np.load(args.trace) as source:
        trace = {name: source[name] for name in source.files}
    env = fetch.Fetch()
    base = env.sys.default_qp()
    target_index = env.sys.body.index["Target"]

    def without_unused_target(qp):
        return qp.replace(
            pos=qp.pos.at[target_index].set(jnp.asarray((1_000.0, 1_000.0, 2.0)))
        )

    initial = base.replace(
        pos=jnp.asarray(trace["initial_qp_pos"]),
        rot=jnp.asarray(trace["initial_qp_rot"]),
        vel=jnp.asarray(trace["initial_qp_vel"]),
        ang=jnp.asarray(trace["initial_qp_ang"]),
    )
    qps = [without_unused_target(initial)]
    for index in range(len(trace["qp_pos"])):
        qps.append(
            without_unused_target(base.replace(
                pos=jnp.asarray(trace["qp_pos"][index]),
                rot=jnp.asarray(trace["qp_rot"][index]),
                vel=jnp.asarray(trace["qp_vel"][index]),
                ang=jnp.asarray(trace["qp_ang"][index]),
            ))
        )
    output = args.output or args.trace.with_suffix(".mp4")
    frames = [
        np.asarray(image.render_array(env.sys, qp, args.size, args.size, ssaa=1))
        for qp in qps
    ]
    imageio.mimwrite(output, frames, fps=50, quality=8)
    output.with_suffix(".html").write_text(html.render(env.sys, qps))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
