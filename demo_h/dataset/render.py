"""Replay and render a physically projected Demo H clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as np
from brax.v1.envs import fetch
from brax.v1.io import html, image

from .contract import DEFAULT_ROOT


def replay(env, archive, index: int):
    base = env.sys.default_qp()
    qp = base.replace(
        pos=jnp.asarray(archive["initial_qp_pos"][index]),
        rot=jnp.asarray(archive["initial_qp_rot"][index]),
        vel=jnp.asarray(archive["initial_qp_vel"][index]),
        ang=jnp.asarray(archive["initial_qp_ang"][index]),
    )
    qps = [qp]
    for control in archive["normalized_control"][index]:
        qp, _ = env.sys.step(qp, jnp.asarray(control))
        qps.append(qp)
    return qps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip-index", type=int)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "manifest.json").read_text())
    row = next(row for row in manifest["sessions"] if row["released_clips"])
    shard = args.dataset_root / row["shard"]
    with np.load(shard) as source:
        archive = {name: source[name] for name in source.files}
    index = (
        int(np.argmax(archive["command"][:, 0]))
        if args.clip_index is None
        else args.clip_index
    )
    env = fetch.Fetch()
    qps = replay(env, archive, index)
    output = args.output or args.dataset_root / "projection_probe.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        np.asarray(image.render_array(env.sys, qp, args.size, args.size, ssaa=1))
        for qp in qps
    ]
    imageio.mimwrite(output, frames, fps=50, quality=8)
    output.with_suffix(".html").write_text(html.render(env.sys, qps))
    command = archive["command"][index]
    print(
        f"wrote {output} | session={row['session']} clip={index} "
        f"command={command.tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
