"""Render retargeted trajectories with the original Brax v1 Fetch body.

This file intentionally imports the historical Brax v1 renderer and should be
run in the same isolated environment as Demo A:

    env -u LD_LIBRARY_PATH uv run --isolated \
      --with 'brax==0.12.3' --with 'jax==0.4.30' --with 'jaxlib==0.4.30' \
      --with 'imageio[ffmpeg]' python -m demo_f.render
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import numpy as np
from PIL import Image, ImageDraw

from brax.v1 import math
from brax.v1.envs import fetch
from brax.v1.io import html, image

from .config import FPS, INSPECTION_CLIPS


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "out" / "retarget"


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), text)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.rounded_rectangle((10, 10, width + 26, height + 22), radius=5, fill=(255, 255, 255, 225))
    draw.text((18, 15), text, fill=(20, 20, 20))
    return np.asarray(canvas)


def make_qps(env, trajectory: dict):
    connected = np.arange(11)
    target_index = env.sys.body.index["Target"]
    qps = []
    for angles, root_position, root_quaternion in zip(
        trajectory["angles"],
        trajectory["root_position"],
        trajectory["root_quaternion"],
        strict=True,
    ):
        # Explicit JAX conversion keeps ``default_qp`` on its immutable-array
        # path; NumPy input would produce arrays without ``.at`` updates.
        qp = env.sys.default_qp(joint_angle=jp.asarray(angles))
        local_position = qp.pos[connected] - qp.pos[0]
        world_position = root_position + jax.vmap(
            math.rotate, in_axes=(0, None)
        )(local_position, root_quaternion)
        world_rotation = jax.vmap(
            math.quat_mul, in_axes=(None, 0)
        )(root_quaternion, qp.rot[connected])
        positions = qp.pos.at[connected].set(world_position)
        rotations = qp.rot.at[connected].set(world_rotation)
        positions = positions.at[target_index].set(jp.asarray((1_000.0, 1_000.0, 2.0)))
        qps.append(qp.replace(pos=positions, rot=rotations))
    return qps


def render_one(path: Path, size: int) -> tuple[Path, list[np.ndarray]]:
    with np.load(path) as data:
        trajectory = {key: data[key] for key in data.files}
    env = fetch.Fetch()
    qps = make_qps(env, trajectory)
    if "requested_source_speed_mps" in trajectory:
        requested = float(trajectory["requested_source_speed_mps"])
        realized = float(trajectory["realized_source_equivalent_speed_mps"])
        command = float(trajectory["fetch_command"][0])
        label = (
            f"Demo F  |  request {requested:.2f} m/s  |  "
            f"realized ~{realized:.2f}  |  Fetch dx={command:.2f}"
        )
    else:
        speed = float(trajectory["source_speed_mps"])
        session = str(trajectory["source_session"])
        start = int(trajectory["source_start"])
        label = f"Coltrane {speed:.3f} m/s  |  {session}@{start}  |  retargeted Fetch"
    frames = []
    for frame_index, qp in enumerate(qps):
        frame = np.asarray(image.render_array(env.sys, qp, size, size, ssaa=1))
        frames.append(_overlay(frame, f"{label}  |  t={frame_index / FPS:.2f}s"))
    output = path.with_name(f"{path.stem}_fetch.mp4")
    imageio.mimwrite(output, frames, fps=int(trajectory["fps"]), quality=8)
    html_path = path.with_name(f"{path.stem}_fetch.html")
    html_path.write_text(html.render(env.sys, qps))
    print(f"wrote {output}", flush=True)
    print(f"wrote {html_path}", flush=True)
    return output, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--mode", choices=("retarget", "generated"), default="retarget")
    args = parser.parse_args()
    if args.mode == "generated":
        paths = sorted(args.input_dir.glob("speed_*.npz"))
    else:
        paths = [args.input_dir / f"{spec.label}.npz" for spec in INSPECTION_CLIPS]
    if not paths:
        raise SystemExit(f"no {args.mode} trajectory artifacts in {args.input_dir}")
    if len(paths) != 4:
        raise SystemExit(
            f"the synchronized workshop grid requires 4 trajectories, found {len(paths)}"
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(
            f"missing retarget artifacts {missing}; run demo_f.retarget first"
        )
    rendered = [render_one(path, args.size) for path in paths]

    # A synchronized 2x2 view makes the stride-frequency/speed relationship
    # directly inspectable without opening four players.
    length = min(len(frames) for _, frames in rendered)
    grid_frames = []
    for frame_index in range(length):
        top = np.concatenate((rendered[0][1][frame_index], rendered[1][1][frame_index]), axis=1)
        bottom = np.concatenate((rendered[2][1][frame_index], rendered[3][1][frame_index]), axis=1)
        grid_frames.append(np.concatenate((top, bottom), axis=0))
    grid_name = "speed_grid_demo_f.mp4" if args.mode == "generated" else "speed_grid_fetch.mp4"
    grid_path = args.input_dir / grid_name
    imageio.mimwrite(grid_path, grid_frames, fps=FPS, quality=8)
    print(f"wrote {grid_path}", flush=True)


if __name__ == "__main__":
    main()
