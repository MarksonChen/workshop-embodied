from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from brax.v1 import math as brax_math
from brax.v1.envs import fetch
from brax.v1.io import image
from PIL import Image, ImageDraw

from .config import FPS, OUT


def make_states(environment, trajectory):
    body_indices = np.arange(11)
    target_index = environment.sys.body.index["Target"]
    states = []
    for angles, root_position, root_quaternion in zip(
        trajectory["angles"],
        trajectory["root_position"],
        trajectory["root_quaternion"],
        strict=True,
    ):
        state = environment.sys.default_qp(joint_angle=jnp.asarray(angles))
        local_position = state.pos[body_indices] - state.pos[0]
        world_position = root_position + jax.vmap(brax_math.rotate, in_axes=(0, None))(
            local_position, root_quaternion
        )
        world_rotation = jax.vmap(brax_math.quat_mul, in_axes=(None, 0))(
            root_quaternion, state.rot[body_indices]
        )
        position = state.pos.at[body_indices].set(world_position)
        rotation = state.rot.at[body_indices].set(world_rotation)
        position = position.at[target_index].set(jnp.asarray((1_000.0, 1_000.0, 2.0)))
        states.append(state.replace(pos=position, rot=rotation))
    return states


def label(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), text)
    draw.rectangle((0, 0, box[2] + 12, box[3] + 10), fill=(255, 255, 255))
    draw.text((6, 5), text, fill=(20, 20, 20))
    return np.asarray(canvas)


def render(
    input_dir: Path = OUT / "generated",
    output: Path = OUT / "motion_sweep.mp4",
    size: int = 320,
    columns: int = 4,
) -> Path:
    paths = sorted(Path(input_dir).glob("speed_*.npz"))
    if not paths:
        raise FileNotFoundError("generate Part 2 motions before rendering")
    environment = fetch.Fetch()
    panels = []
    for path in paths:
        with np.load(path) as archive:
            trajectory = {key: archive[key] for key in archive.files}
        states = make_states(environment, trajectory)
        requested = float(trajectory["requested_source_speed_mps"])
        realized = float(trajectory["realized_source_equivalent_speed_mps"])
        fetch_speed = float(trajectory["realized_fetch_forward_speed"])
        frames = [
            label(
                np.asarray(
                    image.render_array(environment.sys, state, size, size, ssaa=1)
                ),
                f"rodent {requested:.2f}->{realized:.2f} m/s | Fetch {fetch_speed:.2f} u/s",
            )
            for state in states
        ]
        panels.append(frames)
    rows = math.ceil(len(panels) / columns)
    blank = np.full_like(panels[0][0], 255)
    frames = []
    for frame_index in range(min(map(len, panels))):
        cells = [panel[frame_index] for panel in panels]
        cells.extend([blank] * (rows * columns - len(cells)))
        frame_rows = [
            np.concatenate(cells[row * columns : (row + 1) * columns], axis=1)
            for row in range(rows)
        ]
        frames.append(np.concatenate(frame_rows, axis=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, frames, fps=FPS, quality=8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=OUT / "generated")
    parser.add_argument("--output", type=Path, default=OUT / "motion_sweep.mp4")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    print(render(args.input_dir, args.output, args.size, args.columns))


if __name__ == "__main__":
    main()
