"""Render every row of a Demo H speed sweep into one labeled video."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import numpy as np
from brax.v1 import math as brax_math
from brax.v1.envs import fetch
from brax.v1.io import image
from brax.v1.physics.base import vec_to_arr
from PIL import Image, ImageDraw, ImageFont
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light

from demo_h.config import FPS


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _load_qps(env, trace_path: Path):
    with np.load(trace_path) as source:
        trace = {name: source[name] for name in source.files}
    base = env.sys.default_qp()
    target_index = env.sys.body.index["Target"]

    def make_qp(prefix: str, index=None):
        suffix = "initial_qp" if prefix == "initial" else "qp"
        select = (lambda value: value) if index is None else (lambda value: value[index])
        qp = base.replace(
            pos=jnp.asarray(select(trace[f"{suffix}_pos"])),
            rot=jnp.asarray(select(trace[f"{suffix}_rot"])),
            vel=jnp.asarray(select(trace[f"{suffix}_vel"])),
            ang=jnp.asarray(select(trace[f"{suffix}_ang"])),
        )
        return qp.replace(
            pos=qp.pos.at[target_index].set(jnp.asarray((1_000.0, 1_000.0, 2.0)))
        )

    return [make_qp("initial")] + [
        make_qp("step", index) for index in range(len(trace["qp_pos"]))
    ]


def _label(frame: np.ndarray, lines: tuple[str, str], font) -> np.ndarray:
    panel = Image.fromarray(frame)
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel.width, 42), fill=(0, 0, 0, 190))
    draw.text((7, 4), lines[0], fill=(255, 255, 255, 255), font=font)
    draw.text((7, 22), lines[1], fill=(170, 255, 180, 255), font=font)
    return np.asarray(panel)


class _ReusableRenderer:
    """Reuse static geometry instead of rebuilding a Brax scene every frame."""

    def __init__(self, sys, qp, width: int, height: int):
        self.sys = sys
        self.width = width
        self.height = height
        self.scene, self.instances = image._scene(sys, qp)

    def render(self, qp) -> np.ndarray:
        instance_index = 0
        for body_index, body in enumerate(self.sys.config.bodies):
            for collider in body.colliders:
                offset = np.asarray(
                    (
                        collider.position.x,
                        collider.position.y,
                        collider.position.z,
                    )
                )
                position = np.asarray(qp.pos[body_index]) + brax_math.rotate(
                    offset, qp.rot[body_index]
                )
                rotation = brax_math.euler_to_quat(vec_to_arr(collider.rotation))
                rotation = brax_math.quat_mul(qp.rot[body_index], rotation)
                instance = self.instances[instance_index]
                self.scene.set_object_position(instance, list(position))
                self.scene.set_object_orientation(
                    instance,
                    [rotation[1], rotation[2], rotation[3], rotation[0]],
                )
                instance_index += 1

        target = [qp.pos[0, 0], qp.pos[0, 1], 0]
        light = Light(
            direction=[0.57735, -0.57735, 0.57735],
            ambient=0.8,
            diffuse=0.8,
            specular=0.6,
            shadowmap_center=target,
        )
        horizontal_fov = 58.0
        camera = Camera(
            viewWidth=self.width,
            viewHeight=self.height,
            position=image._eye(self.sys, qp),
            target=target,
            up=image._up(self.sys),
            hfov=horizontal_fov,
            vfov=horizontal_fov * self.height / self.width,
        )
        pixels = self.scene.get_camera_image(
            self.instances, light, camera
        ).rgb
        return np.asarray(pixels, dtype=np.uint8).reshape(
            self.height, self.width, -1
        )


def render_sweeps(
    metrics_paths,
    output: Path,
    *,
    panel_size: int = 288,
    columns: int = 3,
    frame_stride: int = 2,
    render_workers: int = 0,
) -> None:
    """Render one or more compatible sweep reports into a tiled MP4."""

    if frame_stride < 1 or render_workers < 0 or columns < 1:
        raise ValueError("stride/columns must be positive and workers non-negative")
    started = time.perf_counter()
    rows = []
    for metrics_path in map(Path, metrics_paths):
        report = json.loads(metrics_path.read_text())
        for source_row in report["speeds"]:
            row = dict(source_row)
            row["group_label"] = report.get("label", report["arm"])
            rows.append(row)
    if not rows:
        raise ValueError("sweep reports contain no trajectories")
    env = fetch.Fetch()
    trajectories = [_load_qps(env, Path(row["trace"])) for row in rows]
    frame_count = min(map(len, trajectories))
    columns = min(columns, len(rows))
    rows_count = math.ceil(len(rows) / columns)
    font = _font(max(11, panel_size // 22))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderers = [
        _ReusableRenderer(env.sys, trajectory[0], panel_size, panel_size)
        for trajectory in trajectories
    ]
    worker_count = render_workers or min(
        len(renderers), max(1, os.cpu_count() or 1), 12
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor, imageio.get_writer(
        output, fps=FPS / frame_stride, quality=8
    ) as writer:
        for frame_index in range(0, frame_count, frame_stride):
            rendered_frames = list(
                executor.map(
                    lambda pair: pair[0].render(pair[1][frame_index]),
                    zip(renderers, trajectories),
                )
            )
            panels = []
            for row, rendered in zip(rows, rendered_frames):
                command = row.get(
                    "commanded_speed_fetch_units_per_s", row.get("commanded_speed_mps")
                )
                actual = row.get(
                    "realized_speed_mean_fetch_units_per_s",
                    row.get("realized_speed_mean_mps"),
                )
                passed = row["four_limb_stride"]["passes_four_limb_stride_gate"]
                panels.append(
                    _label(
                        rendered,
                        (f"{row['group_label']} | command {command:.1f} Fetch units/s",
                         f"actual {actual:.2f} | "
                         f"four-limb stride: {'PASS' if passed else 'FAIL'}"),
                        font,
                    )
                )
            blank = np.zeros_like(panels[0])
            panels.extend([blank] * (rows_count * columns - len(panels)))
            tiled = np.concatenate(
                [
                    np.concatenate(
                        panels[row * columns : (row + 1) * columns], axis=1
                    )
                    for row in range(rows_count)
                ],
                axis=0,
            )
            writer.append_data(tiled)
    print(
        f"wrote {output} in {time.perf_counter() - started:.1f}s "
        f"({len(range(0, frame_count, frame_stride))} frames, "
        f"{worker_count} render workers)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=288)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=0)
    args = parser.parse_args()
    render_sweeps(
        args.metrics,
        args.output,
        panel_size=args.panel_size,
        columns=args.columns,
        frame_stride=args.frame_stride,
        render_workers=args.render_workers,
    )


if __name__ == "__main__":
    main()
