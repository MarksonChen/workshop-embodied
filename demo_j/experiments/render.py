"""Render six held-out target-versus-SNN trajectories in one audit video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo_f.config import FPS
from demo_j.data.physics import XML_PATH
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default()
    )


def _label(frame: np.ndarray, title: str, subtitle: str, font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0, 190))
    draw.text((7, 4), title, fill=(255, 255, 255, 255), font=font)
    draw.text((7, 23), subtitle, fill=(175, 235, 255, 255), font=font)
    return np.asarray(image)


def _speed(qpos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(qpos[..., :2], axis=1), axis=-1).mean(axis=1) * FPS


def _quantile_examples(speed: np.ndarray, count: int = 6) -> np.ndarray:
    quantiles = np.linspace(0.05, 0.95, count)
    targets = np.quantile(speed, quantiles)
    selected = []
    for target in targets:
        order = np.argsort(np.abs(speed - target))
        selected.append(
            next(int(index) for index in order if int(index) not in selected)
        )
    return np.asarray(selected, np.int32)


def render_comparison(
    recording: Path,
    output: Path,
    *,
    reference_root: Path = PROJECTED_ROOT,
    panel_size: int = 240,
    frame_stride: int = 2,
) -> dict[str, object]:
    reference = load_projected_reference("test", reference_root)
    with np.load(recording) as archive:
        policy_qpos = np.asarray(archive["qpos"], np.float32)
    frames = min(policy_qpos.shape[1], reference.frames)
    target_qpos = reference.qpos[:, :frames]
    speeds = _speed(target_qpos)
    policy_speeds = _speed(policy_qpos[:, :frames])
    examples = _quantile_examples(speeds)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=panel_size, width=panel_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 5.6
    camera.azimuth = 135
    camera.elevation = -18
    font = _font(max(11, panel_size // 20))

    def render_one(qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = (qpos[0], qpos[1], 0.75)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=FPS / frame_stride, quality=8) as writer:
        for frame in range(0, frames, frame_stride):
            panels = []
            for rank, clip in enumerate(examples):
                target = _label(
                    render_one(target_qpos[clip, frame]),
                    f"example {rank + 1} | target",
                    f"clip {clip} | {speeds[clip]:.2f} units/s",
                    font,
                )
                policy = _label(
                    render_one(policy_qpos[clip, frame]),
                    f"example {rank + 1} | SNN",
                    f"closed loop | {policy_speeds[clip]:.2f} units/s",
                    font,
                )
                panels.append(np.concatenate((target, policy), axis=1))
            tiled = np.concatenate(
                (
                    np.concatenate(panels[:3], axis=1),
                    np.concatenate(panels[3:], axis=1),
                ),
                axis=0,
            )
            writer.append_data(tiled)
    renderer.close()
    report = {
        "schema": "demo-j-target-policy-video-v1",
        "recording": str(recording),
        "output": str(output),
        "split": "test",
        "examples": [
            {
                "clip": int(clip),
                "target_speed_fetch_units_per_s": float(speeds[clip]),
                "snn_speed_fetch_units_per_s": float(policy_speeds[clip]),
            }
            for clip in examples
        ],
        "frames": len(range(0, frames, frame_stride)),
        "fps": FPS / frame_stride,
        "layout": "2 rows x 3 examples; target and SNN side by side",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    parser.add_argument("--panel-size", type=int, default=240)
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()
    render_comparison(
        args.recording,
        args.output,
        reference_root=args.reference_root,
        panel_size=args.panel_size,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
