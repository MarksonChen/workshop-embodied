"""Render the six-speed aligned rollout and its time-varying speed audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demo_f.config import FPS
from demo_j.artifacts import ALIGNED_OUTPUT_ROOT, write_json
from demo_j.data.physics import XML_PATH


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default()
    )


def _label(frame, title, subtitle, font):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 43), fill=(0, 0, 0, 190))
    draw.text((7, 3), title, fill=(255, 255, 255, 255), font=font)
    draw.text((7, 23), subtitle, fill=(175, 235, 255, 255), font=font)
    return np.asarray(image)


def render(recording: Path, output: Path, *, panel_size: int, frame_stride: int):
    with np.load(recording) as archive:
        qpos = np.asarray(archive["qpos"], np.float32)
        target = np.asarray(archive["target_qpos"], np.float32)
        measured_speed = np.asarray(archive["measured_speed"], np.float32)
        alive = np.asarray(archive["alive"], bool)
        requested = np.asarray(archive["requested_speed"], np.float32)
        reference_speed = np.asarray(archive["reference_speed"], np.float32)
        realized_speed = np.asarray(archive["realized_speed"], np.float32)
    if len(qpos) != 6:
        raise ValueError("the aligned comparison renderer expects six speeds")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=panel_size, width=panel_size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 5.6
    camera.azimuth = 135
    camera.elevation = -18
    font = _font(max(11, panel_size // 20))

    def render_one(values):
        data.qpos[:] = values
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        camera.lookat[:] = (values[0], values[1], 0.75)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output, fps=FPS / frame_stride, quality=8, macro_block_size=2
    ) as writer:
        for frame in range(0, qpos.shape[1], frame_stride):
            panels = []
            for index in range(6):
                target_panel = _label(
                    render_one(target[index, frame]),
                    f"{requested[index]:.2f} requested | periodic target",
                    f"reference {reference_speed[index]:.2f} units/s",
                    font,
                )
                policy_panel = _label(
                    render_one(qpos[index, frame]),
                    f"{requested[index]:.2f} requested | aligned SNN",
                    f"realized {realized_speed[index]:.2f} units/s",
                    font,
                )
                panels.append(np.concatenate((target_panel, policy_panel), axis=1))
            tiled = np.concatenate(
                (
                    np.concatenate(panels[:3], axis=1),
                    np.concatenate(panels[3:], axis=1),
                ),
                axis=0,
            )
            writer.append_data(tiled)
    renderer.close()

    smoothing_bins = max(1, round(0.5 * FPS))
    kernel = np.ones(smoothing_bins, np.float32) / smoothing_bins
    time_axis = np.arange(measured_speed.shape[1], dtype=np.float32) / FPS
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.0), sharex=True, sharey=True)
    for index, axis in enumerate(axes.flat):
        smoothed = np.convolve(measured_speed[index], kernel, mode="same")
        smoothed[~alive[index]] = np.nan
        axis.plot(time_axis, smoothed, color="#007C91", linewidth=1.5)
        axis.axhline(requested[index], color="#D55E00", linestyle="--", linewidth=1.2)
        axis.set_title(
            f"command {requested[index]:.2f}; mean {realized_speed[index]:.2f}"
        )
        axis.grid(alpha=0.18)
    figure.supxlabel("rollout time (s)")
    figure.supylabel("forward speed (Fetch units/s; 0.5 s mean)")
    figure.suptitle("Aligned SNN: instantaneous speed over 1,000 control steps")
    figure.tight_layout()
    speed_plot = output.with_name(f"{output.stem}_speed.png")
    speed_plot_svg = speed_plot.with_suffix(".svg")
    figure.savefig(speed_plot, dpi=220)
    figure.savefig(speed_plot_svg)
    plt.close(figure)
    report = {
        "schema": "demo-j-aligned-periodic-comparison-video-v1",
        "recording": str(recording),
        "output": str(output),
        "steps": int(qpos.shape[1] - 1),
        "rendered_frames": len(range(0, qpos.shape[1], frame_stride)),
        "fps": FPS / frame_stride,
        "layout": "2 rows x 3 speeds; periodic target and SNN side by side",
        "reference_kind": "explicit synthetic periodic extension",
        "speed_plot": str(speed_plot),
        "speed_smoothing_seconds": smoothing_bins / FPS,
    }
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_ROOT / "rollout_1000.mp4"
    )
    parser.add_argument("--panel-size", type=int, default=240)
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args()
    render(
        args.recording,
        args.output,
        panel_size=args.panel_size,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
