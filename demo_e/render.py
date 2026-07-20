"""Render a single or paired rollout produced by :mod:`demo_e.evaluate`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio_ffmpeg
import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .config import ENV, EVAL, OUT
from .env import build_env


def _panel(model, renderer, data, camera, qpos, label):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    camera.lookat[:] = data.xpos[_panel.torso]
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render().copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 27), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--arm", choices=("e0", "e1", "both"), default="both")
    parser.add_argument("--command", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arms = ("e0", "e1") if args.arm == "both" else (args.arm,)
    with np.load(args.rollouts) as source:
        trajectories = {
            arm: source[f"{arm}_c{args.command}_s{args.seed}_qpos"] for arm in arms
        }
        alive = {
            arm: source[f"{arm}_c{args.command}_s{args.seed}_alive"] > 0.5
            for arm in arms
        }

    env = build_env(beta=0.0, score_motion=False)
    model = env.unwrapped.mj_model
    data = mujoco.MjData(model)
    _panel.torso = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent"
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.38
    camera.azimuth = 135
    camera.elevation = -22
    renderer = mujoco.Renderer(model, height=480, width=640)
    command = EVAL.commands[args.command]
    frames = []
    try:
        length = min(len(value) for value in trajectories.values())
        for index in range(length):
            panels = []
            for arm in arms:
                # ``alive`` is the task continuation flag.  A false value can
                # mean height, torso-angle, or numerical termination; calling
                # every held terminal pose a visible "fall" is misleading.
                status = "active" if alive[arm][index] else "terminated"
                name = "task only" if arm == "e0" else "task + motion prior"
                label = (
                    f"{arm.upper()} {name} | vx={command[0]:.2f}, "
                    f"yaw={command[1]:+.2f} | t={index * ENV.control_dt:.2f}s | {status}"
                )
                panels.append(
                    _panel(
                        model,
                        renderer,
                        data,
                        camera,
                        trajectories[arm][index],
                        label,
                    )
                )
            frames.append(np.concatenate(panels, axis=1))
    finally:
        renderer.close()

    output = args.output or OUT / "videos" / (
        f"{args.arm}-command{args.command}-seed{args.seed}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    media.write_video(str(output), np.stack(frames), fps=1 / ENV.control_dt)
    print(output)


if __name__ == "__main__":
    main()
