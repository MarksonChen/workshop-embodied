"""Render direct-retarget references beside their exact-physics projections.

The two panels use the same fixed world-space camera and the same 50 Hz
timeline.  This deliberately keeps displacement loss visible; independently
following each torso would make a slow reproduction look deceptively similar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from brax.v1 import math
from brax.v1.envs import fetch
from brax.v1.io import image
from PIL import Image, ImageDraw, ImageFont

from demo_f.commands import hindsight_command
from demo_h.config import COMMAND_HORIZON_SECONDS, FPS
from demo_h.dataset.contract import DEFAULT_ROOT


HORIZON_SECONDS = COMMAND_HORIZON_SECONDS
CONNECTED_BODIES = jnp.arange(11)
DEFAULT_SPEEDS = (1.5, 2.0, 2.5, 3.0, 4.0)


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _reference_qp(system, target_index, root, quaternion, angles):
    qp = system.default_qp(joint_angle=jnp.asarray(angles))
    root_quaternion = jnp.asarray(quaternion)
    local = qp.pos[CONNECTED_BODIES] - qp.pos[:1]
    world = jax.vmap(math.rotate, in_axes=(0, None))(local, root_quaternion)
    position = qp.pos.at[CONNECTED_BODIES].set(jnp.asarray(root)[None] + world)
    position = position.at[target_index].set(jnp.asarray((1_000.0, 1_000.0, 2.0)))
    rotation = qp.rot.at[CONNECTED_BODIES].set(
        jax.vmap(math.quat_mul, in_axes=(None, 0))(
            root_quaternion, qp.rot[CONNECTED_BODIES]
        )
    )
    return qp.replace(pos=position, rot=rotation)


def _realized_qps(system, archive, index):
    base = system.default_qp()
    qp = base.replace(
        pos=jnp.asarray(archive["initial_qp_pos"][index]),
        rot=jnp.asarray(archive["initial_qp_rot"][index]),
        vel=jnp.asarray(archive["initial_qp_vel"][index]),
        ang=jnp.asarray(archive["initial_qp_ang"][index]),
    )
    output = [qp]
    for control in archive["normalized_control"][index]:
        qp, _ = system.step(qp, jnp.asarray(control))
        output.append(qp)
    return output


def _fixed_camera(system, example_qp, reference_root, realized_root, size):
    roots = np.concatenate((reference_root[:, :2], realized_root[:, :2]), axis=0)
    center = np.asarray(
        ((roots[:, 0].min() + roots[:, 0].max()) / 2,
         (roots[:, 1].min() + roots[:, 1].max()) / 2,
         0.0),
        np.float32,
    )
    initial_target = np.asarray((example_qp.pos[0, 0], example_qp.pos[0, 1], 0.0))
    offset = np.asarray(image._eye(system, example_qp)) - initial_target
    return image.Camera(
        viewWidth=size,
        viewHeight=size,
        position=(center + offset).tolist(),
        target=center.tolist(),
        up=image._up(system),
        hfov=58.0,
        vfov=58.0,
    )


def _panel(frame, heading, detail, font, small_font):
    panel = Image.fromarray(frame)
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel.width, 55), fill=(0, 0, 0, 175))
    draw.text((9, 5), heading, font=font, fill=(255, 255, 255, 255))
    draw.text((9, 31), detail, font=small_font, fill=(235, 235, 235, 255))
    return np.asarray(panel)


def _load_candidates(root: Path, split: str):
    manifest = json.loads((root / "manifest.json").read_text())
    candidates = []
    for row in manifest["sessions"]:
        if row["split"] != split or not row["released_clips"]:
            continue
        path = root / row["shard"]
        with np.load(path) as source:
            reference_command = hindsight_command(
                source["reference_root_position"],
                source["reference_root_quaternion"],
            )
            for index in range(len(reference_command)):
                candidates.append(
                    {
                        "row": row,
                        "path": path,
                        "index": index,
                        "reference_command": reference_command[index],
                        "realized_command": source["command"][index].copy(),
                        "parent_clip_id": int(source["parent_clip_id"][index]),
                        "joint_rmse": float(source["joint_tracking_rmse"][index]),
                        "root_rmse": float(source["root_tracking_rmse"][index]),
                        "saturation": float(source["control_saturation_fraction"][index]),
                    }
                )
    return candidates


def _select(candidates, requested_speeds):
    selected, used = [], set()
    for target in requested_speeds:
        def score(candidate):
            command = candidate["reference_command"]
            forward = command[0] / HORIZON_SECONDS
            lateral = abs(command[1]) / HORIZON_SECONDS
            turn = abs(command[2])
            duplicate = (candidate["path"], candidate["index"]) in used
            return abs(forward - target) + 0.5 * lateral + 0.5 * turn + 1e3 * duplicate

        choice = min(candidates, key=score)
        used.add((choice["path"], choice["index"]))
        selected.append((float(target), choice))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_h/out/projection_comparisons"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--loops", type=int, default=3)
    args = parser.parse_args()

    candidates = _load_candidates(args.dataset_root, args.split)
    selected = _select(candidates, args.speeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = fetch.Fetch()
    system = environment.sys
    target_index = system.body.index["Target"]
    font, small_font = _font(20), _font(14)
    report = []

    for ordinal, (requested_speed, candidate) in enumerate(selected, start=1):
        with np.load(candidate["path"]) as source:
            archive = {name: source[name] for name in source.files}
        index = candidate["index"]
        reference_qps = [
            _reference_qp(
                system,
                target_index,
                archive["reference_root_position"][index, frame],
                archive["reference_root_quaternion"][index, frame],
                archive["reference_joint_angles"][index, frame],
            )
            for frame in range(archive["reference_joint_angles"].shape[1])
        ]
        realized_qps = _realized_qps(system, archive, index)
        reference_speed = float(
            np.linalg.norm(candidate["reference_command"][:2]) / HORIZON_SECONDS
        )
        reference_forward = float(candidate["reference_command"][0] / HORIZON_SECONDS)
        realized_speed = float(
            np.linalg.norm(candidate["realized_command"][:2]) / HORIZON_SECONDS
        )
        camera = _fixed_camera(
            system,
            reference_qps[0],
            archive["reference_root_position"][index],
            archive["realized_root_position"][index],
            args.size,
        )
        frames = []
        for reference_qp, realized_qp in zip(reference_qps, realized_qps, strict=True):
            reference_frame = image.render_array(
                system, reference_qp, args.size, args.size, camera=camera, ssaa=1
            )
            realized_frame = image.render_array(
                system, realized_qp, args.size, args.size, camera=camera, ssaa=1
            )
            reference_frame = _panel(
                reference_frame,
                "KINEMATIC REFERENCE",
                f"hindsight speed {reference_speed:.2f}",
                font,
                small_font,
            )
            realized_frame = _panel(
                realized_frame,
                "PHYSICS REPLAY",
                f"hindsight speed {realized_speed:.2f}",
                font,
                small_font,
            )
            frames.append(np.concatenate((reference_frame, realized_frame), axis=1))
        frames = frames * max(args.loops, 1)
        output = args.output_dir / f"example_{ordinal}_reference-{reference_forward:.2f}.mp4"
        imageio.mimwrite(output, frames, fps=FPS, quality=8)
        row = {
            "example": ordinal,
            "requested_reference_forward_speed": requested_speed,
            "reference_forward_speed": reference_forward,
            "reference_path_speed": reference_speed,
            "realized_path_speed": realized_speed,
            "speed_retention": realized_speed / max(reference_speed, 1e-8),
            "session": candidate["row"]["session"],
            "parent_clip_id": candidate["parent_clip_id"],
            "joint_tracking_rmse": candidate["joint_rmse"],
            "root_tracking_rmse": candidate["root_rmse"],
            "control_saturation_fraction": candidate["saturation"],
            "video": str(output),
        }
        report.append(row)
        print(json.dumps(row), flush=True)

    report_path = args.output_dir / "comparisons.json"
    report_path.write_text(json.dumps({"schema": "demo-h-projection-comparisons-v1", "examples": report}, indent=2) + "\n")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
