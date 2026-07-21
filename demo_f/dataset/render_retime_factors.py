"""Compare modest temporal slowdowns of representative retargeted clips."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from brax.v1.envs import fetch
from brax.v1.io import image
from PIL import Image, ImageDraw, ImageFont

from demo_f.dataset.render_samples import (
    DEFAULT_DEMO_H,
    DEFAULT_PARENT,
    FPS,
    _load_trajectory,
    _retained_rows,
)
from demo_f.render import make_qps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "out" / "retime_factor_comparisons"
DEFAULT_FACTORS = (1.10, 1.25, 1.40, 1.60, 1.80, 2.00)
DEFAULT_QUANTILES = (0.25, 0.50, 0.75)


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _linear(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    lower = np.floor(time).astype(np.int32)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (time - lower).astype(np.float32)
    shape = (len(time),) + (1,) * (values.ndim - 1)
    return values[lower] * (1.0 - weight.reshape(shape)) + values[upper] * weight.reshape(shape)


def _yaw(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quaternion(yaw: np.ndarray) -> np.ndarray:
    output = np.zeros((len(yaw), 4), np.float32)
    output[:, 0] = np.cos(yaw / 2)
    output[:, 3] = np.sin(yaw / 2)
    return output


def retime(trajectory: dict, factor: float) -> dict:
    """Time-dilate one complete trajectory while retaining 50 Hz output."""

    frames = len(trajectory["angles"])
    target_frames = int(math.floor((frames - 1) * factor)) + 1
    source_time = np.arange(target_frames, dtype=np.float64) / factor
    yaw = _linear(_yaw(trajectory["root_quaternion"])[:, None], source_time)[:, 0]
    return {
        "angles": _linear(trajectory["angles"], source_time),
        "root_position": _linear(trajectory["root_position"], source_time),
        "root_quaternion": _yaw_quaternion(yaw),
    }


def _select_quantiles(rows: list[dict], quantiles: tuple[float, ...]) -> list[dict]:
    speeds = np.asarray([row["fetch_path_speed"] for row in rows])
    selected = []
    used = set()
    for quantile in quantiles:
        target = float(np.quantile(speeds, quantile))
        order = np.argsort(np.abs(speeds - target))
        index = next(int(value) for value in order if int(value) not in used)
        used.add(index)
        row = dict(rows[index])
        row["selection_quantile"] = quantile
        row["selection_target_speed"] = target
        selected.append(row)
    return selected


def _label(
    frame: np.ndarray,
    row: dict,
    factor: float,
    *,
    repeat_index: int,
    repeats: int,
    time_seconds: float,
    font,
) -> np.ndarray:
    panel = Image.fromarray(frame)
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel.width, 60), fill=(0, 0, 0, 195))
    draw.text(
        (6, 3),
        f"speed / {factor:.2f} | cadence / {factor:.2f}",
        fill=(255, 235, 150, 255),
        font=font,
    )
    draw.text(
        (6, 22),
        f"Fetch {row['fetch_path_speed']:.2f} -> {row['fetch_path_speed'] / factor:.2f} u/s",
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (6, 41),
        f"same clip | t={time_seconds:.2f}s | replay {repeat_index}/{repeats}",
        fill=(185, 235, 255, 255),
        font=font,
    )
    return np.asarray(panel)


def _render_comparison(
    parent_root: Path,
    row: dict,
    factors: tuple[float, ...],
    output: Path,
    *,
    panel_size: int,
    repeats: int,
) -> None:
    if len(factors) != 6:
        raise ValueError("the 2x3 comparison requires exactly six factors")
    source = _load_trajectory(parent_root, row)
    env = fetch.Fetch()
    trajectories = [make_qps(env, retime(source, factor)) for factor in factors]
    # Stop all panels when the least-slow trajectory ends.  This avoids holding
    # or looping one factor while another is still moving.
    frame_count = min(map(len, trajectories))
    font = _font(max(10, panel_size // 24))
    rendered = [
        [
            np.asarray(
                image.render_array(
                    env.sys,
                    trajectory[frame_index],
                    panel_size,
                    panel_size,
                    ssaa=1,
                )
            )
            for frame_index in range(frame_count)
        ]
        for trajectory in trajectories
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=FPS, quality=8) as writer:
        for repeat_index in range(1, repeats + 1):
            for frame_index in range(frame_count):
                panels = [
                    _label(
                        frames[frame_index],
                        row,
                        factor,
                        repeat_index=repeat_index,
                        repeats=repeats,
                        time_seconds=frame_index / FPS,
                        font=font,
                    )
                    for frames, factor in zip(rendered, factors, strict=True)
                ]
                top = np.concatenate(panels[:3], axis=1)
                bottom = np.concatenate(panels[3:], axis=1)
                writer.append_data(np.concatenate((top, bottom), axis=0))
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--demo-h-root", type=Path, default=DEFAULT_DEMO_H)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--factors", type=float, nargs=6, default=DEFAULT_FACTORS)
    parser.add_argument("--quantiles", type=float, nargs=3, default=DEFAULT_QUANTILES)
    parser.add_argument("--panel-size", type=int, default=288)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    factors = tuple(args.factors)
    if any(factor <= 1.0 for factor in factors):
        raise ValueError("all slowdown factors must exceed one")

    rows = _retained_rows(args.parent_root, args.demo_h_root)
    selected = _select_quantiles(rows, tuple(args.quantiles))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "demo-f-moderate-retiming-inspection-v1",
        "description": (
            "Same exact kinematic clip in every panel; temporal interpolation "
            "divides both translation speed and joint cadence by the factor."
        ),
        "factors": list(factors),
        "quantiles": list(args.quantiles),
        "repeats": args.repeats,
        "clips": selected,
    }
    (args.output_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in selected:
        percentile = int(round(100 * row["selection_quantile"]))
        _render_comparison(
            args.parent_root,
            row,
            factors,
            args.output_dir / f"retime_factors_q{percentile:02d}.mp4",
            panel_size=args.panel_size,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
