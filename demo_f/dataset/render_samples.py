"""Render representative clips from the exact retargeted parent of Demo H.

The output is an inspection aid, not a learned rollout.  It selects only
kinematic Demo F clips whose parent IDs survive into Demo H's direct-scale
training release, stratifies them into six equal-mass Fetch-speed bins, and
draws three non-overlapping examples per bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from brax.v1.envs import fetch
from brax.v1.io import image
from PIL import Image, ImageDraw, ImageFont

from demo_f.render import make_qps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "dataset" / "release"
DEFAULT_DEMO_H = ROOT.parent / "demo_h" / "dataset" / "release_direct"
DEFAULT_OUTPUT = ROOT / "out" / "retargeted_distribution"
FPS = 50


def _font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _retained_rows(parent_root: Path, demo_h_root: Path) -> list[dict]:
    manifest = json.loads((demo_h_root / "manifest.json").read_text())
    rows = []
    for session in manifest["sessions"]:
        if session["split"] != "train":
            continue
        parent_path = parent_root / session["parent_shard"]
        projected_path = demo_h_root / session["shard"]
        with np.load(parent_path) as parent, np.load(projected_path) as projected:
            for parent_index in projected["parent_clip_id"].astype(int):
                root = parent["root_position"][parent_index]
                path_speed = float(
                    np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1).mean() * FPS
                )
                rows.append(
                    {
                        "session": session["session"],
                        "parent_shard": session["parent_shard"],
                        "parent_index": int(parent_index),
                        "source_start": int(parent["source_start"][parent_index]),
                        "fetch_path_speed": path_speed,
                        "source_path_speed_mps": float(
                            parent["source_path_speed_mps"][parent_index]
                        ),
                    }
                )
    return rows


def _select(rows: list[dict], *, draws: int, seed: int) -> tuple[np.ndarray, list[list[dict]]]:
    speeds = np.asarray([row["fetch_path_speed"] for row in rows])
    edges = np.quantile(speeds, np.linspace(0.0, 1.0, 7))
    rng = np.random.default_rng(seed)
    selections = [[] for _ in range(draws)]
    for bin_index in range(6):
        lower, upper = edges[bin_index : bin_index + 2]
        candidates = np.flatnonzero(
            (speeds >= lower)
            & ((speeds <= upper) if bin_index == 5 else (speeds < upper))
        )
        chosen = rng.choice(candidates, size=draws, replace=False)
        for draw_index, row_index in enumerate(chosen):
            row = dict(rows[int(row_index)])
            row["speed_bin"] = bin_index + 1
            row["speed_bin_range"] = [float(lower), float(upper)]
            selections[draw_index].append(row)
    return edges, selections


def _load_trajectory(parent_root: Path, row: dict) -> dict:
    with np.load(parent_root / row["parent_shard"]) as source:
        index = row["parent_index"]
        return {
            "angles": source["joint_angles"][index],
            "root_position": source["root_position"][index],
            "root_quaternion": source["root_quaternion"][index],
        }


def _label(
    frame: np.ndarray,
    row: dict,
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
        f"KINEMATIC DATA | speed bin {row['speed_bin']}/6",
        fill=(255, 235, 150, 255),
        font=font,
    )
    draw.text(
        (6, 22),
        f"Fetch {row['fetch_path_speed']:.2f} u/s | rodent {row['source_path_speed_mps']:.3f} m/s",
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (6, 41),
        f"{row['session']}@{row['source_start']} | t={time_seconds:.2f}s | repeat {repeat_index}/{repeats}",
        fill=(185, 235, 255, 255),
        font=font,
    )
    return np.asarray(panel)


def _render_draw(
    parent_root: Path,
    rows: list[dict],
    output: Path,
    *,
    panel_size: int,
    repeats: int,
) -> None:
    env = fetch.Fetch()
    qps = [make_qps(env, _load_trajectory(parent_root, row)) for row in rows]
    frame_count = min(map(len, qps))
    font = _font(max(10, panel_size // 24))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Render each source frame once; repetition only repeats these exact pixels.
    panels = [
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
        for trajectory in qps
    ]
    with imageio.get_writer(output, fps=FPS, quality=8) as writer:
        for repeat_index in range(1, repeats + 1):
            for frame_index in range(frame_count):
                labeled = [
                    _label(
                        panel[frame_index],
                        row,
                        repeat_index=repeat_index,
                        repeats=repeats,
                        time_seconds=frame_index / FPS,
                        font=font,
                    )
                    for panel, row in zip(panels, rows, strict=True)
                ]
                top = np.concatenate(labeled[:3], axis=1)
                bottom = np.concatenate(labeled[3:], axis=1)
                writer.append_data(np.concatenate((top, bottom), axis=0))
    print(f"wrote {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--demo-h-root", type=Path, default=DEFAULT_DEMO_H)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=288)
    args = parser.parse_args()

    retained = _retained_rows(args.parent_root, args.demo_h_root)
    edges, selections = _select(retained, draws=args.draws, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "demo-f-retargeted-distribution-inspection-v1",
        "description": (
            "Kinematic Demo F parents of clips retained in Demo H direct-scale train; "
            "these are neither learned rollouts nor physics projections."
        ),
        "selection": "one random non-overlapping clip per equal-mass Fetch path-speed bin",
        "seed": args.seed,
        "retained_training_clips": len(retained),
        "fetch_path_speed_bin_edges": edges.tolist(),
        "repeats_of_same_64_frame_clip": args.repeats,
        "draws": selections,
    }
    (args.output_dir / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for draw_index, rows in enumerate(selections, start=1):
        _render_draw(
            args.parent_root,
            rows,
            args.output_dir / f"retargeted_distribution_{draw_index}.mp4",
            panel_size=args.panel_size,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
