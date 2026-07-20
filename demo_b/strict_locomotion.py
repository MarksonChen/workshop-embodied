"""The geometric locomotion subset used by the known-good Demo B model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .constants import CLIP, COMMAND_HORIZON_FRAMES, FPS, FULL_FM
from .features import full_motion_features, quat_to_yaw


DATA_ROOT = Path(os.environ.get("ALDARONDO_ROOT", "/workspace/data/Aldarondo2024"))
CROP_STRIDE = 16
GAIT_WINDOW = CLIP - 1
GAIT_PAIRS = ((64, 72), (22, 63), (16, 71))
MIN_SPEED = 0.10
MIN_GAIT_COORDINATION = 0.0
MAX_TURN_DEGREES = 90.0
MAX_NECK_DRIFT_MM = 10.0


@dataclass
class StrictCropSet:
    features: np.ndarray
    command: np.ndarray
    reset_qpos: np.ndarray
    session_index: np.ndarray
    start: np.ndarray
    sessions: tuple[str, ...]
    session_rows: list[dict]
    seed_features: np.ndarray
    seed_xy: np.ndarray
    seed_yaw: np.ndarray
    seed_name: str


def _correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(1, keepdims=True)
    right = right - right.mean(1, keepdims=True)
    return (left * right).sum(1) / np.sqrt(
        (left ** 2).sum(1) * (right ** 2).sum(1) + 1e-12
    )


def strict_block_starts(qpos: np.ndarray, neck_height_mm: np.ndarray) -> np.ndarray:
    """Non-overlapping 64-frame blocks passing the frozen CANVAS gait rule."""
    speed_frames = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) * FPS
    cumulative = np.concatenate([[0.0], np.cumsum(speed_frames)])
    window_speed = (cumulative[GAIT_WINDOW:] - cumulative[:-GAIT_WINDOW]) / GAIT_WINDOW
    starts = np.arange(0, len(speed_frames) - GAIT_WINDOW + 1, CLIP)

    def windows(values, length, offset=0):
        return values[starts[:, None] + offset + np.arange(length)]

    differences = {
        coordinate: np.diff(qpos[:, coordinate])
        for pair in GAIT_PAIRS
        for coordinate in pair
    }
    gait = -np.nansum(
        np.stack(
            [
                _correlation(
                    windows(differences[left], GAIT_WINDOW),
                    windows(differences[right], GAIT_WINDOW),
                )
                for left, right in GAIT_PAIRS
            ]
        ),
        axis=0,
    )
    yaw = quat_to_yaw(qpos[:, 3:7])
    turn = np.degrees(
        windows(yaw, 8, CLIP - 8).mean(1) - windows(yaw, 8).mean(1)
    )
    turn = (turn + 180) % 360 - 180
    neck_drift = (
        windows(neck_height_mm, 8, CLIP - 8).mean(1)
        - windows(neck_height_mm, 8).mean(1)
    )
    keep = (
        (window_speed[starts] > MIN_SPEED)
        & (gait > MIN_GAIT_COORDINATION)
        & (np.abs(turn) < MAX_TURN_DEGREES)
        & (np.abs(neck_drift) < MAX_NECK_DRIFT_MM)
    )
    return starts[keep]


def merge_blocks(starts: np.ndarray) -> list[tuple[int, int]]:
    if not len(starts):
        return []
    output = []
    first = previous = int(starts[0])
    for value in starts[1:]:
        value = int(value)
        if value == previous + CLIP:
            previous = value
        else:
            output.append((first, previous + CLIP))
            first = previous = value
    output.append((first, previous + CLIP))
    return output


def _commands(xy: np.ndarray, yaw: np.ndarray, starts: np.ndarray) -> np.ndarray:
    frame = starts + CLIP // 2
    future = frame + COMMAND_HORIZON_FRAMES
    delta = xy[future] - xy[frame]
    cosine, sine = np.cos(-yaw[frame]), np.sin(-yaw[frame])
    turn = (yaw[future] - yaw[frame] + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        [
            cosine * delta[:, 0] - sine * delta[:, 1],
            sine * delta[:, 0] + cosine * delta[:, 1],
            turn,
        ],
        axis=-1,
    ).astype(np.float32)


def extract_session(animal: str, session: str) -> dict:
    path = DATA_ROOT / animal / f"{session}.h5"
    with h5py.File(path, "r") as source:
        qpos = source["/pose/qpos"][:].astype(np.float32)
        keypoint_data = source["/pose/keypoints"]
        names = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in keypoint_data.attrs["names"]
        ]
        neck = keypoint_data[:, 2, names.index("SpineF")].astype(np.float64)
        blocks = strict_block_starts(qpos, neck)
        bounds = merge_blocks(blocks)
        crops, commands, reset_qpos, crop_starts, candidates = [], [], [], [], []
        for begin, end in bounds:
            segment_qpos = qpos[begin:end]
            segment_keypoints = keypoint_data[begin:end]
            features = full_motion_features(segment_qpos, segment_keypoints)
            xy = segment_qpos[:, :2]
            yaw = quat_to_yaw(segment_qpos[:, 3:7])
            starts = np.arange(0, len(features) - CLIP + 1, CROP_STRIDE, dtype=np.int64)
            crops.append(
                features[starts[:, None] + np.arange(CLIP)[None]]
            )
            commands.append(_commands(xy, yaw, starts))
            reset_qpos.append(segment_qpos[starts + CLIP // 2 - 1])
            crop_starts.append(starts + begin)
            candidates.append(
                (len(features), features, xy.astype(np.float32), yaw.astype(np.float32), begin)
            )
    if len(blocks) < 3:
        raise ValueError(
            f"{animal}/{session}: only {len(blocks)} blocks pass the strict gait rule"
        )
    features = np.concatenate(crops).astype(np.float32)
    command = np.concatenate(commands).astype(np.float32)
    longest = max(candidates, key=lambda item: item[0])
    duration = COMMAND_HORIZON_FRAMES / FPS
    return {
        "features": features,
        "command": command,
        "reset_qpos": np.concatenate(reset_qpos).astype(np.float32),
        "start": np.concatenate(crop_starts).astype(np.int32),
        "seed": longest,
        "row": {
            "session": session,
            "strict_blocks": int(len(blocks)),
            "segments": int(len(bounds)),
            "crops": int(len(features)),
            "forward_speed_mean": float((command[:, 0] / duration).mean()),
            "planar_speed_mean": float(
                (np.linalg.norm(command[:, :2], axis=-1) / duration).mean()
            ),
        },
    }


def load_strict_crop_set(animal: str, sessions: tuple[str, ...]) -> StrictCropSet:
    groups, retained = [], []
    for session in sessions:
        try:
            group = extract_session(animal, session)
        except ValueError as error:
            print(f"[skip] {error}", flush=True)
            continue
        groups.append(group)
        retained.append(session)
        row = group["row"]
        print(
            f"[{animal}/{session}] {row['strict_blocks']} blocks -> "
            f"{row['crops']} crops | forward={row['forward_speed_mean']:.3f} m/s",
            flush=True,
        )
    if not groups:
        raise ValueError(f"no {animal} sessions in this split pass the strict gait rule")
    best_group = max(groups, key=lambda group: group["seed"][0])
    length, seed_features, seed_xy, seed_yaw, seed_start = best_group["seed"]
    seed_session = best_group["row"]["session"]
    return StrictCropSet(
        features=np.concatenate([group["features"] for group in groups]),
        command=np.concatenate([group["command"] for group in groups]),
        reset_qpos=np.concatenate([group["reset_qpos"] for group in groups]),
        session_index=np.concatenate(
            [
                np.full(len(group["start"]), index, np.int16)
                for index, group in enumerate(groups)
            ]
        ),
        start=np.concatenate([group["start"] for group in groups]),
        sessions=tuple(retained),
        session_rows=[group["row"] for group in groups],
        seed_features=seed_features,
        seed_xy=seed_xy,
        seed_yaw=seed_yaw,
        seed_name=f"{animal}/{seed_session}@{seed_start}:{seed_start + length}",
    )
