"""Session-safe all-session locomotion crops for Demo B.

Motion-mapper labels flicker at 50 Hz, so a useful 1.28 s crop need not be one
perfectly contiguous label run.  The frozen rule requires both locomotion-label
evidence and measured displacement, never stitches disjoint frames, and caps
each session to prevent long recordings from dominating.
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .constants import CLIP, COMMAND_HORIZON_FRAMES, FM, FPS
from .features import motion_features, quat_to_yaw
from .splits import ANIMAL


DATA_ROOT = Path(os.environ.get("ALDARONDO_ROOT", "/workspace/data/Aldarondo2024"))
LOCOMOTION = frozenset(("Amble", "Walk", "WalkFast"))
CROP_STRIDE = 16
MIN_LOCOMOTION_FRACTION = 0.25
MIN_PLANAR_SPEED = 0.08
MAX_CROPS_PER_SESSION = 2048


@dataclass
class CropSet:
    features: np.ndarray
    command: np.ndarray
    reset_qpos: np.ndarray
    session_index: np.ndarray
    start: np.ndarray
    sessions: tuple[str, ...]
    session_rows: list[dict]
    seed_qpos: np.ndarray


def session_path(session: str) -> Path:
    return DATA_ROOT / ANIMAL / f"{session}.h5"


def _decode_names(values) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def candidate_starts(qpos: np.ndarray, behavior: np.ndarray, names: list[str]) -> tuple[np.ndarray, dict]:
    locomotion = np.zeros(len(behavior), bool)
    for label, name in enumerate(names):
        if name in LOCOMOTION:
            locomotion |= behavior == label
    starts = np.arange(0, len(qpos) - CLIP + 1, CROP_STRIDE, dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(locomotion, dtype=np.int64)))
    fraction = (cumulative[starts + CLIP] - cumulative[starts]) / CLIP
    frame_speed = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=-1) * FPS
    speed_sum = np.concatenate(([0.0], np.cumsum(frame_speed, dtype=np.float64)))
    mean_speed = (speed_sum[starts + CLIP - 1] - speed_sum[starts]) / (CLIP - 1)
    keep = (fraction >= MIN_LOCOMOTION_FRACTION) & (mean_speed >= MIN_PLANAR_SPEED)
    diagnostics = {
        "frames": int(len(qpos)),
        "locomotion_frames": int(locomotion.sum()),
        "candidate_crops_before_cap": int(keep.sum()),
        "mean_locomotion_frame_speed": float(
            frame_speed[locomotion[:-1] & locomotion[1:]].mean()
        ),
    }
    return starts[keep], diagnostics


def commands_at(qpos: np.ndarray, starts: np.ndarray) -> np.ndarray:
    frame = starts + CLIP // 2
    future = frame + COMMAND_HORIZON_FRAMES
    yaw = quat_to_yaw(qpos[:, 3:7])
    delta = qpos[future, :2] - qpos[frame, :2]
    cosine, sine = np.cos(-yaw[frame]), np.sin(-yaw[frame])
    local = np.stack(
        [
            cosine * delta[:, 0] - sine * delta[:, 1],
            sine * delta[:, 0] + cosine * delta[:, 1],
        ],
        axis=-1,
    )
    turn = (yaw[future] - yaw[frame] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([local, turn[:, None]], axis=-1).astype(np.float32)


def extract_session(session: str, *, max_crops: int = MAX_CROPS_PER_SESSION) -> dict:
    path = session_path(session)
    if not path.exists():
        raise FileNotFoundError(f"missing {ANIMAL} session {path}; see ref/docs/dataset.md")
    with h5py.File(path, "r") as source:
        animal = source.attrs["animal"]
        animal = animal.decode() if isinstance(animal, (bytes, np.bytes_)) else str(animal)
        if animal != ANIMAL:
            raise ValueError(f"{path} is not a {ANIMAL} session")
        qpos = source["/pose/qpos"][:].astype(np.float32)
        behavior_dataset = source["/behavior/motion_mapper"]
        behavior = behavior_dataset[:]
        names = _decode_names(behavior_dataset.attrs["names"])
    starts, diagnostics = candidate_starts(qpos, behavior, names)
    if not len(starts):
        raise ValueError(f"{session}: no crops pass the frozen locomotion rule")
    if len(starts) > max_crops:
        rng = np.random.default_rng(zlib.crc32(session.encode()))
        starts = np.sort(rng.choice(starts, max_crops, replace=False))
    full_features = motion_features(qpos)
    offsets = np.arange(CLIP, dtype=np.int64)
    crops = full_features[starts[:, None] + offsets[None]].astype(np.float32)
    if crops.shape[-1] != FM:
        raise AssertionError(crops.shape)
    command = commands_at(qpos, starts)
    seed_qpos = qpos[starts[0] : starts[0] + CLIP].copy()
    diagnostics.update(
        {
            "session": session,
            "selected_crops": int(len(starts)),
            "selected_forward_speed_mean": float(
                (command[:, 0] / (COMMAND_HORIZON_FRAMES / FPS)).mean()
            ),
            "selected_planar_command_speed_mean": float(
                (
                    np.linalg.norm(command[:, :2], axis=-1)
                    / (COMMAND_HORIZON_FRAMES / FPS)
                ).mean()
            ),
        }
    )
    return {
        "features": crops,
        "command": command,
        "reset_qpos": qpos[starts + 31].astype(np.float32),
        "start": starts.astype(np.int32),
        "diagnostics": diagnostics,
        "seed_qpos": seed_qpos,
    }


def load_crop_set(sessions: tuple[str, ...], *, max_crops: int = MAX_CROPS_PER_SESSION) -> CropSet:
    groups = []
    for session in sessions:
        group = extract_session(session, max_crops=max_crops)
        groups.append(group)
        row = group["diagnostics"]
        print(
            f"[{session}] {row['selected_crops']} crops | "
            f"forward={row['selected_forward_speed_mean']:.3f} m/s",
            flush=True,
        )
    return CropSet(
        features=np.concatenate([group["features"] for group in groups]),
        command=np.concatenate([group["command"] for group in groups]),
        reset_qpos=np.concatenate([group["reset_qpos"] for group in groups]),
        session_index=np.concatenate(
            [np.full(len(group["start"]), index, np.int16) for index, group in enumerate(groups)]
        ),
        start=np.concatenate([group["start"] for group in groups]),
        sessions=sessions,
        session_rows=[group["diagnostics"] for group in groups],
        seed_qpos=groups[0]["seed_qpos"],
    )
