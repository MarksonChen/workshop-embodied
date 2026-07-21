"""One command convention for retargeted and physics-projected motion."""

from __future__ import annotations

import numpy as np


def yaw_from_quaternion(quaternion: np.ndarray, *, unwrap: bool = True) -> np.ndarray:
    """Convert batched ``wxyz`` quaternions to yaw along the last time axis."""

    quaternion = np.asarray(quaternion, np.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError(f"expected wxyz quaternions, got {quaternion.shape}")
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.unwrap(yaw, axis=-1) if unwrap else yaw


def commands_from_yaw(
    root_position: np.ndarray,
    yaw: np.ndarray,
    start_frames: np.ndarray,
    future_frames: np.ndarray,
) -> np.ndarray:
    """Construct egocentric planar-displacement/yaw commands at causal anchors."""

    root = np.asarray(root_position, np.float32)
    yaw = np.asarray(yaw, np.float32)
    start = np.atleast_1d(np.asarray(start_frames, np.int64))
    future = np.atleast_1d(np.asarray(future_frames, np.int64))
    if root.ndim != 3 or root.shape[-1] != 3:
        raise ValueError(f"expected (clips,time,3) roots, got {root.shape}")
    if yaw.shape != root.shape[:2]:
        raise ValueError(f"root/yaw shapes disagree: {root.shape} vs {yaw.shape}")
    if start.shape != future.shape or start.ndim != 1:
        raise ValueError("start and future frames must be matching 1-D arrays")
    if not len(start) or start.min() < 0 or future.max() >= root.shape[1]:
        raise ValueError("command windows exceed the stored trajectory")
    if np.any(future <= start):
        raise ValueError("every command future must follow its start")

    delta = root[:, future, :2] - root[:, start, :2]
    heading = yaw[:, start]
    cosine, sine = np.cos(-heading), np.sin(-heading)
    turn = (yaw[:, future] - heading + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        (
            cosine * delta[..., 0] - sine * delta[..., 1],
            sine * delta[..., 0] + cosine * delta[..., 1],
            turn,
        ),
        axis=-1,
    ).astype(np.float32)


def hindsight_commands(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    start_frames: np.ndarray,
    horizon_frames: int,
) -> np.ndarray:
    """Return ``(clips, anchors, 3)`` commands with one shared horizon."""

    start = np.atleast_1d(np.asarray(start_frames, np.int64))
    return commands_from_yaw(
        root_position,
        yaw_from_quaternion(root_quaternion),
        start,
        start + int(horizon_frames),
    )


def hindsight_command(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    *,
    start: int = 32,
    future: int = 63,
) -> np.ndarray:
    """Return one ``(clips, 3)`` command for a scalar start/future pair."""

    return commands_from_yaw(
        root_position,
        yaw_from_quaternion(root_quaternion),
        np.asarray((start,)),
        np.asarray((future,)),
    )[:, 0]
