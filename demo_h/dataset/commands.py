"""Pure NumPy command construction shared by data build and model training."""

from __future__ import annotations

import numpy as np


def yaw_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.unwrap(
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        axis=-1,
    )


def hindsight_command(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    start: int = 32,
    future: int = 63,
) -> np.ndarray:
    """Egocentric displacement/yaw command for a batch of realized clips."""

    yaw = yaw_from_quaternion(root_quaternion)
    delta = root_position[:, future, :2] - root_position[:, start, :2]
    heading = yaw[:, start]
    cosine, sine = np.cos(-heading), np.sin(-heading)
    turn = (yaw[:, future] - heading + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        (
            cosine * delta[:, 0] - sine * delta[:, 1],
            sine * delta[:, 0] + cosine * delta[:, 1],
            turn,
        ),
        axis=-1,
    ).astype(np.float32)
