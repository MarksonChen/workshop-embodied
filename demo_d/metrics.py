"""Frozen, unit-testable Demo D scoring rules."""

from __future__ import annotations

import numpy as np

from demo_d.config import COMMAND_ERROR_SCALES, EVAL, TRAINING


COMMAND_SECONDS = TRAINING.command_horizon_frames / 50.0
IMITATION_REWARD_CEILING = 5.0  # root pos, root quat, joints, end effectors, healthy height.


def command_target(command: np.ndarray) -> np.ndarray:
    """Convert Demo B displacement commands to planar/yaw velocities."""
    command = np.asarray(command, dtype=np.float64)
    return command / COMMAND_SECONDS


def command_tracking_score(
    local_velocity: np.ndarray, yaw_rate: np.ndarray, command: np.ndarray
) -> np.ndarray:
    """Bounded [0, 1] score that requires translation and turning jointly."""
    target = command_target(command)
    local_velocity = np.asarray(local_velocity)
    yaw_rate = np.asarray(yaw_rate)
    linear_error = (
        ((local_velocity[..., 0] - target[..., 0]) / COMMAND_ERROR_SCALES[0]) ** 2
        + ((local_velocity[..., 1] - target[..., 1]) / COMMAND_ERROR_SCALES[1]) ** 2
    )
    angular_error = ((yaw_rate - target[..., 2]) / COMMAND_ERROR_SCALES[2]) ** 2
    return np.exp(-0.5 * (linear_error + angular_error))


def reportability(baseline: dict, trained: dict) -> dict:
    imitation_gain = trained["imitation_score"] - baseline["imitation_score"]
    command_gain = trained["command_score"] - baseline["command_score"]
    tolerance = 1e-12
    gates = {
        "imitation_gain": imitation_gain >= EVAL.imitation_gain_min - tolerance,
        "imitation_survival": trained["imitation_survival"] >= EVAL.imitation_survival_min - tolerance,
        "command_gain": command_gain >= EVAL.command_gain_min - tolerance,
        "command_score": trained["command_score"] >= EVAL.command_score_min - tolerance,
        "command_survival": trained["command_survival"] >= EVAL.command_survival_min - tolerance,
    }
    return {
        "imitation_gain": float(imitation_gain),
        "command_gain": float(command_gain),
        "gates": gates,
        "reportable": bool(all(gates.values())),
    }
