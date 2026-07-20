"""Pure NumPy held-out gait metrics shared by Demo G evaluation and tests."""

from __future__ import annotations

import numpy as np

from demo_f.config import FPS


GAIT_FIELDS = (
    "duty_factor",
    "airborne_fraction",
    "max_flight_seconds",
    "contact_switch_hz",
    "stance_foot_speed",
    "stance_world_foot_speed",
    "vertical_accel_rms_g",
    "joint_speed_rms",
    "cyclicity",
)
GAIT_CLIP_FRAMES = 64


def gait_statistics(features: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-clip direct contact/foot statistics on 64-frame clips."""

    features = np.asarray(features, np.float32)
    if features.ndim != 3 or features.shape[1] != GAIT_CLIP_FRAMES:
        raise ValueError(f"expected (clips,{GAIT_CLIP_FRAMES},60), got {features.shape}")
    contacts = features[..., 56:60]
    feet = features[..., 32:44].reshape(len(features), GAIT_CLIP_FRAMES, 4, 3)
    feet_speed = np.linalg.norm(
        features[..., 44:56].reshape(len(features), GAIT_CLIP_FRAMES, 4, 3), axis=-1
    )
    stance_count = np.maximum(contacts.sum(axis=(1, 2)), 1.0)
    stance_speed = (feet_speed * contacts).sum(axis=(1, 2)) / stance_count

    root_velocity = np.zeros((len(features), GAIT_CLIP_FRAMES, 3), np.float32)
    root_velocity[..., :2] = features[..., :2]
    root_height = features[..., 2]
    root_velocity[:, 1:, 2] = np.diff(root_height, axis=1) * FPS
    root_velocity[:, 0, 2] = root_velocity[:, 1, 2]
    root_acceleration_z = np.diff(root_velocity[..., 2], axis=1) * FPS
    angular_velocity = features[..., 9:12]
    feet_velocity = features[..., 44:56].reshape(
        len(features), GAIT_CLIP_FRAMES, 4, 3
    )
    # Norm is rotation invariant, so summing the three root-local velocity
    # terms measures approximate world slip without reconstructing global yaw.
    feet_world_velocity = (
        root_velocity[:, :, None, :]
        + np.cross(angular_velocity[:, :, None, :], feet)
        + feet_velocity
    )
    feet_world_speed = np.linalg.norm(feet_world_velocity, axis=-1)
    stance_world_speed = (
        (feet_world_speed * contacts).sum(axis=(1, 2)) / stance_count
    )
    airborne = contacts.sum(axis=-1) < 0.5

    def longest_run(row: np.ndarray) -> int:
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.flatnonzero(np.diff(padded))
        lengths = edges[1::2] - edges[::2]
        return int(lengths.max()) if len(lengths) else 0

    longest_flight = np.asarray([longest_run(row) for row in airborne], np.float32)

    cyclicity = np.zeros(len(features), np.float32)
    for clip_index, foot_height in enumerate(feet[..., 2]):
        correlations = []
        for foot in range(4):
            signal = foot_height[:, foot] - foot_height[:, foot].mean()
            for lag in range(5, GAIT_CLIP_FRAMES // 2):
                first, second = signal[:-lag], signal[lag:]
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                if denominator > 1e-8:
                    correlations.append(float(np.dot(first, second) / denominator))
        cyclicity[clip_index] = max(correlations, default=0.0)

    return {
        "duty_factor": contacts.mean(axis=(1, 2)),
        "airborne_fraction": airborne.mean(axis=1),
        "max_flight_seconds": longest_flight / FPS,
        "contact_switch_hz": np.abs(np.diff(contacts, axis=1)).mean(axis=(1, 2)) * FPS,
        "stance_foot_speed": stance_speed,
        "stance_world_foot_speed": stance_world_speed,
        "vertical_accel_rms_g": np.sqrt(
            np.mean(np.square(root_acceleration_z), axis=1)
        )
        / 9.8,
        "joint_speed_rms": np.sqrt(
            np.mean(np.square(features[..., 22:32]), axis=(1, 2))
        ),
        "cyclicity": cyclicity,
    }


def gait_distance(metrics: dict, reference: dict) -> float:
    standardized = []
    for name in GAIT_FIELDS:
        target = reference[name]["mean"]
        # A small scale floor prevents a near-constant reference channel from
        # dominating the four equally legible workshop diagnostics.
        scale = max(reference[name]["std"], 0.1 * abs(target), 1e-3)
        standardized.append(abs(metrics[f"gait_{name}"] - target) / scale)
    return float(np.mean(standardized))
