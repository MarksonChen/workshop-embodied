"""Validation-only gait metrics over the shared 60-D Fetch representation.

Nothing in this module is imported by a training environment or used as reward.
The broad metrics compare distributions; the stricter four-limb checks reject
unused limbs, tapping, and high-frequency chatter in workshop videos.
"""

from __future__ import annotations

import numpy as np

from .config import FPS
from .features import FEATURE_DIM, SL


FOOT_NAMES = ("front_right", "front_left", "back_right", "back_left")
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


def _feature_parts(features: np.ndarray):
    features = np.asarray(features, np.float32)
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected final feature dimension {FEATURE_DIM}, got {features.shape}")
    feet = features[..., slice(*SL["feet_local"])].reshape(*features.shape[:-1], 4, 3)
    foot_velocity = features[..., slice(*SL["feet_velocity"])].reshape(
        *features.shape[:-1], 4, 3
    )
    contacts = features[..., slice(*SL["contacts"])]
    return feet, foot_velocity, contacts


def gait_statistics(features: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-clip direct contact/foot statistics on 64-frame clips."""

    features = np.asarray(features, np.float32)
    if features.ndim != 3 or features.shape[1:] != (GAIT_CLIP_FRAMES, FEATURE_DIM):
        raise ValueError(
            f"expected (clips,{GAIT_CLIP_FRAMES},{FEATURE_DIM}), got {features.shape}"
        )
    feet, foot_velocity, contacts = _feature_parts(features)
    feet_speed = np.linalg.norm(foot_velocity, axis=-1)
    stance_count = np.maximum(contacts.sum(axis=(1, 2)), 1.0)
    stance_speed = (feet_speed * contacts).sum(axis=(1, 2)) / stance_count

    root_velocity = np.zeros((len(features), GAIT_CLIP_FRAMES, 3), np.float32)
    root_velocity[..., :2] = features[..., slice(*SL["root_velocity"])]
    root_height = features[..., SL["root_height"][0]]
    root_velocity[:, 1:, 2] = np.diff(root_height, axis=1) * FPS
    root_acceleration_z = np.diff(root_velocity[..., 2], axis=1) * FPS
    angular_velocity = features[..., slice(*SL["root_angular_velocity"])]
    feet_world_velocity = (
        root_velocity[:, :, None, :]
        + np.cross(angular_velocity[:, :, None, :], feet)
        + foot_velocity
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
            np.mean(
                np.square(features[..., slice(*SL["joint_velocity"])]),
                axis=(1, 2),
            )
        ),
        "cyclicity": cyclicity,
    }


def gait_distance(metrics: dict, reference: dict) -> float:
    standardized = []
    for name in GAIT_FIELDS:
        target = reference[name]["mean"]
        scale = max(reference[name]["std"], 0.1 * abs(target), 1e-3)
        standardized.append(abs(metrics[f"gait_{name}"] - target) / scale)
    return float(np.mean(standardized))


def _spectral_summary(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    signal = np.asarray(signal, np.float64)
    time = np.arange(len(signal), dtype=np.float64)
    slope, intercept = np.polyfit(time, signal, 1)
    windowed = (signal - (slope * time + intercept)) * np.hanning(len(signal))
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequency = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    power[frequency < 0.5] = 0.0
    total = float(power.sum()) + 1e-12
    dominant = float(frequency[int(np.argmax(power))])
    stride_fraction = float(power[(frequency >= 0.5) & (frequency < 6.0)].sum() / total)
    buzz_fraction = float(power[frequency >= 8.0].sum() / total)
    return dominant, stride_fraction, buzz_fraction


def _longest_constant_run(values: np.ndarray) -> int:
    if not len(values):
        return 0
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    return int(np.diff(np.concatenate(([0], boundaries, [len(values)]))).max())


def four_limb_contact_metrics(
    contacts: np.ndarray,
    *,
    fps: float = FPS,
    window_seconds: float = 1.0,
) -> dict:
    """Summarize contact participation without prescribing a gait family."""

    contacts = np.asarray(contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != len(FOOT_NAMES):
        raise ValueError(f"expected (time, 4) contacts, got {contacts.shape}")
    if len(contacts) < 2:
        raise ValueError("contact audit needs at least two frames")
    duration = (len(contacts) - 1) / float(fps)
    switches = np.sum(contacts[1:] != contacts[:-1], axis=0)
    switch_hz = switches / max(duration, 1.0 / fps)
    duty = contacts.mean(axis=0)
    probability = np.clip(duty, 1e-6, 1.0 - 1e-6)
    entropy = -(
        probability * np.log2(probability)
        + (1.0 - probability) * np.log2(1.0 - probability)
    )
    longest_constant = np.asarray(
        [_longest_constant_run(contacts[:, index]) for index in range(4)]
    ) / float(fps)

    window = max(int(round(window_seconds * fps)), 2)
    windows = []
    for start in range(0, len(contacts) - 1, window):
        segment = contacts[start : min(start + window + 1, len(contacts))]
        if len(segment) >= 2:
            windows.append(np.any(segment[1:] != segment[:-1], axis=0))
    window_participation = np.stack(windows) if windows else np.zeros((0, 4), bool)
    per_foot_window_fraction = (
        window_participation.mean(axis=0)
        if len(window_participation)
        else np.zeros(4, np.float32)
    )
    all_feet_window_fraction = (
        float(np.all(window_participation, axis=1).mean())
        if len(window_participation)
        else 0.0
    )
    switch_balance = float(np.std(switch_hz) / max(np.mean(switch_hz), 1e-8))
    passes = bool(
        np.min(switch_hz) >= 1.0
        and np.min(entropy) >= 0.30
        and np.max(longest_constant) <= 1.25
        and np.min(per_foot_window_fraction) >= 0.60
        and all_feet_window_fraction >= 0.40
        and switch_balance <= 0.75
    )
    return {
        "foot_order": FOOT_NAMES,
        "switch_count": switches.astype(int).tolist(),
        "switch_hz": switch_hz.tolist(),
        "duty_factor": duty.tolist(),
        "contact_entropy_bits": entropy.tolist(),
        "longest_constant_contact_state_seconds": longest_constant.tolist(),
        "per_foot_active_window_fraction": per_foot_window_fraction.tolist(),
        "all_four_active_window_fraction": all_feet_window_fraction,
        "switch_rate_coefficient_of_variation": switch_balance,
        "minimum_switch_hz": float(np.min(switch_hz)),
        "passes_four_limb_gait_gate": passes,
    }


def four_limb_locomotion_metrics(
    features: np.ndarray,
    *,
    fps: float = FPS,
    warmup_seconds: float = 1.0,
) -> dict:
    """Audit whether every foot strides instead of staying fixed or tapping."""

    features = np.asarray(features, np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (time, {FEATURE_DIM}) features, got {features.shape}")
    warmup = int(round(warmup_seconds * fps))
    if len(features) - warmup < max(int(fps * 2), 16):
        raise ValueError("locomotion audit needs at least two post-warmup seconds")
    features = features[warmup:]
    feet, foot_velocity, contacts = _feature_parts(features)
    contacts = contacts >= 0.5
    contact = four_limb_contact_metrics(contacts, fps=fps)

    fore_aft_excursion = np.quantile(feet[..., 0], 0.95, axis=0) - np.quantile(
        feet[..., 0], 0.05, axis=0
    )
    vertical_excursion = np.quantile(feet[..., 2], 0.95, axis=0) - np.quantile(
        feet[..., 2], 0.05, axis=0
    )
    dominant_frequency = np.zeros(4, np.float64)
    stride_power = np.zeros(4, np.float64)
    buzz_power = np.zeros(4, np.float64)
    for foot in range(4):
        dominant_frequency[foot], stride_power[foot], buzz_power[foot] = (
            _spectral_summary(feet[:, foot, 2], fps)
        )

    stance_forward_velocity = np.zeros(4, np.float64)
    swing_forward_velocity = np.zeros(4, np.float64)
    for foot in range(4):
        stance = contacts[:, foot]
        stance_forward_velocity[foot] = (
            float(foot_velocity[stance, foot, 0].mean()) if np.any(stance) else np.nan
        )
        swing_forward_velocity[foot] = (
            float(foot_velocity[~stance, foot, 0].mean()) if np.any(~stance) else np.nan
        )
    swing_reset_margin = swing_forward_velocity - stance_forward_velocity
    passes_stride = bool(
        contact["passes_four_limb_gait_gate"]
        and np.min(fore_aft_excursion) >= 0.10
        and np.min(vertical_excursion) >= 0.03
        and np.min(stride_power) >= 0.30
        and np.max(buzz_power) <= 0.25
        and np.min(swing_reset_margin) >= 0.15
    )
    return {
        "foot_order": FOOT_NAMES,
        "fore_aft_excursion_fetch_units": fore_aft_excursion.tolist(),
        "vertical_excursion_fetch_units": vertical_excursion.tolist(),
        "dominant_vertical_frequency_hz": dominant_frequency.tolist(),
        "stride_band_power_fraction": stride_power.tolist(),
        "high_frequency_power_fraction": buzz_power.tolist(),
        "stance_local_forward_velocity_fetch_units_per_s": (
            stance_forward_velocity.tolist()
        ),
        "swing_local_forward_velocity_fetch_units_per_s": (
            swing_forward_velocity.tolist()
        ),
        "swing_reset_margin_fetch_units_per_s": swing_reset_margin.tolist(),
        "contact_gate": contact,
        "passes_four_limb_stride_gate": passes_stride,
    }
