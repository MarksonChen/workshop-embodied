"""Metrics that fail when a policy ignores one of Fetch's four feet."""

from __future__ import annotations

import numpy as np

from demo_h.config import FPS


FOOT_NAMES = ("front_right", "front_left", "back_right", "back_left")


def _spectral_summary(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    """Return dominant, stride-band, and high-frequency foot-motion statistics."""

    signal = np.asarray(signal, np.float64)
    time = np.arange(len(signal), dtype=np.float64)
    # Remove slow pose drift before asking whether the remaining motion is a
    # stride or high-frequency chatter.
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
    """Summarize contact participation without averaging feet together."""

    contacts = np.asarray(contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != len(FOOT_NAMES):
        raise ValueError(f"expected (time, 4) contacts, got {contacts.shape}")
    if len(contacts) < 2:
        raise ValueError("contact audit needs at least two frames")
    duration = (len(contacts) - 1) / float(fps)
    switches = np.sum(contacts[1:] != contacts[:-1], axis=0)
    switch_hz = switches / max(duration, 1.0 / fps)
    duty = contacts.mean(axis=0)
    # Binary entropy is zero for a permanently raised or planted foot.
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
    all_feet_window_fraction = float(
        np.all(window_participation, axis=1).mean()
    ) if len(window_participation) else 0.0
    switch_balance = float(
        np.std(switch_hz) / max(np.mean(switch_hz), 1e-8)
    )

    # These broad gates detect unused limbs without prescribing walk, trot, or
    # gallop. At 50 Hz over a five-second audit, each foot must alternate at
    # least once per second and participate in most one-second windows.
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
    """Audit whether every foot makes a stride rather than merely tapping.

    This function is deliberately NumPy-only and is never called by the RL
    environment.  It is a held-out validation gate over Demo F's 60-D feature
    contract: local foot position, local foot velocity, and binary contact.
    """

    features = np.asarray(features, np.float32)
    if features.ndim != 2 or features.shape[1] != 60:
        raise ValueError(f"expected (time, 60) features, got {features.shape}")
    warmup = int(round(warmup_seconds * fps))
    if len(features) - warmup < max(int(fps * 2), 16):
        raise ValueError("locomotion audit needs at least two post-warmup seconds")
    features = features[warmup:]
    feet = features[:, 32:44].reshape(len(features), 4, 3)
    foot_velocity = features[:, 44:56].reshape(len(features), 4, 3)
    contacts = features[:, 56:60] >= 0.5
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
        (
            dominant_frequency[foot],
            stride_power[foot],
            buzz_power[foot],
        ) = _spectral_summary(feet[:, foot, 2], fps)

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

    # Broad validation-only gates.  They reject a raised leg, a planted leg,
    # contact chatter, or vertical tapping without prescribing inter-leg phase.
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
        "fore_aft_excursion_m": fore_aft_excursion.tolist(),
        "vertical_excursion_m": vertical_excursion.tolist(),
        "dominant_vertical_frequency_hz": dominant_frequency.tolist(),
        "stride_band_power_fraction": stride_power.tolist(),
        "high_frequency_power_fraction": buzz_power.tolist(),
        "stance_local_forward_velocity_mps": stance_forward_velocity.tolist(),
        "swing_local_forward_velocity_mps": swing_forward_velocity.tolist(),
        "swing_reset_margin_mps": swing_reset_margin.tolist(),
        "contact_gate": contact,
        "passes_four_limb_stride_gate": passes_stride,
    }
