from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import FEATURE_DIM, FPS


def yaw_from_quaternion(quaternion: np.ndarray, *, unwrap: bool = True) -> np.ndarray:
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
    cosine, sine = (np.cos(-heading), np.sin(-heading))
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
    return commands_from_yaw(
        root_position,
        yaw_from_quaternion(root_quaternion),
        np.asarray((start,)),
        np.asarray((future,)),
    )[:, 0]


SL = {
    "root_velocity": (0, 2),
    "root_height": (2, 3),
    "rotation_delta_6d": (3, 9),
    "root_angular_velocity": (9, 12),
    "joint_angles": (12, 22),
    "joint_velocity": (22, 32),
    "feet_local": (32, 44),
    "feet_velocity": (44, 56),
    "contacts": (56, 60),
}


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def trajectory_features(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_angles: np.ndarray,
    feet_local: np.ndarray,
    contacts: np.ndarray,
    *,
    fps: float = FPS,
) -> np.ndarray:
    root_position = np.asarray(root_position, np.float32)
    quaternion = np.asarray(root_quaternion, np.float32)
    angles = np.asarray(joint_angles, np.float32)
    feet = np.asarray(feet_local, np.float32)
    contacts = np.asarray(contacts, np.float32)
    leading = root_position.shape[:-2]
    time = root_position.shape[-2]
    if root_position.shape[-1] != 3 or quaternion.shape[-2:] != (time, 4):
        raise ValueError("root trajectory shapes disagree")
    xyzw = np.concatenate((quaternion[..., 1:], quaternion[..., :1]), axis=-1)
    rotation = (
        Rotation.from_quat(xyzw.reshape(-1, 4))
        .as_matrix()
        .reshape(leading + (time, 3, 3))
    )
    world_velocity = np.zeros_like(root_position)
    world_velocity[..., 1:, :] = np.diff(root_position, axis=-2) * fps
    world_velocity[..., 0, :] = world_velocity[..., 1, :]
    local_velocity = np.einsum("...tji,...tj->...ti", rotation, world_velocity)
    relative = np.broadcast_to(np.eye(3), leading + (time, 3, 3)).copy()
    relative[..., 1:, :, :] = np.einsum(
        "...tji,...tjk->...tik", rotation[..., :-1, :, :], rotation[..., 1:, :, :]
    )
    rotvec = (
        Rotation.from_matrix(relative.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(leading + (time, 3))
        * fps
    )
    angular_local = rotvec
    joint_velocity = np.zeros_like(angles)
    joint_velocity[..., 1:, :] = np.diff(angles, axis=-2) * fps
    joint_velocity[..., 0, :] = joint_velocity[..., 1, :]
    foot_velocity = np.zeros_like(feet)
    foot_velocity[..., 1:, :, :] = np.diff(feet, axis=-3) * fps
    foot_velocity[..., 0, :, :] = foot_velocity[..., 1, :, :]
    output = np.concatenate(
        (
            local_velocity[..., :2],
            root_position[..., 2:3],
            _rotation_6d(relative),
            angular_local,
            angles,
            joint_velocity,
            feet.reshape(leading + (time, 12)),
            foot_velocity.reshape(leading + (time, 12)),
            contacts,
        ),
        axis=-1,
    ).astype(np.float32)
    if output.shape[-1] != FEATURE_DIM:
        raise AssertionError(output.shape)
    return output


CONTACT_VELOCITY_EPS = 1e-07


def contact_flags(contact_velocity, body_indices):
    selected = jnp.asarray(contact_velocity)[jnp.asarray(body_indices)]
    return jnp.any(jnp.abs(selected) > CONTACT_VELOCITY_EPS, axis=-1)


def quaternion_matrix(quaternion):
    quaternion = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1e-08)
    w, x, y, z = quaternion
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def matrix_rotvec(matrix):
    cosine = jnp.clip((jnp.trace(matrix) - 1) / 2, -1.0, 1.0)
    angle = jnp.arccos(cosine)
    skew = jnp.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        )
    )
    sine = jnp.sin(angle)
    scale = jnp.where(jnp.abs(sine) > 1e-05, angle / (2 * sine), 0.5)
    return skew * scale


def transition_feature(
    previous_root_position,
    root_position,
    previous_root_quaternion,
    root_quaternion,
    previous_joint_angles,
    joint_angles,
    previous_feet_local,
    feet_local,
    contacts,
):
    previous_rotation = quaternion_matrix(previous_root_quaternion)
    rotation = quaternion_matrix(root_quaternion)
    relative = previous_rotation.T @ rotation
    world_velocity = (root_position - previous_root_position) * FPS
    local_velocity = rotation.T @ world_velocity
    feature = jnp.concatenate(
        (
            local_velocity[:2],
            root_position[2:3],
            relative[:, :2].reshape(-1),
            matrix_rotvec(relative) * FPS,
            joint_angles,
            (joint_angles - previous_joint_angles) * FPS,
            feet_local.reshape(-1),
            ((feet_local - previous_feet_local) * FPS).reshape(-1),
            contacts.astype(jnp.float32),
        )
    )
    if feature.shape != (FEATURE_DIM,):
        raise ValueError(
            f"feature contract produced {feature.shape}, expected {(FEATURE_DIM,)}"
        )
    return feature


def world_feet(root_position, yaw, local_feet):
    root_position = np.asarray(root_position, np.float32)
    yaw = np.asarray(yaw, np.float32)
    local_feet = np.asarray(local_feet, np.float32)
    cosine, sine = (np.cos(yaw), np.sin(yaw))
    rotation = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    rotation[..., 0, 0] = cosine
    rotation[..., 0, 1] = -sine
    rotation[..., 1, 0] = sine
    rotation[..., 1, 1] = cosine
    rotation[..., 2, 2] = 1.0
    return root_position[..., None, :] + np.einsum(
        "...tij,...tfj->...tfi", rotation, local_feet
    )


def _rot_x(angle, xp):
    zero = xp.zeros_like(angle)
    one = xp.ones_like(angle)
    cosine, sine = (xp.cos(angle), xp.sin(angle))
    return xp.stack(
        (one, zero, zero, zero, cosine, -sine, zero, sine, cosine), axis=-1
    ).reshape(angle.shape + (3, 3))


def _rot_y(angle, xp):
    zero = xp.zeros_like(angle)
    one = xp.ones_like(angle)
    cosine, sine = (xp.cos(angle), xp.sin(angle))
    return xp.stack(
        (cosine, zero, sine, zero, one, zero, -sine, zero, cosine), axis=-1
    ).reshape(angle.shape + (3, 3))


def _mv(rotation, vector, xp):
    return xp.einsum("...ij,...j->...i", rotation, vector)


def _leg_foot(bar_position, bar_rotation, hip_angle, knee_angle, side, xp):
    hip_rotation = _rot_y(hip_angle, xp)
    upper_rotation = xp.einsum("...ij,...jk->...ik", bar_rotation, hip_rotation)
    upper_parent_offset = xp.asarray((0.0, 0.875 * side, 0.0))
    upper_child_offset = xp.asarray((0.0, 0.0, 0.375))
    upper_position = bar_position + _mv(
        bar_rotation,
        upper_parent_offset - _mv(hip_rotation, upper_child_offset, xp),
        xp,
    )
    knee_rotation = _rot_y(knee_angle, xp)
    lower_rotation = xp.einsum("...ij,...jk->...ik", upper_rotation, knee_rotation)
    lower_parent_offset = xp.asarray((0.0, -0.25 * side, -0.25))
    lower_child_offset = xp.asarray((0.0, 0.0, 0.25))
    lower_position = upper_position + _mv(
        upper_rotation,
        lower_parent_offset - _mv(knee_rotation, lower_child_offset, xp),
        xp,
    )
    return lower_position + _mv(lower_rotation, xp.asarray((0.0, 0.0, -0.5)), xp)


def _fetch_feet(joint_angles, xp):
    joint_angles = xp.asarray(joint_angles)
    if joint_angles.shape[-1] != 10:
        raise ValueError(f"expected (..., 10) joint angles, got {joint_angles.shape}")
    leading = joint_angles.shape[:-1]
    front_position = xp.broadcast_to(xp.asarray((1.0, 0.0, 0.0)), leading + (3,))
    back_position = xp.broadcast_to(xp.asarray((-1.0, 0.0, 0.0)), leading + (3,))
    front_rotation = _rot_x(joint_angles[..., 0], xp)
    back_rotation = _rot_x(joint_angles[..., 1], xp)
    return xp.stack(
        (
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 2],
                joint_angles[..., 3],
                -1.0,
                xp,
            ),
            _leg_foot(
                front_position,
                front_rotation,
                joint_angles[..., 4],
                joint_angles[..., 5],
                1.0,
                xp,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 6],
                joint_angles[..., 7],
                -1.0,
                xp,
            ),
            _leg_foot(
                back_position,
                back_rotation,
                joint_angles[..., 8],
                joint_angles[..., 9],
                1.0,
                xp,
            ),
        ),
        axis=-2,
    )


def fetch_feet(joint_angles):
    return _fetch_feet(joint_angles, jnp)


def fetch_feet_numpy(joint_angles: np.ndarray) -> np.ndarray:
    return np.asarray(_fetch_feet(np.asarray(joint_angles, np.float32), np), np.float32)


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
        raise ValueError(
            f"expected final feature dimension {FEATURE_DIM}, got {features.shape}"
        )
    feet = features[..., slice(*SL["feet_local"])].reshape(*features.shape[:-1], 4, 3)
    foot_velocity = features[..., slice(*SL["feet_velocity"])].reshape(
        *features.shape[:-1], 4, 3
    )
    contacts = features[..., slice(*SL["contacts"])]
    return (feet, foot_velocity, contacts)


def gait_statistics(features: np.ndarray) -> dict[str, np.ndarray]:
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
    stance_world_speed = (feet_world_speed * contacts).sum(axis=(1, 2)) / stance_count
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
                first, second = (signal[:-lag], signal[lag:])
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                if denominator > 1e-08:
                    correlations.append(float(np.dot(first, second) / denominator))
        cyclicity[clip_index] = max(correlations, default=0.0)
    return {
        "duty_factor": contacts.mean(axis=(1, 2)),
        "airborne_fraction": airborne.mean(axis=1),
        "max_flight_seconds": longest_flight / FPS,
        "contact_switch_hz": np.abs(np.diff(contacts, axis=1)).mean(axis=(1, 2)) * FPS,
        "stance_foot_speed": stance_speed,
        "stance_world_foot_speed": stance_world_speed,
        "vertical_accel_rms_g": np.sqrt(np.mean(np.square(root_acceleration_z), axis=1))
        / 9.8,
        "joint_speed_rms": np.sqrt(
            np.mean(np.square(features[..., slice(*SL["joint_velocity"])]), axis=(1, 2))
        ),
        "cyclicity": cyclicity,
    }


def gait_distance(metrics: dict, reference: dict) -> float:
    standardized = []
    for name in GAIT_FIELDS:
        target = reference[name]["mean"]
        scale = max(reference[name]["std"], 0.1 * abs(target), 0.001)
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
    return (dominant, stride_fraction, buzz_fraction)


def _longest_constant_run(values: np.ndarray) -> int:
    if not len(values):
        return 0
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    return int(np.diff(np.concatenate(([0], boundaries, [len(values)]))).max())


def four_limb_contact_metrics(
    contacts: np.ndarray, *, fps: float = FPS, window_seconds: float = 1.0
) -> dict:
    contacts = np.asarray(contacts, dtype=bool)
    if contacts.ndim != 2 or contacts.shape[1] != len(FOOT_NAMES):
        raise ValueError(f"expected (time, 4) contacts, got {contacts.shape}")
    if len(contacts) < 2:
        raise ValueError("contact audit needs at least two frames")
    duration = (len(contacts) - 1) / float(fps)
    switches = np.sum(contacts[1:] != contacts[:-1], axis=0)
    switch_hz = switches / max(duration, 1.0 / fps)
    duty = contacts.mean(axis=0)
    probability = np.clip(duty, 1e-06, 1.0 - 1e-06)
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
    switch_balance = float(np.std(switch_hz) / max(np.mean(switch_hz), 1e-08))
    passes = bool(
        np.min(switch_hz) >= 1.0
        and np.min(entropy) >= 0.3
        and (np.max(longest_constant) <= 1.25)
        and (np.min(per_foot_window_fraction) >= 0.6)
        and (all_feet_window_fraction >= 0.4)
        and (switch_balance <= 0.75)
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
    features: np.ndarray, *, fps: float = FPS, warmup_seconds: float = 1.0
) -> dict:
    features = np.asarray(features, np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"expected (time, {FEATURE_DIM}) features, got {features.shape}"
        )
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
        and np.min(fore_aft_excursion) >= 0.1
        and (np.min(vertical_excursion) >= 0.03)
        and (np.min(stride_power) >= 0.3)
        and (np.max(buzz_power) <= 0.25)
        and (np.min(swing_reset_margin) >= 0.15)
    )
    return {
        "foot_order": FOOT_NAMES,
        "fore_aft_excursion_fetch_units": fore_aft_excursion.tolist(),
        "vertical_excursion_fetch_units": vertical_excursion.tolist(),
        "dominant_vertical_frequency_hz": dominant_frequency.tolist(),
        "stride_band_power_fraction": stride_power.tolist(),
        "high_frequency_power_fraction": buzz_power.tolist(),
        "stance_local_forward_velocity_fetch_units_per_s": stance_forward_velocity.tolist(),
        "swing_local_forward_velocity_fetch_units_per_s": swing_forward_velocity.tolist(),
        "swing_reset_margin_fetch_units_per_s": swing_reset_margin.tolist(),
        "contact_gate": contact,
        "passes_four_limb_stride_gate": passes_stride,
    }
