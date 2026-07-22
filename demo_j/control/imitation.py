"""Teacher-forced sequences for the independent imitation fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from demo_j.data.dataset import ReferenceSet
from demo_j.control.tracking import LAST_TRACK_FRAME, OBS_DIM, REFERENCE_FRAMES
from demo_j.data.physics import joint_qvel_addresses


@dataclass(frozen=True)
class ImitationSequences:
    observation: np.ndarray
    action: np.ndarray
    session_index: np.ndarray

    @property
    def clips(self) -> int:
        return int(self.observation.shape[0])

    @property
    def steps(self) -> int:
        return int(self.observation.shape[1])


def _quaternion_inverse(q: np.ndarray) -> np.ndarray:
    output = np.asarray(q).copy()
    output[..., 1:] *= -1
    return output


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _inverse_rotate(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    q = _quaternion_inverse(quaternion)
    scalar = q[..., :1]
    xyz = q[..., 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + scalar * twice_cross + np.cross(xyz, twice_cross)


def future_reference_observations(
    root_position: np.ndarray,
    root_quaternion: np.ndarray,
    joint_angles: np.ndarray,
    joint_velocities: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """Build Demo J's root-relative five-frame intention target.

    Inputs have shape ``[clips, frames, channels]`` and may come from either
    the projected dataset or an independently recorded locomotion trajectory.
    """

    root_position = np.asarray(root_position, np.float32)
    root_quaternion = np.asarray(root_quaternion, np.float32)
    joint_angles = np.asarray(joint_angles, np.float32)
    joint_velocities = np.asarray(joint_velocities, np.float32)
    shapes = {
        value.shape[:2]
        for value in (
            root_position,
            root_quaternion,
            joint_angles,
            joint_velocities,
        )
    }
    if len(shapes) != 1:
        raise ValueError(shapes)
    clips, frames = root_position.shape[:2]
    if not 1 <= steps <= frames - REFERENCE_FRAMES:
        raise ValueError((steps, frames))
    rows = []
    for frame in range(steps):
        indices = frame + 1 + np.arange(REFERENCE_FRAMES)
        root_target = _inverse_rotate(
            root_position[:, indices] - root_position[:, frame, None],
            root_quaternion[:, frame, None],
        )
        quaternion_target = _quaternion_multiply(
            root_quaternion[:, frame, None],
            _quaternion_inverse(root_quaternion[:, indices]),
        )
        angle_target = joint_angles[:, indices] - joint_angles[:, frame, None]
        velocity_target = (
            joint_velocities[:, indices] - joint_velocities[:, frame, None]
        )
        rows.append(
            np.concatenate(
                (root_target, quaternion_target, angle_target, velocity_target),
                axis=-1,
            ).reshape(clips, -1)
        )
    return np.stack(rows, axis=1).astype(np.float32)


def teacher_forced_sequences(
    reference: ReferenceSet,
    steps: int = LAST_TRACK_FRAME,
) -> ImitationSequences:
    """Build the exact observation/action contract on demonstrated states.

    Action ``t`` is the independent feedback projection control that produced
    demonstrated state ``t + 1``.  These labels are used only by the explicit
    sequence-distillation fallback, never by the state-only PPO path.
    """

    if not 1 <= steps <= reference.teacher_action.shape[1]:
        raise ValueError(steps)
    clips = reference.clips
    previous_action = np.concatenate(
        (
            np.zeros((clips, 1, reference.teacher_action.shape[-1]), np.float32),
            reference.teacher_action[:, : steps - 1],
        ),
        axis=1,
    )
    reference_observation = future_reference_observations(
        reference.root_position,
        reference.root_quaternion,
        reference.joint_angles,
        reference.qvel[..., joint_qvel_addresses()],
        steps=steps,
    )
    observation = np.concatenate(
        (
            reference.features[:, :steps],
            previous_action,
            reference_observation,
        ),
        axis=-1,
    ).astype(np.float32)
    if observation.shape != (clips, steps, OBS_DIM):
        raise ValueError(observation.shape)
    return ImitationSequences(
        observation=observation,
        action=reference.teacher_action[:, :steps].astype(np.float32),
        session_index=reference.session_index,
    )


def normalization(sequences: ImitationSequences) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only affine observation statistics."""

    mean = sequences.observation.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = sequences.observation.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    return mean, std
