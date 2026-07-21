"""Teacher-forced sequences for the independent imitation fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from demo_j.dataset import ReferenceSet
from demo_j.env import LAST_TRACK_FRAME, OBS_DIM, REFERENCE_FRAMES
from demo_j.fetch_mjx import joint_qpos_addresses, joint_qvel_addresses


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
    reference_rows = []
    for frame in range(steps):
        indices = np.minimum(
            frame + 1 + np.arange(REFERENCE_FRAMES), reference.frames - 1
        )
        target_qpos = reference.qpos[:, indices]
        target_qvel = reference.qvel[:, indices]
        current_qpos = reference.qpos[:, frame]
        current_qvel = reference.qvel[:, frame]
        root_target = _inverse_rotate(
            target_qpos[..., :3] - current_qpos[:, None, :3],
            current_qpos[:, None, 3:7],
        )
        # Match brax.math.relative_quat(target, current) exactly.
        quaternion_target = _quaternion_multiply(
            current_qpos[:, None, 3:7],
            _quaternion_inverse(target_qpos[..., 3:7]),
        )
        qpos_address = joint_qpos_addresses()
        qvel_address = joint_qvel_addresses()
        angle_target = (
            target_qpos[..., qpos_address]
            - current_qpos[:, None, qpos_address]
        )
        velocity_target = (
            target_qvel[..., qvel_address]
            - current_qvel[:, None, qvel_address]
        )
        reference_rows.append(
            np.concatenate(
                (root_target, quaternion_target, angle_target, velocity_target),
                axis=-1,
            ).reshape(clips, -1)
        )
    reference_observation = np.stack(reference_rows, axis=1).astype(np.float32)
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
