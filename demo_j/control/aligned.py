"""Long-horizon, token-conditioned data contract for the aligned Demo J SNN.

The accepted Demo F/H release contains independent 64-frame clips.  It does
not contain a continuous 1,000-frame trial.  This module therefore keeps two
claims deliberately separate:

* a train-only PCA encoder learns 16-D tokens from four consecutive body
  features; and
* a 32-frame, wrap-screened segment is repeated as an explicitly synthetic
  periodic reference for recurrent-state stress tests.

No Demo H policy or hidden activation is queried while constructing this
training data.  Session splits and the physics-derived control labels remain
those of the projected Demo F release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from demo_f.config import FPS
from demo_j.control.config import ACTION_DIM, FEATURE_DIM
from demo_j.artifacts import sha256
from demo_j.data.dataset import ReferenceSet
from demo_j.data.physics import joint_qvel_addresses


TOKEN_FRAMES = 4
TOKEN_DIM = 16
CYCLE_FRAMES = 32
DEFAULT_PREVIEW_TOKENS = 8
COMMAND_HORIZON_SECONDS = 31.0 / FPS
PHASE_DIM = 4
SPEED_ANCHORS = np.linspace(1.5, 4.0, 6, dtype=np.float32)
PREVIOUS_ACTION_SLICE = slice(FEATURE_DIM, FEATURE_DIM + ACTION_DIM)


def aligned_input_dim(preview_tokens: int) -> int:
    if preview_tokens < 1:
        raise ValueError("preview_tokens must be positive")
    return FEATURE_DIM + ACTION_DIM + preview_tokens * TOKEN_DIM + PHASE_DIM + 3


@dataclass(frozen=True)
class MotionTokenizer:
    """A transparent, independently fitted whitened-PCA motion encoder."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    block_mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    training_manifest_sha256: str

    @property
    def token_dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, blocks: np.ndarray) -> np.ndarray:
        blocks = np.asarray(blocks, np.float32)
        expected = (TOKEN_FRAMES, FEATURE_DIM)
        if blocks.shape[-2:] != expected:
            raise ValueError(f"expected [..., {expected}], got {blocks.shape}")
        normalized = (blocks - self.feature_mean) / self.feature_std
        flat = normalized.reshape(normalized.shape[:-2] + (-1,))
        token = (flat - self.block_mean) @ self.components
        token /= np.sqrt(np.maximum(self.eigenvalues, 1e-6))
        return token.astype(np.float32)

    def save(self, path: Path) -> dict[str, object]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            block_mean=self.block_mean,
            components=self.components,
            eigenvalues=self.eigenvalues,
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "schema": "demo-j-independent-motion-tokenizer-v1",
                        "method": "train-only whitened PCA",
                        "token_frames": TOKEN_FRAMES,
                        "token_dim": self.token_dim,
                        "training_manifest_sha256": self.training_manifest_sha256,
                    },
                    sort_keys=True,
                )
            ),
        )
        return tokenizer_metadata(path)


def tokenizer_metadata(path: Path) -> dict[str, object]:
    path = Path(path)
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    return {**metadata, "path": str(path), "sha256": sha256(path)}


def load_tokenizer(path: Path) -> MotionTokenizer:
    path = Path(path)
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema") != "demo-j-independent-motion-tokenizer-v1":
            raise ValueError(f"unsupported tokenizer {metadata.get('schema')!r}")
        if metadata.get("token_frames") != TOKEN_FRAMES:
            raise ValueError("token-frame contract mismatch")
        values = {
            name: np.asarray(archive[name], np.float32)
            for name in (
                "feature_mean",
                "feature_std",
                "block_mean",
                "components",
                "eigenvalues",
            )
        }
    return MotionTokenizer(
        **values,
        training_manifest_sha256=metadata["training_manifest_sha256"],
    )


def _contiguous_blocks(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, np.float32)
    clips, frames, channels = features.shape
    if channels != FEATURE_DIM or frames < TOKEN_FRAMES:
        raise ValueError(features.shape)
    windows = np.lib.stride_tricks.sliding_window_view(features, TOKEN_FRAMES, axis=1)
    # sliding_window_view places the window axis last: [N, T, D, 4].
    windows = windows.transpose(0, 1, 3, 2)
    return windows.reshape(clips * (frames - TOKEN_FRAMES + 1), -1)


def fit_tokenizer(
    reference: ReferenceSet, token_dim: int = TOKEN_DIM
) -> MotionTokenizer:
    """Fit the token encoder using only a caller-supplied training split."""

    if reference.split != "train":
        raise ValueError("the tokenizer must be fitted on the training split")
    if not 1 <= token_dim <= TOKEN_FRAMES * FEATURE_DIM:
        raise ValueError(token_dim)
    feature_mean = reference.features.mean(axis=(0, 1), dtype=np.float64)
    feature_std = reference.features.std(axis=(0, 1), dtype=np.float64)
    feature_std = np.maximum(feature_std, 1e-4)
    normalized = (reference.features - feature_mean) / feature_std
    blocks = _contiguous_blocks(normalized).astype(np.float64)
    block_mean = blocks.mean(axis=0)
    centered = blocks - block_mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:token_dim]
    return MotionTokenizer(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        block_mean=block_mean.astype(np.float32),
        components=components[:, order].astype(np.float32),
        eigenvalues=eigenvalues[order].astype(np.float32),
        training_manifest_sha256=reference.manifest_sha256,
    )


def _yaw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def cycle_candidates(
    reference: ReferenceSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose each clip's lowest-discontinuity 32-frame periodic segment.

    The score is used only to construct/evaluate the synthetic stress test.  It
    is never a locomotion reward or checkpoint-selection neural metric.
    """

    if reference.frames < CYCLE_FRAMES + 1:
        raise ValueError(reference.frames)
    starts = np.arange(reference.frames - CYCLE_FRAMES, dtype=np.int32)
    joint_velocity = reference.qvel[..., joint_qvel_addresses()]
    costs = []
    speeds = []
    commands = []
    for start in starts:
        stop = start + CYCLE_FRAMES
        angle_cost = np.sqrt(
            np.mean(
                np.square(
                    (reference.joint_angles[:, stop] - reference.joint_angles[:, start])
                    / 0.35
                ),
                axis=-1,
            )
        )
        velocity_cost = np.sqrt(
            np.mean(
                np.square((joint_velocity[:, stop] - joint_velocity[:, start]) / 5.0),
                axis=-1,
            )
        )
        dot = np.abs(
            np.sum(
                reference.root_quaternion[:, stop]
                * reference.root_quaternion[:, start],
                axis=-1,
            )
        )
        orientation_cost = 2 * np.arccos(np.clip(dot, 0.0, 1.0)) / 0.35
        height_cost = (
            np.abs(
                reference.root_position[:, stop, 2]
                - reference.root_position[:, start, 2]
            )
            / 0.12
        )
        contact_cost = np.mean(
            reference.contacts[:, stop] != reference.contacts[:, start], axis=-1
        )
        costs.append(
            angle_cost
            + 0.35 * velocity_cost
            + orientation_cost
            + height_cost
            + 0.25 * contact_cost
        )

        delta = (
            reference.root_position[:, stop, :2] - reference.root_position[:, start, :2]
        )
        yaw = _yaw(reference.root_quaternion[:, start])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        yaw_delta = _wrap_angle(
            _yaw(reference.root_quaternion[:, stop])
            - _yaw(reference.root_quaternion[:, start])
        )
        seconds = CYCLE_FRAMES / FPS
        speeds.append(local_x / seconds)
        commands.append(
            np.stack(
                (
                    local_x / seconds * COMMAND_HORIZON_SECONDS,
                    local_y / seconds * COMMAND_HORIZON_SECONDS,
                    yaw_delta / seconds * COMMAND_HORIZON_SECONDS,
                ),
                axis=-1,
            )
        )
    cost_matrix = np.stack(costs, axis=1)
    best = np.argmin(cost_matrix, axis=1)
    row = np.arange(reference.clips)
    speed_matrix = np.stack(speeds, axis=1)
    command_matrix = np.stack(commands, axis=1)
    return (
        starts[best],
        cost_matrix[row, best].astype(np.float32),
        np.concatenate(
            (
                speed_matrix[row, best, None],
                command_matrix[row, best],
            ),
            axis=-1,
        ).astype(np.float32),
    )


@dataclass(frozen=True)
class PeriodicSequences:
    """One periodic control cycle per independently projected source clip."""

    observation: np.ndarray
    action: np.ndarray
    cycle_start: np.ndarray
    wrap_score: np.ndarray
    speed: np.ndarray
    command: np.ndarray
    session_index: np.ndarray

    @property
    def clips(self) -> int:
        return int(self.observation.shape[0])

    @property
    def cycle_frames(self) -> int:
        return int(self.observation.shape[1])


def build_periodic_sequences(
    reference: ReferenceSet,
    tokenizer: MotionTokenizer,
    *,
    preview_tokens: int = DEFAULT_PREVIEW_TOKENS,
) -> PeriodicSequences:
    """Encode rolling future-token previews for explicit periodic episodes."""

    if tokenizer.token_dim != TOKEN_DIM:
        raise ValueError(tokenizer.token_dim)
    starts, score, speed_command = cycle_candidates(reference)
    rows = np.arange(reference.clips)[:, None]
    time = np.arange(CYCLE_FRAMES)[None]
    indices = starts[:, None] + time
    feature = reference.features[rows, indices]
    action = reference.teacher_action[rows, indices]
    previous = np.roll(action, 1, axis=1)

    cycle_time = np.arange(CYCLE_FRAMES)
    block_time = (
        cycle_time[:, None] + 1 + np.arange(TOKEN_FRAMES)[None]
    ) % CYCLE_FRAMES
    token_by_time = tokenizer.encode(feature[:, block_time])
    previews = []
    for token_offset in range(preview_tokens):
        token_time = (cycle_time + token_offset * TOKEN_FRAMES) % CYCLE_FRAMES
        previews.append(token_by_time[:, token_time])
    preview = np.concatenate(previews, axis=-1)
    phase = np.eye(PHASE_DIM, dtype=np.float32)[cycle_time % PHASE_DIM]
    phase = np.broadcast_to(phase[None], (reference.clips,) + phase.shape)
    command = speed_command[:, 1:]
    repeated_command = np.broadcast_to(
        command[:, None], (reference.clips, CYCLE_FRAMES, 3)
    )
    observation = np.concatenate(
        (feature, previous, preview, phase, repeated_command), axis=-1
    ).astype(np.float32)
    expected = aligned_input_dim(preview_tokens)
    if observation.shape != (reference.clips, CYCLE_FRAMES, expected):
        raise ValueError(observation.shape)
    return PeriodicSequences(
        observation=observation,
        action=action.astype(np.float32),
        cycle_start=starts,
        wrap_score=score,
        speed=speed_command[:, 0],
        command=command,
        session_index=reference.session_index,
    )


def input_normalization(sequences: PeriodicSequences) -> tuple[np.ndarray, np.ndarray]:
    mean = sequences.observation.mean(axis=(0, 1), dtype=np.float64)
    std = sequences.observation.std(axis=(0, 1), dtype=np.float64)
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def balanced_speed_indices(
    sequences: PeriodicSequences,
    clips_per_speed: int,
    *,
    anchors: np.ndarray = SPEED_ANCHORS,
) -> np.ndarray:
    """Build an equal-size sampling stratum around every speed anchor.

    Sparse strata repeat real clip indices; they never synthesize a trajectory
    or relabel its command.  This makes the intervention auditable and avoids
    silently optimizing only the source distribution's dense low-speed mode.
    """

    if clips_per_speed < 1:
        raise ValueError("clips_per_speed must be positive")
    anchors = np.asarray(anchors, np.float32)
    if anchors.ndim != 1 or len(anchors) < 2 or np.any(np.diff(anchors) <= 0):
        raise ValueError("anchors must be a strictly increasing vector")
    speed = np.asarray(sequences.speed)
    eligible = np.flatnonzero((speed >= anchors[0]) & (speed <= anchors[-1]))
    if not len(eligible):
        raise ValueError("no clips lie inside the requested speed range")
    boundaries = (anchors[:-1] + anchors[1:]) / 2
    strata = np.digitize(speed[eligible], boundaries)
    selected = []
    for index, target in enumerate(anchors):
        candidates = eligible[strata == index]
        if not len(candidates):
            nearest = np.argsort(np.abs(speed[eligible] - target))[:1]
            candidates = eligible[nearest]
        order = np.lexsort(
            (np.abs(speed[candidates] - target), sequences.wrap_score[candidates])
        )
        selected.append(np.resize(candidates[order], clips_per_speed))
    return np.concatenate(selected).astype(np.int32)


def select_speed_examples(
    sequences: PeriodicSequences,
    requested_speeds: np.ndarray,
) -> np.ndarray:
    """Select distinct, wrap-screened examples near requested forward speeds."""

    requested = np.asarray(requested_speeds, np.float32)
    positive = np.flatnonzero(sequences.speed > 0.05)
    if len(positive) < len(requested):
        raise ValueError("not enough positive-speed periodic examples")
    score_scale = max(float(np.quantile(sequences.wrap_score[positive], 0.75)), 1e-3)
    selected: list[int] = []
    for target in requested:
        speed_scale = max(float(target), 0.25)
        objective = (
            np.abs(sequences.speed[positive] - target) / speed_scale
            + 0.08 * sequences.wrap_score[positive] / score_scale
        )
        for candidate in positive[np.argsort(objective)]:
            if int(candidate) not in selected:
                selected.append(int(candidate))
                break
    return np.asarray(selected, np.int32)


def periodic_reference_set(
    reference: ReferenceSet,
    sequences: PeriodicSequences,
    clip_indices: np.ndarray,
    *,
    frames: int,
) -> ReferenceSet:
    """Create explicitly synthetic long references for closed-loop stress tests."""

    clip_indices = np.asarray(clip_indices, np.int32)
    if frames < CYCLE_FRAMES + 1:
        raise ValueError(frames)
    count = len(clip_indices)
    time = np.arange(frames)
    cycle_number = time // CYCLE_FRAMES
    cycle_time = time % CYCLE_FRAMES
    starts = sequences.cycle_start[clip_indices]
    source_index = starts[:, None] + cycle_time[None]
    rows = clip_indices[:, None]

    qpos = reference.qpos[rows, source_index].copy()
    qvel = reference.qvel[rows, source_index].copy()
    features = reference.features[rows, source_index].copy()
    contacts = reference.contacts[rows, source_index].copy()
    angles = reference.joint_angles[rows, source_index].copy()
    quaternions = reference.root_quaternion[rows, source_index].copy()

    stop = starts + CYCLE_FRAMES
    delta = (
        reference.root_position[clip_indices, stop]
        - reference.root_position[clip_indices, starts]
    )
    # A periodic gait may translate in the ground plane, but never accumulate
    # an endpoint height mismatch or an orientation interpolation artifact.
    planar_delta = delta.copy()
    planar_delta[:, 2] = 0.0
    qpos[..., :3] += cycle_number[None, :, None] * planar_delta[:, None]
    roots = qpos[..., :3]

    action_index = starts[:, None] + (np.arange(frames - 1) % CYCLE_FRAMES)[None]
    teacher_action = reference.teacher_action[rows, action_index].copy()
    command = sequences.command[clip_indices].copy()
    provenance_hash = hashlib.sha256(
        (
            reference.manifest_sha256
            + "demo-j-periodic-reference-v1"
            + ",".join(map(str, clip_indices.tolist()))
        ).encode()
    ).hexdigest()
    return replace(
        reference,
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        features=features.astype(np.float32),
        contacts=contacts.astype(np.uint8),
        root_position=roots.astype(np.float32),
        root_quaternion=quaternions.astype(np.float32),
        joint_angles=angles.astype(np.float32),
        command=command.astype(np.float32),
        teacher_action=teacher_action.astype(np.float32),
        session_index=reference.session_index[clip_indices],
        parent_clip_id=reference.parent_clip_id[clip_indices],
        source_start=reference.source_start[clip_indices],
        raw_source_start=reference.raw_source_start[clip_indices],
        source_frame=np.full((count, frames), np.nan, np.float32),
        manifest_sha256=provenance_hash,
        clock="synthetic-periodic-20ms",
    )
