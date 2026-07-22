from __future__ import annotations

import numpy as np

from demo_j.control.aligned import (
    CYCLE_FRAMES,
    TOKEN_DIM,
    TOKEN_FRAMES,
    MotionTokenizer,
    aligned_input_dim,
    balanced_speed_indices,
    build_periodic_sequences,
    periodic_reference_set,
)
from demo_j.data.dataset import ReferenceSet


def _reference() -> ReferenceSet:
    clips, frames = 2, 64
    time = np.arange(frames, dtype=np.float32)
    qpos = np.zeros((clips, frames, 17), np.float32)
    qpos[..., 2] = 1.2
    qpos[..., 3] = 1.0
    qpos[..., 0] = time[None] * np.asarray((0.01, 0.02))[:, None]
    qpos[..., 7:] = 0.1 * np.sin(time[None, :, None] / 5.0)
    qvel = np.zeros((clips, frames, 16), np.float32)
    qvel[..., 0] = np.asarray((0.5, 1.0))[:, None]
    features = np.zeros((clips, frames, 60), np.float32)
    features[..., 0] = time
    features[..., 1] = np.asarray((0.5, 1.0))[:, None]
    contacts = np.zeros((clips, frames, 4), np.uint8)
    contacts[:, ::2, 0] = 1
    action = np.zeros((clips, frames - 1, 10), np.float32)
    action[..., 0] = np.sin(time[:-1] / 4.0)
    return ReferenceSet(
        qpos=qpos,
        qvel=qvel,
        features=features,
        contacts=contacts,
        root_position=qpos[..., :3],
        root_quaternion=qpos[..., 3:7],
        joint_angles=qpos[..., 7:],
        command=np.zeros((clips, 3), np.float32),
        teacher_action=action,
        session_index=np.arange(clips, dtype=np.int16),
        parent_clip_id=np.arange(clips, dtype=np.int32),
        source_start=np.arange(clips, dtype=np.int32) * frames,
        raw_source_start=np.arange(clips, dtype=np.int32) * frames,
        source_frame=np.broadcast_to(time, (clips, frames)).copy(),
        sessions=("a", "b"),
        split="train",
        manifest_sha256="test",
    )


def _tokenizer() -> MotionTokenizer:
    flat = TOKEN_FRAMES * 60
    components = np.zeros((flat, TOKEN_DIM), np.float32)
    components[:TOKEN_DIM] = np.eye(TOKEN_DIM, dtype=np.float32)
    return MotionTokenizer(
        feature_mean=np.zeros(60, np.float32),
        feature_std=np.ones(60, np.float32),
        block_mean=np.zeros(flat, np.float32),
        components=components,
        eigenvalues=np.ones(TOKEN_DIM, np.float32),
        training_manifest_sha256="test",
    )


def test_periodic_sequence_has_rolling_token_and_action_contract() -> None:
    reference = _reference()
    sequences = build_periodic_sequences(reference, _tokenizer(), preview_tokens=4)
    assert sequences.observation.shape == (
        reference.clips,
        CYCLE_FRAMES,
        aligned_input_dim(4),
    )
    assert sequences.action.shape == (reference.clips, CYCLE_FRAMES, 10)
    assert np.isfinite(sequences.observation).all()
    assert np.all(sequences.speed > 0)


def test_periodic_reference_discloses_synthetic_clock_and_translates() -> None:
    reference = _reference()
    sequences = build_periodic_sequences(reference, _tokenizer(), preview_tokens=1)
    long = periodic_reference_set(
        reference, sequences, np.asarray((0,), np.int32), frames=97
    )
    assert long.clock == "synthetic-periodic-20ms"
    assert np.isnan(long.source_frame).all()
    assert long.qpos.shape == (1, 97, 17)
    assert long.teacher_action.shape == (1, 96, 10)
    assert long.root_position[0, 64, 0] > long.root_position[0, 32, 0]


def test_balanced_speed_indices_repeat_only_real_rows() -> None:
    reference = _reference()
    sequences = build_periodic_sequences(reference, _tokenizer(), preview_tokens=1)
    anchors = np.asarray((sequences.speed.min(), sequences.speed.max()))
    selected = balanced_speed_indices(sequences, clips_per_speed=3, anchors=anchors)
    assert selected.shape == (6,)
    assert set(selected.tolist()) <= {0, 1}
    assert np.count_nonzero(selected == 0) == 3
    assert np.count_nonzero(selected == 1) == 3
