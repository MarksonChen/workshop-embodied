from __future__ import annotations

import numpy as np

from demo_j.control.aligned import (
    PREVIOUS_ACTION_SLICE,
    TOKEN_DIM,
    TOKEN_FRAMES,
    MotionTokenizer,
    aligned_input_dim,
    build_clip_sequences,
    clip_observations,
    select_speed_examples,
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


def test_native_sequence_uses_every_transition_once_without_wrapping() -> None:
    reference = _reference()
    sequences = build_clip_sequences(reference, _tokenizer(), preview_tokens=4)
    assert sequences.observation.shape == (
        reference.clips,
        reference.frames - 1,
        aligned_input_dim(4),
    )
    np.testing.assert_array_equal(sequences.action, reference.teacher_action)
    np.testing.assert_array_equal(
        sequences.observation[:, 0, PREVIOUS_ACTION_SLICE], 0.0
    )
    np.testing.assert_array_equal(
        sequences.observation[:, 1:, PREVIOUS_ACTION_SLICE],
        reference.teacher_action[:, :-1],
    )


def test_future_tail_is_zeroed_and_explicitly_masked() -> None:
    sequences = build_clip_sequences(_reference(), _tokenizer(), preview_tokens=4)
    assert np.all(sequences.preview_mask[:, 0] == 1)
    assert np.all(sequences.preview_mask[:, -1] == 0)
    token_start = PREVIOUS_ACTION_SLICE.stop
    token_stop = token_start + 4 * TOKEN_DIM
    np.testing.assert_array_equal(
        sequences.observation[:, -1, token_start:token_stop], 0
    )
    np.testing.assert_array_equal(
        sequences.observation[:, :, token_stop : token_stop + 4],
        sequences.preview_mask,
    )


def test_terminal_state_probe_adds_a_frame_without_a_transition() -> None:
    reference = _reference()
    previous_action = np.concatenate(
        (
            np.zeros((reference.clips, 1, 10), np.float32),
            reference.teacher_action,
        ),
        axis=1,
    )
    observation, mask = clip_observations(
        reference.features,
        previous_action,
        reference.command,
        _tokenizer(),
        preview_tokens=4,
    )
    assert observation.shape == (
        reference.clips,
        reference.frames,
        aligned_input_dim(4),
    )
    np.testing.assert_array_equal(
        observation[:, -1, PREVIOUS_ACTION_SLICE], reference.teacher_action[:, -1]
    )
    assert np.all(mask[:, -1] == 0)


def test_speed_selection_returns_distinct_real_clips() -> None:
    sequences = build_clip_sequences(_reference(), _tokenizer(), preview_tokens=1)
    selected = select_speed_examples(sequences, sequences.speed)
    assert selected.shape == (2,)
    assert set(selected.tolist()) == {0, 1}
