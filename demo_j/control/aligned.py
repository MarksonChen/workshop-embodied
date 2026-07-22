"""Native-clip token contract for the aligned Demo J SNN.

The accepted release contains independent 64-frame clips. Every recurrent
episode therefore follows one clip exactly once and resets at its boundary.
Future-token slots that extend beyond the clip are zeroed and explicitly
masked; no state, action, or intention ever wraps to the clip beginning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demo_f.config import FPS
from demo_j.artifacts import sha256
from demo_j.control.config import ACTION_DIM, FEATURE_DIM
from demo_j.data.dataset import ReferenceSet


TOKEN_FRAMES = 4
TOKEN_DIM = 16
DEFAULT_PREVIEW_TOKENS = 8
COMMAND_DIM = 3
PREVIOUS_ACTION_SLICE = slice(FEATURE_DIM, FEATURE_DIM + ACTION_DIM)


def aligned_input_dim(preview_tokens: int) -> int:
    if preview_tokens < 1:
        raise ValueError("preview_tokens must be positive")
    return (
        FEATURE_DIM
        + ACTION_DIM
        + preview_tokens * TOKEN_DIM
        + preview_tokens
        + COMMAND_DIM
    )


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


@dataclass(frozen=True)
class ClipSequences:
    """One finite recurrent episode per independently recorded motion clip."""

    observation: np.ndarray
    action: np.ndarray
    preview_mask: np.ndarray
    speed: np.ndarray
    command: np.ndarray
    session_index: np.ndarray

    @property
    def clips(self) -> int:
        return int(self.observation.shape[0])

    @property
    def steps(self) -> int:
        return int(self.observation.shape[1])


def _future_tokens(
    features: np.ndarray,
    tokenizer: MotionTokenizer,
    preview_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode only in-clip future blocks and return their explicit mask."""

    features = np.asarray(features, np.float32)
    clips, frames, channels = features.shape
    if channels != FEATURE_DIM or frames < 2:
        raise ValueError(features.shape)
    steps = frames - 1
    tokens = np.zeros((clips, steps, preview_tokens, TOKEN_DIM), np.float32)
    mask = np.zeros((clips, steps, preview_tokens), np.float32)
    for time_index in range(steps):
        for token_index in range(preview_tokens):
            start = time_index + 1 + token_index * TOKEN_FRAMES
            stop = start + TOKEN_FRAMES
            if stop <= frames:
                tokens[:, time_index, token_index] = tokenizer.encode(
                    features[:, start:stop]
                )
                mask[:, time_index, token_index] = 1.0
    return tokens.reshape(clips, steps, -1), mask


def clip_observations(
    features: np.ndarray,
    previous_action: np.ndarray,
    command: np.ndarray,
    tokenizer: MotionTokenizer,
    *,
    preview_tokens: int = DEFAULT_PREVIEW_TOKENS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite observations without wrapping any input at the clip tail."""

    features = np.asarray(features, np.float32)
    previous_action = np.asarray(previous_action, np.float32)
    command = np.asarray(command, np.float32)
    clips, frames, channels = features.shape
    steps = frames - 1
    if channels != FEATURE_DIM:
        raise ValueError(features.shape)
    if previous_action.shape != (clips, steps, ACTION_DIM):
        raise ValueError(previous_action.shape)
    if command.shape == (clips, COMMAND_DIM):
        command = np.broadcast_to(command[:, None], (clips, steps, COMMAND_DIM))
    if command.shape != (clips, steps, COMMAND_DIM):
        raise ValueError(command.shape)
    preview, preview_mask = _future_tokens(features, tokenizer, preview_tokens)
    observation = np.concatenate(
        (
            features[:, :steps],
            previous_action,
            preview,
            preview_mask,
            command,
        ),
        axis=-1,
    ).astype(np.float32)
    expected = aligned_input_dim(preview_tokens)
    if observation.shape != (clips, steps, expected):
        raise ValueError(observation.shape)
    return observation, preview_mask


def _clip_speed(reference: ReferenceSet) -> np.ndarray:
    delta = reference.root_position[:, -1, :2] - reference.root_position[:, 0, :2]
    quaternion = reference.root_quaternion[:, 0]
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    local_x = np.cos(yaw) * delta[:, 0] + np.sin(yaw) * delta[:, 1]
    return (local_x / ((reference.frames - 1) / FPS)).astype(np.float32)


def build_clip_sequences(
    reference: ReferenceSet,
    tokenizer: MotionTokenizer,
    *,
    preview_tokens: int = DEFAULT_PREVIEW_TOKENS,
) -> ClipSequences:
    """Build one non-wrapping action sequence for every native clip."""

    if tokenizer.token_dim != TOKEN_DIM:
        raise ValueError(tokenizer.token_dim)
    steps = reference.teacher_action.shape[1]
    if reference.frames != steps + 1:
        raise ValueError((reference.frames, steps))
    previous_action = np.concatenate(
        (
            np.zeros((reference.clips, 1, ACTION_DIM), np.float32),
            reference.teacher_action[:, :-1],
        ),
        axis=1,
    )
    observation, preview_mask = clip_observations(
        reference.features,
        previous_action,
        reference.command,
        tokenizer,
        preview_tokens=preview_tokens,
    )
    return ClipSequences(
        observation=observation,
        action=reference.teacher_action.astype(np.float32),
        preview_mask=preview_mask,
        speed=_clip_speed(reference),
        command=reference.command.astype(np.float32),
        session_index=reference.session_index,
    )


def input_normalization(sequences: ClipSequences) -> tuple[np.ndarray, np.ndarray]:
    mean = sequences.observation.mean(axis=(0, 1), dtype=np.float64)
    std = sequences.observation.std(axis=(0, 1), dtype=np.float64)
    return mean.astype(np.float32), np.maximum(std, 1e-4).astype(np.float32)


def select_speed_examples(
    sequences: ClipSequences,
    requested_speeds: np.ndarray,
) -> np.ndarray:
    """Select distinct native clips nearest the requested realized speeds."""

    requested = np.asarray(requested_speeds, np.float32)
    positive = np.flatnonzero(sequences.speed > 0.05)
    if len(positive) < len(requested):
        raise ValueError("not enough positive-speed native clips")
    selected: list[int] = []
    for target in requested:
        for candidate in positive[
            np.argsort(np.abs(sequences.speed[positive] - target))
        ]:
            if int(candidate) not in selected:
                selected.append(int(candidate))
                break
    return np.asarray(selected, np.int32)
