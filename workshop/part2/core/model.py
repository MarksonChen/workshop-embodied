from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    FEATURE_CONTRACT_VERSION,
    FEATURE_DIM,
    FPS,
    JOINT_LIMIT,
    LEGACY_FEATURE_CONTRACT_VERSION,
    PriorConfig,
)
from ..data import hindsight_commands
from ..data.contract import COMMAND_FRAME, COMMAND_FUTURE_FRAME
from .motion import SL


class CausalConv1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int = 1):
        super().__init__()
        self.padding = kernel - stride
        self.convolution = nn.Conv1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        return self.convolution(F.pad(values, (self.padding, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, stride: int):
        super().__init__()
        self.trim = kernel - stride
        self.convolution = nn.ConvTranspose1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        output = self.convolution(values)
        return output[..., : -self.trim] if self.trim else output


class MotionAutoencoder(nn.Module):
    def __init__(self, feature_dim: int, hidden: int = 192, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(feature_dim, hidden, 5),
            nn.SiLU(),
            CausalConv1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConv1d(hidden, latent_dim, 4, stride=2),
        )
        self.decoder = nn.Sequential(
            CausalConv1d(latent_dim, hidden, 3),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, feature_dim, 4, stride=2),
        )

    def encode(self, features):
        return self.encoder(features.transpose(1, 2)).transpose(1, 2)

    def decode(self, tokens):
        return self.decoder(tokens.transpose(1, 2)).transpose(1, 2)

    def forward(self, features):
        return self.decode(self.encode(features))


def sinusoidal_positions(length: int, width: int, device) -> torch.Tensor:
    position = torch.arange(length, device=device)[:, None].float()
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-math.log(10000.0) / width)
    )
    output = torch.zeros(length, width, device=device)
    output[:, 0::2] = torch.sin(position * frequency)
    output[:, 1::2] = torch.cos(position * frequency)
    return output


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        future_tokens: int = 8,
        width: int = 192,
        layers: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.future_tokens = future_tokens
        self.input = nn.Linear(latent_dim, width)
        block = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.command = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output = nn.Sequential(
            nn.Linear(width + 64, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, future_tokens * latent_dim),
        )

    def context(self, history):
        hidden = self.input(history)
        hidden = (
            hidden
            + sinusoidal_positions(hidden.shape[1], hidden.shape[2], hidden.device)[
                None
            ]
        )
        return self.norm(self.transformer(hidden))[:, -1]

    def predict_from_context(self, context, command):
        output = self.output(torch.cat((context, self.command(command)), dim=-1))
        return output.view(-1, self.future_tokens, self.latent_dim)

    def predict(self, history, command):
        return self.predict_from_context(self.context(history), command)

    def log_prob(self, history, future, command, sigma):
        sigma = torch.as_tensor(sigma, dtype=history.dtype, device=history.device)
        residual = (future - self.predict(history, command)) / sigma
        return -0.5 * (
            residual.square() + 2 * sigma.log() + math.log(2 * math.pi)
        ).mean(dim=(-1, -2))


def joint_limit_loss(
    normalized_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    *,
    margin: float = 0.95,
) -> torch.Tensor:
    joint_slice = slice(*SL["joint_angles"])
    angles = (
        normalized_features[..., joint_slice] * feature_std[joint_slice]
        + feature_mean[joint_slice]
    )
    excess = torch.relu(angles.abs() - margin * JOINT_LIMIT)
    return excess.square().mean()


def command_frames(
    anchors: np.ndarray, config: PriorConfig
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(anchors, np.int64) * config.downsample
    return (start, start + (COMMAND_FUTURE_FRAME - COMMAND_FRAME))


@torch.inference_mode()
def encode_in_batches(
    model, values: torch.Tensor, batch_size: int = 512
) -> torch.Tensor:
    return torch.cat(
        [
            model.encode(values[offset : offset + batch_size])
            for offset in range(0, len(values), batch_size)
        ]
    )


def predictor_windows(
    tokens, dataset, config: PriorConfig, *, target_tokens: int | None = None
):
    target_tokens = config.future_tokens if target_tokens is None else target_tokens
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    command_horizon = COMMAND_FUTURE_FRAME - COMMAND_FRAME
    last_command_anchor = (
        config.clip_frames - 1 - command_horizon
    ) // config.downsample
    last_target_anchor = tokens.shape[1] - target_tokens
    anchors = np.arange(
        config.history_tokens,
        min(last_command_anchor, last_target_anchor) + 1,
        dtype=np.int64,
    )
    if not len(anchors):
        raise ValueError("model history/future leaves no valid command window")
    history = torch.stack(
        [tokens[:, anchor - config.history_tokens : anchor] for anchor in anchors],
        dim=1,
    ).flatten(0, 1)
    future = torch.stack(
        [tokens[:, anchor : anchor + target_tokens] for anchor in anchors], dim=1
    ).flatten(0, 1)
    command_start, _ = command_frames(anchors, config)
    raw_command = hindsight_commands(
        dataset.root_position, dataset.root_quaternion, command_start, command_horizon
    ).reshape(-1, 3)
    return (history, future, torch.from_numpy(raw_command).to(tokens.device), anchors)


COMMAND_HORIZON_SECONDS = (COMMAND_FUTURE_FRAME - COMMAND_FRAME) / FPS


SPEED_SMOOTHING_FRAMES = 8


@dataclass
class MotionPrior:
    checkpoint: dict
    config: PriorConfig
    tokenizer: MotionAutoencoder
    predictor: ConditionalTransformer

    @property
    def device(self) -> torch.device:
        return next(self.tokenizer.parameters()).device

    def normalize_command(self, command) -> torch.Tensor:
        value = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if "command_mean" in self.checkpoint:
            mean = torch.as_tensor(self.checkpoint["command_mean"], device=self.device)
            std = torch.as_tensor(self.checkpoint["command_std"], device=self.device)
            value = (value - mean) / std
        return value

    @torch.inference_mode()
    def log_prob(self, history, future, command) -> torch.Tensor:
        return self.predictor.log_prob(
            history, future, self.normalize_command(command), self.checkpoint["sigma"]
        )

    def rollout(
        self, seed_features: np.ndarray, command: np.ndarray, frames: int
    ) -> np.ndarray:
        return rollout_features(
            seed_features,
            command,
            frames,
            self.checkpoint,
            self.config,
            self.tokenizer,
            self.predictor,
        )


def load_prior(
    checkpoint_path: str | Path, *, device: str | torch.device | None = None
) -> MotionPrior:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") not in {
        "workshop-part2-prior-v1",
        "workshop-part2-prior-v1",
    }:
        raise ValueError(f"unsupported checkpoint schema {checkpoint.get('schema')!r}")
    feature_contract = checkpoint.get(
        "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
    )
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"checkpoint feature contract {feature_contract!r}; expected {FEATURE_CONTRACT_VERSION!r}"
        )
    config_values = dict(checkpoint["config"])
    config_values.pop("crop_stride", None)
    config = PriorConfig(**config_values)
    tokenizer = MotionAutoencoder(
        FEATURE_DIM, config.hidden_channels, config.latent_dim
    ).to(device)
    predictor = ConditionalTransformer(
        latent_dim=config.latent_dim,
        future_tokens=config.future_tokens,
        width=config.hidden_channels,
        layers=config.transformer_layers,
        heads=config.transformer_heads,
    ).to(device)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    predictor.load_state_dict(checkpoint["predictor"])
    tokenizer.eval()
    predictor.eval()
    return MotionPrior(checkpoint, config, tokenizer, predictor)


def straight_training_mask(command: np.ndarray, source_speed: np.ndarray) -> np.ndarray:
    return (
        (source_speed >= 0.04)
        & (command[:, 0] > 0.0)
        & (np.abs(command[:, 1]) < 0.25)
        & (np.abs(command[:, 2]) < 0.15)
    )


def command_scale(command: np.ndarray, source_speed: np.ndarray) -> float:
    mask = straight_training_mask(command, source_speed)
    if mask.sum() < 20:
        raise ValueError("too few straight training clips to calibrate speed commands")
    return float(np.median(command[mask, 0] / source_speed[mask]))


def dataset_command_calibration(
    manifest: dict, command: np.ndarray, source_speed: np.ndarray
) -> dict:
    dynamic = manifest.get("dynamic_scaling")
    if dynamic is not None:
        scale = float(dynamic["velocity_scale"]) * COMMAND_HORIZON_SECONDS
        return {
            "method": "declared Froude-similar velocity scale times command horizon",
            "fetch_displacement_per_mps": scale,
            "horizon_seconds": COMMAND_HORIZON_SECONDS,
        }
    return {
        "method": "median forward Fetch displacement / source net speed on straight train clips",
        "fetch_displacement_per_mps": command_scale(command, source_speed),
        "horizon_seconds": COMMAND_HORIZON_SECONDS,
    }


def checkpoint_command_scale(checkpoint: dict, train) -> float:
    calibration = checkpoint.get("command_calibration")
    if calibration is not None:
        return float(calibration["fetch_displacement_per_mps"])
    return command_scale(train.command, train.source_speed_mps)


def select_seed(
    dataset, target_speed: float = 0.15, *, scale: float | None = None
) -> int:
    scale = (
        command_scale(dataset.command, dataset.source_speed_mps)
        if scale is None
        else scale
    )
    mask = straight_training_mask(dataset.command, dataset.source_speed_mps)
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        raise ValueError("dataset has no straight locomotion seed candidates")
    clip_speed = dataset.command[candidates, 0] / float(scale)
    score = (
        np.abs(clip_speed - target_speed) / 0.02
        + np.abs(dataset.command[candidates, 1]) / 0.1
        + np.abs(dataset.command[candidates, 2]) / 0.1
    )
    return int(candidates[np.argmin(score)])


@torch.inference_mode()
def rollout_features(
    seed_features: np.ndarray,
    command: np.ndarray,
    frames: int,
    checkpoint: dict,
    config: PriorConfig,
    tokenizer: MotionAutoencoder,
    predictor: ConditionalTransformer,
) -> np.ndarray:
    seed_frames = config.history_tokens * config.downsample
    if frames <= seed_frames:
        raise ValueError(f"frames must exceed the {seed_frames}-frame seed")
    device = next(tokenizer.parameters()).device
    mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    std = torch.as_tensor(checkpoint["feature_std"], device=device)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=device)
    token_std = torch.as_tensor(checkpoint["token_std"], device=device)
    normalized = (
        torch.as_tensor(seed_features, dtype=torch.float32, device=device) - mean
    ) / std
    seed_tokens = tokenizer.encode(normalized[None])
    history = ((seed_tokens - token_mean) / token_std)[:, : config.history_tokens]
    command_tensor = torch.as_tensor(command, dtype=torch.float32, device=device)[None]
    if "command_mean" in checkpoint:
        command_mean = torch.as_tensor(checkpoint["command_mean"], device=device)
        command_std = torch.as_tensor(checkpoint["command_std"], device=device)
        command_tensor = (command_tensor - command_mean) / command_std
    target_tokens = math.ceil(frames / config.downsample)
    stream = [history]
    generated_tokens = config.history_tokens
    while generated_tokens < target_tokens:
        next_token = predictor.predict(history, command_tensor)[:, :1]
        stream.append(next_token)
        history = torch.cat((history, next_token), dim=1)[:, -config.history_tokens :]
        generated_tokens += 1
    normalized_tokens = torch.cat(stream, dim=1)[:, :target_tokens]
    decoded = tokenizer.decode(normalized_tokens * token_std + token_mean) * std + mean
    return decoded[0, :frames].cpu().numpy().astype(np.float32)


def integrate_root(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, np.float32)
    frames = len(features)
    velocity = features[:, slice(*SL["root_velocity"])]
    rotation_columns = features[:, slice(*SL["rotation_delta_6d"])].reshape(
        frames, 3, 2
    )
    delta_yaw = np.arctan2(rotation_columns[:, 1, 0], rotation_columns[:, 0, 0])
    yaw = np.zeros(frames, np.float32)
    yaw[1:] = np.cumsum(delta_yaw[1:]).astype(np.float32)
    root = np.zeros((frames, 3), np.float32)
    root[:, 2] = features[:, SL["root_height"][0]]
    for frame in range(1, frames):
        cosine, sine = (np.cos(yaw[frame]), np.sin(yaw[frame]))
        local_x, local_y = velocity[frame]
        root[frame, 0] = root[frame - 1, 0] + (cosine * local_x - sine * local_y) / FPS
        root[frame, 1] = root[frame - 1, 1] + (sine * local_x + cosine * local_y) / FPS
    quaternion = np.zeros((frames, 4), np.float32)
    quaternion[:, 0] = np.cos(yaw / 2)
    quaternion[:, 3] = np.sin(yaw / 2)
    joint_angles = np.clip(
        features[:, slice(*SL["joint_angles"])], -JOINT_LIMIT, JOINT_LIMIT
    ).astype(np.float32)
    return (joint_angles, root, quaternion)


def trailing_mean(
    values: np.ndarray, window: int = SPEED_SMOOTHING_FRAMES
) -> np.ndarray:
    values = np.asarray(values, np.float32)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid").astype(
        np.float32
    )


def longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest
