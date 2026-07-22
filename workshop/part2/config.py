from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
FPS = 50
FEATURE_DIM = 60
FEATURE_CONTRACT_VERSION = "demo-f-fetch-features-v1"
LEGACY_FEATURE_CONTRACT_VERSION = FEATURE_CONTRACT_VERSION
JOINT_LIMIT = math.radians(60.0)
JOINT_NAMES = (
    "Torso_Shoulders",
    "Torso_Hips",
    "Shoulders_Front Right Upper",
    "Front Right Upper_Front Right Lower",
    "Shoulders_Front Left Upper",
    "Front Left Upper_Front Left Lower",
    "Hips_Back Right Upper",
    "Back Right Upper_Back Right Lower",
    "Hips_Back Left Upper",
    "Back Left Upper_Back Left Lower",
)


@dataclass(frozen=True)
class PriorConfig:
    clip_frames: int = 64
    downsample: int = 4
    latent_dim: int = 16
    history_tokens: int = 4
    future_tokens: int = 1
    hidden_channels: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    tokenizer_steps: int = 1_000
    predictor_steps: int = 2_000
    predictor_validation_interval: int = 100
    tokenizer_batch_size: int = 128
    predictor_batch_size: int = 256
    learning_rate: float = 2e-3
    joint_limit_penalty: float = 10.0
    training_rollout_tokens: int = 4

    def __post_init__(self) -> None:
        if self.future_tokens != 1:
            raise ValueError("Part 2 uses one-step autoregressive prediction")
