"""Configuration owned by Demo F.

Demo F intentionally does not import Demo B's feature or network constants.  The
retargeted Fetch representation and its model are a new experimental object and
can therefore be tuned without changing the behaviorally accepted Demo B demo.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATA_ROOT = Path(os.environ.get("ALDARONDO_ROOT", "/workspace/data/Aldarondo2024"))
ANIMAL = "coltrane"
FPS = 50


@dataclass(frozen=True)
class ClipSpec:
    """A visually inspected, continuous strict-locomotion source clip."""

    label: str
    session: str
    start: int
    frames: int
    source_speed: float


# Closest 128-frame strict-locomotion clips to four well-separated speeds across
# all 38 Coltrane sessions. These are the exact clips accepted at the visual
# inspection gate; preserve them as fixed regression examples.
INSPECTION_CLIPS = (
    ClipSpec("v0100", "2021_08_10_1", 39_488, 128, 0.099979),
    ClipSpec("v0150", "2021_08_01_1", 65_520, 128, 0.149759),
    ClipSpec("v0200", "2021_08_30_1", 4_480, 128, 0.200273),
    ClipSpec("v0217", "2021_08_17_1", 20_864, 128, 0.217071),
)


# Brax v1 Fetch has ten one-axis joints with +/-60 degree limits.  Ordering is
# exactly the actuator/config ordering in brax.v1.envs.fetch.
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
JOINT_LIMIT = math.radians(60.0)

# Source landmarks corresponding to Fetch's lower-limb endpoints.  The Fetch
# ordering is important because the same ordering is used by its kinematics.
SOURCE_FEET = ("HandR", "HandL", "FootR", "FootL")
FETCH_FOOT_NAMES = (
    "Front Right Lower",
    "Front Left Lower",
    "Back Right Lower",
    "Back Left Lower",
)

# Default Fetch foot endpoints relative to its torso.  The x/y values come from
# Brax's unmodified Fetch configuration; z is the 1.375-unit standing height.
FETCH_NOMINAL_FEET = (
    (1.0, -0.625, -1.375),
    (1.0, 0.625, -1.375),
    (-1.0, -0.625, -1.375),
    (-1.0, 0.625, -1.375),
)
FETCH_TRUNK_LENGTH = 2.0
FETCH_STAND_HEIGHT = 1.375


@dataclass(frozen=True)
class RetargetConfig:
    """Sequence-IK settings; these are recorded with every retarget artifact."""

    smoothing_window: int = 7
    contact_height_m: float = 0.008
    contact_speed_mps: float = 0.15
    lateral_residual_scale: float = 0.25
    foot_weights: tuple[float, float, float] = (1.0, 0.15, 1.0)
    contact_target_weight: float = 4.0
    stance_velocity_weight: float = 2.0
    contact_height_weight: float = 2.0
    pose_weight: float = 2e-4
    velocity_weight: float = 5e-3
    acceleration_weight: float = 1e-2
    optimizer_steps: int = 1_500
    learning_rate: float = 3e-2
    seed: int = 0


@dataclass(frozen=True)
class PriorConfig:
    """Deliberately small, independently tunable Demo F model."""

    clip_frames: int = 64
    crop_stride: int = 16
    downsample: int = 4
    latent_dim: int = 16
    history_tokens: int = 8
    future_tokens: int = 8
    hidden_channels: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    tokenizer_steps: int = 1_000
    predictor_steps: int = 2_000
    predictor_validation_interval: int = 100
    tokenizer_batch_size: int = 128
    predictor_batch_size: int = 256
    learning_rate: float = 2e-3
