"""Frozen, readable contracts for Demo H.

The label-generation controller is intentionally a small feedback law rather
than a learned black box.  Its outputs are only pseudo-labels: every state that
Demo H stores comes from replaying the corresponding bounded control in the
unchanged Brax v1 Fetch physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from demo_f.config import FEATURE_DIM, FPS, JOINT_NAMES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

DT = 1.0 / FPS
CLIP_FRAMES = 64
TRANSITIONS = CLIP_FRAMES - 1
STATE_DIM = FEATURE_DIM  # Dataset-schema compatibility name.
ACTION_DIM = len(JOINT_NAMES)
BASE_OBS_DIM = 101
BUFFER_FRAMES = 16
PHASE_DIM = 4
PLAN_DIM = 16
COMMAND_DIM = 3
ACTION_PHASES = 4
HISTORY_TOKENS = BUFFER_FRAMES // ACTION_PHASES
COMMAND_HORIZON_FRAMES = 31
TORQUE_STRENGTH = 300.0
COMMAND_HORIZON_SECONDS = COMMAND_HORIZON_FRAMES / FPS
TARGET_SPEED_FETCH = 3.0
TARGET_COMMAND_X = TARGET_SPEED_FETCH * COMMAND_HORIZON_SECONDS
TASK_SPEED_MIN = 1.5
TASK_SPEED_MAX = 4.0

# Selected on validation projections and retained unchanged for the accepted
# 1.75x-temporally-dilated release.  See experiment/DECISIONS.md.
PD_POSITION_GAIN = 400.0
PD_VELOCITY_GAIN = 10.0

# Fail-closed clip gates frozen before the accepted 1.75x full release.  These
# reject physically broken projections without selecting for a task speed.
MIN_TORSO_HEIGHT = 0.90
MIN_UPRIGHT = 0.50
MAX_CONTROL_SATURATION = 0.15
MAX_JOINT_TRACKING_RMSE = 0.30
MAX_PLANAR_SPEED = 12.0
MAX_YAW_RATE = 6.0
MAX_COMMAND_SPEED = 8.0

# The exact projector can request actuator limits on fast clips.  A Gaussian
# tanh policy cannot represent +/-1 with a finite mean, so likelihood training
# and recurrent inference use this interior support boundary.  The saved
# physical controls remain untouched.
PRIOR_CONTROL_LIMIT = 0.98

FETCH_FOOT_NAMES = (
    "Front Right Lower",
    "Front Left Lower",
    "Back Right Lower",
    "Back Left Lower",
)

# One immutable online observation layout. Keeping the slices beside their
# dimensions makes incompatible tuning fail visibly instead of drifting across
# environment, policy, and evaluation modules.
FEATURE_BUFFER_SLICE = slice(BASE_OBS_DIM, BASE_OBS_DIM + BUFFER_FRAMES * FEATURE_DIM)
PREVIOUS_CONTROL_SLICE = slice(
    FEATURE_BUFFER_SLICE.stop, FEATURE_BUFFER_SLICE.stop + ACTION_DIM
)
PHASE_SLICE = slice(
    PREVIOUS_CONTROL_SLICE.stop, PREVIOUS_CONTROL_SLICE.stop + PHASE_DIM
)
PLAN_SLICE = slice(PHASE_SLICE.stop, PHASE_SLICE.stop + PLAN_DIM)
COMMAND_SLICE = slice(PLAN_SLICE.stop, PLAN_SLICE.stop + COMMAND_DIM)
OBS_DIM = COMMAND_SLICE.stop

# Existing accepted checkpoints predate an explicit field but were trained on
# this exact online layout and Fetch-native contact observation.
LEGACY_OBSERVATION_CONTRACT_VERSION = "demo-h-online-observation-v1"
OBSERVATION_CONTRACT_VERSION = LEGACY_OBSERVATION_CONTRACT_VERSION


@dataclass(frozen=True)
class PriorConfig:
    """Workshop-scale state/action prior."""

    feature_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = PLAN_DIM
    hidden: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    planner_rollout_tokens: int = ACTION_PHASES
    downsample: int = 4
    history_tokens: int = HISTORY_TOKENS
    tokenizer_steps: int = 1_000
    predictor_steps: int = 1_500
    action_steps: int = 2_000
    batch_size: int = 512
    learning_rate: float = 3e-4
    predicted_plan_probability: float = 0.75
    predicted_previous_control_probability: float = 0.75
    plan_noise_std: float = 0.05
    feature_noise_std: float = 0.0
    action_parameterization: str = "previous_control_residual"
    previous_mean_coefficient: float = 1.0

    def validate_online_contract(self) -> None:
        expected = {
            "feature_dim": FEATURE_DIM,
            "action_dim": ACTION_DIM,
            "latent_dim": PLAN_DIM,
            "downsample": ACTION_PHASES,
            "history_tokens": HISTORY_TOKENS,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"Demo H online contract is frozen at {expected}, got {actual}"
            )
        if self.action_parameterization not in {
            "previous_control_residual",
            "leaky_previous",
            "direct",
        }:
            raise ValueError(
                f"unsupported action parameterization {self.action_parameterization!r}"
            )
        if not 0.0 <= self.previous_mean_coefficient <= 1.0:
            raise ValueError(
                f"previous-mean coefficient must be in [0,1], got "
                f"{self.previous_mean_coefficient}"
            )
        for name in (
            "predicted_plan_probability",
            "predicted_previous_control_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")
        if self.plan_noise_std < 0.0:
            raise ValueError(
                f"plan_noise_std must be non-negative, got {self.plan_noise_std}"
            )
        if self.feature_noise_std < 0.0:
            raise ValueError(
                f"feature_noise_std must be non-negative, got "
                f"{self.feature_noise_std}"
            )
        if not ACTION_PHASES <= self.planner_rollout_tokens <= 12:
            raise ValueError(
                "planner_rollout_tokens must cover 4 to 12 tokens, got "
                f"{self.planner_rollout_tokens}"
            )
