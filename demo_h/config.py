"""Frozen, readable contracts for Demo H.

The label-generation controller is intentionally a small feedback law rather
than a learned black box.  Its outputs are only pseudo-labels: every state that
Demo H stores comes from replaying the corresponding bounded control in the
unchanged Brax v1 Fetch physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

FPS = 50
DT = 1.0 / FPS
CLIP_FRAMES = 64
TRANSITIONS = CLIP_FRAMES - 1
STATE_DIM = 60
ACTION_DIM = 10
TORQUE_STRENGTH = 300.0
COMMAND_HORIZON_SECONDS = 0.62
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


@dataclass(frozen=True)
class PriorConfig:
    """Workshop-scale state/action prior."""

    feature_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = 16
    hidden: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
    downsample: int = 4
    history_tokens: int = 4
    tokenizer_steps: int = 1_000
    predictor_steps: int = 1_500
    action_steps: int = 2_000
    batch_size: int = 512
    learning_rate: float = 3e-4
    predicted_plan_probability: float = 0.75
    predicted_previous_control_probability: float = 0.75
    plan_noise_std: float = 0.05
