from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workshop.part2.config import FEATURE_DIM, FPS, JOINT_NAMES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

DT = 1.0 / FPS
CLIP_FRAMES = 64
TRANSITIONS = CLIP_FRAMES - 1
STATE_DIM = FEATURE_DIM
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


PD_POSITION_GAIN = 400.0
PD_VELOCITY_GAIN = 10.0


MIN_TORSO_HEIGHT = 0.90
MIN_UPRIGHT = 0.50
MAX_CONTROL_SATURATION = 0.15
MAX_JOINT_TRACKING_RMSE = 0.30
MAX_PLANAR_SPEED = 12.0
MAX_YAW_RATE = 6.0
MAX_COMMAND_SPEED = 8.0


PRIOR_CONTROL_LIMIT = 0.98

FETCH_FOOT_NAMES = (
    "Front Right Lower",
    "Front Left Lower",
    "Back Right Lower",
    "Back Left Lower",
)


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


LEGACY_OBSERVATION_CONTRACT_VERSION = "workshop-part3-observation-v1"
OBSERVATION_CONTRACT_VERSION = LEGACY_OBSERVATION_CONTRACT_VERSION


@dataclass(frozen=True)
class PriorConfig:
    feature_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    latent_dim: int = PLAN_DIM
    hidden: int = 192
    transformer_layers: int = 4
    transformer_heads: int = 4
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
                f"Part 3 online contract is frozen at {expected}, got {actual}"
            )
