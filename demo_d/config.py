"""Frozen problem definition for one-stage Demo D.

Future mocap is converted to the same egocentric displacement command used by
Demo B.  A single PPO policy sees that command plus proprioception and directly
outputs torques.  There is no pretrained decoder and no second joystick stage.
"""

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_d" / "out"
REFERENCE_DATA = ROOT / "data" / "data" / "rodent" / "rodent_reference_clips.h5"
REFERENCE_HF_FILE = "data/rodent/rodent_reference_clips.h5"
REFERENCE_SHA256 = "c7b02c16d6796f70e62169b5a5aeb65381ea5d42d8e9c75af95cd26b31fb638e"
PIPELINE_VERSION = 5
COMMAND_ERROR_SCALES = (0.06, 0.04, 0.35)  # vx, vy (m/s), yaw rate (rad/s).

# The first 24 clips of each locomotion category in the shuffled public MIMIC
# catalogue are training data; the next eight of each are validation data.  Clip
# IDs are explicit rather than recomputed from metadata at training time.
TRAIN_WALK = (
    2, 8, 19, 30, 33, 38, 43, 54, 56, 75, 96, 97,
    98, 105, 110, 111, 122, 140, 145, 149, 150, 156, 162, 174,
)
TRAIN_FAST_WALK = (
    15, 22, 25, 32, 34, 35, 41, 42, 49, 51, 52, 58,
    61, 67, 70, 73, 76, 82, 83, 102, 104, 107, 109, 112,
)
VAL_WALK = (176, 179, 182, 184, 188, 189, 190, 198)
VAL_FAST_WALK = (116, 117, 123, 124, 125, 129, 131, 139)
TRAIN_CLIPS = TRAIN_WALK + TRAIN_FAST_WALK
VAL_CLIPS = VAL_WALK + VAL_FAST_WALK
ALL_CLIPS = TRAIN_CLIPS + VAL_CLIPS


@dataclass(frozen=True)
class TrainingConfig:
    # Roughly 26.2M realized steps because PPO updates in 327,680-step blocks.
    num_timesteps: int = 25_000_000
    eval_every: int = 5_000_000
    num_envs: int = 4096
    batch_size: int = 1024
    unroll_length: int = 20
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.95
    policy_layers: tuple[int, ...] = (512, 512, 256)
    value_layers: tuple[int, ...] = (512, 512, 256)
    command_horizon_frames: int = 31  # 0.62 s at 50 Hz, exactly Demo B's definition.
    command_reward_weight: float = 2.0
    start_frame_max: int = 20
    # Training may recover from reference drift; held-out imitation evaluation
    # deliberately retains the upstream reference-relative termination.
    termination_mode: str = "physical_fall"


@dataclass(frozen=True)
class EvaluationConfig:
    imitation_steps: int = 300
    command_steps: int = 400
    warmup_steps: int = 50
    # Fixed trial IDs; they choose paired validation clips/start offsets.
    seeds: tuple[int, ...] = (0, 1, 2)
    # [forward displacement, lateral displacement, yaw change] over 0.62 s.
    # These stay in the dense center of the audited Demo D command distribution.
    commands: tuple[tuple[float, float, float], ...] = (
        (0.04, 0.00, 0.00),
        (0.08, 0.00, 0.00),
        (0.12, 0.00, 0.00),
        (0.08, 0.00, -0.30),
        (0.08, 0.00, 0.30),
        (0.08, 0.02, 0.00),
    )
    # Predeclared reportability gates.  These are deliberately modest for a
    # workshop-scale run and are always reported alongside raw diagnostics.
    imitation_gain_min: float = 0.08
    imitation_survival_min: float = 0.70
    command_gain_min: float = 0.08
    command_score_min: float = 0.45
    command_survival_min: float = 0.70


TRAINING = TrainingConfig()
EVAL = EvaluationConfig()


def resolved_config() -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "reference_sha256": REFERENCE_SHA256,
        "command_error_scales": list(COMMAND_ERROR_SCALES),
        "train_clips": list(TRAIN_CLIPS),
        "validation_clips": list(VAL_CLIPS),
        "training": asdict(TRAINING),
        "evaluation": asdict(EVAL),
    }
