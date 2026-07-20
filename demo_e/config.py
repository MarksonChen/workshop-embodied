"""Single frozen configuration for the aligned Demo E experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_e" / "out"
PRIOR_ASSET = ROOT / "demo_b" / "assets" / "motor_prior_demo_e_jax.npz"
TORCH_ASSET = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
MIMIC_CHECKPOINT = (
    ROOT
    / "model_checkpoints"
    / "rodent"
    / "checkpoints-v1"
    / "feedforward_260210_013247_285744"
)

# Version 6 retains v5's literal native task/reset semantics and corrects only
# the Demo B bridge: passive source-constant feature channels are masked, and
# joystick velocities are integrated into the future displacement command.
PIPELINE_VERSION = 6

# The held-out-validation candidate passed the frozen source-data and physical
# transfer gates on two independent training seeds.  Keep the accepted Demo B
# generator untouched; Demo E consumes its separately exported scorer above.
PRIOR_READY_FOR_RL = True


@dataclass(frozen=True)
class EnvironmentConfig:
    sim_dt: float = 0.002
    control_dt: float = 0.01
    feature_dt: float = 0.02
    prior_dt: float = 0.08
    episode_seconds: float = 10.0
    command_seconds: float = 5.12
    forward_range: tuple[float, float] = (0.0, 0.5)
    yaw_range: tuple[float, float] = (-1.0, 1.0)
    zero_command_probability: float = 0.10
    tracking_sigma: float = 0.05
    # Frozen from the promoted prior's physically valid-motion calibration.
    # Across the confirmed 52.4M-step reference controller, the pooled 0th and
    # 100th percentiles were -1.416 and -0.772 nats / latent dimension.  The
    # slightly wider limits avoid clipping valid gait variation while assigning
    # clearly out-of-distribution motion zero bonus.
    prior_logp_floor: float = -1.5
    prior_logp_ceiling: float = -0.75
    beta: float = 1.0

    @property
    def controls_per_feature(self) -> int:
        return round(self.feature_dt / self.control_dt)

    @property
    def controls_per_prior(self) -> int:
        return round(self.prior_dt / self.control_dt)

    @property
    def episode_length(self) -> int:
        return round(self.episode_seconds / self.control_dt)

    @property
    def command_resample_steps(self) -> int:
        return round(self.command_seconds / self.control_dt)


@dataclass(frozen=True)
class TrainingConfig:
    # This reproduces the measured successful reference run.  Brax rounds the
    # requested 50M to 52,428,800 transitions for this batch geometry.
    num_timesteps: int = 50_000_000
    num_envs: int = 8192
    num_eval_envs: int = 512
    num_checkpoints: int = 5
    unroll_length: int = 20
    batch_size: int = 2048
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    learning_rate: float = 1e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.99
    policy_layers: tuple[int, ...] = (1024, 512, 256)
    value_layers: tuple[int, ...] = (1024, 512, 256)
    # A wiring check, not a convergence claim.
    smoke_timesteps: int = 327_680
    smoke_envs: int = 512
    smoke_batch_size: int = 128
    # Minimum report-geometry run realized by four 655,360-step PPO intervals.
    # Measured cold E1 runtime on the H100: 276 s. This exposes live learning
    # but is not a locomotion-convergence budget.
    workshop_timesteps: int = 2_621_440
    workshop_envs: int = 8192
    workshop_batch_size: int = 2048


@dataclass(frozen=True)
class EvaluationConfig:
    # 0.30 m/s retains the user's confirmed steady native-reset video. The
    # lower cells remain useful diagnostics of the standing optimum; no low-
    # speed tracking claim is made until a pipeline-v6 paired run measures it.
    commands: tuple[tuple[float, float], ...] = (
        (0.00, 0.00),
        (0.10, 0.00),
        (0.20, 0.00),
        (0.30, 0.00),
        (0.30, -0.75),
        (0.30, +0.75),
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    trial_batch_size: int = 3
    rollout_seconds: float = 5.0
    warmup_seconds: float = 1.0

    @property
    def rollout_steps(self) -> int:
        return round(self.rollout_seconds / ENV.control_dt)


ENV = EnvironmentConfig()
TRAINING = TrainingConfig()
EVAL = EvaluationConfig()


def resolved_config() -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "prior_ready_for_rl": PRIOR_READY_FOR_RL,
        "environment": asdict(ENV),
        "training": asdict(TRAINING),
        "evaluation": asdict(EVAL),
    }
