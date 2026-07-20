"""Frozen problem definition for Demo C.

The environment, metric, baseline, split seeds, and default step budget live here so
model iterations cannot silently move the comparison. Architecture experiments may
change ``PolicyConfig`` explicitly, but should not edit ``TASK`` or ``PPO`` mid-block.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskConfig:
    # One Demo B transition emits 8 tokens x 4 frames/token at 50 Hz.
    step_seconds: float = 0.64
    horizon: int = 8
    goal_radius_min: float = 0.35
    goal_radius_max: float = 0.75
    goal_bearing_max: float = 1.5707963267948966  # forward semicircle; no out-of-distribution in-place U-turns
    reach_radius: float = 0.12
    forward_min: float = 0.03
    forward_max: float = 0.14
    turn_max: float = 0.55
    progress_scale: float = 10.0
    arrival_bonus: float = 1.0
    time_cost: float = 0.01
    turn_cost: float = 0.01
    invalid_penalty: float = 1.0


@dataclass(frozen=True)
class PPOConfig:
    # Frozen after the 2026-07-19 convergence probe: 262k truncated WAM learning;
    # both variants plateau by ~600--800k. See experiment/DECISIONS.md.
    total_env_steps: int = 786_432
    num_envs: int = 256
    rollout_steps: int = 8
    update_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    max_grad_norm: float = 1.0
    eval_episodes: int = 1024


@dataclass(frozen=True)
class PolicyConfig:
    hidden_size: int = 128
    hidden_layers: int = 2
    initial_log_std: float = -0.6


TASK = TaskConfig()
PPO = PPOConfig()
POLICY = PolicyConfig()
BASE_OBS_SIZE = 8  # goal xy, distance, body vx/vy/yaw-rate, previous action
WAM_CONTEXT_SIZE = 192
VARIANTS = ("goal_only", "wam")
TRAIN_SEEDS = (0, 1, 2)
EVAL_SEED = 10_000


def resolved_config():
    return {"task": asdict(TASK), "ppo": asdict(PPO), "policy": asdict(POLICY)}
