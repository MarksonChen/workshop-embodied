import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from demo_d.config import (
    EVAL,
    PIPELINE_VERSION,
    REFERENCE_DATA,
    REFERENCE_HF_FILE,
    REFERENCE_SHA256,
    TRAINING,
    TRAIN_CLIPS,
    TRAIN_FAST_WALK,
    TRAIN_WALK,
    VAL_CLIPS,
    VAL_FAST_WALK,
    VAL_WALK,
)
from demo_d.env import (
    command_velocity_reward,
    default_config,
    hindsight_command,
    local_planar_velocity,
    physically_alive,
)
from demo_d.evaluate import command_trials
from demo_d.metrics import COMMAND_SECONDS, command_target, command_tracking_score, reportability
from demo_d.provenance import PUBLISHED_CHECKPOINT_FRAGMENT, validate_scratch_metadata
from demo_d.reference import validate_split
from demo_d.waypoint import steer, world_goals, yaw_from_quaternion


def _valid_metadata():
    return {
        "demo_d": {
            "from_scratch": True,
            "parent_checkpoint": None,
            "pipeline_version": PIPELINE_VERSION,
            "reference_sha256": REFERENCE_SHA256,
        }
    }


def test_fixed_split_is_disjoint_and_balanced():
    validate_split()
    assert len(TRAIN_CLIPS) == 48
    assert len(VAL_CLIPS) == 16
    assert set(TRAIN_CLIPS).isdisjoint(VAL_CLIPS)
    assert len(TRAIN_WALK) == len(TRAIN_FAST_WALK) == 24
    assert len(VAL_WALK) == len(VAL_FAST_WALK) == 8


def test_reference_download_layout_resolves_to_the_configured_path():
    assert REFERENCE_DATA.parents[2] / Path(REFERENCE_HF_FILE) == REFERENCE_DATA


def test_hindsight_command_matches_demo_b_geometry():
    yaw_90 = math.pi / 2
    yaw_180 = math.pi
    current = jnp.asarray([0, 0, 0, math.cos(yaw_90 / 2), 0, 0, math.sin(yaw_90 / 2)])
    future = jnp.asarray([0, 1, 0, math.cos(yaw_180 / 2), 0, 0, math.sin(yaw_180 / 2)])
    np.testing.assert_allclose(hindsight_command(current, future), [1, 0, yaw_90], atol=1e-5)


def test_hindsight_planar_command_ignores_root_pitch_and_roll():
    def quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

    current = jnp.asarray([0, 0, 0, *quaternion(0.4, -0.3, math.pi / 2)])
    future = jnp.asarray([0, 1, 0.2, *quaternion(-0.2, 0.25, math.pi)])
    np.testing.assert_allclose(
        hindsight_command(current, future), [1, 0, math.pi / 2], atol=1e-5
    )


def test_measured_velocity_uses_the_same_yaw_only_frame():
    velocity = local_planar_velocity(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.001]),
        jnp.asarray(math.pi / 2),
        0.01,
    )
    np.testing.assert_allclose(velocity, [0.1, 0.0], atol=1e-6)


def test_command_grounding_reward_is_maximal_at_the_hindsight_target():
    command = jnp.asarray([0.062, -0.031, 0.31])
    target = command / (TRAINING.command_horizon_frames / 50.0)
    perfect = command_velocity_reward(target[:2], target[2], command)
    standing = command_velocity_reward(jnp.zeros(2), jnp.asarray(0.0), command)
    assert float(perfect) == pytest.approx(TRAINING.command_reward_weight)
    assert 0.0 <= float(standing) < float(perfect)


def test_command_time_grid_and_tracking_score_are_explicit_and_bounded():
    assert COMMAND_SECONDS == pytest.approx(31 / 50)
    command = np.asarray([0.062, -0.031, 0.31])
    target = command_target(command)
    np.testing.assert_allclose(target, [0.1, -0.05, 0.5])
    perfect = command_tracking_score(target[:2], target[2], command)
    poor = command_tracking_score(np.asarray([2.0, -2.0]), 5.0, command)
    assert perfect == pytest.approx(1.0)
    assert 0.0 <= poor < perfect <= 1.0
    commands = np.asarray(EVAL.commands)
    standing = command_tracking_score(
        np.zeros((len(commands), 2)), np.zeros(len(commands)), commands
    )
    assert float(standing.mean()) < 0.20


def test_observation_design_has_one_compact_future_command():
    config = default_config()
    assert config.reference_length == TRAINING.command_horizon_frames + 1
    assert TRAINING.command_horizon_frames == 31
    assert len(EVAL.commands[0]) == 3


def test_training_uses_physical_falls_but_evaluation_keeps_reference_termination():
    assert default_config(training=True).demo_d_termination == "physical_fall"
    assert default_config().demo_d_termination == "reference"
    qpos = jnp.zeros(59)
    qvel = jnp.zeros(58)
    assert bool(physically_alive(0.1, 1.0, qpos, qvel))
    assert not bool(physically_alive(0.02, 1.0, qpos, qvel))
    assert not bool(physically_alive(0.1, 0.4, qpos, qvel))
    assert not bool(physically_alive(0.1, 1.0, qpos.at[0].set(jnp.nan), qvel))


def test_from_scratch_provenance_rejects_parent_and_published_weights():
    validate_scratch_metadata(_valid_metadata())
    parent = _valid_metadata()
    parent["demo_d"]["parent_checkpoint"] = "/tmp/a_checkpoint"
    with pytest.raises(ValueError, match="parent"):
        validate_scratch_metadata(parent)
    published = _valid_metadata()
    published["accidental_path"] = PUBLISHED_CHECKPOINT_FRAGMENT
    with pytest.raises(ValueError, match="published MIMIC"):
        validate_scratch_metadata(published)


def test_reportability_requires_both_natural_and_commanded_behavior():
    baseline = {
        "imitation_score": 0.10,
        "imitation_survival": 0.0,
        "command_score": 0.05,
        "command_survival": 0.0,
    }
    trained = {
        "imitation_score": 0.10 + EVAL.imitation_gain_min,
        "imitation_survival": EVAL.imitation_survival_min,
        "command_score": EVAL.command_score_min,
        "command_survival": EVAL.command_survival_min,
    }
    assert reportability(baseline, trained)["reportable"]
    trained["command_survival"] -= 0.01
    assert not reportability(baseline, trained)["reportable"]


def test_direct_commands_use_paired_initial_states():
    commands, clips, starts = command_trials()
    n_seeds = len(EVAL.seeds)
    assert commands.shape == (len(EVAL.commands) * n_seeds, 3)
    for condition in range(len(EVAL.commands)):
        rows = slice(condition * n_seeds, (condition + 1) * n_seeds)
        np.testing.assert_array_equal(clips[rows], np.arange(n_seeds))
        np.testing.assert_array_equal(starts[rows], np.asarray(EVAL.seeds) * 5)


def test_waypoint_controller_uses_initial_ego_frame_and_bounded_commands():
    yaw = math.pi / 2
    quaternion = np.asarray([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    assert yaw_from_quaternion(quaternion) == pytest.approx(yaw)
    goals = world_goals(np.asarray([2.0, 3.0]), yaw, ((1.0, 0.0),))
    np.testing.assert_allclose(goals[0], [2.0, 4.0], atol=1e-6)
    command, distance = steer(np.asarray([2.0, 3.0]), yaw, goals[0])
    assert distance == pytest.approx(1.0)
    assert 0.0 < command[0] <= 0.08
    assert command[1] == 0.0
    assert abs(command[2]) <= 0.30
