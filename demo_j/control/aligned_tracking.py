"""Long-horizon functional locomotion task for the aligned Demo J SNN.

The environment supplies rolling motion tokens from the independently fitted
Demo F reference, but its reward is only Demo A-style functional locomotion:
forward-speed tracking, uprightness, and a small actuator cost.  Gait/contact
diagnostics remain validation-only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from brax import math
from brax.envs.base import PipelineEnv, State

from demo_f.jax_features import transition_feature
from demo_f.kinematics import fetch_feet
from demo_j.control.aligned import CYCLE_FRAMES, PeriodicSequences
from demo_j.control.config import ACTION_DIM, FEATURE_DIM
from demo_j.data.dataset import ReferenceSet
from demo_j.data.physics import (
    foot_site_indices,
    joint_angles,
    system,
)


class AlignedLocomotion(PipelineEnv):
    """Demo A's task with a rolling, data-derived motion-token input stream."""

    def __init__(
        self,
        references: ReferenceSet,
        sequences: PeriodicSequences,
        valid_clips: jax.Array,
        *,
        upright_weight: float = 0.1,
        control_weight: float = 1e-3,
    ):
        super().__init__(sys=system(), backend="mjx", n_frames=4)
        self._qpos = jnp.asarray(references.qpos)
        self._qvel = jnp.asarray(references.qvel)
        self._cycle_start = jnp.asarray(sequences.cycle_start)
        self._template = jnp.asarray(sequences.observation)
        self._target_speed = jnp.asarray(sequences.speed)
        self._valid_clips = jnp.asarray(valid_clips, jnp.int32)
        self._foot_sites = jnp.asarray(foot_site_indices())
        self._upright_weight = float(upright_weight)
        self._control_weight = float(control_weight)
        if self._template.shape[0] != references.clips:
            raise ValueError((self._template.shape, references.clips))
        if not len(valid_clips):
            raise ValueError("the locomotion task needs at least one valid clip")

    @property
    def observation_size(self) -> int:
        return int(self._template.shape[-1])

    @property
    def action_size(self) -> int:
        return ACTION_DIM

    def _feature(self, previous_qpos, pipeline_state):
        qpos = pipeline_state.qpos
        previous_angles = joint_angles(previous_qpos)
        angles = joint_angles(qpos)
        previous_feet = fetch_feet(previous_angles)
        feet = fetch_feet(angles)
        contacts = pipeline_state.site_xpos[self._foot_sites, 2] <= 0.025
        return transition_feature(
            previous_qpos[:3],
            qpos[:3],
            previous_qpos[3:7],
            qpos[3:7],
            previous_angles,
            angles,
            previous_feet,
            feet,
            contacts,
        )

    def _observation(self, feature, previous_action, clip, time_index):
        observation = self._template[clip, time_index % CYCLE_FRAMES]
        observation = observation.at[:FEATURE_DIM].set(feature)
        return observation.at[FEATURE_DIM : FEATURE_DIM + ACTION_DIM].set(
            previous_action
        )

    def reset(self, rng: jax.Array) -> State:
        choice = jax.random.randint(rng, (), 0, len(self._valid_clips))
        return self.reset_to(self._valid_clips[choice])

    def reset_to(self, clip: jax.Array) -> State:
        clip = jnp.asarray(clip, jnp.int32)
        start = self._cycle_start[clip]
        pipeline_state = self.pipeline_init(
            self._qpos[clip, start], self._qvel[clip, start]
        )
        feature = self._feature(pipeline_state.qpos, pipeline_state)
        previous_action = jnp.zeros((ACTION_DIM,), jnp.float32)
        quaternion = pipeline_state.qpos[3:7]
        heading = math.rotate(jnp.asarray((1.0, 0.0, 0.0)), quaternion)
        heading = heading.at[2].set(0.0)
        heading = heading / jnp.maximum(jnp.linalg.norm(heading), 1e-6)
        target_speed = jnp.maximum(self._target_speed[clip], 0.1)
        zero = jnp.zeros(())
        metrics = {
            "speed": zero,
            "track": zero,
            "upright": zero,
            "control_cost": zero,
            "torso_height": pipeline_state.qpos[2],
            "target_speed": target_speed,
        }
        info = {
            "clip": clip,
            "time_index": jnp.int32(0),
            "previous_qpos": pipeline_state.qpos,
            "previous_action": previous_action,
            "feature": feature,
            "heading": heading,
            "target_speed": target_speed,
        }
        return State(
            pipeline_state,
            self._observation(feature, previous_action, clip, jnp.int32(0)),
            zero,
            zero,
            metrics,
            info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        displacement = pipeline_state.qpos[:3] - state.pipeline_state.qpos[:3]
        speed = jnp.dot(displacement / self.dt, state.info["heading"])
        target_speed = state.info["target_speed"]
        sigma = jnp.maximum(target_speed / 3.0, 0.15)
        track = jnp.exp(-jnp.square(speed - target_speed) / (2.0 * sigma * sigma))
        upright = 1.0 - 2.0 * (
            jnp.square(pipeline_state.qpos[4]) + jnp.square(pipeline_state.qpos[5])
        )
        control_cost = self._control_weight * jnp.sum(jnp.square(action))
        reward = track + self._upright_weight * upright - control_cost
        finite = jnp.all(jnp.isfinite(pipeline_state.qpos)) & jnp.all(
            jnp.isfinite(pipeline_state.qvel)
        )
        torso_height = pipeline_state.qpos[2]
        done = (torso_height < 0.55) | (upright < 0.0) | ~finite
        feature = self._feature(state.info["previous_qpos"], pipeline_state)
        time_index = state.info["time_index"] + 1
        info = dict(state.info)
        info.update(
            time_index=time_index,
            previous_qpos=pipeline_state.qpos,
            previous_action=action,
            feature=feature,
        )
        metrics = dict(state.metrics)
        metrics.update(
            speed=speed,
            track=track,
            upright=upright,
            control_cost=control_cost,
            torso_height=torso_height,
            target_speed=target_speed,
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=self._observation(feature, action, state.info["clip"], time_index),
            reward=jnp.nan_to_num(reward),
            done=done.astype(jnp.float32),
            metrics=metrics,
            info=info,
        )
