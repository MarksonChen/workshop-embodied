"""Modern-MJX Fetch reference-tracking environment for Demo J."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from brax import math
from brax.envs.base import PipelineEnv, State

from demo_f.jax_features import transition_feature
from demo_f.kinematics import fetch_feet
from demo_j.data.dataset import ReferenceSet
from demo_j.data.physics import (
    foot_site_indices,
    joint_angles,
    joint_velocities,
    system,
)


FEATURE_DIM = 60
ACTION_DIM = 10
REFERENCE_FRAMES = 5
REFERENCE_FRAME_DIM = 27  # root xyz, relative quaternion, joint/qvel errors
REFERENCE_DIM = REFERENCE_FRAMES * REFERENCE_FRAME_DIM
OBS_DIM = FEATURE_DIM + ACTION_DIM + REFERENCE_DIM
LAST_START_FRAME = 32
LAST_TRACK_FRAME = 58


class FetchTracking(PipelineEnv):
    """Score closed-loop tracking of feasible Demo F-derived state sequences."""

    def __init__(
        self,
        references: ReferenceSet,
        *,
        random_start: bool = True,
        reset_noise_scale: float = 0.0,
        track_frames: int = LAST_TRACK_FRAME,
    ):
        super().__init__(sys=system(), backend="mjx", n_frames=4)
        # Keep provenance and session names on the host.  Only the arrays used
        # inside reset/step belong in traced device computations.
        self._qpos = jnp.asarray(references.qpos)
        self._qvel = jnp.asarray(references.qvel)
        self._features = jnp.asarray(references.features)
        self._clips = references.clips
        self._frames = references.frames
        self._random_start = bool(random_start)
        self._reset_noise_scale = float(reset_noise_scale)
        self._last_track_frame = int(track_frames)
        if not 1 <= self._last_track_frame < self._frames:
            raise ValueError(
                f"track_frames must lie in [1, {self._frames - 1}], "
                f"got {self._last_track_frame}"
            )
        self._foot_sites = jnp.asarray(foot_site_indices())

    @property
    def observation_size(self) -> int:
        return OBS_DIM

    @property
    def action_size(self) -> int:
        return ACTION_DIM

    def _reference_observation(self, qpos, qvel, clip, frame):
        indices = jnp.minimum(
            frame + 1 + jnp.arange(REFERENCE_FRAMES), self._frames - 1
        )
        target_qpos = self._qpos[clip, indices]
        target_qvel = self._qvel[clip, indices]
        root = qpos[:3]
        quaternion = qpos[3:7]
        root_target = jax.vmap(lambda value: math.inv_rotate(value - root, quaternion))(
            target_qpos[:, :3]
        )
        quat_target = jax.vmap(lambda value: math.relative_quat(value, quaternion))(
            target_qpos[:, 3:7]
        )
        angle_target = jax.vmap(joint_angles)(target_qpos) - joint_angles(qpos)
        velocity_target = jax.vmap(joint_velocities)(target_qvel) - joint_velocities(
            qvel
        )
        return jnp.concatenate(
            (root_target, quat_target, angle_target, velocity_target), axis=-1
        ).reshape(-1)

    def _observation(self, state, feature, previous_action, clip, frame):
        reference = self._reference_observation(state.qpos, state.qvel, clip, frame)
        observation = jnp.concatenate((feature, previous_action, reference))
        if observation.shape != (OBS_DIM,):
            raise ValueError(observation.shape)
        return observation

    def reset(self, rng: jax.Array) -> State:
        clip_rng, start_rng, noise_rng = jax.random.split(rng, 3)
        clip = jax.random.randint(clip_rng, (), 0, self._clips)
        start = jnp.where(
            self._random_start,
            jax.random.randint(start_rng, (), 0, LAST_START_FRAME + 1),
            jnp.int32(0),
        )
        qvel = self._qvel[clip, start]
        if self._reset_noise_scale:
            qvel = qvel + self._reset_noise_scale * jax.random.normal(
                noise_rng, qvel.shape
            )
        return self._reset_to(clip, start, qvel)

    def reset_to(self, clip: jax.Array, start: jax.Array = 0) -> State:
        """Deterministically initialize a requested clip for evaluation."""

        clip = jnp.asarray(clip, jnp.int32)
        start = jnp.asarray(start, jnp.int32)
        return self._reset_to(clip, start, self._qvel[clip, start])

    def _reset_to(self, clip, start, qvel) -> State:
        qpos = self._qpos[clip, start]
        pipeline_state = self.pipeline_init(qpos, qvel)
        previous_action = jnp.zeros((ACTION_DIM,), jnp.float32)
        feature = self._features[clip, start]
        obs = self._observation(pipeline_state, feature, previous_action, clip, start)
        zero = jnp.zeros(())
        metrics = {
            "tracking_reward": zero,
            "root_error": zero,
            "root_angle_error_deg": zero,
            "joint_error": zero,
            "joint_velocity_error": zero,
            "foot_error": zero,
            "control_cost": zero,
            "control_difference_cost": zero,
            "current_frame": start.astype(jnp.float32),
            "completed": zero,
        }
        info = {
            "clip": clip,
            "frame": start,
            "previous_qpos": qpos,
            "previous_action": previous_action,
            "feature": feature,
        }
        return State(pipeline_state, obs, zero, zero, metrics, info)

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

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        clip = state.info["clip"]
        frame = state.info["frame"] + 1
        target_qpos = self._qpos[clip, frame]
        target_qvel = self._qvel[clip, frame]

        root_error = jnp.linalg.norm(pipeline_state.qpos[:3] - target_qpos[:3])
        dot = jnp.dot(pipeline_state.qpos[3:7], target_qpos[3:7])
        root_angle = jnp.rad2deg(jnp.arccos(jnp.clip(2.0 * dot * dot - 1.0, -1.0, 1.0)))
        angle_error = jnp.linalg.norm(
            joint_angles(pipeline_state.qpos) - joint_angles(target_qpos)
        )
        velocity_error = jnp.linalg.norm(
            joint_velocities(pipeline_state.qvel) - joint_velocities(target_qvel)
        )
        feet = fetch_feet(joint_angles(pipeline_state.qpos))
        target_feet = fetch_feet(joint_angles(target_qpos))
        foot_error = jnp.linalg.norm(feet - target_feet)

        root_reward = jnp.exp(-0.5 * jnp.square(root_error / 0.20))
        quaternion_reward = jnp.exp(-0.5 * jnp.square(root_angle / 40.0))
        joint_reward = jnp.exp(-0.5 * jnp.square(angle_error / 1.40))
        velocity_reward = jnp.exp(-0.5 * jnp.square(velocity_error / 8.0))
        foot_reward = jnp.exp(-0.5 * jnp.square(foot_error / 0.30))
        control_cost = 0.002 * jnp.sum(jnp.square(action))
        difference_cost = 0.005 * jnp.sum(
            jnp.square(action - state.info["previous_action"])
        )
        tracking_reward = (
            root_reward
            + quaternion_reward
            + joint_reward
            + velocity_reward
            + foot_reward
        )
        reward = tracking_reward - control_cost - difference_cost

        finite = jnp.all(jnp.isfinite(pipeline_state.qpos)) & jnp.all(
            jnp.isfinite(pipeline_state.qvel)
        )
        completed = frame >= self._last_track_frame
        failed = (
            (root_error > 1.0)
            | (root_angle > 120.0)
            | (angle_error > 4.5)
            | (pipeline_state.qpos[2] < 0.5)
            | ~finite
        )
        done = completed | failed
        reward = jnp.nan_to_num(reward)
        feature = self._feature(state.info["previous_qpos"], pipeline_state)
        obs = self._observation(pipeline_state, feature, action, clip, frame)
        metrics = dict(state.metrics)
        metrics.update(
            {
                "tracking_reward": tracking_reward,
                "root_error": root_error,
                "root_angle_error_deg": root_angle,
                "joint_error": angle_error,
                "joint_velocity_error": velocity_error,
                "foot_error": foot_error,
                "control_cost": control_cost,
                "control_difference_cost": difference_cost,
                "current_frame": frame.astype(jnp.float32),
                "completed": completed.astype(jnp.float32),
            }
        )
        # Episode/Vmap/AutoReset wrappers append bookkeeping fields to these
        # dictionaries.  Preserve them so scan carries keep a fixed pytree.
        info = dict(state.info)
        info.update(
            {
                "clip": clip,
                "frame": frame,
                "previous_qpos": pipeline_state.qpos,
                "previous_action": action,
                "feature": feature,
            }
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done.astype(jnp.float32),
            metrics=metrics,
            info=info,
        )
