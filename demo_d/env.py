"""A one-stage hindsight-command imitation environment.

The simulator and imitation reward are inherited from VNL's RodentImitation.
Only the policy observation changes: instead of seeing future joint/body targets,
it sees the three-number command used by Demo B plus current proprioception.
"""

from __future__ import annotations

import collections
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from vnl_playground.tasks.rodent import imitation

from demo_d.config import COMMAND_ERROR_SCALES, TRAINING


COMMAND_SIZE = 3


def quaternion_yaw(quaternion):
    """Demo B's MuJoCo-wxyz yaw formula, expressed in JAX."""
    w, x, y, z = quaternion
    return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def local_planar_velocity(previous_xy, current_xy, previous_yaw, dt):
    """Measure root velocity in Demo B's yaw-only egocentric frame."""
    delta_world = (current_xy - previous_xy) / dt
    cosine, sine = jp.cos(-previous_yaw), jp.sin(-previous_yaw)
    return jp.stack(
        (
            cosine * delta_world[..., 0] - sine * delta_world[..., 1],
            sine * delta_world[..., 0] + cosine * delta_world[..., 1],
        ),
        axis=-1,
    )


def command_velocity_reward(local_velocity, yaw_rate, command):
    """Reward physical velocity that realizes a hindsight displacement command."""
    target = command / (TRAINING.command_horizon_frames / 50.0)
    actual = jp.array([local_velocity[0], local_velocity[1], yaw_rate])
    scaled_error = (actual - target) / jp.asarray(COMMAND_ERROR_SCALES)
    return TRAINING.command_reward_weight / (1.0 + jp.sum(scaled_error**2))


def hindsight_command(current_qpos, future_qpos):
    """Convert two recorded root poses to Demo B's egocentric command.

    Both poses use MuJoCo's ``[x, y, z, qw, qx, qy, qz, ...]`` layout.  The
    result is ``[forward displacement, lateral displacement, yaw change]``.
    """
    current_yaw = quaternion_yaw(current_qpos[3:7])
    future_yaw = quaternion_yaw(future_qpos[3:7])
    delta_world = future_qpos[:2] - current_qpos[:2]
    cosine, sine = jp.cos(-current_yaw), jp.sin(-current_yaw)
    forward = cosine * delta_world[0] - sine * delta_world[1]
    lateral = sine * delta_world[0] + cosine * delta_world[1]
    delta_yaw = (future_yaw - current_yaw + jp.pi) % (2 * jp.pi) - jp.pi
    return jp.array([forward, lateral, delta_yaw])


def default_config(*, training: bool = False) -> config_dict.ConfigDict:
    cfg = imitation.default_config()
    # reference_length also makes the inherited episode boundary leave enough
    # future frames to construct the command.
    cfg.reference_length = TRAINING.command_horizon_frames + 1
    cfg.reference_stride = 1
    cfg.start_frame_range = [0, TRAINING.start_frame_max]
    cfg.qvel_init = "reference"
    cfg.domain_randomization = config_dict.create(
        use_domain_randomization=False,
        floor_friction=[1.0, 1.0],
        static_friction_scale=[1.0, 1.0],
        armature_scale=[1.0, 1.0],
        com_jitter=[0.0, 0.0],
        link_mass_scale=[1.0, 1.0],
        torso_mass_jitter=[0.0, 0.0],
        qpos0_jitter=[0.0, 0.0],
    )
    # Four visible tracking rewards + physical regularizers.  Velocity/body
    # duplicates are disabled as in the maintained track-mjx feedforward config.
    cfg.reward_terms.joints_vel.weight = 0.0
    cfg.reward_terms.bodies_pos.weight = 0.0
    cfg.demo_d_termination = TRAINING.termination_mode if training else "reference"
    return cfg


def physically_alive(torso_z, torso_up, qpos, qvel):
    """The standard rodent locomotion fall criterion, kept reference-free."""
    finite = jp.all(jp.isfinite(qpos)) & jp.all(jp.isfinite(qvel))
    return (torso_z > 0.03) & (torso_up > 0.5) & finite


class HindsightCommandImitation(imitation.Imitation):
    """Rodent imitation where future mocap supplies a compact command."""

    def reset(self, rng, clip_idx=None, start_frame=None):
        state = super().reset(rng, clip_idx=clip_idx, start_frame=start_frame)
        info = state.info.copy()
        info["previous_root_xy"] = state.data.qpos[:2]
        info["previous_root_yaw"] = quaternion_yaw(state.data.qpos[3:7])
        info["hindsight_command"] = state.obs["state"][:COMMAND_SIZE]
        info["command_velocity_ema"] = info["hindsight_command"] / (
            TRAINING.command_horizon_frames / 50.0
        )
        return state.replace(info=info)

    def _hindsight_command(self, data, info):
        current_frame = self._get_cur_frame(data, info)
        future_frame = current_frame + TRAINING.command_horizon_frames
        current = self.reference_clips.at(info["reference_clip"], current_frame)
        future = self.reference_clips.at(info["reference_clip"], future_frame)

        return hindsight_command(current.qpos, future.qpos)

    def _command_observation(self, data, info, command):
        proprioception = self._get_proprioception(data, info, flatten=True)
        return collections.OrderedDict(
            state=jp.nan_to_num(jp.concatenate([command, proprioception]))
        )

    def _get_obs(self, data, info):
        # Keep a dict top level so Brax can explicitly route the "state" input.
        return self._command_observation(data, info, self._hindsight_command(data, info))

    def _get_reward(self, data, info, metrics):
        """Combine mocap imitation with one hindsight-command grounding term."""
        imitation_reward = super()._get_reward(data, info, metrics)
        if "previous_root_xy" in info:
            yaw = quaternion_yaw(data.qpos[3:7])
            local_velocity = local_planar_velocity(
                info["previous_root_xy"],
                data.qpos[:2],
                info["previous_root_yaw"],
                float(self._config.ctrl_dt),
            )
            delta_yaw = (yaw - info["previous_root_yaw"] + jp.pi) % (2 * jp.pi) - jp.pi
            yaw_rate = delta_yaw / float(self._config.ctrl_dt)
            command = info["hindsight_command"]
            instantaneous = jp.array([local_velocity[0], local_velocity[1], yaw_rate])
            smoothing = float(self._config.ctrl_dt) / (
                TRAINING.command_horizon_frames / 50.0
            )
            smoothed = info["command_velocity_ema"] + smoothing * (
                instantaneous - info["command_velocity_ema"]
            )
            grounding_reward = command_velocity_reward(smoothed[:2], smoothed[2], command)
            command_error = jp.linalg.norm(
                smoothed - command / (TRAINING.command_horizon_frames / 50.0)
            )
            info["previous_root_xy"] = data.qpos[:2]
            info["previous_root_yaw"] = yaw
            info["hindsight_command"] = self._hindsight_command(data, info)
            info["command_velocity_ema"] = smoothed
        else:
            # ``super().reset`` asks for a reward before the history fields can
            # be attached. Reset rewards are not PPO transitions.
            grounding_reward = jp.array(0.0)
            command_error = jp.array(0.0)
        metrics["rewards/command_tracking"] = grounding_reward
        metrics["command_velocity_error"] = command_error
        return imitation_reward + grounding_reward

    def _is_done(self, data, info, metrics):
        """Allow recovery from reference drift only in the training environment."""
        if self._config.demo_d_termination == "reference":
            return super()._is_done(data, info, metrics)

        torso = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_z = torso.xpos[2]
        torso_up = torso.xmat.reshape(3, 3)[2, 2]
        finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
        fallen = (torso_z <= 0.03) | (torso_up <= 0.5)
        terminated = fallen | ~finite

        # Preserve the upstream metric tree so Brax train/eval wrappers see the
        # same leaves even though their termination semantics differ.
        metrics["terminations/root_too_far"] = jp.array(0.0)
        metrics["terminations/root_too_rotated"] = jp.array(0.0)
        metrics["terminations/pose_error"] = jp.array(0.0)
        metrics["terminations/nan_termination"] = (~finite).astype(float)
        metrics["terminations/any"] = terminated.astype(float)
        return terminated

    def command_step(self, state, action, command):
        """Advance physics using only an external command and proprioception.

        Training uses :meth:`step`, whose hidden reward compares against mocap.
        Deployment uses this method after a reference pose initializes the body;
        no future frame, imitation target, or reward is read here.
        """
        command = jp.asarray(command)
        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)
        data = mjx_env.step(self.mjx_model, state.data, action, n_steps)
        info = state.info.copy()
        info["prev_action"] = state.info["action"]
        info["action"] = action
        obs = self._command_observation(data, info, command)

        torso = data.bind(self.mjx_model, self._spec.body(f"torso{self._suffix}"))
        torso_z = torso.xpos[2]
        torso_up = torso.xmat.reshape(3, 3)[2, 2]
        alive = physically_alive(torso_z, torso_up, data.qpos, data.qvel)
        done = jp.logical_or(state.done > 0.5, jp.logical_not(alive))
        return state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=jp.zeros_like(state.reward),
            done=done.astype(float),
        )

    def render_commands(self, trajectory, height=480, width=640, camera=None):
        """Render a command rollout without indexing the reference dataset."""
        data = mujoco.MjData(self.mj_model)
        renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        camera = camera or self._default_render_camera
        frames = []
        try:
            for state in trajectory:
                data.qpos[:] = np.asarray(state.data.qpos)
                data.qvel[:] = np.asarray(state.data.qvel)
                mujoco.mj_forward(self.mj_model, data)
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
        finally:
            renderer.close()
        return frames


def replace_command(state, command):
    """Replace the first three observation values for direct deployment."""
    command = jp.asarray(command)
    if command.shape != (COMMAND_SIZE,):
        raise ValueError(f"command must have shape ({COMMAND_SIZE},), got {command.shape}")
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:COMMAND_SIZE].set(command)
    return state.replace(obs=obs)
