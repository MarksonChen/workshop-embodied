"""TRACK-MJX-aligned RodentJoystick environment for Demo E.

Both experimental arms use the published, frozen 16-D intention decoder as a
shared low-level motor system.  PPO trains only the high-level policy.  E1 adds
Demo B's frozen conditional motion likelihood; E0 sets its coefficient to
zero while retaining the same body, observations, resets, and score path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as playground_wrapper
from vnl_playground.tasks import wrappers as vnl_wrappers
from vnl_playground.tasks.rodent import joystick

from track_mjx.agent import checkpointing
from track_mjx.agent.ff_ppo import ppo_networks as ff_ppo_networks

from .config import ENV, MIMIC_CHECKPOINT
from .features import (
    full_motion_feature,
    joystick_to_hindsight_command,
    keypoint_local,
)
from .prior import FrozenMotionPrior, load_prior


# Demo B conditions on displacement over frames 32 -> 63 at 50 Hz.
COMMAND_HORIZON_SECONDS = 31.0 / 50.0


def normalize_logp(raw_logp):
    """Frozen monotone conversion from physical log likelihood to reward units."""
    return jp.clip(
        (raw_logp - ENV.prior_logp_floor)
        / (ENV.prior_logp_ceiling - ENV.prior_logp_floor),
        0.0,
        1.0,
    )


def default_config() -> config_dict.ConfigDict:
    """The reproduced upstream joystick settings, with explicit constants."""
    config = joystick.default_config()
    config.ctrl_dt = ENV.control_dt
    config.sim_dt = ENV.sim_dt
    config.torque_actuators = True
    config.rescale_factor = 0.9
    config.iterations = 5
    config.ls_iterations = 5
    config.episode_length = ENV.episode_length
    config.lin_vel_x = ENV.forward_range
    config.ang_vel_yaw = ENV.yaw_range
    config.command_config.resample_interval = ENV.command_resample_steps
    config.command_config.zero_prob = ENV.zero_command_probability
    config.reward_config.tracking_sigma = ENV.tracking_sigma
    return config


class LikelihoodJoystick(joystick.Joystick):
    """100-Hz torque task with the native reset and a 12.5-Hz score."""

    def __init__(
        self,
        *,
        beta: float,
        score_motion: bool,
        prior: FrozenMotionPrior | None = None,
        config: config_dict.ConfigDict | None = None,
    ):
        self.beta = float(beta)
        self.score_motion = bool(score_motion)
        self.prior = (prior or load_prior()) if self.score_motion else None
        super().__init__(config=config or default_config())
        if self.action_size != 38 or self.mjx_model.nq != 74:
            raise ValueError(
                "the frozen decoder requires the 74-qpos, 38-actuator rodent"
            )

        if not self.score_motion:
            return

        body_names = self.prior.metadata["keypoint_bodies"]
        body_ids = [
            mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{name}{self._suffix}"
            )
            for name in body_names
        ]
        if any(index < 0 for index in body_ids):
            raise ValueError("joystick model is missing a calibrated keypoint body")
        self._keypoint_body_ids = jp.asarray(body_ids, jp.int32)
        self._keypoint_offsets = self.prior.norm["keypoint_offsets"]

        # Infer the fixed relation between the attached walker's free-joint
        # coordinates and world coordinates.  It is zero for the current XML,
        # but making it explicit guards a future arena/spawn change.
        cpu_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, cpu_data)
        root_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"walker{self._suffix}"
        )
        self._root_spawn_offset = jp.asarray(
            cpu_data.xpos[root_id] - cpu_data.qpos[:3], jp.float32
        )

    def _world_qpos(self, data: mjx.Data):
        return data.qpos.at[:3].add(self._root_spawn_offset)

    def _local_keypoints(self, data: mjx.Data, world_qpos):
        return keypoint_local(
            data,
            self._keypoint_body_ids,
            self._keypoint_offsets,
            world_qpos[:3],
            world_qpos[3:7],
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Keep the physical task reset byte-for-byte on the native Joystick
        # path.  In particular, the published trainer starts from mjx.make_data
        # without forwarding it first.  Forward only a private copy used to
        # initialize Demo B's kinematic history; never substitute that copy
        # into the task state.
        state = super().reset(rng)
        info = dict(state.info)
        metrics: dict[str, Any] = dict(state.metrics)
        if self.score_motion:
            feature_data = mjx.forward(self.mjx_model, state.data)
            world_qpos = self._world_qpos(feature_data)
            keypoints = self._local_keypoints(feature_data, world_qpos)
            stationary = full_motion_feature(
                world_qpos, world_qpos, keypoints, keypoints
            )
            normalized = self.prior.normalize_features(stationary)
            feature_buffer = jp.repeat(normalized[None], 24, axis=0)
            initial_latent = self.prior.encode_last(feature_buffer)
            initial_history = jp.repeat(initial_latent[None], 8, axis=0)
            prior_command = joystick_to_hindsight_command(
                info["command"], COMMAND_HORIZON_SECONDS
            )
            info.update(
                {
                    "previous_feature_qpos": world_qpos,
                    "previous_keypoints": keypoints,
                    "feature_buffer": feature_buffer,
                    "latent_history": initial_history,
                    "predicted_future": self.prior.predict(
                        initial_history, self.prior.normalize_command(prior_command)
                    ),
                    "realized_chunk": jp.zeros((8, 16), jp.float32),
                    "last_prior_realized": initial_latent,
                    # Do not reward the synthetic stationary context. Eight
                    # realized tokens fully replace it before E1 is scored.
                    "prior_tokens": jp.asarray(0, jp.int32),
                }
            )
        metrics.update(
            {
                "r_task": state.reward,
                "r_prior_raw": jp.asarray(0.0),
                "r_prior_score": jp.asarray(0.0),
                "r_prior_weighted": jp.asarray(0.0),
                "prior_update": jp.asarray(0.0),
            }
        )
        return state.replace(info=info, metrics=metrics)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        next_state = super().step(state, action)
        info = dict(next_state.info)
        metrics = dict(next_state.metrics)
        if not self.score_motion:
            metrics.update(
                {
                    "r_task": next_state.reward,
                    "r_prior_raw": jp.asarray(0.0),
                    "r_prior_score": jp.asarray(0.0),
                    "r_prior_weighted": jp.asarray(0.0),
                    "prior_update": jp.asarray(0.0),
                }
            )
            return next_state.replace(info=info, metrics=metrics)

        sample_feature = (info["step"] % ENV.controls_per_feature) == 0
        update_prior = (info["step"] % ENV.controls_per_prior) == 0

        world_qpos = self._world_qpos(next_state.data)
        keypoints = self._local_keypoints(next_state.data, world_qpos)
        feature = full_motion_feature(
            info["previous_feature_qpos"],
            world_qpos,
            info["previous_keypoints"],
            keypoints,
        )
        normalized = self.prior.normalize_features(feature)
        candidate_buffer = jp.concatenate(
            [info["feature_buffer"][1:], normalized[None]], axis=0
        )
        info["feature_buffer"] = jp.where(
            sample_feature, candidate_buffer, info["feature_buffer"]
        )
        info["previous_feature_qpos"] = jp.where(
            sample_feature, world_qpos, info["previous_feature_qpos"]
        )
        info["previous_keypoints"] = jp.where(
            sample_feature, keypoints, info["previous_keypoints"]
        )

        realized = self.prior.encode_last(info["feature_buffer"])
        command = joystick_to_hindsight_command(
            info["command"], COMMAND_HORIZON_SECONDS
        )
        token_index = info["prior_tokens"] % 8
        predicted = info["predicted_future"][token_index]
        raw_logp = self.prior.log_prob(realized, predicted)
        score = normalize_logp(raw_logp)
        candidate_chunk = info["realized_chunk"].at[token_index].set(realized)
        end_block = update_prior & (token_index == 7)
        info["latent_history"] = jp.where(
            end_block, candidate_chunk, info["latent_history"]
        )
        info["realized_chunk"] = jp.where(
            update_prior, candidate_chunk, info["realized_chunk"]
        )
        next_prediction = self.prior.predict(
            candidate_chunk, self.prior.normalize_command(command)
        )
        info["predicted_future"] = jp.where(
            end_block, next_prediction, info["predicted_future"]
        )
        info["last_prior_realized"] = jp.where(
            update_prior, realized, info["last_prior_realized"]
        )
        next_tokens = info["prior_tokens"] + update_prior.astype(jp.int32)
        info["prior_tokens"] = next_tokens
        warmed_up = next_tokens > 8
        prior_reward = jp.where(
            update_prior & warmed_up, self.beta * score, 0.0
        )
        metrics.update(
            {
                "r_task": next_state.reward,
                "r_prior_raw": jp.where(
                    update_prior & warmed_up, raw_logp, 0.0
                ),
                "r_prior_score": jp.where(
                    update_prior & warmed_up, score, 0.0
                ),
                "r_prior_weighted": prior_reward,
                "prior_update": update_prior.astype(jp.float32),
            }
        )
        return next_state.replace(
            info=info,
            metrics=metrics,
            reward=jp.nan_to_num(next_state.reward + prior_reward),
        )


@lru_cache(maxsize=1)
def load_decoder():
    """Load the frozen imitation decoder and its training configuration."""
    # The pinned TRACK-MJX checkpoint loader predates typed JAX keys and
    # branches on ``key.ndim``. Keep this compatibility shim at the boundary so
    # every caller (including tests and renderers) gets the same behavior.
    jax.random.key = jax.random.PRNGKey
    checkpoint = str(MIMIC_CHECKPOINT.resolve())
    config = checkpointing.load_config_from_checkpoint(checkpoint)
    decoder = ff_ppo_networks.make_decoder_policy_fn(checkpoint)
    return config, decoder


def build_env(*, beta: float, score_motion: bool):
    """Compose joystick + Demo B score + the shared frozen motor decoder."""
    mimic_config, decoder = load_decoder()
    config = default_config()
    config.ctrl_dt = float(mimic_config["env_config"]["ctrl_dt"])
    low_level = LikelihoodJoystick(
        beta=beta, score_motion=score_motion, config=config
    )
    return vnl_wrappers.HighLevelWrapper(
        low_level,
        decoder_inference_fn=decoder,
        latent_size=int(mimic_config["network_config"]["intention_size"]),
        policy_obs_key="state",
        value_obs_key="state",
        highlvl_obs_key="task_obs",
        lowlvl_obs_key="proprioception",
    )


def replace_command(state: mjx_env.State, command: jax.Array) -> mjx_env.State:
    """Pin every command copy used by reward, actor, and frozen decoder."""
    command = jp.asarray(command)
    info = dict(state.info)
    info["command"] = command
    if "_full_obs" in info:
        full_obs = dict(info["_full_obs"])
        full_state = dict(full_obs["state"])
        full_state["task_obs"] = full_state["task_obs"].at[-2:].set(command)
        full_obs["state"] = full_state
        info["_full_obs"] = full_obs
    obs = state.obs
    if isinstance(obs, dict) and "state" in obs:
        obs = dict(obs)
        if isinstance(obs["state"], dict):
            nested = dict(obs["state"])
            nested["task_obs"] = nested["task_obs"].at[-2:].set(command)
            obs["state"] = nested
        else:
            obs["state"] = obs["state"].at[-2:].set(command)
    return state.replace(info=info, obs=obs)


_PRIOR_RESET_FIELDS = (
    "previous_feature_qpos",
    "previous_keypoints",
    "feature_buffer",
    "latent_history",
    "predicted_future",
    "realized_chunk",
    "last_prior_realized",
    "prior_tokens",
)


class CausalAutoResetWrapper(playground_wrapper.BraxAutoResetWrapper):
    """Reset only reward-side history in addition to upstream autoreset.

    The native wrapper deliberately resets cached data/observations but leaves
    task info untouched.  Preserving that behavior is required for an exact E0
    reproduction; E1 additionally needs its causal prior buffers cleared when
    an episode ends.
    """

    _causal_key = "DemoE_first_info"

    def reset(self, rng):
        state = super().reset(rng)
        state.info[self._causal_key] = {
            key: state.info[key]
            for key in _PRIOR_RESET_FIELDS
            if key in state.info
        }
        return state

    def step(self, state, action):
        state = super().step(state, action)

        def where_done(first, current):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [done.shape[0]] + [1] * (current.ndim - 1))
            return jp.where(done, first, current)

        for key, first in state.info[self._causal_key].items():
            state.info[key] = jax.tree.map(where_done, first, state.info[key])
        return state


def wrap_for_training(env, episode_length=ENV.episode_length, action_repeat=1, **_):
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return CausalAutoResetWrapper(env, full_reset=False)
