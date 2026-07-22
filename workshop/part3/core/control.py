from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen

from brax.envs.base import Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.training import distribution, networks
from brax.training.agents.ppo import networks as ppo_networks
from brax.v1 import jumpy as jp

from workshop.part1.environment import FetchRun, RawKeyVmapWrapper
from workshop.part2.core.motion import fetch_feet, transition_feature
from workshop.part3.config import (
    ACTION_DIM,
    BASE_OBS_DIM,
    BUFFER_FRAMES,
    COMMAND_DIM,
    COMMAND_HORIZON_SECONDS,
    COMMAND_SLICE,
    FEATURE_BUFFER_SLICE,
    FEATURE_DIM,
    FETCH_FOOT_NAMES,
    HISTORY_TOKENS,
    OBS_DIM,
    PHASE_DIM,
    PHASE_SLICE,
    PLAN_DIM,
    PLAN_SLICE,
    PREVIOUS_CONTROL_SLICE,
    TARGET_SPEED_FETCH,
)


class PriorFetchRun(FetchRun):
    def __init__(
        self, command=None, task_speed_min=None, task_speed_max=None, **kwargs
    ):
        kwargs.setdefault("target_speed", TARGET_SPEED_FETCH)
        kwargs.setdefault("speed_width", TARGET_SPEED_FETCH / 3.0)
        target_speed = float(kwargs["target_speed"])
        super().__init__(**kwargs)
        if (task_speed_min is None) != (task_speed_max is None):
            raise ValueError("task speed range needs both endpoints")
        self.task_speed_min = (
            target_speed if task_speed_min is None else float(task_speed_min)
        )
        self.task_speed_max = (
            target_speed if task_speed_max is None else float(task_speed_max)
        )
        if not 0 < self.task_speed_min <= self.task_speed_max:
            raise ValueError((self.task_speed_min, self.task_speed_max))
        self.command_override = (
            None if command is None else jp.array(command, dtype=jp.float32)
        )
        self.foot_indices = jp.array(
            tuple((self.sys.body.index[name] for name in FETCH_FOOT_NAMES))
        )

    def _command(self, target_speed):
        if self.command_override is not None:
            return self.command_override
        return jp.array(
            (target_speed * COMMAND_HORIZON_SECONDS, 0.0, 0.0), dtype=jp.float32
        )

    @property
    def observation_size(self):
        return OBS_DIM

    def _feature(self, previous_qp, qp, observation):
        previous_angles, _ = self.sys.joints[0].angle_vel(previous_qp)
        angles, _ = self.sys.joints[0].angle_vel(qp)
        previous_feet = fetch_feet(previous_angles)
        feet = fetch_feet(angles)
        contacts = observation[-qp.pos.shape[0] :][self.foot_indices]
        return transition_feature(
            previous_qp.pos[self.torso_idx],
            qp.pos[self.torso_idx],
            previous_qp.rot[self.torso_idx],
            qp.rot[self.torso_idx],
            previous_angles,
            angles,
            previous_feet,
            feet,
            contacts,
        )

    def _observation(self, base_observation, info):
        return jp.concatenate(
            (
                base_observation,
                info["h_feature_buffer"].reshape(-1),
                info["h_previous_control"],
                jax.nn.one_hot(info["h_phase"], PHASE_DIM),
                info["h_plan"],
                info["h_command"],
            )
        )

    def reset(self, rng):
        task_rng, environment_rng = jp.random_split(rng)
        state = super().reset(environment_rng)
        target_speed = jp.random_uniform(
            task_rng, low=self.task_speed_min, high=self.task_speed_max
        )
        target_speed = jp.where(
            self.task_speed_min == self.task_speed_max,
            jp.float32(self.task_speed_min),
            target_speed,
        )
        standing = self._feature(state.qp, state.qp, state.obs)
        info = dict(state.info)
        info.update(
            h_feature_buffer=jp.repeat(standing[None], BUFFER_FRAMES, axis=0),
            h_previous_control=jp.zeros(ACTION_DIM),
            h_phase=jp.int32(0),
            h_plan=jp.zeros(PLAN_DIM),
            h_target_speed=target_speed,
            h_command=self._command(target_speed),
        )
        metrics = dict(state.metrics)
        zero = state.reward
        metrics.update(
            task_reward=zero,
            reference_logp=zero,
            reference_reward=zero,
            target_speed=target_speed,
            torso_height=state.qp.pos[self.torso_idx, 2],
            speed_reward=zero,
        )
        return state.replace(
            obs=self._observation(state.obs, info), info=info, metrics=metrics
        )

    def step(self, state, action):
        stepped = super().step(state, action)
        target_speed = state.info["h_target_speed"]
        sigma = target_speed / 3.0
        speed = stepped.metrics["speed"]
        track = jp.exp(-((speed - target_speed) ** 2) / (2.0 * sigma**2))
        upright = stepped.metrics["upright"]
        ctrl_cost = self.control_weight * jp.sum(jp.square(action))
        feature = self._feature(state.qp, stepped.qp, stepped.obs)
        speed_reward = track + self.upright_weight * upright - ctrl_cost
        task_reward = speed_reward
        info = dict(stepped.info)
        buffer = jnp.roll(state.info["h_feature_buffer"], -1, axis=0)
        info.update(
            h_feature_buffer=buffer.at[-1].set(feature),
            h_previous_control=action,
            h_phase=(state.info["h_phase"] + 1) % PHASE_DIM,
            h_plan=state.info["h_plan"],
            h_target_speed=target_speed,
            h_command=state.info["h_command"],
        )
        metrics = dict(stepped.metrics)
        metrics.update(
            speed=speed,
            track=track,
            upright=upright,
            ctrl_cost=ctrl_cost,
            task_reward=task_reward,
            target_speed=target_speed,
            torso_height=stepped.qp.pos[self.torso_idx, 2],
            speed_reward=speed_reward,
        )
        return stepped.replace(
            obs=self._observation(stepped.obs, info),
            reward=task_reward,
            info=info,
            metrics=metrics,
        )


RESIDUAL_MEAN_SCALE = 2.0


RESIDUAL_SCALE_LOGIT = 1.0


class BoundedResidualMLP(linen.Module):
    hidden: int = 128

    @linen.compact
    def __call__(self, values):
        values = linen.silu(linen.Dense(self.hidden)(values))
        values = linen.silu(linen.Dense(self.hidden)(values))
        return linen.Dense(
            2 * ACTION_DIM,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )(values)


def inverse_softplus(value):
    return jnp.log(jnp.expm1(value))


def unpack_observation(observation):
    leading = observation.shape[:-1]
    buffer = observation[..., FEATURE_BUFFER_SLICE].reshape(
        leading + (BUFFER_FRAMES, FEATURE_DIM)
    )
    return (
        observation[..., :BASE_OBS_DIM],
        buffer,
        observation[..., PREVIOUS_CONTROL_SLICE],
        observation[..., PHASE_SLICE],
        observation[..., PLAN_SLICE],
        observation[..., COMMAND_SLICE],
    )


def compute_plans(prior, observation):
    _, buffer, _, _, _, command = unpack_observation(observation)
    leading = observation.shape[:-1]
    flat_buffer = buffer.reshape((-1, BUFFER_FRAMES, FEATURE_DIM))
    flat_command = command.reshape((-1, COMMAND_DIM))

    def one(feature_buffer, raw_command):
        tokens = prior.encode(feature_buffer)
        return prior.predict_plan(tokens[-HISTORY_TOKENS:], raw_command)

    plan = jax.vmap(one)(flat_buffer, flat_command)
    return plan.reshape(leading + (prior.metadata["config"]["latent_dim"],))


def frozen_context(prior, observation):
    base_observation, buffer, previous, phase, plan, command = unpack_observation(
        observation
    )
    mean = prior.action_mean(buffer[..., -1, :], plan, previous, phase, command)
    latest = (buffer[..., -1, :] - prior.feature_mean) / prior.feature_std
    normalized_command = (command - prior.command_mean) / prior.command_std
    compact = jnp.concatenate(
        (base_observation, latest, plan, phase, normalized_command, mean), axis=-1
    )
    return (mean, plan, compact)


def reference_parameters(prior, observation):
    mean, _, _ = frozen_context(prior, observation)
    std = jnp.exp(prior.action_log_std)
    raw_scale = inverse_softplus(jnp.maximum(std - 0.001, 1e-06))
    return jnp.concatenate((mean, jnp.broadcast_to(raw_scale, mean.shape)), axis=-1)


def make_residual_ppo_networks(
    observation_size, action_size, preprocess_observations_fn=lambda x, _: x, *, prior
):
    del preprocess_observations_fn
    if action_size != ACTION_DIM:
        raise ValueError(action_size)
    if isinstance(observation_size, dict):
        raise TypeError("Part 3 uses one flat observation")
    observation_dim = int(observation_size[-1])
    residual = BoundedResidualMLP()
    dummy_obs = jnp.zeros((1, observation_dim), dtype=jnp.float32)
    _, _, dummy_compact = frozen_context(prior, dummy_obs)

    def policy_init(key):
        return residual.init(key, dummy_compact)

    def policy_apply(_, params, observation):
        base_mean, _, compact = frozen_context(prior, observation)
        output = residual.apply(params, compact)
        delta_mean, delta_scale = jnp.split(output, 2, axis=-1)
        mean = base_mean + RESIDUAL_MEAN_SCALE * jnp.tanh(delta_mean)
        std = jnp.exp(prior.action_log_std)
        base_scale = inverse_softplus(jnp.maximum(std - 0.001, 1e-06))
        scale = base_scale + RESIDUAL_SCALE_LOGIT * jnp.tanh(delta_scale)
        return jnp.concatenate((mean, scale), axis=-1)

    policy_network = networks.FeedForwardNetwork(init=policy_init, apply=policy_apply)
    value_module = networks.MLP(layer_sizes=(256, 256, 1), activation=linen.silu)

    def value_init(key):
        return value_module.init(key, dummy_compact)

    def value_apply(_, params, observation):
        _, _, compact = frozen_context(prior, observation)
        return value_module.apply(params, compact).squeeze(-1)

    value_network = networks.FeedForwardNetwork(init=value_init, apply=value_apply)
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=distribution.NormalTanhDistribution(
            event_size=ACTION_DIM
        ),
    )


def diagonal_gaussian_kl(parameters, reference_parameters_):
    mean, raw_scale = jnp.split(parameters, 2, axis=-1)
    reference_mean, reference_raw_scale = jnp.split(reference_parameters_, 2, axis=-1)
    scale = jax.nn.softplus(raw_scale) + 0.001
    reference_scale = jax.nn.softplus(reference_raw_scale) + 0.001
    return 0.5 * jnp.mean(
        2 * (jnp.log(reference_scale) - jnp.log(scale))
        + (jnp.square(scale) + jnp.square(mean - reference_mean))
        / jnp.square(reference_scale)
        - 1,
        axis=-1,
    )


class BatchedPlanWrapper(Wrapper):
    def __init__(self, env, prior):
        super().__init__(env)
        self.prior = prior

    def _set_plan(self, state, plan):
        observation = state.obs.at[..., PLAN_SLICE].set(plan)
        pipeline = state.pipeline_state
        pipeline_info = dict(pipeline.info)
        pipeline_info["h_plan"] = plan
        pipeline = pipeline.replace(
            obs=pipeline.obs.at[..., PLAN_SLICE].set(plan), info=pipeline_info
        )
        return state.replace(obs=observation, pipeline_state=pipeline)

    def reset(self, rng):
        state = self.env.reset(rng)
        return self._set_plan(state, compute_plans(self.prior, state.obs))

    def step(self, state, action):
        old_plan = state.obs[..., PLAN_SLICE]
        stepped = self.env.step(state, action)
        phase = jnp.argmax(stepped.obs[..., PHASE_SLICE], axis=-1)
        refresh = phase == 0
        candidate = jax.lax.cond(
            jnp.any(refresh),
            lambda observation: compute_plans(self.prior, observation),
            lambda _: old_plan,
            stepped.obs,
        )
        plan = jnp.where(refresh[..., None], candidate, old_plan)
        return self._set_plan(stepped, plan)


class BatchedReferenceWrapper(Wrapper):
    def __init__(self, env, prior, beta):
        super().__init__(env)
        self.prior = prior
        self.beta = float(beta)
        self.distribution = distribution.NormalTanhDistribution(ACTION_DIM)

    def step(self, state, action):
        if self.beta:
            parameters = reference_parameters(self.prior, state.obs)
            raw_action = jnp.arctanh(jnp.clip(action, -0.999999, 0.999999))
            reference_logp = (
                self.distribution.log_prob(parameters, raw_action) / ACTION_DIM
            )
            reference_reward = self.beta * reference_logp
        else:
            reference_logp = jnp.zeros_like(state.reward)
            reference_reward = jnp.zeros_like(state.reward)
        stepped = self.env.step(state, action)
        task_reward = stepped.reward
        metrics = dict(stepped.metrics)
        metrics.update(
            task_reward=task_reward,
            reference_logp=reference_logp,
            reference_reward=reference_reward,
        )
        return stepped.replace(reward=task_reward + reference_reward, metrics=metrics)


def wrap_training(env, episode_length=1000, action_repeat=1, *, prior, beta, **_):
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = RawKeyVmapWrapper(env)
    env = BatchedPlanWrapper(env, prior)
    env = BatchedReferenceWrapper(env, prior, beta)
    return AutoResetWrapper(env)
