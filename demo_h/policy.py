"""Frozen prior plus the accepted zero-initialized bounded PPO adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen

from brax.training import distribution, networks
from brax.training.agents.ppo import networks as ppo_networks

from demo_h.config import (
    ACTION_DIM,
    BASE_OBS_DIM,
    BUFFER_FRAMES,
    COMMAND_DIM,
    COMMAND_SLICE,
    FEATURE_BUFFER_SLICE,
    FEATURE_DIM,
    HISTORY_TOKENS,
    PHASE_SLICE,
    PLAN_SLICE,
    PREVIOUS_CONTROL_SLICE,
)


# The accepted adapter stays close enough to the controller for stable,
# workshop-scale post-training. It is deliberately bounded and must not be
# described as an unconstrained policy initialized from the prior.
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
    mean = prior.action_mean(
        buffer[..., -1, :], plan, previous, phase, command
    )
    latest = (buffer[..., -1, :] - prior.feature_mean) / prior.feature_std
    normalized_command = (command - prior.command_mean) / prior.command_std
    compact = jnp.concatenate(
        (
            base_observation,
            latest,
            plan,
            phase,
            normalized_command,
            mean,
        ),
        axis=-1,
    )
    return mean, plan, compact


def reference_parameters(prior, observation):
    mean, _, _ = frozen_context(prior, observation)
    std = jnp.exp(prior.action_log_std)
    raw_scale = inverse_softplus(jnp.maximum(std - 0.001, 1e-6))
    return jnp.concatenate((mean, jnp.broadcast_to(raw_scale, mean.shape)), axis=-1)


def make_residual_ppo_networks(
    observation_size,
    action_size,
    preprocess_observations_fn=lambda x, _: x,
    *,
    prior,
):
    # Brax supplies this callback as part of the network-factory protocol.  The
    # frozen prior instead needs raw observations and applies its own exported
    # feature/command normalization inside ``frozen_context``.
    del preprocess_observations_fn
    if action_size != ACTION_DIM:
        raise ValueError(action_size)
    if isinstance(observation_size, dict):
        raise TypeError("Demo H uses one flat observation")
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
        base_scale = inverse_softplus(jnp.maximum(std - 0.001, 1e-6))
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
