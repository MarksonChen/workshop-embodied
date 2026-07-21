"""TRACK-MJX-style intention encoder with a BrainPy LSNN decoder."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from demo_j.config import ACTION_DIM, FEATURE_DIM, INTENTION_DIM, SNNConfig
from demo_j.env import REFERENCE_DIM
from demo_j.snn import LSNNParams, LSNNState, init_params, sequence


CURRENT_DIM = FEATURE_DIM + ACTION_DIM
REFERENCE_HIDDEN_DIM = 64
SNN_INPUT_DIM = CURRENT_DIM + INTENTION_DIM


class PolicyParams(NamedTuple):
    reference_weight_1: jax.Array
    reference_bias_1: jax.Array
    reference_weight_2: jax.Array
    reference_bias_2: jax.Array
    snn: LSNNParams
    log_standard_deviation: jax.Array


def _lecun(key: jax.Array, fan_in: int, fan_out: int) -> jax.Array:
    return jax.random.normal(key, (fan_in, fan_out)) / jnp.sqrt(fan_in)


def init_policy(key: jax.Array, config: SNNConfig) -> PolicyParams:
    """Initialize the deterministic intention encoder and spiking decoder."""

    key_1, key_2, key_snn = jax.random.split(key, 3)
    return PolicyParams(
        reference_weight_1=_lecun(key_1, REFERENCE_DIM, REFERENCE_HIDDEN_DIM),
        reference_bias_1=jnp.zeros((REFERENCE_HIDDEN_DIM,), jnp.float32),
        reference_weight_2=_lecun(
            key_2, REFERENCE_HIDDEN_DIM, INTENTION_DIM
        ),
        reference_bias_2=jnp.zeros((INTENTION_DIM,), jnp.float32),
        snn=init_params(key_snn, SNN_INPUT_DIM, ACTION_DIM, config),
        log_standard_deviation=jnp.full((ACTION_DIM,), -1.5, jnp.float32),
    )


def encode_observation(params: PolicyParams, observation: jax.Array) -> jax.Array:
    """Map future reference frames to intention and retain current features."""

    current = observation[..., :CURRENT_DIM]
    reference = observation[..., CURRENT_DIM:]
    hidden = jax.nn.silu(
        reference @ params.reference_weight_1 + params.reference_bias_1
    )
    intention = jax.nn.silu(
        hidden @ params.reference_weight_2 + params.reference_bias_2
    )
    return jnp.concatenate((current, intention), axis=-1)


def policy_sequence(
    params: PolicyParams,
    state: LSNNState,
    observation: jax.Array,
    config: SNNConfig,
) -> tuple[LSNNState, tuple[jax.Array, jax.Array]]:
    """Run time-major observations and return bounded means and hard spikes."""

    encoded = encode_observation(params, observation)
    state, (unbounded, spikes) = sequence(params.snn, state, encoded, config)
    return state, (jnp.tanh(unbounded), spikes)
