from __future__ import annotations

import jax
import jax.numpy as jnp

from demo_j.config import SNNConfig
from demo_j.env import OBS_DIM
from demo_j.policy import encode_observation, init_policy, policy_sequence
from demo_j.snn import initial_state


def test_policy_shapes_and_bounded_means() -> None:
    config = SNNConfig(neurons=16, adaptive_neurons=8)
    params = init_policy(jax.random.key(0), config)
    observation = jnp.ones((3, 2, OBS_DIM))
    assert encode_observation(params, observation).shape == (3, 2, 102)
    state = initial_state((2,), config)
    _, (mean, spikes) = policy_sequence(params, state, observation, config)
    assert mean.shape == (3, 2, 10)
    assert spikes.shape == (3, config.substeps, 2, config.neurons)
    assert bool(jnp.all(jnp.abs(mean) <= 1.0))
