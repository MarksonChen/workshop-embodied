from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("brainpy")

from demo_j.control.config import SNNConfig  # noqa: E402
from demo_j.control.snn import (  # noqa: E402
    control_step,
    init_params,
    initial_state,
    reset_where,
    sequence,
)


def test_control_step_emits_integer_20ms_counts() -> None:
    config = SNNConfig(neurons=16, adaptive_neurons=8)
    params = init_params(jax.random.key(0), 5, 2, config)
    state = initial_state((3,), config)
    _, (output, spikes) = control_step(params, state, jnp.ones((3, 5)), config)
    assert output.shape == (3, 2)
    assert spikes.shape == (config.substeps, 3, config.neurons)
    counts = spikes.sum(axis=0)
    assert jnp.issubdtype(counts.dtype, jnp.floating)
    assert bool(jnp.all(counts == jnp.floor(counts)))
    assert bool(jnp.all((0 <= counts) & (counts <= config.substeps)))


def test_surrogate_gradient_is_finite_and_nonzero() -> None:
    config = SNNConfig(neurons=16, adaptive_neurons=8)
    params = init_params(jax.random.key(1), 4, 2, config)
    state = initial_state((2,), config)
    inputs = jnp.ones((8, 2, 4))

    def loss(candidate):
        _, (outputs, _) = sequence(candidate, state, inputs, config)
        return jnp.square(outputs - 0.5).mean()

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert float(sum(jnp.linalg.norm(leaf) for leaf in leaves)) > 0.0


def test_reset_clears_every_state_field_for_done_environments() -> None:
    config = SNNConfig(neurons=8, adaptive_neurons=4)
    state = initial_state((3,), config)
    state = jax.tree.map(lambda value: value + 1.0, state)
    reset = reset_where(state, jnp.array([False, True, False]))
    for value in jax.tree.leaves(reset):
        assert bool(jnp.all(value[1] == 0.0))
        assert bool(jnp.all(value[jnp.array([0, 2])] == 1.0))
