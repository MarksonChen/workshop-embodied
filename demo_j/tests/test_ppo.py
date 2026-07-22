from __future__ import annotations

import jax
import jax.numpy as jnp

from demo_j.control.ppo import (
    critic_value,
    generalized_advantage_estimate,
    init_critic,
    normalize_advantage,
    tanh_normal_log_probability,
)


def test_tanh_normal_log_probability_is_finite() -> None:
    raw = jnp.asarray([[0.0, 0.5], [-0.2, 1.0]])
    action = jnp.tanh(raw)
    value = tanh_normal_log_probability(
        raw,
        action,
        jnp.zeros_like(raw),
        jnp.asarray([-1.0, -1.0]),
    )
    assert value.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(value)))


def test_gae_stops_at_terminal_and_masks_padding() -> None:
    reward = jnp.asarray([[1.0], [1.0], [9.0]])
    value = jnp.zeros_like(reward)
    done = jnp.asarray([[0.0], [1.0], [1.0]])
    valid = jnp.asarray([[1.0], [1.0], [0.0]])
    advantage, target = generalized_advantage_estimate(
        reward,
        value,
        done,
        valid,
        discount=1.0,
        gae_lambda=1.0,
    )
    assert advantage[:, 0].tolist() == [2.0, 1.0, 0.0]
    assert target[:, 0].tolist() == [2.0, 1.0, 0.0]


def test_critic_and_advantage_helpers_have_finite_gradients() -> None:
    critic = init_critic(jax.random.key(0), observation_dim=7, hidden=8)
    observation = jax.random.normal(jax.random.key(1), (5, 3, 7))
    gradient = jax.grad(
        lambda params: jnp.mean(critic_value(params, observation) ** 2)
    )(critic)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradient))
    valid = jnp.asarray([[1.0, 1.0], [1.0, 0.0]])
    normalized = normalize_advantage(jnp.asarray([[1.0, 3.0], [5.0, 9.0]]), valid)
    assert abs(float((normalized * valid).sum())) < 1e-5
