"""Minimal PPO math shared by the aligned recurrent controller."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


LOG_TWO_PI = float(jnp.log(2.0 * jnp.pi))


class CriticParams(NamedTuple):
    weight_1: jax.Array
    bias_1: jax.Array
    weight_2: jax.Array
    bias_2: jax.Array
    weight_out: jax.Array
    bias_out: jax.Array


def _orthogonal(key: jax.Array, rows: int, columns: int, scale: float) -> jax.Array:
    matrix = jax.random.normal(key, (max(rows, columns), min(rows, columns)))
    q, r = jnp.linalg.qr(matrix)
    q *= jnp.sign(jnp.diag(r))
    if rows < columns:
        q = q.T
    return q[:rows, :columns] * scale


def init_critic(
    key: jax.Array,
    observation_dim: int,
    hidden: int = 256,
) -> CriticParams:
    """Initialize the non-spiking critic excluded from neural analyses."""

    key_1, key_2, key_out = jax.random.split(key, 3)
    return CriticParams(
        weight_1=_orthogonal(key_1, observation_dim, hidden, jnp.sqrt(2.0)),
        bias_1=jnp.zeros((hidden,), jnp.float32),
        weight_2=_orthogonal(key_2, hidden, hidden, jnp.sqrt(2.0)),
        bias_2=jnp.zeros((hidden,), jnp.float32),
        weight_out=_orthogonal(key_out, hidden, 1, 1.0),
        bias_out=jnp.zeros((1,), jnp.float32),
    )


def critic_value(params: CriticParams, observation: jax.Array) -> jax.Array:
    hidden = jax.nn.silu(observation @ params.weight_1 + params.bias_1)
    hidden = jax.nn.silu(hidden @ params.weight_2 + params.bias_2)
    return (hidden @ params.weight_out + params.bias_out)[..., 0]


def tanh_normal_log_probability(
    raw_action: jax.Array,
    action: jax.Array,
    mean: jax.Array,
    log_standard_deviation: jax.Array,
) -> jax.Array:
    """Log density of a diagonal Gaussian followed by tanh."""

    log_standard_deviation = jnp.clip(log_standard_deviation, -5.0, 1.0)
    inverse_variance = jnp.exp(-2.0 * log_standard_deviation)
    normal = -0.5 * (
        jnp.square(raw_action - mean) * inverse_variance
        + 2.0 * log_standard_deviation
        + LOG_TWO_PI
    )
    log_jacobian = jnp.log(jnp.maximum(1.0 - jnp.square(action), 1e-6))
    return jnp.sum(normal - log_jacobian, axis=-1)


def generalized_advantage_estimate(
    reward: jax.Array,
    value: jax.Array,
    done: jax.Array,
    valid: jax.Array,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute masked GAE for time-major, independently terminated episodes."""

    next_value = jnp.concatenate((value[1:], jnp.zeros_like(value[:1])), axis=0)

    def backward(carry, row):
        reward_t, value_t, next_value_t, done_t, valid_t = row
        continuation = discount * (1.0 - done_t)
        delta = reward_t + continuation * next_value_t - value_t
        advantage = (delta + continuation * gae_lambda * carry) * valid_t
        return advantage, advantage

    _, advantage = jax.lax.scan(
        backward,
        jnp.zeros_like(value[0]),
        (reward, value, next_value, done, valid),
        reverse=True,
    )
    return advantage, advantage + value


def normalize_advantage(advantage: jax.Array, valid: jax.Array) -> jax.Array:
    count = jnp.maximum(valid.sum(), 1.0)
    mean = (advantage * valid).sum() / count
    variance = (jnp.square(advantage - mean) * valid).sum() / count
    return (advantage - mean) / jnp.sqrt(variance + 1e-8)
