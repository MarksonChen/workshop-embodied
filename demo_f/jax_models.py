"""Small pure-JAX neural-network operations shared by Demo F/G/H priors."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def linear(values, weight, bias):
    return values @ weight.T + bias


def layer_norm(values, weight, bias, epsilon=1e-5):
    mean = values.mean(axis=-1, keepdims=True)
    variance = jnp.square(values - mean).mean(axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + epsilon) * weight + bias


def causal_conv(values, weight, bias, stride):
    kernel = weight.shape[-1]
    padded = jnp.pad(values, ((kernel - stride, 0), (0, 0)))
    output = jax.lax.conv_general_dilated(
        padded[None],
        weight.transpose(2, 1, 0),
        window_strides=(stride,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )[0]
    return output + bias


def sinusoidal_positions(length, width, dtype):
    position = jnp.arange(length, dtype=dtype)[:, None]
    frequency = jnp.exp(
        jnp.arange(0, width, 2, dtype=dtype) * (-math.log(10_000.0) / width)
    )
    output = jnp.zeros((length, width), dtype=dtype)
    output = output.at[:, 0::2].set(jnp.sin(position * frequency))
    return output.at[:, 1::2].set(jnp.cos(position * frequency))
