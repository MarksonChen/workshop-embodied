"""Functional LSNN dynamics using BrainPy's differentiable spike operator.

BrainPy's object transforms are useful for standalone simulations, whereas a
Brax policy must pass parameters and recurrent state explicitly through JAX
transforms. This module therefore keeps the neuron state functional while
using BrainPy's tested hard-forward/surrogate-backward spike primitive. The
same equations are used for sequence training, inference, and recorded hard
spike counts.
"""

from __future__ import annotations

from typing import NamedTuple

import brainpy.math as bm
import jax
import jax.numpy as jnp

from demo_j.control.config import SNNConfig


Array = jax.Array


class LSNNState(NamedTuple):
    """Per-environment recurrent state."""

    membrane: Array
    current: Array
    adaptation: Array
    filtered_spikes: Array
    spikes: Array


class LSNNParams(NamedTuple):
    """Trainable current encoder, recurrent weights, and linear readout."""

    input_weight: Array
    recurrent_weight: Array
    bias: Array
    readout_weight: Array
    readout_bias: Array


def _orthogonal(key: Array, shape: tuple[int, int], scale: float) -> Array:
    value = jax.random.normal(key, shape)
    q, r = jnp.linalg.qr(value)
    q = q * jnp.sign(jnp.diag(r))[None, :]
    return q * scale


def init_params(
    key: Array,
    input_dim: int,
    output_dim: int,
    config: SNNConfig,
) -> LSNNParams:
    """Initialize a stable recurrent population and small action readout."""

    config.validate()
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    key_in, key_rec, key_out = jax.random.split(key, 3)
    input_weight = jax.random.normal(key_in, (input_dim, config.neurons))
    input_weight *= jnp.sqrt(2.0 / input_dim)
    recurrent_weight = _orthogonal(
        key_rec, (config.neurons, config.neurons), config.recurrent_scale
    )
    recurrent_weight -= jnp.diag(jnp.diag(recurrent_weight))
    readout_weight = jax.random.normal(key_out, (config.neurons, output_dim))
    readout_weight *= 0.01 / jnp.sqrt(config.neurons)
    return LSNNParams(
        input_weight=input_weight,
        recurrent_weight=recurrent_weight,
        bias=jnp.zeros((config.neurons,)),
        readout_weight=readout_weight,
        readout_bias=jnp.zeros((output_dim,)),
    )


def initial_state(batch_shape: tuple[int, ...], config: SNNConfig) -> LSNNState:
    """Return a zero neuronal state for arbitrary leading batch dimensions."""

    config.validate()
    shape = batch_shape + (config.neurons,)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    return LSNNState(zeros, zeros, zeros, zeros, zeros)


def reset_where(state: LSNNState, done: Array) -> LSNNState:
    """Reset every neuronal variable for environments selected by ``done``."""

    mask = jnp.asarray(done, dtype=bool)[..., None]
    return jax.tree.map(lambda value: jnp.where(mask, 0.0, value), state)


def _decay(step_ms: float, tau_ms: float) -> Array:
    return jnp.exp(jnp.asarray(-step_ms / tau_ms, dtype=jnp.float32))


def substep(
    params: LSNNParams,
    state: LSNNState,
    input_current: Array,
    config: SNNConfig,
) -> tuple[LSNNState, Array]:
    """Advance the recurrent LIF/ALIF population by one hard-spike step."""

    syn_decay = _decay(config.step_ms, config.tau_synapse_ms)
    mem_decay = _decay(config.step_ms, config.tau_membrane_ms)
    adapt_decay = _decay(config.step_ms, config.tau_adaptation_ms)
    readout_decay = _decay(config.step_ms, config.tau_readout_ms)

    drive = (
        input_current @ params.input_weight
        + state.spikes @ params.recurrent_weight
        + params.bias
    )
    current = syn_decay * state.current + (1.0 - syn_decay) * drive
    membrane = mem_decay * state.membrane + (1.0 - mem_decay) * current

    adaptive_mask = jnp.arange(config.neurons) < config.adaptive_neurons
    threshold = config.threshold + config.adaptation_strength * jnp.where(
        adaptive_mask, state.adaptation, 0.0
    )
    spike_fn = bm.surrogate.InvSquareGrad(alpha=config.surrogate_alpha)
    spikes = spike_fn(membrane - threshold)
    membrane = membrane - jax.lax.stop_gradient(spikes) * config.threshold
    adaptation = adapt_decay * state.adaptation + spikes
    filtered = readout_decay * state.filtered_spikes + (1.0 - readout_decay) * spikes
    next_state = LSNNState(membrane, current, adaptation, filtered, spikes)
    output = filtered @ params.readout_weight + params.readout_bias
    return next_state, output


def control_step(
    params: LSNNParams,
    state: LSNNState,
    inputs: Array,
    config: SNNConfig,
) -> tuple[LSNNState, tuple[Array, Array]]:
    """Advance one 20 ms control bin and return output plus substep spikes."""

    def body(carry: LSNNState, _: None):
        next_state, output = substep(params, carry, inputs, config)
        return next_state, (output, next_state.spikes)

    next_state, (outputs, spikes) = jax.lax.scan(
        body, state, xs=None, length=config.substeps
    )
    return next_state, (outputs[-1], spikes)


def sequence(
    params: LSNNParams,
    state: LSNNState,
    inputs: Array,
    config: SNNConfig,
) -> tuple[LSNNState, tuple[Array, Array]]:
    """Run time-major inputs and return 20 ms outputs and hard substep spikes."""

    def body(carry: LSNNState, value: Array):
        return control_step(params, carry, value, config)

    return jax.lax.scan(body, state, inputs)
