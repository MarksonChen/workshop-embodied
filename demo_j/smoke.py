"""Fail-fast BrainPy/JAX/GPU and differentiable-LSNN smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import brainpy
import brax
import jax
import jax.numpy as jnp

from demo_j.config import SNNConfig
from demo_j.snn import init_params, initial_state, sequence


def run_smoke(*, steps: int = 1_000, batch_size: int = 16) -> dict[str, object]:
    """Differentiate through a long recurrent sequence and report invariants."""

    config = SNNConfig(neurons=64, adaptive_neurons=32)
    input_dim = 12
    output_dim = 3
    key_params, key_inputs = jax.random.split(jax.random.key(0))
    params = init_params(key_params, input_dim, output_dim, config)
    state = initial_state((batch_size,), config)
    inputs = jax.random.normal(key_inputs, (steps, batch_size, input_dim)) * 0.8
    target = jnp.tanh(inputs[..., :output_dim])

    def loss_fn(candidate):
        _, (outputs, spikes) = sequence(candidate, state, inputs, config)
        task = jnp.mean(jnp.square(outputs - target))
        activity = jnp.mean(spikes)
        return task + 1e-4 * activity, (task, activity, outputs, spikes)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
    started = time.perf_counter()
    (loss, (task, activity, outputs, spikes)), gradients = value_and_grad(params)
    jax.block_until_ready(loss)
    elapsed = time.perf_counter() - started

    leaves = jax.tree.leaves(gradients)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
    finite = all(bool(jnp.all(jnp.isfinite(x))) for x in (*leaves, outputs, spikes))
    hard_spikes = bool(jnp.all((spikes == 0.0) | (spikes == 1.0)))
    devices = [str(device) for device in jax.devices()]
    result: dict[str, object] = {
        "brainpy_version": brainpy.__version__,
        "brax_version": brax.__version__,
        "jax_version": jax.__version__,
        "devices": devices,
        "steps": steps,
        "batch_size": batch_size,
        "neurons": config.neurons,
        "substeps": config.substeps,
        "loss": float(loss),
        "task_loss": float(task),
        "mean_spike_probability_per_5ms": float(activity),
        "gradient_norm": float(grad_norm),
        "finite": finite,
        "hard_forward_spikes": hard_spikes,
        "compile_and_run_seconds": elapsed,
    }
    if not finite:
        raise RuntimeError("non-finite LSNN value or gradient")
    if not hard_spikes:
        raise RuntimeError("BrainPy surrogate did not emit hard forward spikes")
    if not float(grad_norm) > 0.0:
        raise RuntimeError("surrogate gradient vanished across the smoke sequence")
    if not any("cuda" in device.lower() for device in devices):
        raise RuntimeError(f"Demo J smoke test did not use a CUDA device: {devices}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke(steps=args.steps, batch_size=args.batch_size)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

