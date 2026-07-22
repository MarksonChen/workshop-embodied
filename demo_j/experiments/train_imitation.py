"""Sequence-distill the independent feedback controller into a BrainPy SNN."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from demo_j.control.config import SNNConfig
from demo_j.artifacts import OUTPUT_ROOT, save_pickle, sha256, write_json
from demo_j.control.imitation import normalization, teacher_forced_sequences
from demo_j.control.policy import init_policy, policy_sequence
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.control.snn import initial_state


TARGET_SPIKE_PROBABILITY = 0.02  # broad anti-silence target per 5 ms substep
ACTIVITY_COEFFICIENT = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    steps = 20 if args.smoke else args.steps

    training_reference = load_projected_reference("train", args.reference_root)
    validation_reference = load_projected_reference("validation", args.reference_root)
    training = teacher_forced_sequences(training_reference)
    validation = teacher_forced_sequences(validation_reference)
    mean, std = normalization(training)
    training_observation = jnp.asarray(
        np.clip((training.observation - mean) / std, -10.0, 10.0)
    )
    validation_observation = jnp.asarray(
        np.clip((validation.observation - mean) / std, -10.0, 10.0)
    )
    training_action = jnp.asarray(training.action)
    validation_action = jnp.asarray(validation.action)

    config = SNNConfig()
    key = jax.random.key(args.seed)
    key, parameter_key = jax.random.split(key)
    params = init_policy(parameter_key, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.learning_rate, weight_decay=1e-5),
    )
    optimizer_state = optimizer.init(params)

    def objective(candidate, observation, action):
        batch = observation.shape[0]
        state = initial_state((batch,), config)
        _, (prediction, spikes) = policy_sequence(
            candidate,
            state,
            observation.swapaxes(0, 1),
            config,
        )
        prediction = prediction.swapaxes(0, 1)
        action_mse = jnp.mean(jnp.square(prediction - action))
        per_neuron_activity = jnp.mean(spikes, axis=(0, 1, 2))
        activity_loss = jnp.mean(
            jnp.square(per_neuron_activity - TARGET_SPIKE_PROBABILITY)
        )
        loss = action_mse + ACTIVITY_COEFFICIENT * activity_loss
        metrics = {
            "loss": loss,
            "action_mse": action_mse,
            "spike_probability": jnp.mean(spikes),
            "silent_neuron_fraction": jnp.mean(per_neuron_activity == 0),
            "action_saturation_fraction": jnp.mean(jnp.abs(prediction) >= 0.99),
        }
        return loss, metrics

    @jax.jit
    def update(candidate, state, observation, action):
        (_, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(
            candidate, observation, action
        )
        updates, state = optimizer.update(gradient, state, candidate)
        candidate = optax.apply_updates(candidate, updates)
        metrics["gradient_norm"] = optax.global_norm(gradient)
        return candidate, state, metrics

    @jax.jit
    def validate(candidate):
        count = min(64, validation.clips)
        return objective(
            candidate,
            validation_observation[:count],
            validation_action[:count],
        )[1]

    started = time.perf_counter()
    progress = []
    report_every = max(1, steps // 10)
    for step in range(1, steps + 1):
        key, batch_key = jax.random.split(key)
        indices = jax.random.randint(batch_key, (args.batch_size,), 0, training.clips)
        params, optimizer_state, metrics = update(
            params,
            optimizer_state,
            training_observation[indices],
            training_action[indices],
        )
        if step == 1 or step % report_every == 0 or step == steps:
            validation_metrics = validate(params)
            row = {
                "step": step,
                "seconds": time.perf_counter() - started,
                **{f"train_{key}": float(value) for key, value in metrics.items()},
                **{
                    f"validation_{key}": float(value)
                    for key, value in validation_metrics.items()
                },
            }
            progress.append(row)
            print(json.dumps(row), flush=True)

    elapsed = time.perf_counter() - started
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint = OUTPUT_ROOT / f"snn_distilled_seed{args.seed}_{stamp}.pkl"
    payload = {
        "schema": "demo-j-snn-distillation-v1",
        "params": jax.device_get(params),
        "observation_mean": mean,
        "observation_std": std,
        "config": config.as_dict(),
        "training_reference_manifest_sha256": training_reference.manifest_sha256,
        "training_steps": steps,
        "seed": args.seed,
    }
    save_pickle(checkpoint, payload)
    report_path = checkpoint.with_suffix(".json")
    report = {
        "schema": "demo-j-snn-distillation-training-v1",
        "method": "sequence behavior cloning from independent feedback projection",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": args.seed,
        "steps": steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_seconds": elapsed,
        "clips_per_second": steps * args.batch_size / elapsed,
        "activity_coefficient": ACTIVITY_COEFFICIENT,
        "target_spike_probability_per_5ms": TARGET_SPIKE_PROBABILITY,
        "progress": progress,
    }
    write_json(report_path, report)
    print(f"wrote {checkpoint} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
