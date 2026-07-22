"""Fit motion tokens and train the native-clip recurrent Demo J SNN."""

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

from demo_f.config import FPS
from demo_j.artifacts import (
    ALIGNED_OUTPUT_ROOT,
    REPOSITORY_ROOT,
    load_pickle,
    save_pickle,
    sha256,
    write_json,
)
from demo_j.control.aligned import (
    ACTION_DIM,
    DEFAULT_PREVIEW_TOKENS,
    FEATURE_DIM,
    PREVIOUS_ACTION_SLICE,
    TOKEN_DIM,
    TOKEN_FRAMES,
    aligned_input_dim,
    build_clip_sequences,
    fit_tokenizer,
    input_normalization,
    load_tokenizer,
)
from demo_j.control.config import SNNConfig
from demo_j.control.snn import LSNNParams, control_step, init_params, initial_state
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference


DEFAULT_TOKENIZER = ALIGNED_OUTPUT_ROOT / "motion_tokenizer.npz"
TARGET_SPIKE_PROBABILITY = 0.02
ACTIVITY_COEFFICIENT = 1e-3


def load_clip_checkpoint(path: Path):
    saved = load_pickle(path, ("demo-j-native-clip-snn-v1",))
    tokenizer_path = Path(saved["tokenizer"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = REPOSITORY_ROOT / tokenizer_path
    if sha256(tokenizer_path) != saved["tokenizer_sha256"]:
        raise ValueError("motion tokenizer hash mismatch")
    tokenizer = load_tokenizer(tokenizer_path)
    config = SNNConfig(**saved["config"])
    params = jax.tree.map(jnp.asarray, saved["params"])
    return saved, tokenizer, config, params


# Preserve the notebook API name while rejecting obsolete periodic checkpoints.
load_aligned_checkpoint = load_clip_checkpoint


def fit_tokenizer_command(reference_root: Path, output: Path) -> dict[str, object]:
    training = load_projected_reference("train", reference_root)
    tokenizer = fit_tokenizer(training)
    metadata = tokenizer.save(output)
    result = {
        **metadata,
        "training_clips": training.clips,
        "whitened_token_variance_mean": float(
            np.mean(tokenizer.eigenvalues / np.maximum(tokenizer.eigenvalues, 1e-6))
        ),
        "top_components_variance_fraction": float(
            tokenizer.eigenvalues.sum() / (TOKEN_FRAMES * FEATURE_DIM)
        ),
        "independent_of_demo_h_weights": True,
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))
    return result


def _run_sequence(
    params: LSNNParams,
    state,
    observation,
    previous_action,
    mean,
    std,
    config: SNNConfig,
):
    """Autoregress actions while teacher-forcing body and intention inputs."""

    previous_mean = mean[PREVIOUS_ACTION_SLICE]
    previous_std = std[PREVIOUS_ACTION_SLICE]

    def advance(carry, normalized):
        neuronal_state, previous = carry
        normalized = normalized.at[..., PREVIOUS_ACTION_SLICE].set(
            (previous - previous_mean) / previous_std
        )
        neuronal_state, (logits, spikes) = control_step(
            params, neuronal_state, normalized, config
        )
        action = jnp.tanh(logits)
        return (neuronal_state, action), (action, spikes)

    return jax.lax.scan(advance, (state, previous_action), observation)


def _sequence_metrics(prediction, target, spikes):
    per_neuron = jnp.mean(spikes, axis=(0, 1, 2))
    action_mse = jnp.mean(jnp.square(prediction - target))
    activity_loss = jnp.mean(jnp.square(per_neuron - TARGET_SPIKE_PROBABILITY))
    return action_mse + ACTIVITY_COEFFICIENT * activity_loss, {
        "action_mse": action_mse,
        "spike_probability": jnp.mean(spikes),
        "silent_neuron_fraction": jnp.mean(per_neuron == 0),
        "action_saturation_fraction": jnp.mean(jnp.abs(prediction) >= 0.99),
    }


def train(
    *,
    tokenizer_path: Path,
    reference_root: Path,
    preview_tokens: int,
    seed: int,
    batch_size: int,
    updates: int,
    learning_rate: float,
    output_dir: Path,
) -> dict[str, object]:
    tokenizer = load_tokenizer(tokenizer_path)
    training = build_clip_sequences(
        load_projected_reference("train", reference_root),
        tokenizer,
        preview_tokens=preview_tokens,
    )
    validation = build_clip_sequences(
        load_projected_reference("validation", reference_root),
        tokenizer,
        preview_tokens=preview_tokens,
    )
    if training.steps != validation.steps:
        raise ValueError((training.steps, validation.steps))
    mean, std = input_normalization(training)
    training_observation = jnp.asarray(
        np.clip((training.observation - mean) / std, -10.0, 10.0)
    )
    training_action = jnp.asarray(training.action)
    validation_observation = jnp.asarray(
        np.clip((validation.observation - mean) / std, -10.0, 10.0)
    )
    validation_action = jnp.asarray(validation.action)

    config = SNNConfig()
    key = jax.random.key(seed)
    key, parameter_key = jax.random.split(key)
    params = init_params(
        parameter_key, aligned_input_dim(preview_tokens), ACTION_DIM, config
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=1e-5),
    )
    optimizer_state = optimizer.init(params)
    mean_device = jnp.asarray(mean)
    std_device = jnp.asarray(std)

    def objective(candidate, observation, action):
        batch = observation.shape[0]
        (_, _), (prediction, spikes) = _run_sequence(
            candidate,
            initial_state((batch,), config),
            observation.swapaxes(0, 1),
            jnp.zeros((batch, ACTION_DIM), jnp.float32),
            mean_device,
            std_device,
            config,
        )
        prediction = prediction.swapaxes(0, 1)
        spikes = spikes.transpose(2, 0, 1, 3)
        return _sequence_metrics(prediction, action, spikes)

    @jax.jit
    def update(candidate, opt_state, observation, action):
        (_, metrics), gradient = jax.value_and_grad(objective, has_aux=True)(
            candidate, observation, action
        )
        updates_, opt_state = optimizer.update(gradient, opt_state, candidate)
        candidate = optax.apply_updates(candidate, updates_)
        metrics["gradient_norm"] = optax.global_norm(gradient)
        return candidate, opt_state, metrics

    @jax.jit
    def validate(candidate):
        return objective(candidate, validation_observation, validation_action)[1]

    progress = []
    best = None
    best_mse = float("inf")
    started = time.perf_counter()
    report_every = max(1, updates // 10)
    compile_seconds = None
    for update_index in range(1, updates + 1):
        key, index_key = jax.random.split(key)
        indices = jax.random.randint(index_key, (batch_size,), 0, training.clips)
        params, optimizer_state, metrics = update(
            params,
            optimizer_state,
            training_observation[indices],
            training_action[indices],
        )
        jax.block_until_ready(params.readout_weight)
        if compile_seconds is None:
            compile_seconds = time.perf_counter() - started
        if (
            update_index == 1
            or update_index % report_every == 0
            or update_index == updates
        ):
            validation_metrics = validate(params)
            jax.block_until_ready(validation_metrics["action_mse"])
            row = {
                "update": update_index,
                "seconds": time.perf_counter() - started,
                **{f"train_{name}": float(value) for name, value in metrics.items()},
                **{
                    f"validation_{name}": float(value)
                    for name, value in validation_metrics.items()
                },
            }
            progress.append(row)
            if row["validation_action_mse"] < best_mse:
                best_mse = row["validation_action_mse"]
                best = jax.device_get(params)
            print(json.dumps(row), flush=True)

    elapsed = time.perf_counter() - started
    if best is None or compile_seconds is None:
        raise RuntimeError("no native-clip checkpoint candidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint = output_dir / f"snn_native_clip_seed{seed}_{stamp}.pkl"
    tokenizer_path = Path(tokenizer_path).resolve()
    payload = {
        "schema": "demo-j-native-clip-snn-v1",
        "params": best,
        "observation_mean": mean,
        "observation_std": std,
        "config": config.as_dict(),
        "preview_tokens": preview_tokens,
        "token_frames": TOKEN_FRAMES,
        "token_dim": TOKEN_DIM,
        "episode_steps": training.steps,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "training_manifest_sha256": tokenizer.training_manifest_sha256,
        "seed": seed,
    }
    save_pickle(checkpoint, payload)
    report = {
        "schema": "demo-j-native-clip-snn-training-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": seed,
        "preview_tokens": preview_tokens,
        "maximum_preview_milliseconds": preview_tokens * TOKEN_FRAMES * 1_000 / FPS,
        "episode_steps": training.steps,
        "updates": updates,
        "batch_size": batch_size,
        "training_seconds_including_compile": elapsed,
        "compile_and_first_update_seconds": compile_seconds,
        "sequence_bins_per_second_including_compile": (
            updates * batch_size * training.steps / elapsed
        ),
        "best_validation_action_mse": best_mse,
        "valid_preview_fraction_train": float(training.preview_mask.mean()),
        "training_method": (
            "autoregressive-action sequence distillation on one finite native clip "
            "per recurrent episode; state resets at clip boundaries"
        ),
        "future_tail_contract": (
            "future blocks beyond the 64-frame clip are zeroed and explicitly masked"
        ),
        "periodic_extension_used": False,
        "demo_h_policy_used_for_training": False,
        "progress": progress,
    }
    write_json(checkpoint.with_suffix(".json"), report)
    print(f"wrote {checkpoint}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    tokenizer_parser = commands.add_parser("fit-tokenizer")
    tokenizer_parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    tokenizer_parser.add_argument("--output", type=Path, default=DEFAULT_TOKENIZER)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    train_parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    train_parser.add_argument(
        "--preview-tokens", type=int, default=DEFAULT_PREVIEW_TOKENS
    )
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--updates", type=int, default=2_000)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--output-dir", type=Path, default=ALIGNED_OUTPUT_ROOT)

    args = parser.parse_args()
    if args.command == "fit-tokenizer":
        fit_tokenizer_command(args.reference_root, args.output)
    else:
        train(
            tokenizer_path=args.tokenizer,
            reference_root=args.reference_root,
            preview_tokens=args.preview_tokens,
            seed=args.seed,
            batch_size=args.batch_size,
            updates=args.updates,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
