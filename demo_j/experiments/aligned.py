"""Fit motion tokens and pretrain the long-horizon aligned Demo J SNN."""

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
from demo_j.control.aligned import (
    ACTION_DIM,
    CYCLE_FRAMES,
    DEFAULT_PREVIEW_TOKENS,
    FEATURE_DIM,
    PREVIOUS_ACTION_SLICE,
    TOKEN_DIM,
    TOKEN_FRAMES,
    aligned_input_dim,
    balanced_speed_indices,
    build_periodic_sequences,
    fit_tokenizer,
    input_normalization,
    load_tokenizer,
)
from demo_j.control.config import SNNConfig
from demo_j.artifacts import (
    ALIGNED_OUTPUT_ROOT,
    REPOSITORY_ROOT,
    load_pickle,
    save_pickle,
    sha256,
    write_json,
)
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.control.snn import LSNNParams, control_step, init_params, initial_state


DEFAULT_TOKENIZER = ALIGNED_OUTPUT_ROOT / "motion_tokenizer.npz"
TARGET_SPIKE_PROBABILITY = 0.02
ACTIVITY_COEFFICIENT = 1e-3


def load_aligned_checkpoint(path: Path):
    saved = load_pickle(path, ("demo-j-aligned-snn-v1",))
    tokenizer_path = Path(saved["tokenizer"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = REPOSITORY_ROOT / tokenizer_path
    if sha256(tokenizer_path) != saved["tokenizer_sha256"]:
        raise ValueError("aligned tokenizer hash mismatch")
    tokenizer = load_tokenizer(tokenizer_path)
    config = SNNConfig(**saved["config"])
    params = jax.tree.map(jnp.asarray, saved["params"])
    return saved, tokenizer, config, params


def fit_tokenizer_command(reference_root: Path, output: Path) -> dict[str, object]:
    training = load_projected_reference("train", reference_root)
    tokenizer = fit_tokenizer(training)
    metadata = tokenizer.save(output)
    total_variance = TOKEN_FRAMES * FEATURE_DIM
    result = {
        **metadata,
        "training_clips": training.clips,
        "whitened_token_variance_mean": float(
            np.mean(tokenizer.eigenvalues / np.maximum(tokenizer.eigenvalues, 1e-6))
        ),
        "top_components_variance_fraction": float(
            tokenizer.eigenvalues.sum() / total_variance
        ),
        "independent_of_demo_h_weights": True,
    }
    write_json(output.with_suffix(".json"), result)
    print(json.dumps(result, indent=2))
    return result


def _episode_batch(base, offsets, steps):
    index = (jnp.arange(steps)[None] + offsets[:, None]) % CYCLE_FRAMES
    return jnp.take_along_axis(base, index[..., None], axis=1)


def _run_sequence(
    params: LSNNParams,
    state,
    observation,
    previous_action,
    mean,
    std,
    config: SNNConfig,
):
    """Autoregress actions while teacher-forcing only body/reference inputs."""

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
    episode_batches: int,
    episode_steps: int,
    chunk_steps: int,
    learning_rate: float,
    clips_per_speed: int,
    output_dir: Path,
) -> dict[str, object]:
    if episode_steps % chunk_steps:
        raise ValueError("episode_steps must be divisible by chunk_steps")
    tokenizer = load_tokenizer(tokenizer_path)
    training = build_periodic_sequences(
        load_projected_reference("train", reference_root),
        tokenizer,
        preview_tokens=preview_tokens,
    )
    validation = build_periodic_sequences(
        load_projected_reference("validation", reference_root),
        tokenizer,
        preview_tokens=preview_tokens,
    )
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

    def objective(candidate, neuronal_state, previous_action, observation, action):
        (next_state, next_previous), (prediction, spikes) = _run_sequence(
            candidate,
            neuronal_state,
            observation.swapaxes(0, 1),
            previous_action,
            jnp.asarray(mean),
            jnp.asarray(std),
            config,
        )
        prediction = prediction.swapaxes(0, 1)
        spikes = spikes.transpose(2, 0, 1, 3)
        loss, metrics = _sequence_metrics(prediction, action, spikes)
        return loss, (metrics, next_state, next_previous)

    @jax.jit
    def update(
        candidate, opt_state, neuronal_state, previous_action, observation, action
    ):
        (_, (metrics, next_state, next_previous)), gradient = jax.value_and_grad(
            objective, has_aux=True
        )(
            candidate,
            neuronal_state,
            previous_action,
            observation,
            action,
        )
        updates, opt_state = optimizer.update(gradient, opt_state, candidate)
        candidate = optax.apply_updates(candidate, updates)
        metrics["gradient_norm"] = optax.global_norm(gradient)
        next_state = jax.tree.map(jax.lax.stop_gradient, next_state)
        next_previous = jax.lax.stop_gradient(next_previous)
        return candidate, opt_state, next_state, next_previous, metrics

    training_pool = jnp.asarray(balanced_speed_indices(training, clips_per_speed))
    validation_pool = balanced_speed_indices(validation, 10)
    validation_count = len(validation_pool)
    validation_indices = jnp.asarray(validation_pool)
    validation_offsets = jnp.arange(validation_count) % CYCLE_FRAMES

    @jax.jit
    def validate(candidate):
        observation = _episode_batch(
            validation_observation[validation_indices],
            validation_offsets,
            episode_steps,
        )
        action = _episode_batch(
            validation_action[validation_indices],
            validation_offsets,
            episode_steps,
        )
        (_, _), (prediction, spikes) = _run_sequence(
            candidate,
            initial_state((validation_count,), config),
            observation.swapaxes(0, 1),
            jnp.zeros((validation_count, ACTION_DIM)),
            jnp.asarray(mean),
            jnp.asarray(std),
            config,
        )
        prediction = prediction.swapaxes(0, 1)
        spikes = spikes.transpose(2, 0, 1, 3)
        _, metrics = _sequence_metrics(prediction, action, spikes)
        quarter = episode_steps // 4
        metrics.update(
            first_quarter_action_mse=jnp.mean(
                jnp.square(prediction[:, :quarter] - action[:, :quarter])
            ),
            last_quarter_action_mse=jnp.mean(
                jnp.square(prediction[:, -quarter:] - action[:, -quarter:])
            ),
            first_quarter_spike_probability=jnp.mean(spikes[:, :quarter]),
            last_quarter_spike_probability=jnp.mean(spikes[:, -quarter:]),
        )
        return metrics

    chunks = episode_steps // chunk_steps
    progress = []
    best = None
    best_mse = float("inf")
    started = time.perf_counter()
    report_every = max(1, episode_batches // 8)
    compiled = False
    for episode in range(1, episode_batches + 1):
        key, index_key, offset_key = jax.random.split(key, 3)
        pool_indices = jax.random.randint(
            index_key, (batch_size,), 0, len(training_pool)
        )
        indices = training_pool[pool_indices]
        offsets = jax.random.randint(offset_key, (batch_size,), 0, CYCLE_FRAMES)
        observation = _episode_batch(
            training_observation[indices], offsets, episode_steps
        )
        action = _episode_batch(training_action[indices], offsets, episode_steps)
        neuronal_state = initial_state((batch_size,), config)
        previous_action = jnp.zeros((batch_size, ACTION_DIM))
        rows = []
        for chunk in range(chunks):
            start = chunk * chunk_steps
            stop = start + chunk_steps
            params, optimizer_state, neuronal_state, previous_action, metrics = update(
                params,
                optimizer_state,
                neuronal_state,
                previous_action,
                observation[:, start:stop],
                action[:, start:stop],
            )
            rows.append(metrics)
        jax.block_until_ready(params.readout_weight)
        if not compiled:
            compiled = True
            compile_seconds = time.perf_counter() - started
        if episode == 1 or episode % report_every == 0 or episode == episode_batches:
            validation_metrics = validate(params)
            jax.block_until_ready(validation_metrics["action_mse"])
            averaged = {
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in rows[0]
            }
            row = {
                "episode_batch": episode,
                "optimizer_updates": episode * chunks,
                "seconds": time.perf_counter() - started,
                **{f"train_{name}": value for name, value in averaged.items()},
                **{
                    f"validation_{name}": float(value)
                    for name, value in validation_metrics.items()
                },
            }
            progress.append(row)
            validation_mse = row["validation_action_mse"]
            if validation_mse < best_mse:
                best_mse = validation_mse
                best = jax.device_get(params)
            print(json.dumps(row), flush=True)

    elapsed = time.perf_counter() - started
    if best is None:
        raise RuntimeError("no aligned checkpoint candidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    checkpoint = output_dir / (
        f"snn_aligned_preview{preview_tokens}_seed{seed}_{stamp}.pkl"
    )
    tokenizer_path = Path(tokenizer_path).resolve()
    payload = {
        "schema": "demo-j-aligned-snn-v1",
        "params": best,
        "observation_mean": mean,
        "observation_std": std,
        "config": config.as_dict(),
        "preview_tokens": preview_tokens,
        "token_frames": TOKEN_FRAMES,
        "token_dim": TOKEN_DIM,
        "cycle_frames": CYCLE_FRAMES,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256(tokenizer_path),
        "training_manifest_sha256": tokenizer.training_manifest_sha256,
        "seed": seed,
    }
    save_pickle(checkpoint, payload)
    report = {
        "schema": "demo-j-aligned-snn-training-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": seed,
        "preview_tokens": preview_tokens,
        "preview_milliseconds": preview_tokens * TOKEN_FRAMES * 1_000 / FPS,
        "episode_batches": episode_batches,
        "batch_size": batch_size,
        "episode_steps": episode_steps,
        "chunk_steps": chunk_steps,
        "optimizer_updates": episode_batches * chunks,
        "training_seconds_including_compile": elapsed,
        "compile_and_first_episode_seconds": compile_seconds,
        "sequence_bins_per_second_including_compile": (
            episode_batches * batch_size * episode_steps / elapsed
        ),
        "best_validation_action_mse": best_mse,
        "balanced_speed_sampling": True,
        "clips_per_speed": clips_per_speed,
        "training_method": (
            "autoregressive-action sequence distillation with synthetic periodic "
            "extensions; recurrent state reset only at 1000-step episode boundary; "
            "truncated BPTT at chunk boundaries"
        ),
        "synthetic_periodicity_disclosure": (
            "the source release has no continuous 1000-frame clips; 32-frame "
            "wrap-screened segments are repeated without claiming biological continuity"
        ),
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
    train_parser.add_argument("--episode-batches", type=int, default=64)
    train_parser.add_argument("--episode-steps", type=int, default=1000)
    train_parser.add_argument("--chunk-steps", type=int, default=50)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--clips-per-speed", type=int, default=64)
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
            episode_batches=args.episode_batches,
            episode_steps=args.episode_steps,
            chunk_steps=args.chunk_steps,
            learning_rate=args.learning_rate,
            clips_per_speed=args.clips_per_speed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
