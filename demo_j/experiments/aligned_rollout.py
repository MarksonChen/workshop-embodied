"""Evaluate aligned controllers and record their fixed-input spike streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_f.config import FPS
from demo_f.features import SL
from demo_h.config import COMMAND_SLICE, PHASE_SLICE
from demo_j.artifacts import ALIGNED_OUTPUT_ROOT, sha256, write_json
from demo_j.control.aligned import (
    ACTION_DIM,
    CYCLE_FRAMES,
    FEATURE_DIM,
    PREVIOUS_ACTION_SLICE,
    TOKEN_FRAMES,
    aligned_input_dim,
    build_periodic_sequences,
    periodic_reference_set,
    select_speed_examples,
)
from demo_j.control.aligned_tracking import AlignedLocomotion
from demo_j.control.snn import control_step, initial_state
from demo_j.control.tracking import FetchTracking
from demo_j.data.physics import joint_angles
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.experiments.aligned import load_aligned_checkpoint


def evaluate(
    checkpoint: Path,
    *,
    reference_root: Path,
    speeds: tuple[float, ...],
    steps: int,
    output: Path,
) -> dict[str, object]:
    """Roll out one held-out example for each requested speed."""

    saved, tokenizer, config, params = load_aligned_checkpoint(checkpoint)
    reference = load_projected_reference("test", reference_root)
    sequences = build_periodic_sequences(
        reference, tokenizer, preview_tokens=int(saved["preview_tokens"])
    )
    selected = select_speed_examples(sequences, np.asarray(speeds, np.float32))
    long_reference = periodic_reference_set(
        reference,
        sequences,
        selected,
        frames=steps + 1,
    )
    functional_rollout = saved.get("rl_finetune") is not None
    environment = (
        AlignedLocomotion(reference, sequences, selected)
        if functional_rollout
        else FetchTracking(
            long_reference,
            random_start=False,
            track_frames=steps,
        )
    )
    reset = jax.jit(jax.vmap(environment.reset_to))
    physics_step = jax.jit(jax.vmap(environment.step))
    examples = len(selected)
    state = reset(
        jnp.asarray(selected)
        if functional_rollout
        else jnp.arange(examples, dtype=jnp.int32)
    )
    neuronal_state = initial_state((examples,), config)
    previous_action = jnp.zeros((examples, ACTION_DIM), jnp.float32)
    base = jnp.asarray(sequences.observation[selected])
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])
    target_qpos_device = jnp.asarray(long_reference.qpos[:, : steps + 1])

    def advance(carry, time_index):
        state, neuronal_state, previous_action, alive = carry
        if functional_rollout:
            raw = state.obs
        else:
            raw = base[:, time_index % CYCLE_FRAMES]
            raw = raw.at[:, :FEATURE_DIM].set(state.info["feature"])
            raw = raw.at[:, PREVIOUS_ACTION_SLICE].set(previous_action)
        normalized = jnp.clip((raw - mean) / std, -10.0, 10.0)
        neuronal_state, (logits, spikes) = control_step(
            params, neuronal_state, normalized, config
        )
        action = jnp.tanh(logits)
        state = physics_step(state, action)
        if functional_rollout:
            failed = state.done.astype(bool)
            root_error = jnp.linalg.norm(
                state.pipeline_state.qpos[:, :3]
                - target_qpos_device[:, time_index + 1, :3],
                axis=-1,
            )
            joint_error = jnp.linalg.norm(
                joint_angles(state.pipeline_state.qpos)
                - joint_angles(target_qpos_device[:, time_index + 1]),
                axis=-1,
            )
            speed = state.metrics["speed"]
            track = state.metrics["track"]
            upright = state.metrics["upright"]
        else:
            failed = state.done.astype(bool) & ~state.metrics["completed"].astype(bool)
            root_error = state.metrics["root_error"]
            joint_error = state.metrics["joint_error"]
            speed = jnp.zeros((examples,))
            track = jnp.zeros((examples,))
            upright = jnp.zeros((examples,))
        alive = alive & ~failed
        return (state, neuronal_state, action, alive), (
            state.pipeline_state.qpos,
            action,
            spikes,
            state.info["feature"],
            root_error,
            joint_error,
            speed,
            track,
            upright,
            state.done,
            alive,
        )

    (_, _, _, final_alive), stream = jax.lax.scan(
        advance,
        (
            state,
            neuronal_state,
            previous_action,
            jnp.ones((examples,), bool),
        ),
        jnp.arange(steps),
    )
    (
        qpos,
        action,
        spikes,
        feature,
        root_error,
        joint_error,
        measured_speed,
        track,
        upright,
        done,
        alive,
    ) = map(np.asarray, jax.device_get(stream))
    (
        qpos,
        action,
        feature,
        root_error,
        joint_error,
        measured_speed,
        track,
        upright,
        done,
        alive,
    ) = [
        np.swapaxes(value, 0, 1)
        for value in (
            qpos,
            action,
            feature,
            root_error,
            joint_error,
            measured_speed,
            track,
            upright,
            done,
            alive,
        )
    ]
    spikes = spikes.transpose(2, 0, 1, 3)
    qpos = np.concatenate(
        (np.asarray(state.pipeline_state.qpos)[:, None], qpos), axis=1
    )
    target_qpos = long_reference.qpos[:, : steps + 1]
    actual_speed = np.asarray(
        [
            measured_speed[index, alive[index]].mean() if np.any(alive[index]) else 0.0
            for index in range(examples)
        ],
        np.float32,
    )
    target_speed = sequences.speed[selected]
    contacts = feature[..., slice(*SL["contacts"])] >= 0.5
    contact_switches = np.sum(contacts[:, 1:] != contacts[:, :-1], axis=1)
    counts = spikes.sum(axis=2)
    quarter = steps // 4
    rows = []
    for index, clip in enumerate(selected):
        alive_mask = alive[index]
        alive_count = max(int(alive_mask.sum()), 1)
        rows.append(
            {
                "requested_speed_fetch_units_per_s": float(speeds[index]),
                "reference_speed_fetch_units_per_s": float(target_speed[index]),
                "realized_speed_fetch_units_per_s": float(actual_speed[index]),
                "test_clip": int(clip),
                "cycle_start": int(sequences.cycle_start[clip]),
                "wrap_score": float(sequences.wrap_score[clip]),
                "physics_nonterminated_all_steps": bool(final_alive[index]),
                "physics_nontermination_fraction": float(alive_mask.mean()),
                "root_error_mean_alive": float(
                    (root_error[index] * alive_mask).sum() / alive_count
                ),
                "joint_rmse_mean_alive_rad": float(
                    (joint_error[index] / np.sqrt(ACTION_DIM) * alive_mask).sum()
                    / alive_count
                ),
                "task_track_mean_alive": float(
                    (track[index] * alive_mask).sum() / alive_count
                ),
                "upright_mean_alive": float(
                    (upright[index] * alive_mask).sum() / alive_count
                ),
                "contact_switches_per_foot": contact_switches[index].tolist(),
                "first_quarter_rate_hz": float(counts[index, :quarter].mean() * FPS),
                "last_quarter_rate_hz": float(counts[index, -quarter:].mean() * FPS),
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        target_qpos=target_qpos.astype(np.float32),
        action=action.astype(np.float32),
        spikes_5ms=spikes.astype(np.uint8),
        spike_counts_20ms=counts.astype(np.uint8),
        feature=feature.astype(np.float32),
        measured_speed=measured_speed.astype(np.float32),
        track=track.astype(np.float32),
        upright=upright.astype(np.float32),
        alive=alive.astype(np.uint8),
        done=done.astype(np.uint8),
        selected_clip=selected,
        requested_speed=np.asarray(speeds, np.float32),
        reference_speed=target_speed.astype(np.float32),
        realized_speed=actual_speed.astype(np.float32),
    )
    report = {
        "schema": "demo-j-aligned-snn-periodic-rollout-v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "steps": steps,
        "seconds": steps / FPS,
        "episodes": rows,
        "physics_nontermination_fraction": float(np.mean(final_alive)),
        "termination_metric_scope": (
            "coarse environment termination only; nontermination is not successful "
            "locomotion"
        ),
        "functional_success_claimed": False,
        "mean_firing_rate_hz": float(counts.mean() * FPS),
        "silent_neuron_fraction": float(np.mean(counts.sum(axis=(0, 1)) == 0)),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.99)),
        "rollout_task": (
            "functional speed/upright locomotion"
            if functional_rollout
            else "periodic reference tracking"
        ),
        "reference_kind": "explicit synthetic periodic token stream",
        "continuity_claim": False,
    }
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return report


def record(
    checkpoint: Path,
    trace: Path,
    *,
    steps: int,
    output: Path,
) -> dict[str, object]:
    """Record aligned SNN spikes on a saved long Demo H trajectory bank."""

    saved, tokenizer, config, params = load_aligned_checkpoint(checkpoint)
    preview_tokens = int(saved["preview_tokens"])
    preview_frames = preview_tokens * TOKEN_FRAMES
    with np.load(trace) as archive:
        h_observation = np.asarray(archive["observation"], np.float32)
        feature = np.asarray(archive["feature"], np.float32)
        target_speed = np.asarray(archive["target_speed"], np.float32)
        reset_seed = np.asarray(archive["reset_seed"], np.int32)
    episodes, frames = feature.shape[:2]
    if frames < steps + preview_frames:
        raise ValueError(
            f"trace has {frames} frames; {steps + preview_frames} required"
        )
    previews = []
    for token_offset in range(preview_tokens):
        offsets = (
            np.arange(steps)[:, None]
            + 1
            + token_offset * TOKEN_FRAMES
            + np.arange(TOKEN_FRAMES)[None]
        )
        previews.append(tokenizer.encode(feature[:, offsets]))
    preview = np.concatenate(previews, axis=-1)
    phase = h_observation[:, :steps, PHASE_SLICE]
    command = h_observation[:, :steps, COMMAND_SLICE]
    static = np.concatenate(
        (
            feature[:, :steps],
            np.zeros((episodes, steps, ACTION_DIM), np.float32),
            preview,
            phase,
            command,
        ),
        axis=-1,
    ).astype(np.float32)
    if static.shape[-1] != aligned_input_dim(preview_tokens):
        raise ValueError(static.shape)
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])
    neuronal_state = initial_state((episodes,), config)

    def advance(carry, raw):
        neuronal_state, previous_action = carry
        raw = raw.at[:, PREVIOUS_ACTION_SLICE].set(previous_action)
        normalized = jnp.clip((raw - mean) / std, -10.0, 10.0)
        neuronal_state, (logits, spikes) = control_step(
            params, neuronal_state, normalized, config
        )
        action = jnp.tanh(logits)
        return (neuronal_state, action), (raw, action, spikes)

    (_, _), (behavior, action, spikes) = jax.lax.scan(
        advance,
        (neuronal_state, jnp.zeros((episodes, ACTION_DIM), jnp.float32)),
        jnp.asarray(static).swapaxes(0, 1),
    )
    behavior = np.asarray(behavior).transpose(1, 0, 2)
    action = np.asarray(action).transpose(1, 0, 2)
    spikes = np.asarray(spikes).transpose(2, 0, 1, 3)
    counts = spikes.sum(axis=2).astype(np.uint8)
    input_norm = np.linalg.norm(
        np.asarray(saved["params"].input_weight), axis=0
    ).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        behavior=behavior.astype(np.float32),
        action=action.astype(np.float32),
        spikes_5ms=spikes.astype(np.uint8),
        spike_counts_20ms=counts,
        input_weight_norm=input_norm,
        adaptive_neuron=(np.arange(config.neurons) < config.adaptive_neurons).astype(
            np.uint8
        ),
        target_speed=target_speed,
        reset_seed=reset_seed,
    )
    normalized = (behavior - np.asarray(mean)) / np.asarray(std)
    report = {
        "schema": "demo-j-aligned-fixed-trajectory-recording-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "snn_seed": int(saved["seed"]),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "episodes": episodes,
        "bins_per_episode": steps,
        "clock_ms": 20,
        "preview_tokens": preview_tokens,
        "preview_milliseconds": preview_frames * 1_000 / FPS,
        "mean_firing_rate_hz": float(counts.mean() * FPS),
        "silent_neuron_fraction": float(np.mean(counts.sum(axis=(0, 1)) == 0)),
        "normalization_clip_fraction": float(np.mean(np.abs(normalized) > 10)),
        "behavior_array_is_exact_raw_snn_input": True,
        "snn_previous_action_autoregressive": True,
        "recurrent_state_reset_only_at_episode_boundary": True,
        "demo_h_policy_used_for_snn_training": False,
    }
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    evaluate_parser.add_argument(
        "--speeds", type=float, nargs=6, default=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    )
    evaluate_parser.add_argument("--steps", type=int, default=1000)
    evaluate_parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_ROOT / "rollout_1000.npz"
    )

    record_parser = commands.add_parser("record")
    record_parser.add_argument("--checkpoint", type=Path, required=True)
    record_parser.add_argument("--trace", type=Path, required=True)
    record_parser.add_argument("--steps", type=int, default=1000)
    record_parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_ROOT / "fixed_trace.npz"
    )

    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(
            args.checkpoint,
            reference_root=args.reference_root,
            speeds=tuple(args.speeds),
            steps=args.steps,
            output=args.output,
        )
    else:
        record(args.checkpoint, args.trace, steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()
