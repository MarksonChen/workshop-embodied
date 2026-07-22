"""Evaluate native clips and record matched finite-trial SNN spike streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_f.config import FPS
from demo_f.features import SL
from demo_h.config import COMMAND_SLICE
from demo_j.artifacts import ALIGNED_OUTPUT_ROOT, sha256, write_json
from demo_j.control.aligned import (
    ACTION_DIM,
    FEATURE_DIM,
    PREVIOUS_ACTION_SLICE,
    build_clip_sequences,
    clip_observations,
    select_speed_examples,
)
from demo_j.control.snn import control_step, initial_state
from demo_j.control.tracking import FetchTracking
from demo_j.data.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.data.projection import load_projected_reference
from demo_j.experiments.aligned import load_clip_checkpoint


def _forward_speed(qpos: np.ndarray) -> np.ndarray:
    """Measure forward speed along each clip's initial heading."""

    quaternion = qpos[:, 0, 3:7]
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    heading = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
    velocity = np.diff(qpos[..., :2], axis=1) * FPS
    return np.sum(velocity * heading[:, None], axis=-1).astype(np.float32)


def evaluate(
    checkpoint: Path,
    *,
    reference_root: Path,
    speeds: tuple[float, ...],
    output: Path,
) -> dict[str, object]:
    """Audit every test clip and retain six examples for visualization."""

    saved, tokenizer, config, params = load_clip_checkpoint(checkpoint)
    reference = load_projected_reference("test", reference_root)
    sequences = build_clip_sequences(
        reference, tokenizer, preview_tokens=int(saved["preview_tokens"])
    )
    selected = select_speed_examples(sequences, np.asarray(speeds, np.float32))
    steps = sequences.steps
    if steps != int(saved["episode_steps"]):
        raise ValueError((steps, saved["episode_steps"]))

    environment = FetchTracking(
        reference,
        random_start=False,
        track_frames=steps,
    )
    reset = jax.jit(jax.vmap(environment.reset_to))
    physics_step = jax.jit(jax.vmap(environment.step))
    examples = reference.clips
    state = reset(jnp.arange(examples, dtype=jnp.int32))
    neuronal_state = initial_state((examples,), config)
    previous_action = jnp.zeros((examples, ACTION_DIM), jnp.float32)
    template = jnp.asarray(sequences.observation)
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])

    def advance(carry, time_index):
        state, neuronal_state, previous_action, alive = carry
        raw = template[:, time_index]
        raw = raw.at[:, :FEATURE_DIM].set(state.info["feature"])
        raw = raw.at[:, PREVIOUS_ACTION_SLICE].set(previous_action)
        normalized = jnp.clip((raw - mean) / std, -10.0, 10.0)
        neuronal_state, (logits, spikes) = control_step(
            params, neuronal_state, normalized, config
        )
        action = jnp.tanh(logits)
        state = physics_step(state, action)
        failed = state.done.astype(bool) & ~state.metrics["completed"].astype(bool)
        alive = alive & ~failed
        return (state, neuronal_state, action, alive), (
            state.pipeline_state.qpos,
            action,
            spikes,
            state.info["feature"],
            state.metrics["root_error"],
            state.metrics["joint_error"],
            state.metrics["foot_error"],
            state.done,
            alive,
        )

    (_, _, _, final_alive), stream = jax.lax.scan(
        advance,
        (state, neuronal_state, previous_action, jnp.ones((examples,), bool)),
        jnp.arange(steps),
    )
    (
        qpos,
        action,
        spikes,
        feature,
        root_error,
        joint_error,
        foot_error,
        done,
        alive,
    ) = map(np.asarray, jax.device_get(stream))
    qpos, action, feature, root_error, joint_error, foot_error, done, alive = [
        np.swapaxes(value, 0, 1)
        for value in (
            qpos,
            action,
            feature,
            root_error,
            joint_error,
            foot_error,
            done,
            alive,
        )
    ]
    spikes = spikes.transpose(2, 0, 1, 3)
    qpos = np.concatenate((reference.qpos[:, :1], qpos), axis=1)
    measured_speed = _forward_speed(qpos)
    realized_speed = np.asarray(
        [
            measured_speed[index, alive[index]].mean() if np.any(alive[index]) else 0.0
            for index in range(examples)
        ],
        np.float32,
    )
    target_speed = sequences.speed
    contacts = feature[..., slice(*SL["contacts"])] >= 0.5
    contact_switches = np.sum(contacts[:, 1:] != contacts[:, :-1], axis=1)
    target_contacts = reference.features[:, 1:, slice(*SL["contacts"])] >= 0.5
    target_contact_switches = np.sum(
        target_contacts[:, 1:] != target_contacts[:, :-1], axis=1
    )
    counts = spikes.sum(axis=2)
    alive_count = np.maximum(alive.sum(axis=1), 1)
    root_error_mean = (root_error * alive).sum(axis=1) / alive_count
    joint_rmse_mean = (joint_error / np.sqrt(ACTION_DIM) * alive).sum(
        axis=1
    ) / alive_count
    foot_rmse_mean = (foot_error / np.sqrt(12.0) * alive).sum(axis=1) / alive_count
    positive_speed = target_speed > 0.05
    rows = []
    for requested, clip in zip(speeds, selected, strict=True):
        rows.append(
            {
                "requested_speed_fetch_units_per_s": float(requested),
                "reference_speed_fetch_units_per_s": float(target_speed[clip]),
                "realized_speed_fetch_units_per_s": float(realized_speed[clip]),
                "test_clip": int(clip),
                "completed_native_clip": bool(final_alive[clip]),
                "completion_fraction": float(alive[clip].mean()),
                "root_error_mean": float(root_error_mean[clip]),
                "joint_rmse_mean_rad": float(joint_rmse_mean[clip]),
                "foot_rmse_mean": float(foot_rmse_mean[clip]),
                "contact_switches_per_foot": contact_switches[clip].tolist(),
                "reference_contact_switches_per_foot": target_contact_switches[
                    clip
                ].tolist(),
            }
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos[selected].astype(np.float32),
        target_qpos=reference.qpos[selected].astype(np.float32),
        action=action[selected].astype(np.float32),
        spikes_5ms=spikes[selected].astype(np.uint8),
        spike_counts_20ms=counts[selected].astype(np.uint8),
        feature=feature[selected].astype(np.float32),
        measured_speed=measured_speed[selected].astype(np.float32),
        alive=alive[selected].astype(np.uint8),
        done=done[selected].astype(np.uint8),
        selected_clip=selected,
        requested_speed=np.asarray(speeds, np.float32),
        reference_speed=target_speed[selected].astype(np.float32),
        realized_speed=realized_speed[selected].astype(np.float32),
    )
    report = {
        "schema": "demo-j-native-clip-rollout-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "steps": steps,
        "seconds": steps / FPS,
        "episodes": rows,
        "selection_rule": "nearest distinct positive-speed native test clips",
        "selected_completion_fraction": float(np.mean(final_alive[selected])),
        "full_test_audit": {
            "clips": reference.clips,
            "completion_fraction": float(np.mean(final_alive)),
            "median_completion_fraction": float(np.median(alive.mean(axis=1))),
            "median_root_error": float(np.median(root_error_mean)),
            "median_joint_rmse_rad": float(np.median(joint_rmse_mean)),
            "median_foot_rmse": float(np.median(foot_rmse_mean)),
            "median_forward_speed_absolute_error": float(
                np.median(
                    np.abs(
                        realized_speed[positive_speed] - target_speed[positive_speed]
                    )
                )
            ),
            "positive_speed_clips": int(positive_speed.sum()),
        },
        "mean_firing_rate_hz": float(counts.mean() * FPS),
        "silent_neuron_fraction": float(np.mean(counts.sum(axis=(0, 1)) == 0)),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.99)),
        "reference_kind": "finite native 64-frame held-out clips",
        "recurrent_state_reset_at_clip_boundary": True,
        "periodic_extension_used": False,
    }
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))
    return report


def record(
    checkpoint: Path,
    trace: Path,
    *,
    output: Path,
) -> dict[str, object]:
    """Record one native-length SNN trial from each fixed Demo H episode."""

    saved, tokenizer, config, params = load_clip_checkpoint(checkpoint)
    preview_tokens = int(saved["preview_tokens"])
    action_steps = int(saved["episode_steps"])
    recording_bins = action_steps + 1
    with np.load(trace) as archive:
        h_observation = np.asarray(archive["observation"], np.float32)
        feature = np.asarray(archive["feature"], np.float32)
        driver_action = np.asarray(archive["driver_action"], np.float32)
        target_speed = np.asarray(archive["target_speed"], np.float32)
        reset_seed = np.asarray(archive["reset_seed"], np.int32)
    episodes, frames = feature.shape[:2]
    if frames < recording_bins:
        raise ValueError(f"trace has {frames} frames; {recording_bins} required")
    previous_action = np.concatenate(
        (
            np.zeros((episodes, 1, ACTION_DIM), np.float32),
            driver_action[:, :action_steps],
        ),
        axis=1,
    )
    static, preview_mask = clip_observations(
        feature[:, :recording_bins],
        previous_action,
        h_observation[:, :recording_bins, COMMAND_SLICE],
        tokenizer,
        preview_tokens=preview_tokens,
    )
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])

    def advance(carry, raw):
        neuronal_state, previous = carry
        raw = raw.at[:, PREVIOUS_ACTION_SLICE].set(previous)
        normalized = jnp.clip((raw - mean) / std, -10.0, 10.0)
        neuronal_state, (logits, spikes) = control_step(
            params, neuronal_state, normalized, config
        )
        action = jnp.tanh(logits)
        return (neuronal_state, action), (raw, action, spikes)

    (_, _), (behavior, action, spikes) = jax.lax.scan(
        advance,
        (initial_state((episodes,), config), jnp.zeros((episodes, ACTION_DIM))),
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
        action_applied_to_physics=np.concatenate(
            (np.ones(action_steps, np.uint8), np.zeros(1, np.uint8))
        ),
        spikes_5ms=spikes.astype(np.uint8),
        spike_counts_20ms=counts,
        preview_mask=preview_mask.astype(np.uint8),
        input_weight_norm=input_norm,
        adaptive_neuron=(np.arange(config.neurons) < config.adaptive_neurons).astype(
            np.uint8
        ),
        target_speed=target_speed,
        reset_seed=reset_seed,
    )
    normalized = (behavior - np.asarray(mean)) / np.asarray(std)
    report = {
        "schema": "demo-j-native-clip-fixed-trajectory-recording-v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "snn_seed": int(saved["seed"]),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "episodes": episodes,
        "bins_per_episode": recording_bins,
        "state_frames_per_episode": recording_bins,
        "physical_actions_per_native_clip": action_steps,
        "terminal_probe_action_used": False,
        "clock_ms": 20,
        "preview_tokens": preview_tokens,
        "valid_preview_fraction": float(preview_mask.mean()),
        "mean_firing_rate_hz": float(counts.mean() * FPS),
        "silent_neuron_fraction": float(np.mean(counts.sum(axis=(0, 1)) == 0)),
        "normalization_clip_fraction": float(np.mean(np.abs(normalized) > 10)),
        "behavior_array_is_exact_raw_snn_input": True,
        "activity_alignment": "one SNN update at each of 64 saved state frames",
        "recurrent_state_reset_at_each_native_length_trial": True,
        "periodic_extension_used": False,
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
    evaluate_parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_ROOT / "native_clip_rollout.npz"
    )

    record_parser = commands.add_parser("record")
    record_parser.add_argument("--checkpoint", type=Path, required=True)
    record_parser.add_argument("--trace", type=Path, required=True)
    record_parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_ROOT / "native_fixed_trace.npz"
    )

    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(
            args.checkpoint,
            reference_root=args.reference_root,
            speeds=tuple(args.speeds),
            output=args.output,
        )
    else:
        record(args.checkpoint, args.trace, output=args.output)


if __name__ == "__main__":
    main()
