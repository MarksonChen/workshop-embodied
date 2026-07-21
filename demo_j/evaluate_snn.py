"""Evaluate a distilled Demo J SNN in closed-loop modern MJX physics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_j.config import SNNConfig
from demo_j.dataset import take_references
from demo_j.env import LAST_TRACK_FRAME, FetchTracking
from demo_j.policy import policy_sequence
from demo_j.projection import DEFAULT_ROOT as PROJECTED_ROOT
from demo_j.projection import load_projected_reference
from demo_j.snn import initial_state


def evaluate(
    checkpoint: Path,
    split: str,
    max_clips: int | None,
    output: Path,
    reference_root: Path = PROJECTED_ROOT,
):
    with Path(checkpoint).open("rb") as stream:
        saved = pickle.load(stream)
    if saved.get("schema") != "demo-j-snn-distillation-v1":
        raise ValueError(f"unsupported checkpoint schema {saved.get('schema')!r}")
    reference = load_projected_reference(split, reference_root)
    if max_clips is not None:
        reference = take_references(
            reference, np.arange(min(max_clips, reference.clips))
        )
    environment = FetchTracking(reference, random_start=False)
    reset = jax.jit(jax.vmap(environment.reset_to))
    physics_step = jax.jit(jax.vmap(environment.step))
    clips = reference.clips
    state = reset(jnp.arange(clips, dtype=jnp.int32))
    neuronal_state = initial_state((clips,), SNNConfig(**saved["config"]))
    params = jax.tree.map(jnp.asarray, saved["params"])
    mean = jnp.asarray(saved["observation_mean"])
    std = jnp.asarray(saved["observation_std"])
    config = SNNConfig(**saved["config"])

    def advance(carry, _):
        state, neuronal_state, alive = carry
        observation = jnp.clip((state.obs - mean) / std, -10.0, 10.0)
        neuronal_state, (action, spikes) = policy_sequence(
            params, neuronal_state, observation[None], config
        )
        action = action[0]
        state = physics_step(state, action)
        failed = state.done.astype(bool) & ~state.metrics["completed"].astype(bool)
        alive = alive & ~failed
        return (state, neuronal_state, alive), (
            state.pipeline_state.qpos,
            action,
            spikes[0],
            state.reward,
            state.done,
            state.metrics["root_error"],
            state.metrics["joint_error"],
            state.metrics["foot_error"],
            alive,
        )

    (state, neuronal_state, alive), stream = jax.lax.scan(
        advance,
        (state, neuronal_state, jnp.ones((clips,), bool)),
        xs=None,
        length=LAST_TRACK_FRAME,
    )
    del state, neuronal_state
    (
        qpos,
        action,
        spikes,
        reward,
        done,
        root_error,
        joint_error,
        foot_error,
        alive_history,
    ) = jax.device_get(stream)
    qpos = np.swapaxes(qpos, 0, 1)
    qpos = np.concatenate((reference.qpos[:, :1], qpos), axis=1)
    action = np.swapaxes(action, 0, 1)
    spikes = np.swapaxes(spikes, 0, 1).astype(np.uint8)
    reward, done, root_error, joint_error, foot_error, alive_history = map(
        lambda value: np.swapaxes(value, 0, 1),
        (reward, done, root_error, joint_error, foot_error, alive_history),
    )
    completed = np.asarray(alive)
    action_mse = np.mean(
        np.square(action - reference.teacher_action[:, :LAST_TRACK_FRAME]),
        axis=(1, 2),
    )
    report = {
        "schema": "demo-j-snn-closed-loop-evaluation-v1",
        "checkpoint": str(checkpoint),
        "split": split,
        "clips": clips,
        "completion_fraction": float(completed.mean()),
        "action_mse_median": float(np.median(action_mse)),
        "root_error_median_per_frame": float(np.median(root_error)),
        "joint_rmse_median_per_frame_rad": float(
            np.median(joint_error / np.sqrt(10.0))
        ),
        "foot_rmse_median_per_frame": float(
            np.median(foot_error / np.sqrt(12.0))
        ),
        "return_median": float(np.median(reward.sum(axis=1))),
        "spike_probability_per_5ms": float(spikes.mean()),
        "mean_firing_rate_hz": float(spikes.mean() / (config.step_ms / 1000.0)),
        "silent_neuron_fraction": float(
            np.mean(spikes.sum(axis=(0, 1, 2)) == 0)
        ),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.99)),
        "finite": bool(
            np.isfinite(qpos).all()
            and np.isfinite(action).all()
            and np.isfinite(reward).all()
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        action=action.astype(np.float32),
        spikes_5ms=spikes,
        spike_counts_20ms=spikes.sum(axis=2).astype(np.uint8),
        reward=reward.astype(np.float32),
        done=done.astype(np.uint8),
        alive=alive_history.astype(np.uint8),
        clip=np.arange(clips, dtype=np.int32),
        session_index=reference.session_index,
        source_frame=reference.source_frame,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--reference-root", type=Path, default=PROJECTED_ROOT)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "out" / "snn_closed_loop.npz"
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.split,
        args.max_clips,
        args.output,
        args.reference_root,
    )


if __name__ == "__main__":
    main()
