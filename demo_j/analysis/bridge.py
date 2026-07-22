"""Export fixed Demo H trajectories and residual-policy activations.

This module intentionally runs in Demo H's legacy JAX/Brax environment.  It
only writes versioned NumPy files, which form the explicit runtime boundary to
the modern BrainPy/MJX Demo J environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from demo_j.artifacts import OUTPUT_ROOT, REPOSITORY_ROOT, sha256
from demo_j.analysis.contracts import checkpoint_contract


DEFAULT_PRIOR = REPOSITORY_ROOT / "demo_h" / "out" / "prior_retime_1p75_jax.npz"


def record_trace(
    driver_checkpoint: Path,
    prior_path: Path,
    output: Path,
    *,
    target_speeds: tuple[float, ...],
    repeats: int,
    steps: int,
    require_sweep: bool,
) -> dict[str, object]:
    """Record one healthy Demo H rollout bank used by every beta policy."""

    from demo_h.config import FEATURE_BUFFER_SLICE, FEATURE_DIM
    from demo_h.evaluate import _set_target_speeds, make_actor, make_environment
    from demo_h.prior import load_prior

    driver_report = checkpoint_contract(driver_checkpoint, require_sweep=require_sweep)
    if driver_report["arm"] != "h1":
        raise ValueError("the fixed-trajectory driver must be the beta=0 H1 arm")
    if repeats < 1 or steps < 8:
        raise ValueError((repeats, steps))
    speeds = np.repeat(np.asarray(target_speeds, np.float32), repeats)
    seeds = np.arange(10_001, 10_001 + len(speeds), dtype=np.int32)
    prior = load_prior(prior_path)
    environment = make_environment(prior, float(speeds[0]))
    action_fn, _ = make_actor("h1", driver_checkpoint, prior)
    reset_keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
    initial = environment.reset(reset_keys)
    initial = _set_target_speeds(environment, prior, initial, speeds)
    legacy_system = environment.env.env._env.sys

    def rollout(initial_state):
        def step(carry, _):
            state, key = carry
            key, action_key = jax.random.split(key)
            qp = state.pipeline_state.qp
            angles, angle_velocities = jax.vmap(legacy_system.joints[0].angle_vel)(qp)
            action = action_fn(state.obs, action_key)
            feature = state.obs[..., FEATURE_BUFFER_SLICE].reshape(
                (len(speeds), -1, FEATURE_DIM)
            )[:, -1]
            output_row = (
                state.obs,
                feature,
                action,
                qp.pos[:, 0],
                qp.rot[:, 0],
                angles,
                angle_velocities,
                state.done,
                state.metrics["speed"],
                state.metrics["upright"],
            )
            return (environment.step(state, action), key), output_row

        return jax.lax.scan(
            step,
            (initial_state, jax.random.PRNGKey(91_337)),
            xs=None,
            length=steps,
        )[1]

    stream = jax.jit(rollout)(initial)
    jax.block_until_ready(stream[0])
    (
        observation,
        feature,
        action,
        root_position,
        root_quaternion,
        joint_angles,
        joint_velocities,
        done,
        measured_speed,
        upright,
    ) = [np.swapaxes(np.asarray(value), 0, 1) for value in stream]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observation=observation.astype(np.float32),
        feature=feature.astype(np.float32),
        driver_action=action.astype(np.float32),
        root_position=root_position.astype(np.float32),
        root_quaternion=root_quaternion.astype(np.float32),
        joint_angles=joint_angles.astype(np.float32),
        joint_velocities=joint_velocities.astype(np.float32),
        done=done.astype(np.uint8),
        measured_speed=measured_speed.astype(np.float32),
        upright=upright.astype(np.float32),
        target_speed=speeds,
        reset_seed=seeds,
    )
    warmup = min(32, steps // 4)
    report = {
        "schema": "demo-j-fixed-demo-h-trace-v1",
        "runtime": "legacy Demo H JAX/Brax",
        "driver_checkpoint": str(driver_checkpoint),
        "driver_checkpoint_sha256": sha256(driver_checkpoint),
        "driver_seed": int(driver_report["seed"]),
        "prior": str(prior_path),
        "prior_sha256": prior.artifact_sha256,
        "episodes": int(len(speeds)),
        "steps": int(steps),
        "target_speeds": list(map(float, target_speeds)),
        "repeats_per_speed": int(repeats),
        "warmup_bins_for_summary": int(warmup),
        "survival_fraction": float(np.mean(done == 0)),
        "mean_speed_after_warmup": float(measured_speed[:, warmup:].mean()),
        "minimum_upright_after_warmup": float(upright[:, warmup:].min()),
        "fixed_input_contract": (
            "all beta policies receive the identical saved 1094-D observations"
        ),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def extract_activations(
    trace: Path,
    checkpoint: Path,
    prior_path: Path,
    output: Path,
    *,
    require_sweep: bool,
) -> dict[str, object]:
    """Evaluate one residual actor on saved observations without simulation."""

    from demo_h.artifacts import load_policy_checkpoint
    from demo_h.policy import (
        BoundedResidualMLP,
        RESIDUAL_MEAN_SCALE,
        frozen_context,
        make_residual_ppo_networks,
    )
    from demo_h.prior import load_prior

    report = checkpoint_contract(checkpoint, require_sweep=require_sweep)
    prior = load_prior(prior_path)
    params, envelope = load_policy_checkpoint(
        checkpoint,
        expected_arm=report["arm"],
        expected_prior_sha256=prior.artifact_sha256,
    )
    with np.load(trace) as archive:
        observation = np.asarray(archive["observation"], np.float32)
        target_speed = np.asarray(archive["target_speed"], np.float32)
    leading = observation.shape[:2]
    flat = jnp.asarray(observation.reshape((-1, observation.shape[-1])))
    residual_module = BoundedResidualMLP()

    def apply(values):
        base_mean, _, compact = frozen_context(prior, values)
        residual, captured = residual_module.apply(
            params[1],
            compact,
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        hidden_1 = jax.nn.silu(captured["intermediates"]["Dense_0"]["__call__"][0])
        hidden_2 = jax.nn.silu(captured["intermediates"]["Dense_1"]["__call__"][0])
        delta_mean, delta_scale = jnp.split(residual, 2, axis=-1)
        correction = RESIDUAL_MEAN_SCALE * jnp.tanh(delta_mean)
        policy_mean = base_mean + correction
        return hidden_1, hidden_2, correction, delta_scale, policy_mean

    arrays = [np.asarray(value) for value in apply(flat)]
    network = make_residual_ppo_networks((observation.shape[-1],), 10, prior=prior)
    expected_all = np.asarray(network.policy_network.apply(params[0], params[1], flat))[
        ..., :10
    ]
    expected = expected_all.reshape(leading + (-1,))[0, :8]
    extracted = arrays[-1].reshape(leading + (-1,))[0, :8]
    np.testing.assert_allclose(expected, extracted, atol=2e-6)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = (
        "hidden_1",
        "hidden_2",
        "residual_mean_correction",
        "raw_scale_correction",
        "policy_mean",
    )
    np.savez_compressed(
        output,
        **{
            name: value.reshape(leading + (value.shape[-1],)).astype(np.float32)
            for name, value in zip(names, arrays, strict=True)
        },
        target_speed=target_speed,
    )
    result = {
        "schema": "demo-j-fixed-demo-h-activations-v1",
        "sweep_id": report.get("sweep_id"),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_envelope": envelope,
        "arm": report["arm"],
        "beta": float(report["beta"]),
        "seed": int(report["seed"]),
        "prior_sha256": prior.artifact_sha256,
        "episodes": int(leading[0]),
        "steps": int(leading[1]),
        "manual_activation_parity_max_abs": float(np.max(np.abs(expected - extracted))),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("--driver-checkpoint", type=Path, required=True)
    trace_parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    trace_parser.add_argument(
        "--output", type=Path, default=OUTPUT_ROOT / "h_fixed_trace.npz"
    )
    trace_parser.add_argument(
        "--target-speeds",
        type=float,
        nargs="+",
        default=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    )
    trace_parser.add_argument("--repeats", type=int, default=5)
    trace_parser.add_argument("--steps", type=int, default=256)
    trace_parser.add_argument("--allow-pilot", action="store_true")

    activation_parser = subparsers.add_parser("activations")
    activation_parser.add_argument("--trace", type=Path, required=True)
    activation_parser.add_argument("--checkpoint", type=Path, required=True)
    activation_parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    activation_parser.add_argument("--output", type=Path, required=True)
    activation_parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    if args.command == "trace":
        record_trace(
            args.driver_checkpoint,
            args.prior,
            args.output,
            target_speeds=tuple(args.target_speeds),
            repeats=args.repeats,
            steps=args.steps,
            require_sweep=not args.allow_pilot,
        )
    else:
        extract_activations(
            args.trace,
            args.checkpoint,
            args.prior,
            args.output,
            require_sweep=not args.allow_pilot,
        )


if __name__ == "__main__":
    main()
