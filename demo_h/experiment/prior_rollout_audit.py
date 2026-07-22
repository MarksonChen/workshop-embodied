"""Frozen closed-loop acceptance metric for Demo H prior iteration.

Run in the pinned legacy Brax environment.  The audit starts every held-out
test clip at frame 15, rolls the prior without reference states, and compares
both its native-horizon trajectory and its long-run occupancy with the exact
physics demonstrations.  No diagnostic in this file is a training reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.v1.envs import fetch

from demo_f.commands import hindsight_command
from demo_f.features import SL
from demo_f.metrics import GAIT_FIELDS, gait_statistics
from demo_h.config import COMMAND_HORIZON_SECONDS
from demo_h.dataset.contract import DEFAULT_ROOT
from demo_h.dataset.loader import load_split
from demo_h.evaluate_physics import make_rollout
from demo_h.prior import DEFAULT_PRIOR, load_prior


START_FRAME = 15
NATIVE_STEPS = 48
DEFAULT_LONG_STEPS = 250
PROJECTIONS = 64


def _initial_qps(root: Path, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text())
    names = (
        "initial_qp_pos",
        "initial_qp_rot",
        "initial_qp_vel",
        "initial_qp_ang",
    )
    parts = {name: [] for name in names}
    for row in manifest["sessions"]:
        if row["split"] != split or not row["released_clips"]:
            continue
        with np.load(root / row["shard"]) as archive:
            for name in names:
                parts[name].append(np.asarray(archive[name], np.float32))
    return {name: np.concatenate(values) for name, values in parts.items()}


def _warm_start(system, initial, controls):
    def one(qp, actions):
        def step(state, action):
            state, _ = system.step(state, action)
            return state, None

        return jax.lax.scan(step, qp, actions)[0]

    return jax.jit(jax.vmap(one))(initial, controls)


def _sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    projections: int = PROJECTIONS,
    quantiles: int = 257,
) -> float:
    """Deterministic squared sliced-Wasserstein distance."""

    first = np.asarray(first, np.float32).reshape(-1, first.shape[-1])
    second = np.asarray(second, np.float32).reshape(-1, second.shape[-1])
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((first.shape[1], projections)).astype(np.float32)
    directions /= np.maximum(np.linalg.norm(directions, axis=0), 1e-8)
    levels = np.linspace(0.0, 1.0, quantiles)
    first_quantiles = np.quantile(first @ directions, levels, axis=0)
    second_quantiles = np.quantile(second @ directions, levels, axis=0)
    return float(np.mean(np.square(first_quantiles - second_quantiles)))


def _contact_js(first: np.ndarray, second: np.ndarray) -> float:
    probabilities = []
    for contacts in (first, second):
        contacts = np.asarray(contacts, bool).reshape(-1, 4)
        code = np.sum(contacts * (1 << np.arange(4)), axis=-1)
        probability = np.bincount(code, minlength=16).astype(np.float64) + 1e-9
        probabilities.append(probability / probability.sum())
    first_probability, second_probability = probabilities
    midpoint = 0.5 * (first_probability + second_probability)
    return float(
        0.5 * np.sum(first_probability * np.log(first_probability / midpoint))
        + 0.5 * np.sum(second_probability * np.log(second_probability / midpoint))
    )


def _gait_summary(features: np.ndarray) -> dict[str, dict[str, float]]:
    """Summarize complete 64-frame chunks without joining clip boundaries."""

    chunks = features.shape[1] // 64
    values = gait_statistics(
        np.concatenate(
            [features[:, index * 64 : (index + 1) * 64] for index in range(chunks)]
        )
    )
    return {
        name: {"mean": float(values[name].mean()), "std": float(values[name].std())}
        for name in GAIT_FIELDS
    }


def evaluate(
    prior_path: Path,
    *,
    dataset_root: Path,
    steps: int,
    output: Path,
    save_trace: bool = False,
) -> dict:
    if steps < 64:
        raise ValueError("long-rollout audit needs at least 64 steps")
    prior = load_prior(prior_path)
    test = load_split(
        "test", dataset_root, expected_variant=prior.metadata["dataset_variant"]
    )
    validation = load_split(
        "validation", dataset_root, expected_variant=prior.metadata["dataset_variant"]
    )
    environment = fetch.Fetch()
    count = len(test.features)
    base = environment.sys.default_qp()
    initial_arrays = _initial_qps(Path(dataset_root), "test")
    initial = jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value)[None], count, axis=0), base
    ).replace(
        pos=jnp.asarray(initial_arrays["initial_qp_pos"]),
        rot=jnp.asarray(initial_arrays["initial_qp_rot"]),
        vel=jnp.asarray(initial_arrays["initial_qp_vel"]),
        ang=jnp.asarray(initial_arrays["initial_qp_ang"]),
    )
    initial = _warm_start(
        environment.sys,
        initial,
        jnp.asarray(test.normalized_control[:, :START_FRAME]),
    )
    command = hindsight_command(
        test.root_position,
        test.root_quaternion,
        start=START_FRAME,
        future=START_FRAME + 31,
    )
    rollout = jax.jit(jax.vmap(make_rollout(environment.sys, prior, steps)))
    stream = rollout(
        initial,
        jnp.asarray(test.features[:, : START_FRAME + 1]),
        jnp.asarray(test.normalized_control[:, START_FRAME - 1]),
        jnp.asarray(command),
    )
    jax.block_until_ready(stream[1])
    qps, controls, contacts, _, features = jax.device_get(stream)
    features = np.asarray(features, np.float32)
    controls = np.asarray(controls, np.float32)
    contacts = np.asarray(contacts, np.uint8)

    feature_mean = np.asarray(prior.feature_mean)
    feature_std = np.asarray(prior.feature_std)
    generated_z = (features - feature_mean) / feature_std
    test_z = (test.features - feature_mean) / feature_std
    validation_z = (validation.features - feature_mean) / feature_std
    native_target = test_z[:, START_FRAME + 1 :]
    native_generated = generated_z[:, :NATIVE_STEPS]
    initial_z = test_z[:, START_FRAME : START_FRAME + 1]
    native_mse = float(np.mean(np.square(native_generated - native_target)))
    persistence_mse = float(np.mean(np.square(initial_z - native_target)))
    native_action_target = test.normalized_control[:, START_FRAME:]
    native_action_mse = float(
        np.mean(np.square(controls[:, :NATIVE_STEPS] - native_action_target))
    )

    motion_slice = slice(0, SL["contacts"][0])
    feature_swd = _sliced_wasserstein(
        generated_z[..., motion_slice], test_z[..., motion_slice], seed=0
    )
    split_feature_swd = _sliced_wasserstein(
        validation_z[..., motion_slice], test_z[..., motion_slice], seed=0
    )
    action_swd = _sliced_wasserstein(
        controls, test.normalized_control, seed=1, projections=32
    )
    split_action_swd = _sliced_wasserstein(
        validation.normalized_control,
        test.normalized_control,
        seed=1,
        projections=32,
    )
    root = np.asarray(qps.pos[:, :, 0])
    quaternion = np.asarray(qps.rot[:, :, 0])
    upright = 1.0 - 2.0 * (
        np.square(quaternion[..., 1]) + np.square(quaternion[..., 2])
    )
    survives = np.all((root[..., 2] >= 0.6875) & (upright >= 0.0), axis=1)
    commanded_speed = command[:, 0] / COMMAND_HORIZON_SECONDS
    realized_speed = features[..., SL["root_velocity"][0]].mean(axis=1)

    # Frozen scalar: lower is better.  Distribution distances are normalized
    # by the honest validation-vs-test split gap; the native term prevents a
    # marginal-distribution match from hiding immediate trajectory divergence.
    feature_excess = feature_swd / max(split_feature_swd, 1e-8)
    action_excess = action_swd / max(split_action_swd, 1e-8)
    native_ratio = native_mse / max(persistence_mse, 1e-8)
    objective = 0.50 * feature_excess + 0.25 * action_excess + 0.25 * native_ratio
    generated_gait = _gait_summary(features)
    target_gait = _gait_summary(test.features)
    gait_standardized_distance = float(
        np.mean(
            [
                abs(generated_gait[name]["mean"] - target_gait[name]["mean"])
                / max(
                    target_gait[name]["std"], 0.1 * abs(target_gait[name]["mean"]), 1e-3
                )
                for name in GAIT_FIELDS
            ]
        )
    )
    report = {
        "schema": "demo-h-frozen-prior-rollout-audit-v1",
        "prior": str(prior_path),
        "prior_sha256": prior.artifact_sha256,
        "dataset_manifest_sha256": prior.metadata["dataset_manifest_sha256"],
        "split": "test",
        "clips": count,
        "start_frame": START_FRAME,
        "native_steps": NATIVE_STEPS,
        "long_steps": steps,
        "frozen_objective_lower_is_better": objective,
        "hard_gates": {
            "all_finite": bool(
                np.isfinite(features).all() and np.isfinite(controls).all()
            ),
            "survival_fraction_at_least_0p99": bool(survives.mean() >= 0.99),
            "native_skill_positive": bool(native_mse < persistence_mse),
        },
        "passes_hard_gates": bool(
            np.isfinite(features).all()
            and np.isfinite(controls).all()
            and survives.mean() >= 0.99
            and native_mse < persistence_mse
        ),
        "native_trajectory": {
            "normalized_feature_mse": native_mse,
            "persistence_mse": persistence_mse,
            "skill_over_persistence": 1.0 - native_ratio,
            "action_mse": native_action_mse,
        },
        "long_occupancy": {
            "motion_sliced_wasserstein": feature_swd,
            "validation_test_motion_sliced_wasserstein": split_feature_swd,
            "motion_excess_over_split_gap": feature_excess,
            "action_sliced_wasserstein": action_swd,
            "validation_test_action_sliced_wasserstein": split_action_swd,
            "action_excess_over_split_gap": action_excess,
            "contact_pattern_js_nats": _contact_js(
                contacts, test.contacts[:, START_FRAME + 1 :]
            ),
            "gait_standardized_distance": gait_standardized_distance,
            "survival_fraction": float(survives.mean()),
            "command_speed_mae": float(
                np.mean(np.abs(realized_speed - commanded_speed))
            ),
            "command_speed_correlation": float(
                np.corrcoef(realized_speed, commanded_speed)[0, 1]
            ),
            "generated_gait": generated_gait,
            "target_gait": target_gait,
        },
        "note": (
            "Validation-only audit. No distribution, contact, gait, or survival "
            "diagnostic is used as a training reward."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if save_trace:
        np.savez_compressed(
            output.with_suffix(".npz"),
            features=features,
            controls=controls,
            contacts=contacts,
            command=command.astype(np.float32),
            height=root[..., 2].astype(np.float32),
            upright=upright.astype(np.float32),
            realized_speed=realized_speed.astype(np.float32),
            survives=survives.astype(np.uint8),
        )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--steps", type=int, default=DEFAULT_LONG_STEPS)
    parser.add_argument(
        "--output", type=Path, default=Path("demo_h/out/prior_iteration/audit.json")
    )
    parser.add_argument("--save-trace", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.prior,
        dataset_root=args.dataset_root,
        steps=args.steps,
        output=args.output,
        save_trace=args.save_trace,
    )


if __name__ == "__main__":
    main()
