"""Eligibility gates for Demo B's frozen conditional likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .constants import DEV, FPS
from .geometry import sixd2mat
from .models import load_motor


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
DEFAULT_REPORT = ROOT / "demo_b" / "assets" / "eligibility.json"
COMMAND_SECONDS = 31 / FPS


@torch.inference_mode()
def decoded_future_statistics(motion, transition, bundle, history, raw_command):
    batch = history.shape[0]
    cmean = torch.as_tensor(bundle["cmean"], device=DEV)
    cstd = torch.as_tensor(bundle["cstd"], device=DEV)
    zmean = torch.as_tensor(bundle["zmean"], device=DEV)
    zstd = torch.as_tensor(bundle["zstd"], device=DEV)
    mmean = np.asarray(bundle["mmean"])
    mstd = np.asarray(bundle["mstd"])
    command = torch.as_tensor(raw_command, device=DEV).expand(batch, -1)
    predicted = transition.predict(history, (command - cmean) / cstd)
    tokens = torch.cat([history, predicted], dim=1) * zstd + zmean
    decoded = motion.decode(tokens).cpu().numpy() * mstd + mmean
    future = decoded[:, 32:]
    forward_velocity = future[..., 0].mean(1)
    rotations = sixd2mat(future[..., 3:9].reshape(-1, 6)).reshape(
        batch, -1, 3, 3
    )
    yaw_increment = np.arctan2(rotations[..., 1, 0], rotations[..., 0, 0])
    yaw_rate = yaw_increment.sum(1) / (future.shape[1] / FPS)
    return forward_velocity, yaw_rate


@torch.inference_mode()
def speed_likelihood_audit(transition, bundle) -> dict:
    """Score held-out motion under matching and counterfactual speeds.

    History, realized future, egocentric travel direction, and yaw command stay
    fixed; only conditioned planar speed changes.  Thus a diagonal maximum
    cannot be explained by comparing different motions across columns.
    """
    bank = bundle["evaluation_bank"]
    history = torch.as_tensor(bank["history"], device=DEV)
    future = torch.as_tensor(bank["future"], device=DEV)[:, 0]
    raw = np.asarray(bank["command_raw"], np.float32)
    planar = raw[:, :2]
    speed = np.linalg.norm(planar, axis=-1) / COMMAND_SECONDS
    # Preserve each recorded egocentric direction and yaw; only the magnitude
    # of the planar speed is changed.  Exclude nearly stationary commands whose
    # direction is undefined.
    indices = np.flatnonzero(speed > 0.02)
    history, future, raw, speed, planar = (
        history[indices],
        future[indices],
        raw[indices],
        speed[indices],
        planar[indices],
    )
    direction = planar / np.maximum(np.linalg.norm(planar, axis=-1, keepdims=True), 1e-6)
    centers = np.quantile(speed, [0.10, 0.30, 0.50, 0.70, 0.90]).astype(np.float32)
    actual_bin = np.argmin(np.abs(speed[:, None] - centers[None]), axis=1)
    cmean = torch.as_tensor(bundle["cmean"], device=DEV)
    cstd = torch.as_tensor(bundle["cstd"], device=DEV)
    sigma = torch.as_tensor(bundle["sigma"], device=DEV)
    columns = []
    for conditioned_speed in centers:
        command = raw.copy()
        command[:, :2] = direction * conditioned_speed * COMMAND_SECONDS
        normalized = (torch.as_tensor(command, device=DEV) - cmean) / cstd
        columns.append(
            transition.log_prob_next(history, future, normalized, sigma).cpu().numpy()
        )
    scores = np.stack(columns, axis=1)
    matrix = np.stack(
        [scores[actual_bin == row].mean(0) for row in range(len(centers))]
    )
    row_argmax = matrix.argmax(1)
    diagonal = np.diag(matrix)
    best_off_diagonal = np.max(
        np.where(np.eye(len(centers), dtype=bool), -np.inf, matrix), axis=1
    )

    # A per-example relative curve asks the literal question p(motion at x | x
    # + delta).  Use only the central half of support to avoid clipped offsets.
    q25, q75 = np.quantile(speed, [0.25, 0.75])
    central = (speed >= q25) & (speed <= q75)
    spacing = max(float(q75 - q25) / 6, 1e-3)
    offsets = spacing * np.asarray([-2, -1, 0, 1, 2], np.float32)
    relative = []
    for offset in offsets:
        command = raw[central].copy()
        command[:, :2] = (
            direction[central]
            * (speed[central] + offset)[:, None]
            * COMMAND_SECONDS
        )
        normalized = (torch.as_tensor(command, device=DEV) - cmean) / cstd
        relative.append(
            transition.log_prob_next(
                history[central], future[central], normalized, sigma
            ).mean().item()
        )
    relative = np.asarray(relative)
    return {
        "n_examples": int(len(speed)),
        "speed_centers_m_per_s": centers.tolist(),
        "mean_logp_matrix_actual_rows_conditioned_columns": matrix.tolist(),
        "row_argmax_conditioned_bin": row_argmax.tolist(),
        "diagonal_wins": int(np.sum(row_argmax == np.arange(len(centers)))),
        "diagonal_margin_by_speed": (diagonal - best_off_diagonal).tolist(),
        "sample_top1_speed_bin_accuracy": float(
            np.mean(scores.argmax(1) == actual_bin)
        ),
        "chance_accuracy": 1 / len(centers),
        "relative_speed_offsets_m_per_s": offsets.tolist(),
        "relative_mean_logp": relative.tolist(),
        "relative_peak_offset_m_per_s": float(offsets[relative.argmax()]),
        "relative_peak_at_match": bool(relative.argmax() == len(offsets) // 2),
    }


def evaluate(asset: Path = DEFAULT_ASSET) -> dict:
    bundle = torch.load(asset, map_location="cpu", weights_only=False)
    if bundle.get("format_version") == 4 and bundle.get("animal") == "coltrane":
        metrics = bundle["metrics"]
        speed = metrics["speed_likelihood"]
        diagnostics = {
            "animal": "coltrane",
            "feature_dim": int(bundle["feature_dim"]),
            "calibration_scope": metrics["calibration_scope"],
            "real_minus_shuffled_logp": float(
                metrics["logp_mean"] - metrics["logp_shuffled_mean"]
            ),
            "speed_likelihood": speed,
            "source_asset_sha256": bundle["source_asset_sha256"],
        }
        gates = {
            "full_281_representation": diagnostics["feature_dim"] == 281,
            "beats_shuffled_future": diagnostics["real_minus_shuffled_logp"] > 0,
            "speed_likelihood_diagonal": speed["diagonal_wins"] == 5,
            "speed_likelihood_peaks_at_match": speed["relative_peak_at_match"],
        }
        return {
            "eligible": all(gates.values()),
            "gates": gates,
            "diagnostics": diagnostics,
            "limitation": (
                "Likelihood calibration uses the standalone training windows; "
                "a held-out Coltrane/reset-bank rebuild is required before Demo E PPO."
            ),
        }
    if bundle.get("animal") != "freddie" or bundle.get("format_version") != 3:
        raise ValueError("eligibility requires the format-v3 all-session Freddie asset")
    motion, transition, _, _ = load_motor(asset)
    history = torch.as_tensor(bundle["reset_banks"]["val"]["history"][:128], device=DEV)
    forward_support = np.asarray(bundle["command_support_velocity"])[:, 0]
    slow_speed, fast_speed = np.quantile(forward_support, [0.25, 0.75])
    slow_v, _ = decoded_future_statistics(
        motion,
        transition,
        bundle,
        history,
        np.asarray([COMMAND_SECONDS * slow_speed, 0.0, 0.0], np.float32),
    )
    fast_v, _ = decoded_future_statistics(
        motion,
        transition,
        bundle,
        history,
        np.asarray([COMMAND_SECONDS * fast_speed, 0.0, 0.0], np.float32),
    )
    _, left_yaw = decoded_future_statistics(
        motion,
        transition,
        bundle,
        history,
        np.asarray([COMMAND_SECONDS * 0.15, 0.0, -COMMAND_SECONDS * 0.75], np.float32),
    )
    _, right_yaw = decoded_future_statistics(
        motion,
        transition,
        bundle,
        history,
        np.asarray([COMMAND_SECONDS * 0.15, 0.0, COMMAND_SECONDS * 0.75], np.float32),
    )
    speed_audit = speed_likelihood_audit(transition, bundle)
    test = bundle["metrics"]["test"]
    diagnostics = {
        "animal": bundle["animal"],
        "sessions": {key: len(value) for key, value in bundle["split_ids"].items()},
        "test_skill_over_persistence": float(test["skill_over_persistence"]),
        "test_real_minus_shuffled_logp": float(
            test["logp_mean"] - test["logp_shuffled_mean"]
        ),
        "forward_intervention_m_per_s": float(np.mean(fast_v - slow_v)),
        "yaw_intervention_rad_per_s": float(np.mean(right_yaw - left_yaw)),
        "decoded_slow_forward_m_per_s": float(np.mean(slow_v)),
        "decoded_fast_forward_m_per_s": float(np.mean(fast_v)),
        "decoded_left_yaw_rad_per_s": float(np.mean(left_yaw)),
        "decoded_right_yaw_rad_per_s": float(np.mean(right_yaw)),
        "speed_likelihood": speed_audit,
    }
    gates = {
        "all_freddie_sessions_used": sum(diagnostics["sessions"].values()) == 25,
        "session_split_is_held_out": diagnostics["sessions"] == {
            "train": 17,
            "validation": 4,
            "test": 4,
        },
        "beats_persistence": diagnostics["test_skill_over_persistence"] > 0,
        "beats_shuffled_future": diagnostics["test_real_minus_shuffled_logp"] > 0,
        "forward_command_used": diagnostics["forward_intervention_m_per_s"] > 0,
        "yaw_command_used": diagnostics["yaw_intervention_rad_per_s"] > 0,
        "speed_likelihood_diagonal": speed_audit["diagonal_wins"] >= 4,
        "speed_likelihood_peaks_at_match": speed_audit["relative_peak_at_match"],
    }
    return {"eligible": all(gates.values()), "gates": gates, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate(args.asset)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["eligible"]:
        raise SystemExit("Demo B Freddie likelihood failed an eligibility gate")


if __name__ == "__main__":
    main()
