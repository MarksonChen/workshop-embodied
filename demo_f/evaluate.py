"""Frozen quantitative gate for Demo F predictor iterations.

Model selection uses three fixed held-out-session histories and four speed
interventions.  Lower objective is better; gates prevent a rollout improvement
from hiding worse-than-persistence prediction or geometric collapse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import FPS, OUT
from .dataset import load_split
from .dataset.contract import DYNAMIC_ROOT
from .features import SL
from .generate import (
    COMMAND_HORIZON_SECONDS,
    SPEED_SMOOTHING_FRAMES,
    checkpoint_command_scale,
    integrate_root,
    load_prior,
    rollout_features,
    straight_training_mask,
    trailing_mean,
)
from .windows import encode_in_batches, predictor_windows


EVAL_SPEEDS = np.asarray((0.10, 0.15, 0.20, 0.25), np.float32)
EVAL_SECONDS = 4.0
EVAL_SEEDS = 3


@torch.inference_mode()
def likelihood_report(
    checkpoint, config, tokenizer, predictor, train, evaluation
) -> dict:
    """Audit whether real next tokens prefer their matching speed command."""

    device = next(tokenizer.parameters()).device
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=device)
    features = (
        torch.as_tensor(evaluation.features, device=device) - feature_mean
    ) / feature_std
    tokens = encode_in_batches(tokenizer, features)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=device)
    token_std = torch.as_tensor(checkpoint["token_std"], device=device)
    history, future, raw_command, _ = predictor_windows(
        (tokens - token_mean) / token_std, evaluation, config
    )
    command_mean = torch.as_tensor(
        checkpoint.get("command_mean", np.zeros(3, np.float32)), device=device
    )
    command_std = torch.as_tensor(
        checkpoint.get("command_std", np.ones(3, np.float32)), device=device
    )
    command = (raw_command - command_mean) / command_std
    sigma = checkpoint["sigma"]

    real_logp = predictor.log_prob(history, future, command, sigma)
    generator = torch.Generator(device=device).manual_seed(13)
    permutation = torch.randperm(len(future), device=device, generator=generator)
    shuffled_logp = predictor.log_prob(
        history, future[permutation], command, sigma
    )

    raw = raw_command.cpu().numpy()
    planar = raw[:, :2]
    scale = checkpoint_command_scale(checkpoint, train)
    speed = np.linalg.norm(planar, axis=1) / scale
    keep = speed > 0.02
    history, future = history[keep], future[keep]
    raw, planar, speed = raw[keep], planar[keep], speed[keep]
    direction = planar / np.maximum(
        np.linalg.norm(planar, axis=1, keepdims=True), 1e-6
    )
    centers = np.quantile(speed, (0.10, 0.30, 0.50, 0.70, 0.90)).astype(
        np.float32
    )
    actual_bin = np.argmin(np.abs(speed[:, None] - centers[None]), axis=1)
    columns = []
    for conditioned_speed in centers:
        counterfactual = raw.copy()
        counterfactual[:, :2] = direction * (conditioned_speed * scale)
        normalized = (
            torch.as_tensor(counterfactual, device=device) - command_mean
        ) / command_std
        columns.append(
            predictor.log_prob(history, future, normalized, sigma).cpu().numpy()
        )
    scores = np.stack(columns, axis=1)
    matrix = np.stack(
        [scores[actual_bin == row].mean(0) for row in range(len(centers))]
    )
    row_argmax = matrix.argmax(1)

    q25, q75 = np.quantile(speed, (0.25, 0.75))
    central = (speed >= q25) & (speed <= q75)
    spacing = max(float(q75 - q25) / 6, 1e-3)
    offsets = spacing * np.asarray((-2, -1, 0, 1, 2), np.float32)
    relative = []
    for offset in offsets:
        counterfactual = raw[central].copy()
        requested = np.maximum(speed[central] + offset, 1e-3)
        counterfactual[:, :2] = (
            direction[central] * (requested * scale)[:, None]
        )
        normalized = (
            torch.as_tensor(counterfactual, device=device) - command_mean
        ) / command_std
        relative.append(
            predictor.log_prob(
                history[central], future[central], normalized, sigma
            ).mean().item()
        )
    relative = np.asarray(relative, np.float32)
    return {
        "evaluation_contract": "demo-f-likelihood-v1",
        "n_windows": int(len(real_logp)),
        "mean_logp": float(real_logp.mean()),
        "shuffled_future_mean_logp": float(shuffled_logp.mean()),
        "real_minus_shuffled_logp": float(
            real_logp.mean() - shuffled_logp.mean()
        ),
        "speed_centers_mps": centers.tolist(),
        "mean_logp_matrix_actual_rows_conditioned_columns": matrix.tolist(),
        "row_argmax_conditioned_bin": row_argmax.tolist(),
        "diagonal_wins": int(np.sum(row_argmax == np.arange(len(centers)))),
        "sample_top1_speed_bin_accuracy": float(
            np.mean(scores.argmax(1) == actual_bin)
        ),
        "chance_accuracy": 1 / len(centers),
        "relative_speed_offsets_mps": offsets.tolist(),
        "relative_mean_logp": relative.tolist(),
        "relative_peak_offset_mps": float(offsets[relative.argmax()]),
        "relative_peak_at_match": bool(relative.argmax() == len(offsets) // 2),
    }


def fixed_seed_indices(dataset, count: int = EVAL_SEEDS) -> list[int]:
    """Choose nearly straight 0.15 m/s histories from distinct sessions."""

    mask = straight_training_mask(dataset.command, dataset.source_speed_mps)
    candidates = np.flatnonzero(mask)
    score = (
        np.abs(dataset.source_speed_mps[candidates] - 0.15) / 0.02
        + np.abs(dataset.command[candidates, 1]) / 0.10
        + np.abs(dataset.command[candidates, 2]) / 0.10
    )
    selected, seen_sessions = [], set()
    for index in candidates[np.argsort(score)]:
        session = dataset.sessions[int(dataset.session_index[index])]
        if session in seen_sessions:
            continue
        selected.append(int(index))
        seen_sessions.add(session)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only found {len(selected)} distinct validation seed sessions")
    return selected


def rollout_report(checkpoint, config, tokenizer, predictor, train, validation) -> dict:
    """Evaluate already-loaded models; used for validation checkpoint selection."""

    scale = checkpoint_command_scale(checkpoint, train)
    seed_indices = fixed_seed_indices(validation)
    frames = int(EVAL_SECONDS * FPS)
    seed_frames = config.history_tokens * config.downsample
    rows, realized_by_seed = [], []

    for seed_index in seed_indices:
        session = validation.sessions[int(validation.session_index[seed_index])]
        realized = []
        for requested_speed in EVAL_SPEEDS:
            command = np.asarray((scale * requested_speed, 0.0, 0.0), np.float32)
            features = rollout_features(
                validation.features[seed_index],
                command,
                frames,
                checkpoint,
                config,
                tokenizer,
                predictor,
            )
            angles, root, _ = integrate_root(features)
            path_speed = np.zeros(frames, np.float32)
            path_speed[1:] = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) * FPS
            path_equivalent = trailing_mean(
                path_speed * COMMAND_HORIZON_SECONDS / scale
            )
            forward_equivalent = trailing_mean(
                features[:, SL["root_velocity"][0]]
                * COMMAND_HORIZON_SECONDS
                / scale
            )
            start = seed_frames + SPEED_SMOOTHING_FRAMES
            path_post = path_equivalent[start:]
            forward_post = forward_equivalent[start:]
            median_forward = float(np.median(forward_post))
            realized.append(median_forward)
            rows.append(
                {
                    "session": session,
                    "source_start": int(validation.source_start[seed_index]),
                    "requested_speed_mps": float(requested_speed),
                    "median_forward_speed_mps": median_forward,
                    "median_path_speed_mps": float(np.median(path_post)),
                    "low_speed_dwell_fraction": float(
                        np.mean(path_post < 0.25 * requested_speed)
                    ),
                    "root_height_min": float(root[start:, 2].min()),
                    "root_height_max": float(root[start:, 2].max()),
                    "joint_limit_fraction": float(
                        np.mean(np.abs(angles[start:]) >= np.pi / 3 - 1e-6)
                    ),
                }
            )
        realized_by_seed.append(realized)

    realized = np.asarray(realized_by_seed, np.float32)
    tracking_mae = float(np.mean(np.abs(realized - EVAL_SPEEDS[None])))
    monotonic_violation = float(np.maximum(realized[:, :-1] - realized[:, 1:], 0).mean())
    pause_fraction = float(np.mean([row["low_speed_dwell_fraction"] for row in rows]))
    # Frozen dimensionless objective: tracking dominates, while pauses and
    # command-order reversals break ties. Lower is better.
    objective = (
        tracking_mae / 0.15
        + 0.5 * pause_fraction
        + 2.0 * monotonic_violation / 0.05
    )
    return {
        "evaluation_contract": "demo-f-rollout-v1",
        "config": checkpoint["config"],
        "seeds": [
            {
                "session": validation.sessions[int(validation.session_index[index])],
                "source_start": int(validation.source_start[index]),
            }
            for index in seed_indices
        ],
        "speeds_mps": EVAL_SPEEDS.tolist(),
        "objective": objective,
        "tracking_mae_mps": tracking_mae,
        "monotonic_violation_mps": monotonic_violation,
        "low_speed_dwell_fraction": pause_fraction,
        "realized_forward_speed_mps": realized.tolist(),
        "trials": rows,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_root: Path = DYNAMIC_ROOT,
    evaluation_split: str = "validation",
) -> dict:
    checkpoint, config, tokenizer, predictor = load_prior(checkpoint_path)
    train = load_split("train", dataset_root)
    validation = load_split(evaluation_split, dataset_root)
    report = rollout_report(
        checkpoint, config, tokenizer, predictor, train, validation
    )
    likelihood = likelihood_report(
        checkpoint, config, tokenizer, predictor, train, validation
    )
    metrics = checkpoint["metrics"]
    rows = report["trials"]
    realized = np.asarray(report["realized_forward_speed_mps"])
    gates = {
        "finite": bool(np.isfinite(realized).all()),
        "beats_persistence": metrics["validation_mse"] < metrics["persistence_mse"],
        "command_win_rate": metrics["command_vs_reversed_win_rate"] >= 0.60,
        "beats_shuffled_future": likelihood["real_minus_shuffled_logp"] > 0,
        "speed_likelihood_diagonal": likelihood["diagonal_wins"]
        == len(likelihood["speed_centers_mps"]),
        "speed_likelihood_peaks_at_match": likelihood["relative_peak_at_match"],
        "root_height": min(row["root_height_min"] for row in rows) >= 1.0
        and max(row["root_height_max"] for row in rows) <= 1.8,
        "joint_limits": max(row["joint_limit_fraction"] for row in rows) <= 0.01,
    }
    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "split": evaluation_split,
            "prediction_skill_over_persistence": 1
            - metrics["validation_mse"] / metrics["persistence_mse"],
            "command_win_rate": metrics["command_vs_reversed_win_rate"],
            "likelihood": likelihood,
            "gates": gates,
            "passed_gates": all(gates.values()),
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=OUT / "evaluation.json")
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.dataset_root, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"objective={report['objective']:.4f} | tracking={report['tracking_mae_mps']:.4f} "
        f"m/s | monotonic={report['monotonic_violation_mps']:.4f} m/s | "
        f"low-speed={report['low_speed_dwell_fraction']:.1%} | "
        f"likelihood={report['likelihood']['diagonal_wins']}/5 | "
        f"gates={report['passed_gates']}",
        flush=True,
    )
    print(np.asarray(report["realized_forward_speed_mps"]), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
