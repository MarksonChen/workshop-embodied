"""Score frozen transition candidates on Demo E development tokens.

The token artifact is generated once from native-reset physical rollouts with
development seeds 101--103.  This command changes neither the tokens nor model
weights; it only evaluates conditional Gaussian scores for candidate bundles.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from demo_b.constants import DEV
from demo_b.models import load_motor
from demo_e.config import OUT, TORCH_ASSET


DEFAULT_TOKENS = (
    OUT / "reproduction" / "development-masked-constant-tokens-native-v5.npz"
)


def _integrate_hindsight_command(
    normalized_features: np.ndarray,
    feature_updates: np.ndarray,
    alive: np.ndarray,
    start: int,
    mmean: np.ndarray,
    mstd: np.ndarray,
) -> np.ndarray | None:
    """Recover Demo B's future command from a frozen physical trajectory."""
    sample_steps = start + 2 * np.arange(1, 32)
    if sample_steps[-1] >= len(normalized_features):
        return None
    if not np.all(feature_updates[sample_steps] > 0.5):
        raise ValueError("development anchor violates the 50-Hz feature clock")
    if not np.all(alive[sample_steps] > 0.5):
        return None
    features = normalized_features[sample_steps] * mstd + mmean
    delta_yaw = np.arctan2(features[:, 4], features[:, 3])
    relative_yaw = np.cumsum(delta_yaw)
    cosine, sine = np.cos(relative_yaw), np.sin(relative_yaw)
    local_velocity = features[:, :2]
    displacement = 0.02 * np.stack(
        [
            cosine * local_velocity[:, 0] - sine * local_velocity[:, 1],
            sine * local_velocity[:, 0] + cosine * local_velocity[:, 1],
        ],
        axis=-1,
    ).sum(0)
    return np.asarray([*displacement, delta_yaw.sum()], np.float32)


def _actual_command_grid(
    anchor_path: Path,
    matching: np.ndarray,
    token_seed: np.ndarray,
    token_index: np.ndarray,
    mmean: np.ndarray,
    mstd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Return per-token counterfactuals from realized reference motion.

    The six joystick labels are task requests, not motion measurements.  For a
    fair physical transfer check, every candidate condition is therefore the
    actual future displacement realized by that reference rollout at the same
    seed and block time.
    """
    with np.load(anchor_path, allow_pickle=False) as anchor:
        trial_command = np.asarray(anchor["trial_command"], np.int64)
        trial_seed = np.asarray(anchor["trial_seed"], np.int64)
        features = np.asarray(anchor["feature"], np.float32)
        feature_updates = np.asarray(anchor["feature_update"], np.float32)
        prior_updates = np.asarray(anchor["prior_update"], np.float32)
        alive = np.asarray(anchor["alive"], np.float32)
        online_index = np.asarray(anchor["online_index"], np.int64)

    lookup: dict[tuple[int, int, int], np.ndarray] = {}
    trial_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for trial in range(len(trial_command)):
        steps = np.flatnonzero(
            (prior_updates[trial] > 0.5)
            & (alive[trial] > 0.5)
            & (np.arange(prior_updates.shape[1]) >= 100)
        )
        indices = online_index[trial, steps]
        starts = steps - (indices + 1) * 8
        trial_rows.append((indices, starts))
        for start in np.unique(starts):
            command = _integrate_hindsight_command(
                features[trial],
                feature_updates[trial],
                alive[trial],
                int(start),
                mmean,
                mstd,
            )
            if command is not None:
                lookup[
                    (int(trial_command[trial]), int(trial_seed[trial]), int(start))
                ] = command

    keep: list[int] = []
    grids: list[np.ndarray] = []
    cursor = 0
    for trial, (indices, starts) in enumerate(trial_rows):
        stop = cursor + len(indices)
        if not np.array_equal(token_index[cursor:stop], indices):
            raise ValueError("token artifact is not aligned with its physical anchor")
        if not np.all(matching[cursor:stop] == trial_command[trial]):
            raise ValueError("token task labels disagree with the physical anchor")
        if not np.all(token_seed[cursor:stop] == trial_seed[trial]):
            raise ValueError("token seeds disagree with the physical anchor")
        for offset, start in enumerate(starts):
            keys = [
                (command_index, int(trial_seed[trial]), int(start))
                for command_index in range(6)
            ]
            if all(key in lookup for key in keys):
                keep.append(cursor + offset)
                grids.append(np.stack([lookup[key] for key in keys]))
        cursor = stop
    if cursor != len(matching):
        raise ValueError("token artifact has trailing rows")

    command_means = []
    for command_index in range(6):
        values = [
            value
            for (index, _, _), value in lookup.items()
            if index == command_index
        ]
        command_means.append(np.mean(values, axis=0).tolist())
    return (
        np.asarray(keep, np.int64),
        np.asarray(grids, np.float32),
        command_means,
    )


def _source_gates(bundle: dict) -> dict[str, bool]:
    metrics = bundle.get("metrics", {})
    test = metrics.get("test", {})
    speed = metrics.get("speed_likelihood", {})
    return {
        "heldout_test_available": bool(test),
        "beats_persistence": float(test.get("skill_over_persistence", -np.inf)) > 0,
        "beats_shuffled": float(test.get("logp_mean", -np.inf))
        > float(test.get("logp_shuffled_mean", np.inf)),
        "speed_diagonal_at_least_4_of_5": int(speed.get("diagonal_wins", 0)) >= 4,
        "relative_peak_at_match": bool(speed.get("relative_peak_at_match", False)),
    }


def _summarize(scores: np.ndarray, matching: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == matching[:, None], axis=1) + 1
    predicted = order[:, 0]
    matched = scores[np.arange(len(scores)), matching]
    off_diagonal = np.max(
        np.where(np.eye(scores.shape[1], dtype=bool)[matching], -np.inf, scores),
        axis=1,
    )
    moving = matching >= 3
    standing = matching == 0
    by_command = []
    for command_index in range(scores.shape[1]):
        mask = matching == command_index
        by_command.append(
            {
                "command_index": command_index,
                "tokens": int(mask.sum()),
                "matched_logp": float(matched[mask].mean()),
                "top1": float(np.mean(rank[mask] == 1)),
                "rank": float(rank[mask].mean()),
                "margin": float((matched[mask] - off_diagonal[mask]).mean()),
            }
        )
    return {
        "moving_top1": float(np.mean(rank[moving] == 1)),
        "moving_rank": float(rank[moving].mean()),
        "moving_margin": float((matched[moving] - off_diagonal[moving]).mean()),
        "moving_minus_zero_command": float(
            (matched[moving] - scores[moving, 0]).mean()
        ),
        "moving_minus_standing_logp": float(
            matched[moving].mean() - matched[standing].mean()
        ),
        "mean_logp_matrix_actual_rows_conditioned_columns": [
            scores[matching == command_index].mean(0).tolist()
            for command_index in range(scores.shape[1])
        ],
        "top1_confusion_actual_rows_predicted_columns": [
            np.bincount(
                predicted[matching == command_index], minlength=scores.shape[1]
            ).tolist()
            for command_index in range(scores.shape[1])
        ],
        "by_command": by_command,
    }


@torch.inference_mode()
def audit_candidate(
    path: Path,
    *,
    history_raw: np.ndarray,
    realized_raw: np.ndarray,
    index: np.ndarray,
    matching: np.ndarray,
    raw_commands: np.ndarray,
) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    _, transition, norms, _ = load_motor(path)
    zmean = np.asarray(norms["zmean"], np.float32).reshape(-1)
    zstd = np.asarray(norms["zstd"], np.float32).reshape(-1)
    history = torch.as_tensor((history_raw - zmean) / zstd, device=DEV)
    realized = torch.as_tensor((realized_raw - zmean) / zstd, device=DEV)
    token_index = torch.as_tensor(index, device=DEV)
    row = torch.arange(len(history), device=DEV)
    sigma = torch.as_tensor(norms["sigma"], device=DEV)
    context = transition.context(history)
    columns = []
    if raw_commands.ndim == 2:
        raw_commands = np.broadcast_to(
            raw_commands[None], (len(history),) + raw_commands.shape
        )
    for command_index in range(raw_commands.shape[1]):
        raw_command = raw_commands[:, command_index]
        command = torch.as_tensor(
            (raw_command - norms["cmean"]) / norms["cstd"], device=DEV
        )
        predicted = transition.predict_from_context(context, command)[row, token_index]
        error = (realized - predicted) / sigma
        columns.append(
            (
                -0.5
                * (
                    error.square()
                    + 2 * sigma.log()
                    + math.log(2 * math.pi)
                ).mean(-1)
            )
            .cpu()
            .numpy()
        )
    physical = _summarize(np.stack(columns, axis=1), matching)
    source = _source_gates(bundle)
    physical_gates = {
        "moving_top1_at_least_half": physical["moving_top1"] >= 0.50,
        "moving_margin_positive": physical["moving_margin"] > 0.0,
        "matched_beats_zero_command": physical["moving_minus_zero_command"] > 0.0,
        "moving_not_less_likely_than_standing": physical[
            "moving_minus_standing_logp"
        ]
        >= 0.0,
    }
    return {
        "name": path.stem,
        "path": str(path),
        "source_gates": source,
        "physical": physical,
        "physical_gates": physical_gates,
        "eligible": all(source.values()) and all(physical_gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.tokens, allow_pickle=False) as source:
        role = str(source["role"])
        seeds = sorted(map(int, np.unique(source["token_seed"])))
        if "model-selection" not in role or seeds != [101, 102, 103]:
            raise ValueError("candidate audit requires the frozen development anchor")
        history = np.asarray(source["history"], np.float32)
        realized = np.asarray(source["realized"], np.float32)
        index = np.asarray(source["index"], np.int64)
        matching = np.asarray(source["matching"], np.int64)
        token_seed = np.asarray(source["token_seed"], np.int64)
        anchor_path = Path(str(source["anchor"]))

    base = torch.load(TORCH_ASSET, map_location="cpu", weights_only=False)
    base_mean = np.asarray(base["zmean"], np.float32).reshape(-1)
    base_std = np.asarray(base["zstd"], np.float32).reshape(-1)
    keep, raw_commands, command_means = _actual_command_grid(
        anchor_path,
        matching,
        token_seed,
        index,
        np.asarray(base["mmean"], np.float32),
        np.asarray(base["mstd"], np.float32),
    )
    history = history[keep]
    realized = realized[keep]
    index = index[keep]
    matching = matching[keep]
    history_raw = history * base_std + base_mean
    realized_raw = realized * base_std + base_mean
    rows = [
        audit_candidate(
            path,
            history_raw=history_raw,
            realized_raw=realized_raw,
            index=index,
            matching=matching,
            raw_commands=raw_commands,
        )
        for path in args.candidates
    ]
    report = {
        "pipeline_version": 6,
        "command_bridge": "realized egocentric hindsight displacement over 0.62 s",
        "reference_actual_command_mean": command_means,
        "role": role,
        "development_seeds": seeds,
        "tokens": str(args.tokens),
        "candidates": rows,
    }
    summary = [
        {
            "name": row["name"],
            "eligible": row["eligible"],
            **{
                key: row["physical"][key]
                for key in (
                    "moving_top1",
                    "moving_rank",
                    "moving_margin",
                    "moving_minus_zero_command",
                    "moving_minus_standing_logp",
                )
            },
        }
        for row in rows
    ]
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
