"""Autoregressively visualize the trained Demo F conditional motion prior.

The model is trained in Fetch coordinates, while workshop speed labels refer to
the source animal in metres per second.  A robust scale fitted on straight
training clips maps source speed to the Fetch-space hindsight displacement used
as the conditioning variable.  Both values are saved with every trajectory.

    uv run --extra workshop python -m demo_f.generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import FPS, JOINT_LIMIT, OUT, PriorConfig
from .dataset import load_manifest, load_split
from .dataset.contract import COMMAND_FRAME, COMMAND_FUTURE_FRAME, DYNAMIC_ROOT
from .features import FEATURE_DIM, SL
from .models import ConditionalTransformer, MotionAutoencoder


DEFAULT_SPEEDS = (0.10, 0.15, 0.20, 0.25)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMMAND_HORIZON_SECONDS = (COMMAND_FUTURE_FRAME - COMMAND_FRAME) / FPS
SPEED_SMOOTHING_FRAMES = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def straight_training_mask(command: np.ndarray, source_speed: np.ndarray) -> np.ndarray:
    """Select forward clips suitable for a robust cross-morphology scale."""

    return (
        (source_speed >= 0.04)
        & (command[:, 0] > 0.0)
        & (np.abs(command[:, 1]) < 0.25)
        & (np.abs(command[:, 2]) < 0.15)
    )


def command_scale(command: np.ndarray, source_speed: np.ndarray) -> float:
    """Median Fetch displacement per source ``m/s`` on straight train clips."""

    mask = straight_training_mask(command, source_speed)
    if mask.sum() < 20:
        raise ValueError("too few straight training clips to calibrate speed commands")
    return float(np.median(command[mask, 0] / source_speed[mask]))


def dataset_command_calibration(
    manifest: dict,
    command: np.ndarray,
    source_speed: np.ndarray,
) -> dict:
    """Resolve the declared source-speed-to-command map for one release."""

    dynamic = manifest.get("dynamic_scaling")
    if dynamic is not None:
        scale = float(dynamic["velocity_scale"]) * COMMAND_HORIZON_SECONDS
        return {
            "method": "declared Froude-similar velocity scale times command horizon",
            "fetch_displacement_per_mps": scale,
            "horizon_seconds": COMMAND_HORIZON_SECONDS,
        }
    return {
        "method": "median forward Fetch displacement / source net speed on straight train clips",
        "fetch_displacement_per_mps": command_scale(command, source_speed),
        "horizon_seconds": COMMAND_HORIZON_SECONDS,
    }


def checkpoint_command_scale(checkpoint: dict, train) -> float:
    calibration = checkpoint.get("command_calibration")
    if calibration is not None:
        return float(calibration["fetch_displacement_per_mps"])
    return command_scale(train.command, train.source_speed_mps)


def select_seed(dataset, target_speed: float = 0.15) -> int:
    """Choose one fixed, nearly straight on-manifold history for every command."""

    mask = straight_training_mask(dataset.command, dataset.source_speed_mps)
    candidates = np.flatnonzero(mask)
    score = (
        np.abs(dataset.source_speed_mps[candidates] - target_speed) / 0.02
        + np.abs(dataset.command[candidates, 1]) / 0.10
        + np.abs(dataset.command[candidates, 2]) / 0.10
    )
    return int(candidates[np.argmin(score)])


def load_prior(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {"demo-f-prior-v1", "demo-f-prior-v2"}:
        raise ValueError(f"unsupported checkpoint schema {checkpoint.get('schema')!r}")
    config = PriorConfig(**checkpoint["config"])
    tokenizer = MotionAutoencoder(
        FEATURE_DIM, config.hidden_channels, config.latent_dim
    ).to(DEVICE)
    predictor = ConditionalTransformer(
        latent_dim=config.latent_dim,
        future_tokens=config.future_tokens,
        width=config.hidden_channels,
        layers=config.transformer_layers,
        heads=config.transformer_heads,
    ).to(DEVICE)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    predictor.load_state_dict(checkpoint["predictor"])
    tokenizer.eval()
    predictor.eval()
    return checkpoint, config, tokenizer, predictor


@torch.inference_mode()
def rollout_features(
    seed_features: np.ndarray,
    command: np.ndarray,
    frames: int,
    checkpoint: dict,
    config: PriorConfig,
    tokenizer: MotionAutoencoder,
    predictor: ConditionalTransformer,
) -> np.ndarray:
    """Roll deterministic means one token at a time and decode continuously."""

    seed_frames = config.history_tokens * config.downsample
    if frames <= seed_frames:
        raise ValueError(f"frames must exceed the {seed_frames}-frame seed")
    mean = torch.as_tensor(checkpoint["feature_mean"], device=DEVICE)
    std = torch.as_tensor(checkpoint["feature_std"], device=DEVICE)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=DEVICE)
    token_std = torch.as_tensor(checkpoint["token_std"], device=DEVICE)
    normalized = (
        torch.as_tensor(seed_features, dtype=torch.float32, device=DEVICE) - mean
    ) / std
    seed_tokens = tokenizer.encode(normalized[None])
    history = ((seed_tokens - token_mean) / token_std)[:, :config.history_tokens]
    command_tensor = torch.as_tensor(command, dtype=torch.float32, device=DEVICE)[None]
    if "command_mean" in checkpoint:
        command_mean = torch.as_tensor(checkpoint["command_mean"], device=DEVICE)
        command_std = torch.as_tensor(checkpoint["command_std"], device=DEVICE)
        command_tensor = (command_tensor - command_mean) / command_std

    target_tokens = math.ceil(frames / config.downsample)
    stream = [history]
    generated_tokens = config.history_tokens
    while generated_tokens < target_tokens:
        proposal = predictor.predict(history, command_tensor)
        next_token = proposal[:, :1]
        stream.append(next_token)
        history = torch.cat((history, next_token), dim=1)[:, -config.history_tokens:]
        generated_tokens += 1
    normalized_tokens = torch.cat(stream, dim=1)[:, :target_tokens]
    decoded = tokenizer.decode(normalized_tokens * token_std + token_mean) * std + mean
    return decoded[0, :frames].cpu().numpy().astype(np.float32)


def integrate_root(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the yaw-only Fetch root and bounded joints from generated features."""

    features = np.asarray(features, np.float32)
    frames = len(features)
    velocity = features[:, slice(*SL["root_velocity"])]
    rotation_columns = features[:, slice(*SL["rotation_delta_6d"])].reshape(
        frames, 3, 2
    )
    delta_yaw = np.arctan2(rotation_columns[:, 1, 0], rotation_columns[:, 0, 0])
    yaw = np.zeros(frames, np.float32)
    yaw[1:] = np.cumsum(delta_yaw[1:]).astype(np.float32)

    root = np.zeros((frames, 3), np.float32)
    root[:, 2] = features[:, SL["root_height"][0]]
    for frame in range(1, frames):
        cosine, sine = np.cos(yaw[frame]), np.sin(yaw[frame])
        local_x, local_y = velocity[frame]
        root[frame, 0] = root[frame - 1, 0] + (
            cosine * local_x - sine * local_y
        ) / FPS
        root[frame, 1] = root[frame - 1, 1] + (
            sine * local_x + cosine * local_y
        ) / FPS

    quaternion = np.zeros((frames, 4), np.float32)
    quaternion[:, 0] = np.cos(yaw / 2)
    quaternion[:, 3] = np.sin(yaw / 2)
    joint_angles = np.clip(
        features[:, slice(*SL["joint_angles"])], -JOINT_LIMIT, JOINT_LIMIT
    ).astype(np.float32)
    return joint_angles, root, quaternion


def trailing_mean(values: np.ndarray, window: int = SPEED_SMOOTHING_FRAMES) -> np.ndarray:
    """Causal moving average with edge padding and unchanged length."""

    values = np.asarray(values, np.float32)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid").astype(np.float32)


def longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def plot_speed_traces(traces: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=150, sharex=True)
    for axis, trace in zip(axes.flat, traces, strict=True):
        time = np.arange(len(trace["instantaneous_speed"])) / FPS
        requested = trace["requested_speed"]
        axis.plot(
            time,
            trace["instantaneous_speed"],
            color="#7b8ff5",
            alpha=0.28,
            linewidth=0.8,
            label="50 Hz instantaneous",
        )
        axis.plot(
            time,
            trace["smoothed_speed"],
            color="#273c75",
            linewidth=1.8,
            label=f"{SPEED_SMOOTHING_FRAMES / FPS:.2f} s trailing mean",
        )
        axis.axhline(requested, color="#e06e8f", linestyle="--", linewidth=1.2, label="request")
        axis.axvline(trace["seed_frames"] / FPS, color="#777", linestyle=":", linewidth=1.0)
        axis.set_title(
            f"request {requested:.2f} m/s | low-speed dwell {trace['pause_fraction']:.0%}"
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("source-equivalent speed (m/s)")
    axes[1, 0].set_ylabel("source-equivalent speed (m/s)")
    figure.suptitle("Demo F autoregressive root-speed diagnostics (seed ends at dotted line)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT / "generated")
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed-speed", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    manifest_path = args.dataset_root / "manifest.json"
    manifest = load_manifest(args.dataset_root)
    checkpoint, config, tokenizer, predictor = load_prior(args.checkpoint)
    if checkpoint["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("checkpoint was not trained from this dataset manifest")

    train = load_split("train", args.dataset_root)
    scale = checkpoint_command_scale(checkpoint, train)
    seed_index = select_seed(train, args.seed_speed)
    seed_session = train.sessions[int(train.session_index[seed_index])]
    seed_start = int(train.source_start[seed_index])
    frames = int(round(args.seconds * FPS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)
    rows, traces = [], []
    print(
        f"device={DEVICE} | scale={scale:.4f} Fetch displacement/(m/s) | "
        f"seed={seed_session}@{seed_start}",
        flush=True,
    )
    for requested_speed in args.speeds:
        command = np.asarray((scale * requested_speed, 0.0, 0.0), np.float32)
        features = rollout_features(
            train.features[seed_index],
            command,
            frames,
            checkpoint,
            config,
            tokenizer,
            predictor,
        )
        seed_frames = config.history_tokens * config.downsample
        angles, root, quaternion = integrate_root(features)
        generated_root = root[seed_frames:]
        generated_path_speed = float(
            np.linalg.norm(np.diff(generated_root[:, :2], axis=0), axis=1).mean() * FPS
        )
        generated_forward_speed = float(
            features[seed_frames:, SL["root_velocity"][0]].mean()
        )
        equivalent_speed = generated_path_speed * COMMAND_HORIZON_SECONDS / scale
        instantaneous_fetch_speed = np.zeros(frames, np.float32)
        instantaneous_fetch_speed[1:] = (
            np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) * FPS
        )
        instantaneous_equivalent_speed = (
            instantaneous_fetch_speed * COMMAND_HORIZON_SECONDS / scale
        ).astype(np.float32)
        smoothed_equivalent_speed = trailing_mean(instantaneous_equivalent_speed)
        joint_activity = np.zeros(frames, np.float32)
        joint_activity[1:] = np.mean(np.abs(np.diff(angles, axis=0)), axis=1) * FPS
        evaluation_start = seed_frames + SPEED_SMOOTHING_FRAMES
        evaluated_speed = smoothed_equivalent_speed[evaluation_start:]
        low_speed = evaluated_speed < 0.25 * requested_speed
        pause_fraction = float(np.mean(low_speed))
        longest_low_speed_seconds = longest_true_run(low_speed) / FPS
        speed_quantiles = np.quantile(evaluated_speed, (0.1, 0.5, 0.9))
        # The previous visualizer reset at these 32-frame boundaries. Ratios near
        # one (or above) demonstrate that the new rollout does not systematically
        # pause at the obsolete boundary locations.
        old_boundaries = np.arange(seed_frames + COMMAND_FRAME, frames, COMMAND_FRAME)
        if len(old_boundaries):
            boundary_index = np.unique(
                np.concatenate(
                    [
                        np.arange(max(evaluation_start, frame - 2), min(frames, frame + 3))
                        for frame in old_boundaries
                    ]
                )
            )
            post_seed_index = np.arange(evaluation_start, frames)
            ordinary_index = post_seed_index[~np.isin(post_seed_index, boundary_index)]
            boundary_speed_ratio = float(
                smoothed_equivalent_speed[boundary_index].mean()
                / smoothed_equivalent_speed[ordinary_index].mean()
            )
            boundary_joint_activity_ratio = float(
                joint_activity[boundary_index].mean() / joint_activity[ordinary_index].mean()
            )
        else:
            boundary_speed_ratio = math.nan
            boundary_joint_activity_ratio = math.nan
        saturation = float((np.abs(angles) >= JOINT_LIMIT - 1e-6).mean())
        label = f"speed_{int(round(requested_speed * 100)):03d}"
        artifact = args.output_dir / f"{label}.npz"
        np.savez_compressed(
            artifact,
            angles=angles,
            root_position=root,
            root_quaternion=quaternion,
            features=features,
            fetch_command=command,
            requested_source_speed_mps=np.float32(requested_speed),
            realized_source_equivalent_speed_mps=np.float32(equivalent_speed),
            realized_fetch_path_speed=np.float32(generated_path_speed),
            realized_fetch_forward_speed=np.float32(generated_forward_speed),
            instantaneous_source_equivalent_speed_mps=instantaneous_equivalent_speed,
            smoothed_source_equivalent_speed_mps=smoothed_equivalent_speed,
            instantaneous_joint_activity_rad_s=joint_activity,
            command_scale_fetch_displacement_per_mps=np.float32(scale),
            command_horizon_seconds=np.float32(COMMAND_HORIZON_SECONDS),
            seed_session=np.asarray(seed_session),
            seed_start=np.int32(seed_start),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            dataset_repository_id=np.asarray(manifest["repository_id"]),
            fps=np.int32(FPS),
        )
        row = {
            "label": label,
            "requested_source_speed_mps": requested_speed,
            "fetch_command": command.tolist(),
            "realized_fetch_path_speed": generated_path_speed,
            "realized_fetch_forward_speed": generated_forward_speed,
            "realized_source_equivalent_speed_mps": equivalent_speed,
            "speed_equivalent_p10_p50_p90_mps": speed_quantiles.tolist(),
            "low_speed_dwell_fraction": pause_fraction,
            "longest_low_speed_run_seconds": longest_low_speed_seconds,
            "old_32_frame_boundary_speed_ratio": boundary_speed_ratio,
            "old_32_frame_boundary_joint_activity_ratio": boundary_joint_activity_ratio,
            "joint_limit_fraction": saturation,
            "artifact": str(artifact),
        }
        rows.append(row)
        traces.append(
            {
                "requested_speed": requested_speed,
                "instantaneous_speed": instantaneous_equivalent_speed,
                "smoothed_speed": smoothed_equivalent_speed,
                "joint_activity": joint_activity,
                "root_height": root[:, 2],
                "pause_fraction": pause_fraction,
                "seed_frames": seed_frames,
            }
        )
        print(
            f"[{label}] request={requested_speed:.2f} m/s -> command dx={command[0]:.3f} "
            f"Fetch | realized~{equivalent_speed:.3f} m/s equivalent | "
            f"p10/50/90={speed_quantiles[0]:.3f}/{speed_quantiles[1]:.3f}/"
            f"{speed_quantiles[2]:.3f} | low-speed={pause_fraction:.1%}, "
            f"longest={longest_low_speed_seconds:.2f}s | boundary ratio="
            f"{boundary_speed_ratio:.2f} | {artifact}",
            flush=True,
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_repository_id": manifest["repository_id"],
        "command_calibration": checkpoint.get(
            "command_calibration",
            {
                "method": "legacy empirical calibration",
                "fetch_displacement_per_mps": scale,
                "horizon_seconds": COMMAND_HORIZON_SECONDS,
            },
        ),
        "seed": {
            "session": seed_session,
            "source_start": seed_start,
            "source_speed_mps": float(train.source_speed_mps[seed_index]),
            "command": train.command[seed_index].tolist(),
        },
        "rollout": "deterministic conditional Gaussian mean; one-token receding horizon",
        "frames": frames,
        "seed_frames": config.history_tokens * config.downsample,
        "fps": FPS,
        "videos": rows,
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    speed_plot = args.output_dir / "speed_timeseries.png"
    plot_speed_traces(traces, speed_plot)
    print(f"wrote {report_path}", flush=True)
    print(f"wrote {speed_plot}", flush=True)


if __name__ == "__main__":
    main()
