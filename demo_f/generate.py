"""Generate and diagnose command-conditioned Demo F motion rollouts.

Pure prior logic lives in :mod:`demo_f.prior`; this module is deliberately a
thin artifact/plotting front that can be replaced by notebook cells later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import sha256
from .config import FPS, JOINT_LIMIT, OUT
from .dataset import load_manifest, load_split
from .dataset.contract import COMMAND_FRAME, DYNAMIC_ROOT
from .features import SL
from .prior import (
    COMMAND_HORIZON_SECONDS,
    SPEED_SMOOTHING_FRAMES,
    checkpoint_command_scale,
    integrate_root,
    load_prior,
    longest_true_run,
    select_seed,
    trailing_mean,
)


DEFAULT_SPEEDS = (0.10, 0.15, 0.20, 0.25)


def plot_speed_traces(traces: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=150, sharex=True)
    if len(traces) != len(axes.flat):
        raise ValueError("the workshop plot expects exactly four speed traces")
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
        axis.axhline(requested, color="#e06e8f", linestyle="--", linewidth=1.2)
        axis.axvline(trace["seed_frames"] / FPS, color="#777", linestyle=":")
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
    figure.suptitle("Demo F autoregressive root-speed diagnostics")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def generate_rollouts(
    checkpoint_path: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    speeds=DEFAULT_SPEEDS,
    seconds: float = 4.0,
    seed_speed: float = 0.15,
) -> dict:
    """Generate the four workshop trajectories and return their report."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    manifest_path = dataset_root / "manifest.json"
    manifest = load_manifest(dataset_root)
    prior = load_prior(checkpoint_path)
    checkpoint, config = prior.checkpoint, prior.config
    if checkpoint["dataset_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("checkpoint was not trained from this dataset manifest")

    train = load_split("train", dataset_root)
    scale = checkpoint_command_scale(checkpoint, train)
    seed_index = select_seed(train, seed_speed, scale=scale)
    seed_session = train.sessions[int(train.session_index[seed_index])]
    seed_start = int(train.source_start[seed_index])
    frames = int(round(seconds * FPS))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(checkpoint_path)
    rows, traces = [], []
    print(
        f"device={prior.device} | scale={scale:.4f} Fetch displacement/(m/s) | "
        f"seed={seed_session}@{seed_start}",
        flush=True,
    )
    for requested_speed in speeds:
        command = np.asarray((scale * requested_speed, 0.0, 0.0), np.float32)
        features = prior.rollout(train.features[seed_index], command, frames)
        seed_frames = config.history_tokens * config.downsample
        angles, root, quaternion = integrate_root(features)
        generated_root = root[seed_frames:]
        generated_path_speed = float(
            np.linalg.norm(np.diff(generated_root[:, :2], axis=0), axis=1).mean()
            * FPS
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
        old_boundaries = np.arange(
            seed_frames + COMMAND_FRAME, frames, COMMAND_FRAME
        )
        if len(old_boundaries):
            boundary_index = np.unique(
                np.concatenate(
                    [
                        np.arange(
                            max(evaluation_start, frame - 2),
                            min(frames, frame + 3),
                        )
                        for frame in old_boundaries
                    ]
                )
            )
            post_seed_index = np.arange(evaluation_start, frames)
            ordinary_index = post_seed_index[
                ~np.isin(post_seed_index, boundary_index)
            ]
            boundary_speed_ratio = float(
                smoothed_equivalent_speed[boundary_index].mean()
                / smoothed_equivalent_speed[ordinary_index].mean()
            )
            boundary_joint_activity_ratio = float(
                joint_activity[boundary_index].mean()
                / joint_activity[ordinary_index].mean()
            )
        else:
            boundary_speed_ratio = math.nan
            boundary_joint_activity_ratio = math.nan
        saturation = float((np.abs(angles) >= JOINT_LIMIT - 1e-6).mean())
        label = f"speed_{int(round(requested_speed * 100)):03d}"
        artifact = output_dir / f"{label}.npz"
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
                "pause_fraction": pause_fraction,
                "seed_frames": seed_frames,
            }
        )
        print(
            f"[{label}] request={requested_speed:.2f} m/s -> dx={command[0]:.3f} | "
            f"realized~{equivalent_speed:.3f} m/s | low-speed={pause_fraction:.1%}",
            flush=True,
        )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_repository_id": manifest["repository_id"],
        "command_calibration": checkpoint.get("command_calibration"),
        "seed": {
            "session": seed_session,
            "source_start": seed_start,
            "clip_command": train.command[seed_index].tolist(),
        },
        "rollout": "deterministic conditional Gaussian mean; one-token receding horizon",
        "frames": frames,
        "seed_frames": config.history_tokens * config.downsample,
        "fps": FPS,
        "videos": rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_speed_traces(traces, output_dir / "speed_timeseries.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT / "generated")
    parser.add_argument("--speeds", type=float, nargs=4, default=DEFAULT_SPEEDS)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--seed-speed", type=float, default=0.15)
    args = parser.parse_args()
    report = generate_rollouts(
        args.checkpoint,
        args.dataset_root,
        args.output_dir,
        speeds=args.speeds,
        seconds=args.seconds,
        seed_speed=args.seed_speed,
    )
    print(f"wrote {args.output_dir / 'metrics.json'} ({len(report['videos'])} speeds)")


if __name__ == "__main__":
    main()
