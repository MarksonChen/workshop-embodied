"""Roll out Demo H across commands and optionally render one comparison video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_f.features import SL
from demo_f.metrics import four_limb_contact_metrics, four_limb_locomotion_metrics
from demo_h.config import DT, OUT
from demo_h.evaluate import rollout_arm, save_trace, summarize
from demo_h.prior import DEFAULT_PRIOR, load_prior


DEFAULT_SPEEDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def speed_label(speed: float) -> str:
    return f"{speed:.3f}".replace(".", "p")


def rollout_speeds(
    checkpoint: Path,
    prior_path: Path,
    output_dir: Path,
    *,
    arm: str = "h2",
    speeds=DEFAULT_SPEEDS,
    steps: int = 250,
    seed: int = 101,
    label: str | None = None,
) -> dict:
    """Generate traces and validation-only naturalness metrics for one policy."""

    if arm not in {"h1", "h2"}:
        raise ValueError("Demo H policy sweeps support h1 or h2")
    checkpoint, prior_path, output_dir = map(Path, (checkpoint, prior_path, output_dir))
    prior = load_prior(prior_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_speeds = np.asarray(speeds, dtype=np.float32)
    seeds = tuple(seed for _ in target_speeds)
    _, initial, stream = rollout_arm(
        arm,
        checkpoint,
        prior,
        seeds,
        steps,
        target_speed=target_speeds,
    )
    episodes = summarize(stream, seeds, steps, target_speed=target_speeds)
    rows = []
    for index, (target_speed, episode) in enumerate(
        zip(target_speeds, episodes, strict=True)
    ):
        target_speed = float(target_speed)
        trace_path = output_dir / f"{arm}_command-{speed_label(target_speed)}ups.npz"
        save_trace(trace_path, initial, stream, batch_index=index)
        speed = np.asarray(stream[2])[:, index]
        features = np.asarray(stream[7])[:, index]
        contacts = features[:, slice(*SL["contacts"])]
        alive = np.concatenate(
            ((True,), np.cumprod(~np.asarray(stream[1])[:-1, index].astype(bool)).astype(bool))
        )
        row = {
            "commanded_speed_fetch_units_per_s": target_speed,
            "realized_speed_mean_fetch_units_per_s": float(speed[alive].mean()),
            "realized_speed_std_fetch_units_per_s": float(speed[alive].std()),
            "forward_displacement_fetch_units": float(speed[alive].sum() * DT),
            "trace": str(trace_path),
            "four_limb_gait": four_limb_contact_metrics(contacts[alive]),
            "four_limb_stride": four_limb_locomotion_metrics(features[alive]),
            **episode["metrics"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    sidecar = checkpoint.with_suffix(".json")
    training = json.loads(sidecar.read_text()) if sidecar.is_file() else None
    if label is None and training is not None:
        label = f"beta={training['beta']:g}"
    report = {
        "schema": "demo-h-speed-sweep-v2",
        "arm": arm,
        "label": label or arm,
        "checkpoint": str(checkpoint),
        "prior": str(prior_path),
        "training": training,
        "seed": seed,
        "steps": steps,
        "seconds": steps * DT,
        "note": "Qualitative command sweep; gait diagnostics are validation-only.",
        "speeds": rows,
    }
    output = output_dir / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("h1", "h2"), default="h2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--label")
    parser.add_argument("--output-dir", type=Path, default=OUT / "speed_sweep")
    args = parser.parse_args()
    rollout_speeds(
        args.checkpoint,
        args.prior,
        args.output_dir,
        arm=args.arm,
        speeds=args.speeds,
        steps=args.steps,
        seed=args.seed,
        label=args.label,
    )


if __name__ == "__main__":
    main()
