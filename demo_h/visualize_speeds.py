"""Roll out one Demo H policy at several commanded speeds.

This is intentionally separate from the matched H0/H1/H2 evaluator: it is a
qualitative generalization probe, not part of the fixed-target algorithm score.
Each trace records the commanded and realized speed before it is rendered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_h.config import DT, OUT
from demo_h.evaluate import rollout_arm, save_trace, summarize
from demo_h.gait_metrics import (
    four_limb_contact_metrics,
    four_limb_locomotion_metrics,
)
from demo_h.prior import DEFAULT_PRIOR, load_prior


DEFAULT_SPEEDS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def speed_label(speed: float) -> str:
    return f"{speed:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("h1", "h2"), default="h1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--label", type=str)
    parser.add_argument("--output-dir", type=Path, default=OUT / "speed_sweep")
    args = parser.parse_args()

    prior = load_prior(args.prior)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_speeds = np.asarray(args.speeds, dtype=np.float32)
    seeds = tuple(args.seed for _ in target_speeds)
    _, initial, stream = rollout_arm(
        args.arm,
        args.checkpoint,
        prior,
        seeds,
        args.steps,
        target_speed=target_speeds,
    )
    episodes = summarize(
        stream,
        seeds,
        args.steps,
        target_speed=target_speeds,
    )
    rows = []
    for index, (target_speed, episode) in enumerate(zip(target_speeds, episodes)):
        target_speed = float(target_speed)
        trace_path = args.output_dir / (
            f"{args.arm}_command-{speed_label(target_speed)}mps.npz"
        )
        save_trace(trace_path, initial, stream, batch_index=index)
        speed = np.asarray(stream[2])[:, index]
        contacts = np.asarray(stream[7])[:, index, 56:60]
        features = np.asarray(stream[7])[:, index]
        alive = np.concatenate(
            (
                (True,),
                np.cumprod(
                    ~np.asarray(stream[1])[:-1, index].astype(bool)
                ).astype(bool),
            )
        )
        row = {
            "commanded_speed_mps": float(target_speed),
            "realized_speed_mean_mps": float(speed[alive].mean()),
            "realized_speed_std_mps": float(speed[alive].std()),
            "forward_displacement_m": float(speed[alive].sum() * DT),
            "trace": str(trace_path),
            "four_limb_gait": four_limb_contact_metrics(contacts[alive]),
            "four_limb_stride": four_limb_locomotion_metrics(features[alive]),
            **episode["metrics"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    training_report_path = args.checkpoint.with_suffix(".json")
    training = (
        json.loads(training_report_path.read_text())
        if training_report_path.exists()
        else None
    )
    label = args.label
    if label is None and training is not None:
        label = f"beta={training['beta']:g}"
    report = {
        "schema": "demo-h-speed-sweep-v1",
        "arm": args.arm,
        "label": label or args.arm,
        "checkpoint": str(args.checkpoint),
        "prior": str(args.prior),
        "training": training,
        "seed": args.seed,
        "steps": args.steps,
        "seconds": args.steps * DT,
        "note": "Qualitative command sweep; gait diagnostics do not enter training.",
        "speeds": rows,
    }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
