"""Summarize the frozen three-seed matched PPO comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_c.config import TRAIN_SEEDS, VARIANTS

OUT = Path(__file__).resolve().parent / "out"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(TRAIN_SEEDS))
    args = parser.parse_args()
    summary = {}
    for variant in VARIANTS:
        rows = []
        for seed in args.seeds:
            path = OUT / "metrics" / f"{variant}_seed{seed}.json"
            if not path.exists():
                raise SystemExit(f"missing {path}; run demo_c.train first")
            rows.append(json.loads(path.read_text())["metrics"])
        summary[variant] = {}
        for metric in ("success_rate", "return_mean", "final_distance_mean", "invalid_rate"):
            values = np.array([row[metric] for row in rows])
            summary[variant][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    delta = summary["wam"]["success_rate"]["mean"] - summary["goal_only"]["success_rate"]["mean"]
    noise = 2 * max(summary[v]["success_rate"]["std"] for v in VARIANTS)
    summary["decision"] = {
        "wam_minus_goal_only_success": delta,
        "eta_2sigma": noise,
        "interpretation": "beyond_noise" if abs(delta) > noise else "within_noise",
    }
    path = OUT / "comparison.json"; path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
