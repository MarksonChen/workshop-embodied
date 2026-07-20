"""Plot the four diagnostics used to audit a Demo D PPO run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from demo_d.config import OUT
from demo_d.runtime import latest_metrics_path, resolve_run


def _values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = resolve_run(args.run)
    log_path = latest_metrics_path(run)
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty progress log: {log_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = _values(rows, "step") / 1e6
    panels = (
        ("eval/avg_episode_length", "held-out episode length", "control steps"),
        ("eval/episode_reward", "held-out episodic return", "return"),
        ("training/kl_mean", "PPO KL diagnostic", "KL"),
        ("training/sps", "training throughput", "physics steps/s"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), dpi=150, sharex=True)
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        axis.plot(steps, _values(rows, key), marker="o", color="#35a7a0", linewidth=1.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("physics steps (millions)")
    fig.suptitle(f"Demo D learning audit — {run.name}")
    fig.tight_layout()
    output = Path(args.out) if args.out else OUT / "figures" / f"learning-{run.name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
