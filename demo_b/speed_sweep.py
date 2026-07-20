"""Render the same frozen Demo B model at several straight-line speeds.

This intentionally loads the checkpoint once so every panel differs only in
the command.  It also accepts the pre-Demo-E ``rl_standalone`` asset for exact
regression comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .constants import FPS, n_steps_for_seconds
    from .foot_metrics import ground, report
    from .models import load_motor
    from .rollout import render, roll
except ImportError:  # pragma: no cover - direct workshop script entry point.
    from constants import FPS, n_steps_for_seconds
    from foot_metrics import ground, report
    from models import load_motor
    from rollout import render, roll


OUT = Path(__file__).resolve().parent / "out"
COMMAND_FRAMES = 31


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.10, 0.15, 0.20, 0.25))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--prefix", default="restored_coltrane")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    motion, transition, norms, seed = load_motor(args.asset) if args.asset else load_motor()
    steps = n_steps_for_seconds(args.seconds)
    print(
        f"seed {seed['name']} | feature_dim={len(norms['mmean'])} | "
        f"{steps} rollout steps (~{args.seconds:g}s)",
        flush=True,
    )
    rows = []
    for speed in args.speeds:
        command = np.asarray([speed * COMMAND_FRAMES / FPS, 0.0, 0.0], np.float32)
        qpos = roll(
            transition,
            motion,
            norms,
            seed,
            steps,
            command="fixed",
            cmd_raw=command,
        )
        metrics = report(qpos, f"v={speed:.2f}")
        achieved = float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).mean() * FPS)
        net = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
        print(
            f"  command={speed:.2f} m/s -> realized={achieved:.3f} m/s, "
            f"net={net:.2f} m, jerk={metrics['jerk']:.1f}",
            flush=True,
        )
        rows.append((speed, achieved, qpos))
        if args.render:
            tag = f"v{round(speed * 100):03d}"
            render(
                ground(qpos, alpha=0.03),
                f"restored Coltrane | cmd {speed:.2f} m/s",
                OUT / f"{args.prefix}_{tag}_straight.mp4",
            )

    output = OUT / f"{args.prefix}_speed_metrics.npz"
    np.savez(
        output,
        commanded=np.asarray([row[0] for row in rows], np.float32),
        realized=np.asarray([row[1] for row in rows], np.float32),
    )
    print(f"  [metrics] {output}", flush=True)


if __name__ == "__main__":
    main()
