"""Drive the motor model straight at a set of TARGET SPEEDS (m/s) and compare commanded vs achieved.

    python rl_standalone/speed_sweep.py [--speeds 0.10 0.15 0.20 0.25] [--seconds 10] [--render]

The command is an egocentric displacement over the 32-frame (0.64 s) step, so cmd_dx = v * 32/FPS.
Renders a 2x2 grid of grounded rollouts + a path/speed plot.
"""
import argparse
from pathlib import Path
import numpy as np
from constants import FPS, n_steps_for_seconds
from models import load_motor
from rollout import roll, render
import foot_metrics as FM

OUT = Path(__file__).resolve().parent / "out"
STEP_FRAMES = 32                                                  # frames spanned by one command (see rollout.cmd_at)
COLS = ["#e0a45e", "#35d0bf", "#7b8ff5", "#e06e8f"]


def plot(rows, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=130)
    for (v, gq, sp, col) in rows:
        ax.plot(gq[:, 0], gq[:, 1], color=col, lw=2.2, label=f"{v:.2f} m/s cmd")
        ax.scatter(gq[0, 0], gq[0, 1], color=col, s=42, zorder=5, edgecolor="white", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.25, lw=.6)
    ax.legend(frameon=False, fontsize=8); ax.set_title("root path, straight command", fontsize=11)
    cmd = [r[0] for r in rows]; ach = [r[2] for r in rows]
    lim = [0, max(cmd + ach) * 1.15]
    bx.plot(lim, lim, color="#888", lw=1, ls="--", label="ideal")
    bx.plot(cmd, ach, "o-", color="#35d0bf", lw=2, ms=7)
    bx.set_xlim(lim); bx.set_ylim(lim); bx.set_aspect("equal")
    bx.set_xlabel("commanded (m/s)"); bx.set_ylabel("achieved (m/s)"); bx.grid(alpha=.25, lw=.6)
    bx.legend(frameon=False, fontsize=8); bx.set_title("command tracking", fontsize=11)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out); print(f"  [plot] {out}")


def render_grid(items, out):
    """items = [(label, gq)]; tile into a square-ish grid, one shared video."""
    import mujoco, numpy as np
    from rollout import _frames, _writer
    from mujoco_rodent import build_model, RH, RW
    media = _writer(out); model = build_model(); d = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=RH, width=RW)
    panels = [_frames(r, model, d, gq, lab) for lab, gq in items]; r.close()
    n = len(panels); ncol = 2 if n > 1 else 1; nrow = -(-n // ncol)
    T = min(len(p) for p in panels)
    blank = np.zeros_like(panels[0][0])
    grid = [np.concatenate([np.concatenate([(panels[i * ncol + c][t] if i * ncol + c < n else blank)
                                            for c in range(ncol)], 1) for i in range(nrow)], 0) for t in range(T)]
    media.write_video(str(out), np.stack(grid), fps=FPS); print(f"  [video] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25])
    ap.add_argument("--seconds", type=float, default=10.0); ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    mv, m, norms, seed = load_motor()
    n = n_steps_for_seconds(a.seconds)
    print(f"seed {seed['name']}  {n} steps (~{a.seconds:.0f}s)  step = {STEP_FRAMES}f = {STEP_FRAMES/FPS:.2f}s", flush=True)
    rows, vids = [], []
    for v, col in zip(a.speeds, COLS * 4):
        cmd = np.array([v * STEP_FRAMES / FPS, 0.0, 0.0], np.float32)
        gq = roll(m, mv, norms, seed, n, command="fixed", cmd_raw=cmd)
        FM.report(gq, f"v={v:.2f}")
        ach = float(np.linalg.norm(np.diff(gq[:, :2], axis=0), axis=1).mean() * FPS)
        net = float(np.linalg.norm(gq[-1, :2] - gq[0, :2]))
        print(f"    cmd {v:.2f} m/s (dx_ego={cmd[0]:.3f} m/step) -> achieved {ach:.3f} m/s, net {net:.2f} m", flush=True)
        rows.append((v, gq, ach, col)); vids.append((f"cmd {v:.2f} m/s", FM.ground(gq, alpha=0.03)))
    plot(rows, OUT / "speed_sweep.png")
    if a.render:
        render_grid(vids, OUT / "speed_sweep.mp4")


if __name__ == "__main__":
    main()
