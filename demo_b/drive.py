"""Drive the motor model with a FIXED command: go straight / walk in cycles. Renders raw | re-anchored + a path plot.

    python demo_b/drive.py [--seconds 16] [--render]
"""
import argparse, math
from pathlib import Path
import numpy as np
from constants import n_steps_for_seconds
from models import load_motor
from rollout import roll, cmd_at, render_compare
import foot_metrics as FM

OUT = Path(__file__).resolve().parent / "out"


def plot_path(paths, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    for label, gq, col in paths:
        ax.plot(gq[:, 0], gq[:, 1], color=col, lw=2.2, label=label)
        ax.scatter(gq[0, 0], gq[0, 1], color=col, s=42, zorder=5, edgecolor="white", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=.25, lw=.6)
    ax.legend(frameon=False, fontsize=9); ax.set_title("root path under a fixed command", fontsize=11)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out); print(f"  [plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16.0); ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    mv, m, norms, seed = load_motor()
    cs = np.array([cmd_at(seed["xy"], seed["yaw"], f0) for f0 in range(0, len(seed["feat"]) - 32, 16)])
    fwd = float(np.linalg.norm(cs[:, :2], axis=1).mean()); n = n_steps_for_seconds(a.seconds); turn = 2 * math.pi / n
    print(f"seed {seed['name']}  fwd={fwd:.3f} m/step  turn={turn:.3f} rad/step  {n} steps", flush=True)
    cmds = {"straight": np.array([fwd, 0.0, 0.0], np.float32), "circle": np.array([fwd, 0.0, turn], np.float32)}
    paths = []
    for name, cmd in cmds.items():
        gq = roll(m, mv, norms, seed, n, command="fixed", cmd_raw=cmd)
        FM.report(gq, f"drive/{name}")
        sp = gq[:, :2].max(0) - gq[:, :2].min(0)
        print(f"    {name}: span {sp[0]:.2f} x {sp[1]:.2f} m, net {np.linalg.norm(gq[-1,:2]-gq[0,:2]):.2f} m", flush=True)
        paths.append((name, gq, "#e0a45e" if name == "straight" else "#35d0bf"))
        if a.render:                                                  # left: raw (floor only, drifts) | right: re-anchored
            render_compare(FM.fix_floor(gq), FM.ground(gq, alpha=0.03), f"drive:{name}", OUT / f"drive_{name}.mp4")
    plot_path(paths, OUT / "drive_path.png")


if __name__ == "__main__":
    main()
