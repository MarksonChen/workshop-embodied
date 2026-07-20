"""Drive the motor model with a fixed straight or circular command.

By default the videos show the floor-aligned/re-anchored kinematic generation.
Pass ``--compare`` to render the raw integrated trajectory beside it.

    python demo_b/drive.py [--seconds 16] [--render]
"""
import argparse
from pathlib import Path
import numpy as np
from constants import n_steps_for_seconds
from models import load_motor
from rollout import roll, cmd_at, render, render_compare
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
    ap.add_argument("--speed", type=float, help="commanded forward speed in m/s")
    ap.add_argument("--yaw-rate", type=float, default=0.75, help="circle yaw command in rad/s")
    ap.add_argument("--prefix", default="drive")
    ap.add_argument("--compare", action="store_true", help="show raw and re-anchored views side by side")
    ap.add_argument("--asset", type=Path, help="alternate Demo B checkpoint (legacy 281-D or format-v3 85-D)")
    ap.add_argument("--mode", choices=("straight", "circle", "both"), default="both")
    a = ap.parse_args()
    mv, m, norms, seed = load_motor(a.asset) if a.asset else load_motor()
    cs = np.array([cmd_at(seed["xy"], seed["yaw"], f0) for f0 in range(0, len(seed["feat"]) - 32, 16)])
    horizon = 31 / 50
    fwd = float(np.linalg.norm(cs[:, :2], axis=1).mean()) if a.speed is None else horizon * a.speed
    n = n_steps_for_seconds(a.seconds)
    turn = horizon * a.yaw_rate
    print(
        f"seed {seed['name']}  command_v={fwd / horizon:.3f} m/s  "
        f"circle_yaw={a.yaw_rate:.3f} rad/s  {n} steps",
        flush=True,
    )
    cmds = {
        "straight": np.array([fwd, 0.0, 0.0], np.float32),
        "circle": np.array([fwd, 0.0, turn], np.float32),
    }
    if a.mode != "both":
        cmds = {a.mode: cmds[a.mode]}
    paths = []
    for name, cmd in cmds.items():
        gq = roll(m, mv, norms, seed, n, command="fixed", cmd_raw=cmd)
        FM.report(gq, f"drive/{name}")
        sp = gq[:, :2].max(0) - gq[:, :2].min(0)
        print(f"    {name}: span {sp[0]:.2f} x {sp[1]:.2f} m, net {np.linalg.norm(gq[-1,:2]-gq[0,:2]):.2f} m", flush=True)
        paths.append((name, gq, "#e0a45e" if name == "straight" else "#35d0bf"))
        if a.render:
            destination = OUT / f"{a.prefix}_{name}.mp4"
            label = f"{a.prefix}:{name} cmd={fwd / horizon:.2f}m/s"
            grounded = FM.ground(gq, alpha=0.03)
            if a.compare:
                render_compare(FM.fix_floor(gq), grounded, label, destination)
            else:
                render(grounded, label, destination)
    plot_path(paths, OUT / f"{a.prefix}_path.png")


if __name__ == "__main__":
    main()
