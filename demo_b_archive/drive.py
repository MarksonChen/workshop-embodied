"""demo_b/drive.py -- drive the canonical motor model with a FIXED egocentric command and visualize the path.

Applies one constant command every rollout step (the DART "keep walking" test):
  straight : [fwd, 0, 0]         -- forward, no turn
  circle   : [fwd, 0, turn]      -- forward + constant yaw rate -> walks in a loop
`fwd` is the seed's typical per-step planar displacement (in-distribution); `turn` is sized so the roll traces
~one loop. Renders the gait (close camera) + a top-down root-path plot (the tracking camera can't show the loop),
and scores foot quality. No neural.

Run:  uv run python -m demo_b.drive [--seconds 16] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse, math
import numpy as np, torch
import canvas as C
from canvas.train import ddim, H, NLAT, DM, CLIP, FM, DEV
from canvas.autoresearch.stage3.motor_rollout import _cmd_at, _q0
from canvas.prepare import reconstruct_qpos
from canvas import utils as U
from demo_b import foot_metrics as FM_
from demo_b.rollout import load_motor, seed_segment, render_compare


@torch.no_grad()
def roll_cmd(m, mv, norms, seg, n_steps, cmd_raw, cfg=1.5):
    """autoregressive rollout applying the fixed egocentric command cmd_raw (3,) every step -> qpos(T,74)."""
    feat, xy, yaw, sid = seg["feat"], seg["xy"], seg["yaw"], seg["sid"]
    mm, ms = norms["mmean"], norms["mstd"]
    zmean, zstd = torch.tensor(norms["zmean"], device=DEV), torch.tensor(norms["zstd"], device=DEV)
    cmean, cstd = torch.tensor(norms["cmean"], device=DEV), torch.tensor(norms["cstd"], device=DEV)
    sidx = torch.full((1,), sid, device=DEV)
    zm0 = mv.encode(torch.tensor((feat[:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    z = ((zm0 - zmean[0]) / zstd[0])[:H]; stream = [z]
    cmd = (torch.tensor(cmd_raw[None], dtype=torch.float32, device=DEV) - cmean) / cstd
    for _ in range(n_steps):
        hist = torch.cat(stream, 0)[-H:][None]
        stream.append(ddim(m, m.context(hist), cmd, sidx, cfg=cfg)[0])
    Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
    gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
    return reconstruct_qpos(gfeat, _q0(gfeat[0], xy[0], yaw[0]))


def plot_path(paths, out):
    """top-down root-xy trajectories -> png."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    for label, gq, col in paths:
        ax.plot(gq[:, 0], gq[:, 1], color=col, lw=2.2, label=label)
        ax.scatter(gq[0, 0], gq[0, 1], color=col, s=42, zorder=5, edgecolor="white", linewidth=1)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25, lw=.6)
    ax.set_title("root path under a fixed command", fontsize=11)
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print(f"  [plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    mv, m, norms = load_motor()
    seg = seed_segment(min_len=CLIP * 4)
    cs = np.array([_cmd_at(seg["xy"], seg["yaw"], f0) for f0 in range(0, len(seg["feat"]) - 32, 16)])
    fwd = float(np.linalg.norm(cs[:, :2], axis=1).mean())              # typical per-step planar displacement
    n_steps = U.n_steps_for_seconds(a.seconds)
    turn = 2 * math.pi / n_steps                                      # ~one full loop over the rollout
    print(f"seed {seg['name']} @ {seg['start']}; fwd={fwd:.4f} m/step, turn={turn:.3f} rad/step, {n_steps} steps",
          flush=True)
    cmds = {"straight": np.array([fwd, 0.0, 0.0], np.float32),
            "circle": np.array([fwd, 0.0, turn], np.float32)}
    outdir = C.HERE.parent / "demo_b" / "out"; paths = []
    for name, cmd in cmds.items():
        gq = roll_cmd(m, mv, norms, seg, n_steps, cmd)
        FM_.report(gq, label=f"drive/{name}")
        span = gq[:, :2].max(0) - gq[:, :2].min(0)
        print(f"    {name}: path span {span[0]:.2f} x {span[1]:.2f} m, net displacement "
              f"{np.linalg.norm(gq[-1,:2]-gq[0,:2]):.2f} m", flush=True)
        paths.append((name, gq, "#e0a45e" if name == "straight" else "#35d0bf"))
        if a.render:                                                    # left: raw (open-loop, flies) | right: re-anchored
            render_compare(FM_.fix_floor(gq), FM_.ground(gq, alpha=0.03), f"drive:{name}", outdir / f"drive_{name}.mp4")
    plot_path(paths, outdir / "drive_path.png")


if __name__ == "__main__":
    main()
