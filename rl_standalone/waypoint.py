"""Closed-loop HEURISTIC waypoint controller (no RL): steer the egocentric command to custom goals.

Each step, decode the stream so far to read the current (xy, yaw), compute the goal bearing in the root frame,
and steer -- turn toward the goal (clamped to the feasible range) while always walking forward a little (arcs,
since in-place turns are out of the loco distribution). Closed-loop, so imperfect command adherence is corrected.

    python rl_standalone/waypoint.py [--shape square] [--max_steps 120] [--render]
"""
import argparse, math
from pathlib import Path
import numpy as np, torch
from constants import DEV, CLIP, H, NLAT, DM, FM, FPS
from geometry import reconstruct_qpos, q0_from, quat2yaw
from models import load_motor
from rollout import cmd_at, render
import foot_metrics as met

OUT = Path(__file__).resolve().parent / "out"
SHAPES = {
    "square": [(1.4, 0.0), (1.4, 1.4), (0.0, 1.4), (0.0, 0.0)],
    "zigzag": [(1.2, 0.6), (2.4, -0.6), (3.6, 0.6), (4.8, 0.0)],
    "star": [(1.5, 0.0), (0.5, 1.4), (-0.4, 0.2), (1.2, 0.9), (0.0, -0.4)],
}


def steer(cur_xy, cur_yaw, goal, fwd_max, turn_max, stop_dist=0.5):
    d = np.asarray(goal) - cur_xy; dist = float(np.hypot(*d))
    err = (math.atan2(d[1], d[0]) - cur_yaw + math.pi) % (2 * math.pi) - math.pi
    dpsi = float(np.clip(err, -turn_max, turn_max)); align = max(0.0, math.cos(err))
    fwd = fwd_max * (0.3 + 0.7 * align) * min(1.0, dist / stop_dist)
    return np.array([fwd, 0.0, dpsi], np.float32), dist


@torch.no_grad()
def run(m, mv, norms, seed, goals, fwd_max, turn_max, max_steps=120, reach=0.18):
    mm, ms = norms["mmean"], norms["mstd"]
    zmean = torch.tensor(norms["zmean"], device=DEV); zstd = torch.tensor(norms["zstd"], device=DEV)
    cmean = torch.tensor(norms["cmean"], device=DEV); cstd = torch.tensor(norms["cstd"], device=DEV)
    sidx = torch.full((1,), seed["sid"], device=DEV)
    zm0 = mv.encode(torch.tensor((seed["feat"][:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    stream = [((zm0 - zmean[0]) / zstd[0])[:H]]

    def state():
        Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
        if nn_ == 0:
            return np.zeros((1, 74)), np.zeros(2), 0.0
        gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
        gq = reconstruct_qpos(gfeat, q0_from(gfeat[0], np.zeros(2), 0.0))
        return gq, gq[-1, :2], float(quat2yaw(gq[-1, 3:7]))

    gi = 0; log = []; gq = np.zeros((1, 74))
    for step in range(max_steps):
        gq, cur_xy, cur_yaw = state()
        cmd, dist = steer(cur_xy, cur_yaw, goals[gi], fwd_max, turn_max)
        if dist < reach:
            log.append((gi, step, len(gq) / FPS, round(dist, 3))); gi += 1
            if gi >= len(goals):
                break
            continue
        cmdn = (torch.tensor(cmd[None], device=DEV) - cmean) / cstd
        stream.append(m.predict(torch.cat(stream, 0)[-H:][None], cmdn, sidx)[0])
    reached = {r[0] for r in log}
    finals = [round(float(np.hypot(*(np.asarray(g) - gq[:, :2]).T).min()), 3) for g in goals]
    return gq, log, reached, finals


def plot(gq, goals, reached, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=130)
    ax.plot(gq[:, 0], gq[:, 1], color="#35d0bf", lw=2.0, zorder=2, label="path")
    ax.scatter(gq[0, 0], gq[0, 1], color="#35d0bf", s=48, zorder=4, edgecolor="white", lw=1, label="start")
    for i, g in enumerate(goals):
        ax.scatter(*g, s=150, marker="*", zorder=5, color="#79d38f" if i in reached else "#e0655e",
                   edgecolor="white", lw=1)
        ax.annotate(f"{i+1}", g, textcoords="offset points", xytext=(7, 6), fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.25, lw=.6); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"heuristic waypoint reaching  ({len(reached)}/{len(goals)})", fontsize=11)
    ax.legend(frameon=False, fontsize=9); fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out); print(f"  [plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="square", choices=list(SHAPES))
    ap.add_argument("--max_steps", type=int, default=120); ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    mv, m, norms, seed = load_motor()
    cs = np.array([cmd_at(seed["xy"], seed["yaw"], f0) for f0 in range(0, len(seed["feat"]) - 32, 16)])
    fwd_max = float(np.percentile(np.linalg.norm(cs[:, :2], axis=1), 85))
    turn_max = float(np.percentile(np.abs(cs[:, 2]), 90))
    goals = SHAPES[a.shape]
    print(f"shape={a.shape}  {len(goals)} wp  fwd_max={fwd_max:.3f}  turn_max={turn_max:.3f}", flush=True)
    gq, log, reached, finals = run(m, mv, norms, seed, goals, fwd_max, turn_max, a.max_steps)
    for gi, step, t, dist in log:
        print(f"  reached wp{gi+1} at step {step:2d} ({t:4.1f}s), dist {dist:.3f} m", flush=True)
    print(f"REACHED {len(reached)}/{len(goals)}  |  final min-dist (m): {finals}", flush=True)
    met.report(gq, f"wp/{a.shape}")
    plot(gq, goals, reached, OUT / f"waypoint_{a.shape}.png")
    if a.render:
        render(met.ground(gq, alpha=0.03), f"waypoints:{a.shape} ({len(reached)}/{len(goals)})", OUT / f"waypoint_{a.shape}.mp4")


if __name__ == "__main__":
    main()
