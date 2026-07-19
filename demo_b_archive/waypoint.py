"""demo_b/waypoint.py -- closed-loop HEURISTIC waypoint controller (no RL) over the motor command.

The honest baseline before any RL: reaching a waypoint is a go-to-goal controller on the egocentric command
gamma=[dx_ego, dy_ego, dpsi]. Each step we decode the stream so far to read the current (xy, yaw), compute the
goal bearing in the root frame, and steer -- turn toward the goal (clamped to the feasible per-step turn) while
always walking forward a little (so it ARCS rather than stalls; in-place turns are out of the loco distribution).
Closed-loop, so imperfect command adherence is corrected by feedback. This measures how far pure steering gets;
what it can't do localizes what RL would have to add. No neural.

Run:  uv run python -m demo_b.waypoint [--shape square] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse, math
import numpy as np, torch
import canvas as C
from canvas.train import ddim, H, NLAT, DM, CLIP, FM, DEV
from canvas.prepare import reconstruct_qpos, quat2yaw
from canvas.autoresearch.stage3.motor_rollout import _cmd_at, _q0
from demo_b.rollout import load_motor, seed_segment, render
from demo_b import foot_metrics as FM_

SHAPES = {
    "square": [(1.4, 0.0), (1.4, 1.4), (0.0, 1.4), (0.0, 0.0)],
    "zigzag": [(1.2, 0.6), (2.4, -0.6), (3.6, 0.6), (4.8, 0.0)],
    "star": [(1.5, 0.0), (0.5, 1.4), (-0.4, 0.2), (1.2, 0.9), (0.0, -0.4)],
}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def steer(cur_xy, cur_yaw, goal, fwd_max, turn_max, stop_dist=0.5):
    """egocentric go-to-goal command: turn toward the goal (clamped), always some forward (arc), slow near goal."""
    d = np.asarray(goal) - cur_xy; dist = float(np.hypot(*d))
    err = _wrap(math.atan2(d[1], d[0]) - cur_yaw)
    dpsi = float(np.clip(err, -turn_max, turn_max))
    align = max(0.0, math.cos(err))                                  # 1 facing goal, 0 perpendicular
    fwd = fwd_max * (0.3 + 0.7 * align) * min(1.0, dist / stop_dist)  # never 0 (in-place turn is OOD); ease in near goal
    return np.array([fwd, 0.0, dpsi], np.float32), dist


@torch.no_grad()
def run_waypoints(m, mv, norms, seg, goals, fwd_max, turn_max, max_steps=90, reach=0.18):
    """closed-loop rollout steering to each goal in turn. Returns qpos(T,74) + per-goal reach log."""
    mm, ms = norms["mmean"], norms["mstd"]
    zmean = torch.tensor(norms["zmean"], device=DEV); zstd = torch.tensor(norms["zstd"], device=DEV)
    cmean, cstd = torch.tensor(norms["cmean"], device=DEV), torch.tensor(norms["cstd"], device=DEV)
    sidx = torch.full((1,), seg["sid"], device=DEV)
    zm0 = mv.encode(torch.tensor((seg["feat"][:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    stream = [((zm0 - zmean[0]) / zstd[0])[:H]]

    def decode_state():
        Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
        if nn_ == 0:                                                  # <NLAT latents yet -> still at the origin start
            return None, np.zeros(2), 0.0
        gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
        gq = reconstruct_qpos(gfeat, _q0(gfeat[0], np.zeros(2), 0.0))
        return gq, gq[-1, :2], float(quat2yaw(gq[-1, 3:7]))

    gi = 0; log = []; gq = np.zeros((1, 74))
    for step in range(max_steps):
        gq_new, cur_xy, cur_yaw = decode_state()
        if gq_new is not None: gq = gq_new
        cmd, dist = steer(cur_xy, cur_yaw, goals[gi], fwd_max, turn_max)
        if dist < reach:
            log.append(dict(goal=gi, step=step, frames=len(gq), dist=round(dist, 3))); gi += 1
            if gi >= len(goals): break
            continue
        cmdn = (torch.tensor(cmd[None], device=DEV) - cmean) / cstd
        stream.append(ddim(m, m.context(torch.cat(stream, 0)[-H:][None]), cmdn, sidx, cfg=1.5)[0])
    # final min distance to each unreached goal
    reached = {r["goal"] for r in log}
    finals = [round(float(np.hypot(*(np.asarray(g) - gq[:, :2]).T).min()), 3) for g in goals]
    return gq, log, reached, finals


def plot(gq, goals, reached, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=130)
    ax.plot(gq[:, 0], gq[:, 1], color="#35d0bf", lw=2.0, zorder=2, label="path")
    ax.scatter(gq[0, 0], gq[0, 1], color="#35d0bf", s=48, zorder=4, edgecolor="white", lw=1, label="start")
    for i, g in enumerate(goals):
        hit = i in reached
        ax.scatter(*g, s=150, marker="*", zorder=5, color="#79d38f" if hit else "#e0655e",
                   edgecolor="white", lw=1)
        ax.annotate(f"{i+1}", g, textcoords="offset points", xytext=(7, 6), fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.25, lw=.6); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"heuristic waypoint reaching  ({len(reached)}/{len(goals)} reached)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(out); plt.close(fig); print(f"  [plot] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="square", choices=list(SHAPES))
    ap.add_argument("--max_steps", type=int, default=90)
    ap.add_argument("--render", action="store_true"); a = ap.parse_args()
    mv, m, norms = load_motor()
    seg = seed_segment(min_len=CLIP * 4)
    cs = np.array([_cmd_at(seg["xy"], seg["yaw"], f0) for f0 in range(0, len(seg["feat"]) - 32, 16)])
    fwd_max = float(np.percentile(np.linalg.norm(cs[:, :2], axis=1), 85))
    turn_max = float(np.percentile(np.abs(cs[:, 2]), 90))
    goals = SHAPES[a.shape]
    print(f"shape={a.shape}  {len(goals)} waypoints  fwd_max={fwd_max:.3f} m/step  turn_max={turn_max:.3f} rad/step",
          flush=True)
    gq, log, reached, finals = run_waypoints(m, mv, norms, seg, goals, fwd_max, turn_max, max_steps=a.max_steps)
    for r in log:
        print(f"  reached wp{r['goal']+1} at step {r['step']:2d} ({r['frames']/C.FPS:4.1f}s), dist {r['dist']:.3f} m", flush=True)
    print(f"REACHED {len(reached)}/{len(goals)}  |  final min-dist to each wp (m): {finals}", flush=True)
    FM_.report(gq, label=f"wp/{a.shape}")
    outdir = C.HERE.parent / "demo_b" / "out"
    plot(gq, goals, reached, outdir / f"waypoint_{a.shape}.png")
    if a.render:
        render(FM_.ground(gq, alpha=0.03), f"waypoints:{a.shape} ({len(reached)}/{len(goals)})",
               outdir / f"waypoint_{a.shape}.mp4")


if __name__ == "__main__":
    main()
