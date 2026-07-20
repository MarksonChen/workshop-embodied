"""demo_b/foot_metrics.py -- DART-style foot-contact quality metrics for rodent kinematic rollouts.

Side project (demo_b/): exploring DART-style locomotion. NOT part of CANVAS; reuses the CANVAS package
only as a library (the MuJoCo rodent model + constants). No neural activity here.

Given a qpos(T,74) trajectory in the CANVAS/vnl rodent convention, we run MuJoCo forward kinematics to
get world-frame paw-site positions, then compute the artifacts that plague kinematic motion generation
(ref/docs/lit/motor_generation.md):
  - skate       : DART's calc_skate -- FULL-3D paw displacement between frames, gated by paw height so it
                  only fires when the paw is near the floor (weight 1 on floor -> 0 above `thr`).
  - penetration : paw depth below the floor plane (z = 0, the arena 'floor' geom).
  - jerk        : 3rd time-difference of paw positions (a jitter proxy, DART scores jerk on joints).
  - fix_floor   : DART rollout hard-snap -- shift each frame vertically so the lowest paw rests on the floor.

The arena floor is at z = 0; real STAC-fit motion sits ~0 (paw soles dip to ~-20 mm from marker-fit slack),
so __main__ prints a REAL-motion reference to contextualize generated numbers. The rodent walker is attached
with a "-rodent" suffix by C.utils.build_model(); _site_ids() resolves both bare and suffixed names.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")               # WSL2 headless render backend
import glob
import numpy as np
import canvas as C

# ground-contact paw tips: hind toes + fore fingers (the four points that actually meet the floor)
CONTACT_SITES = ("toe_L", "toe_R", "finger_L", "finger_R")
FLOOR_Z = 0.0                                              # arena 'floor' plane
SKATE_THR = 0.03                                           # DART RL/paper eq-18 threshold (m); optimizer uses 0.033

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = C.utils.build_model()
    return _MODEL


def _site_ids(model, sites):
    import mujoco
    ids = []
    for s in sites:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        if i < 0:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s + "-rodent")
        if i < 0:
            raise KeyError(f"site {s!r} not found (tried {s!r} and {s + '-rodent'!r})")
        ids.append(i)
    return ids


def paw_positions(gq, sites=CONTACT_SITES):
    """qpos(T,74) -> world paw positions (T,S,3) via MuJoCo forward kinematics."""
    import mujoco
    model = _model(); d = mujoco.MjData(model); ids = _site_ids(model, sites)
    gq = np.asarray(gq, np.float64); out = np.empty((len(gq), len(ids), 3))
    for t in range(len(gq)):
        d.qpos[:] = gq[t]; mujoco.mj_forward(model, d)
        out[t] = d.site_xpos[ids]
    return out


def skate(paws, thr=SKATE_THR):
    """DART calc_skate: full-3D paw displacement between consecutive frames, weighted by a factor that is 1
    when the paw is on the floor and ->0 once its (consecutive-max) height exceeds `thr`. (mean, per-frame)."""
    disp = np.linalg.norm(np.diff(paws, axis=0), axis=-1)          # (T-1,S) full 3D displacement
    h = np.maximum(paws[:-1, :, 2], paws[1:, :, 2])                # consecutive-max height
    w = 2 - 2 ** np.clip(h / thr, 0, 1)                            # 1 on floor, 0 above thr
    s = disp * w
    return float(s.mean()), s


def penetration(paws, floor=FLOOR_Z):
    """paw depth below the floor plane. dict(mean_depth, max_depth, frac_frames_penetrating)."""
    depth = np.clip(floor - paws[..., 2], 0, None)                 # (T,S) >= 0
    per_frame = depth.max(1)                                       # worst paw per frame
    return dict(mean_depth=float(depth.mean()), max_depth=float(depth.max()),
                frac_pen=float((per_frame > 1e-4).mean()))


def jerk(paws):
    """3rd time-difference of paw positions (jitter proxy). (mean, peak), in m/s^3."""
    j = np.linalg.norm(np.diff(paws, n=3, axis=0), axis=-1) * C.FPS ** 3
    return float(j.mean()), float(j.max())


def fix_floor(gq, sites=CONTACT_SITES, floor=FLOOR_Z):
    """DART rollout fix_floor: hard per-frame vertical snap so the lowest paw rests on the floor. Returns a
    COPY of gq with the root height qpos[:,2] shifted by (min_paw_z - floor) each frame. Removes both
    penetration and floating; unsuitable for jumps/rears by construction."""
    paws = paw_positions(gq, sites)
    out = np.array(gq, np.float64, copy=True)
    out[:, 2] -= (paws[..., 2].min(1) - floor)                     # snap lowest paw to floor each frame
    return out


def fix_upright(gq):
    """post-hoc: remove accumulated root pitch/roll drift, keeping heading (yaw). The orientation analog of
    fix_floor. reconstruct_qpos INTEGRATES the 6D orientation delta open-loop (R = R @ d6), so over a long
    rollout a tiny per-frame bias accumulates and the body rears nose-up ('flies to the sky') -- nothing
    anchors pitch/roll (no physics, no command). For a rodent on flat ground the trunk stays ~level, so we
    flatten the root orientation to yaw-only. Leg gait (jq) is untouched. NOT a model fix -- see scheduled
    sampling; this only cleans the render, like fix_floor."""
    from canvas.prepare import quat2yaw
    out = np.array(gq, np.float64, copy=True)
    yaw = quat2yaw(gq[:, 3:7])
    out[:, 3] = np.cos(yaw / 2); out[:, 4] = 0.0; out[:, 5] = 0.0; out[:, 6] = np.sin(yaw / 2)
    return out


def _axis_angle(ax, ang):
    import math
    x, y, z = ax; c, s = math.cos(ang), math.sin(ang); k = 1 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def anchor_orientation(gq, alpha=0.03):
    """ROLLOUT-TIME re-anchoring: re-integrate the trunk orientation from the original per-frame d6 deltas with a
    gentle pull toward gravity each frame, so accumulated pitch/roll drift ('flying') can't build up while fast
    gait bob survives (a leaky-integrator fix_upright). `alpha` = fraction of the body-up->world-up correction
    applied per frame; steady-state tilt ~ per-frame-drift / alpha. XY uses only yaw, so re-leveling never moves
    the path. Unlike fix_upright (hard yaw-only), this keeps natural body pitch oscillation."""
    import math
    from canvas.prepare import quat2mat, mat2quat
    Rs = quat2mat(np.asarray(gq[:, 3:7], np.float64))              # (T,3,3)
    out = np.array(gq, np.float64, copy=True); Rc = Rs[0].copy()
    for t in range(len(gq)):
        if t > 0:
            Rc = Rc @ (Rs[t - 1].T @ Rs[t])                       # replay the ORIGINAL frame-to-frame rotation
        bu = Rc[:, 2]; ax = np.cross(bu, [0, 0, 1.0]); s = np.linalg.norm(ax)
        if s > 1e-8:
            Rc = _axis_angle(ax / s, alpha * math.atan2(s, bu[2])) @ Rc   # pull alpha of the way to level
        out[t, 3:7] = mat2quat(Rc)
    return out


def ground(gq, sites=CONTACT_SITES, floor=FLOOR_Z, alpha=None):
    """kinematic re-grounding for rendering: re-anchor orientation THEN snap paws to the floor. alpha=None uses
    the hard yaw-only fix_upright; a float uses the leaky anchor_orientation (keeps gait bob). Order matters --
    fix_floor reads FK paw positions, which depend on orientation."""
    up = fix_upright(gq) if alpha is None else anchor_orientation(gq, alpha)
    return fix_floor(up, sites, floor)


def report(gq, label="", sites=CONTACT_SITES, floor=FLOOR_Z, thr=SKATE_THR):
    """compute + print the full foot-quality battery for one qpos trajectory; return a dict."""
    paws = paw_positions(gq, sites)
    sk, _ = skate(paws, thr); pen = penetration(paws, floor); jm, jp = jerk(paws)
    r = dict(label=label, frames=len(gq), seconds=len(gq) / C.FPS, skate=sk,
             pen_mean_mm=pen["mean_depth"] * 1000, pen_max_mm=pen["max_depth"] * 1000, pen_frac=pen["frac_pen"],
             jerk_mean=jm, jerk_peak=jp, min_paw_z_mm=float(paws[..., 2].min()) * 1000)
    print(f"[{label:14s}] {r['frames']:4d}f ({r['seconds']:4.1f}s)  skate {sk:.4f}  "
          f"pen mean {r['pen_mean_mm']:5.2f}mm max {r['pen_max_mm']:6.2f}mm ({pen['frac_pen'] * 100:3.0f}% frames)  "
          f"jerk {jm:7.1f}  minz {r['min_paw_z_mm']:+6.1f}mm")
    return r


def _real_windows(n_frames=600, k=2):
    """pull k moving real-motion windows from a cached raw session (validation + reference baseline)."""
    f = sorted(glob.glob(str(C.CACHE) + "/raw_*.npz"))
    if not f:
        raise FileNotFoundError(f"no raw_*.npz cache under {C.CACHE}")
    q = np.load(f[0], allow_pickle=True)["qpos"]                   # (T,74)
    spd = np.zeros(len(q)); spd[1:] = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)  # planar step
    win = np.convolve(spd, np.ones(n_frames) / n_frames, "valid")  # windowed mean speed
    starts = np.argsort(win)[::-1]                                 # fastest (most locomotor) windows first
    picks, used = [], []
    for s in starts:
        if all(abs(s - u) > n_frames for u in used):
            used.append(s); picks.append((int(s), q[s:s + n_frames]))
        if len(picks) >= k:
            break
    return f[0].split("/")[-1], picks


if __name__ == "__main__":
    name, wins = _real_windows()
    print(f"REAL-motion reference ({name}) -- the target a good rollout should approach:")
    for s, q in wins:
        report(q, label=f"real@{s}")
        report(fix_floor(q), label="real+fixfloor")
    print("\nInstrument OK. Feed a generated qpos(T,74) to report(gq) to score a rollout (Phase 0b).")
