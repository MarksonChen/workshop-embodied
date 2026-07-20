"""Foot-contact quality metrics via MuJoCo forward kinematics + kinematic re-grounding (floor / upright / anchor).

Given a qpos(T,74) trajectory in the rodent convention: FK to world paw positions, then
  skate       -- DART-style foot slide (full-3D paw displacement gated by paw height)
  penetration -- paw depth below the floor plane (z=0)
  jerk        -- 3rd time-difference of paw positions
  fix_floor   -- snap the lowest paw to the floor (removes penetration)
  fix_upright -- hard yaw-only trunk (removes accumulated pitch/roll drift)
  anchor_orientation -- leaky gravity re-anchor (removes drift, keeps gait bob)
  ground      -- upright/anchor + floor, for rendering
"""
import math
import numpy as np
try:
    from .constants import FPS
    from .mujoco_rodent import build_model
    from .geometry import quat2mat, mat2quat
except ImportError:  # pragma: no cover
    from constants import FPS
    from mujoco_rodent import build_model
    from geometry import quat2mat, mat2quat

CONTACT_SITES = ("toe_L", "toe_R", "finger_L", "finger_R")   # the four ground-contact paw tips
FLOOR_Z = 0.0
SKATE_THR = 0.03

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = build_model()
    return _MODEL


def _site_ids(model, sites):
    import mujoco
    ids = []
    for s in sites:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
        if i < 0:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s + "-rodent")   # attach suffix
        if i < 0:
            raise KeyError(f"site {s!r} not found")
        ids.append(i)
    return ids


def paw_positions(gq, sites=CONTACT_SITES):
    """qpos(T,74) -> world paw positions (T,S,3) via MuJoCo forward kinematics."""
    import mujoco
    model = _model(); d = mujoco.MjData(model); ids = _site_ids(model, sites)
    gq = np.asarray(gq, np.float64); out = np.empty((len(gq), len(ids), 3))
    for t in range(len(gq)):
        d.qpos[:] = gq[t]; mujoco.mj_forward(model, d); out[t] = d.site_xpos[ids]
    return out


def skate(paws, thr=SKATE_THR):
    disp = np.linalg.norm(np.diff(paws, axis=0), axis=-1)
    h = np.maximum(paws[:-1, :, 2], paws[1:, :, 2]); w = 2 - 2 ** np.clip(h / thr, 0, 1)
    return float((disp * w).mean())


def penetration(paws, floor=FLOOR_Z):
    depth = np.clip(floor - paws[..., 2], 0, None); per_frame = depth.max(1)
    return dict(mean_mm=float(depth.mean()) * 1000, max_mm=float(depth.max()) * 1000,
                frac=float((per_frame > 1e-4).mean()))


def jerk(paws):
    j = np.linalg.norm(np.diff(paws, n=3, axis=0), axis=-1) * FPS ** 3
    return float(j.mean())


def fix_floor(gq, sites=CONTACT_SITES, floor=FLOOR_Z):
    """snap the lowest paw to the floor each frame (removes penetration + floating)."""
    paws = paw_positions(gq, sites); out = np.array(gq, np.float64, copy=True)
    out[:, 2] -= (paws[..., 2].min(1) - floor); return out


def fix_upright(gq):
    """hard re-level: flatten trunk pitch/roll to yaw-only (kills orientation drift and any gait bob)."""
    from geometry import quat2yaw
    out = np.array(gq, np.float64, copy=True); yaw = quat2yaw(gq[:, 3:7])
    out[:, 3] = np.cos(yaw / 2); out[:, 4] = 0.0; out[:, 5] = 0.0; out[:, 6] = np.sin(yaw / 2)
    return out


def _axis_angle(ax, ang):
    x, y, z = ax; c, s = math.cos(ang), math.sin(ang); k = 1 - c
    return np.array([[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
                     [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
                     [z * x * k - y * s, z * y * k + x * s, c + z * z * k]])


def anchor_orientation(gq, alpha=0.03):
    """leaky gravity re-anchor: re-integrate orientation from the original per-frame deltas with a gentle pull
    toward level each frame -> drift can't accumulate while fast gait bob survives. XY uses only yaw."""
    Rs = quat2mat(np.asarray(gq[:, 3:7], np.float64)); out = np.array(gq, np.float64, copy=True); Rc = Rs[0].copy()
    for t in range(len(gq)):
        if t > 0:
            Rc = Rc @ (Rs[t - 1].T @ Rs[t])
        bu = Rc[:, 2]; ax = np.cross(bu, [0, 0, 1.0]); s = np.linalg.norm(ax)
        if s > 1e-8:
            Rc = _axis_angle(ax / s, alpha * math.atan2(s, bu[2])) @ Rc
        out[t, 3:7] = mat2quat(Rc)
    return out


def ground(gq, sites=CONTACT_SITES, floor=FLOOR_Z, alpha=0.03):
    """render-ready: re-anchor orientation (leaky if alpha, hard if None) then snap paws to the floor."""
    up = fix_upright(gq) if alpha is None else anchor_orientation(gq, alpha)
    return fix_floor(up, sites, floor)


def report(gq, label="", sites=CONTACT_SITES, floor=FLOOR_Z, thr=SKATE_THR):
    paws = paw_positions(gq, sites); pen = penetration(paws, floor)
    r = dict(label=label, frames=len(gq), seconds=len(gq) / FPS, skate=skate(paws, thr),
             pen_mm=pen["mean_mm"], jerk=jerk(paws))
    print(f"[{label:14s}] {r['frames']:4d}f ({r['seconds']:4.1f}s)  skate {r['skate']:.4f}  "
          f"pen {pen['mean_mm']:5.2f}mm ({pen['frac']*100:3.0f}%)  jerk {r['jerk']:7.1f}")
    return r
