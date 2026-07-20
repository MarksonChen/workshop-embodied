"""Rotation helpers + qpos reconstruction (copied from CANVAS prepare.py). Pure numpy, no external imports."""
import math
import numpy as np
try:
    from .constants import ACTIVE_JOINTS, FPS
except ImportError:  # pragma: no cover
    from constants import ACTIVE_JOINTS, FPS


def quat2yaw(q):                                            # q: (...,4) wxyz
    w_, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w_ * z + x * y), 1 - 2 * (y * y + z * z))


def quat2mat(q):                                            # wxyz -> (...,3,3)
    w_, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w_ * z), 2 * (x * z + w_ * y),
                  2 * (x * y + w_ * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w_ * x),
                  2 * (x * z - w_ * y), 2 * (y * z + w_ * x), 1 - 2 * (x * x + y * y)], -1)
    return n.reshape(q.shape[:-1] + (3, 3))


def mat2quat(m):                                            # (3,3) -> wxyz
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w_ = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w_ = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w_ = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w_ = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w_, x, y, z]); return q / (np.linalg.norm(q) + 1e-9)


def sixd2mat(d):                                            # (...,6) -> (...,3,3) gram-schmidt
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], -1)


def reconstruct_qpos(feat, q0):
    """integrate root-local vel + 6D orientation delta back to global qpos(74); joint angles read directly."""
    Ln = len(feat); qpos = np.zeros((Ln, 74))
    full_joints = feat.shape[-1] >= 143
    xy = q0[:2].copy(); R = quat2mat(q0[3:7][None])[0]
    for t in range(Ln):
        vlx, vly = feat[t, 0], feat[t, 1]; h = feat[t, 2]; d6 = feat[t, 3:9]
        yaw = math.atan2(R[1, 0], R[0, 0]); c, s = math.cos(yaw), math.sin(yaw)
        xy = xy + np.array([c * vlx - s * vly, s * vlx + c * vly]) / FPS
        R = R @ sixd2mat(d6[None])[0]
        qpos[t, :2] = xy; qpos[t, 2] = h; qpos[t, 3:7] = mat2quat(R)
        if full_joints:
            qpos[t, 7:] = feat[t, 9:76]
        else:
            qpos[t, 7 + np.asarray(ACTIVE_JOINTS)] = feat[t, 9:47]
    return qpos


def q0_from(f0, xy0, yaw0):
    """initial qpos(74) from the first feature frame + a world (xy, yaw) placement."""
    q = np.zeros(74); q[:2] = xy0; q[2] = f0[2]; q[3] = math.cos(yaw0 / 2); q[6] = math.sin(yaw0 / 2)
    if len(f0) >= 143:
        q[7:] = f0[9:76]
    else:
        q[7 + np.asarray(ACTIVE_JOINTS)] = f0[9:47]
    return q
