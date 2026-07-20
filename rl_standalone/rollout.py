"""Autoregressive rollout of the motor model + MuJoCo rendering. Self-contained."""
import math
import numpy as np, torch
from constants import DEV, CLIP, H, K, NLAT, DM, FM, FPS
from geometry import reconstruct_qpos, q0_from
from mujoco_rodent import build_model, RH, RW


def cmd_at(xy, yaw, f0):
    """egocentric hindsight command over frames [f0, f0+31]: root-local dxy + wrapped yaw delta."""
    y0 = yaw[f0]; cc, sn = math.cos(-y0), math.sin(-y0); dxy = xy[f0 + 31] - xy[f0]
    return [cc * dxy[0] - sn * dxy[1], sn * dxy[0] + cc * dxy[1],
            (yaw[f0 + 31] - yaw[f0] + np.pi) % (2 * np.pi) - np.pi]


@torch.no_grad()
def roll(m, mv, norms, seed, n_steps, command="constant", cmd_raw=None):
    """roll K latents/step under `command` ('hindsight' from the seed window, 'constant' = its mean, or a fixed
    cmd_raw=[dx_ego,dy_ego,dpsi]); decode via the frozen tokenizer -> qpos(T,74)."""
    feat, xy, yaw, sid = seed["feat"], seed["xy"], seed["yaw"], seed["sid"]
    mm, ms = norms["mmean"], norms["mstd"]; Ls = len(feat)
    zmean = torch.tensor(norms["zmean"], device=DEV); zstd = torch.tensor(norms["zstd"], device=DEV)
    cmean = torch.tensor(norms["cmean"], device=DEV); cstd = torch.tensor(norms["cstd"], device=DEV)
    sidx = torch.full((1,), sid, device=DEV)
    zm0 = mv.encode(torch.tensor((feat[:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    stream = [((zm0 - zmean[0]) / zstd[0])[:H]]
    if command == "constant" and cmd_raw is None:
        cmd_raw = np.mean([cmd_at(xy, yaw, f0) for f0 in range(0, Ls - 32, 16)], 0)
    for s in range(n_steps):
        c = cmd_at(xy, yaw, min(4 * H + 4 * K * s, Ls - 32)) if command == "hindsight" else cmd_raw
        cmd = (torch.tensor(np.asarray([c], np.float32), device=DEV) - cmean) / cstd
        stream.append(m.predict(torch.cat(stream, 0)[-H:][None], cmd, sidx)[0])
    Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
    gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
    return reconstruct_qpos(gfeat, q0_from(gfeat[0], xy[0], yaw[0]))


# ------------------------------------------------------------------------------- MuJoCo render (lazy deps)
def _frames(r, model, d, gq, label):
    import mujoco
    from PIL import Image, ImageDraw
    spd = float(np.linalg.norm(np.diff(gq[:, :2], axis=0), axis=1).mean() * FPS); out = []
    for t in range(len(gq)):
        d.qpos[:] = gq[t]; mujoco.mj_forward(model, d); r.update_scene(d, camera="close_profile-rodent")
        im = Image.fromarray(r.render().copy())
        ImageDraw.Draw(im).text((6, 4), f"{label} | {len(gq)/FPS:.1f}s | {spd:.2f} m/s", fill=(255, 255, 255))
        out.append(np.array(im))
    return out


def _writer(out):
    import mediapy as media, imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe()); out.parent.mkdir(parents=True, exist_ok=True)
    return media


def render(gq, name, out):
    import mujoco
    media = _writer(out); model = build_model(); d = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=RH, width=RW); fr = _frames(r, model, d, gq, name); r.close()
    media.write_video(str(out), np.stack(fr), fps=FPS); print(f"  [video] {out}")


def render_compare(gq_a, gq_b, name, out):
    """side-by-side render of two qpos trajectories (e.g. raw | grounded)."""
    import mujoco
    media = _writer(out); model = build_model(); d = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=RH, width=RW)
    A = _frames(r, model, d, gq_a, f"{name} L"); B = _frames(r, model, d, gq_b, f"{name} R"); r.close()
    media.write_video(str(out), np.stack([np.concatenate([a, b], 1) for a, b in zip(A, B)]), fps=FPS)
    print(f"  [video] {out}")
