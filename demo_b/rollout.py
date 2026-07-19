"""demo_b/rollout.py -- Phase 0b: generate a baseline locomotion rollout from the CANVAS motor model and SCORE it.

Side project (demo_b/): DART-style locomotion. Reuses CANVAS as a library; no neural. Loads the motor-only
x0-diffusion transition (MotionTrans) from canvas/out/motor_ckpt.pt (frozen Stage-1 tokenizer + this
transition), seeds from a real loco window, and rolls out ~10 s under:
  - "hindsight" : the real segment's per-step egocentric command (reproduces the shipped rollout / the artifact)
  - "constant"  : a single fixed command = the seed's MEAN hindsight command (the DART-style "keep walking
                  straight" test, and the answer to "does a constant command still walk?")
Each rollout is scored with demo_b.foot_metrics (skate / penetration / jerk) against the real reference, and again
after fix_floor. Optional MuJoCo render (--render) writes mp4s to demo_b/out/.

Run:  uv run python -m demo_b.rollout [--seconds 10] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import glob, argparse
import numpy as np, torch
import canvas as C
from canvas.train import ddim, H, K, NLAT, DM, CLIP, FM, DEV
from canvas.prepare import reconstruct_qpos, quat2yaw
from canvas.autoresearch.stage3.motor_rollout import MotionTrans, _cmd_at, _q0
from canvas import utils as U
from demo_b import foot_metrics as FM_


REPRO_CKPT = C.HERE.parent / "demo_b" / "out" / "motor_repro.pt"       # faithful autoresearch baseline (demo_b.reproduce)
PHASE1_CKPT = C.HERE.parent / "demo_b" / "out" / "motor_phase1_vc.pt"  # CANONICAL: Phase-1 (+ anti-jitter loss, demo_b.train_phase1)


def load_motor(path=None, tok_ckpt=C.OUT / "ms_ckpt.pt"):
    """frozen Stage-1 tokenizer + the trained MotionTrans transition + its bundled norm constants.
    Default = the CANONICAL Phase-1 model (demo_b/out/motor_phase1_vc.pt: velocity-consistency anti-jitter loss)
    if present, else the faithful autoresearch baseline (motor_repro.pt) -- NEVER the live_link motor_ckpt.pt
    (a different, worse artifact). The tokenizer is loaded from ms_ckpt.pt['motion'] -- verified the SAME frozen
    mint (identical mmean); out/stage1_motion_tokenizer.pt is absent on this box."""
    if path is None:
        path = PHASE1_CKPT if PHASE1_CKPT.exists() else REPRO_CKPT
    ck = torch.load(path, map_location=DEV, weights_only=False)
    tck = torch.load(tok_ckpt, map_location=DEV, weights_only=False)
    mv = C.MotionVAE().to(DEV); mv.load_state_dict(tck["motion"]); mv.eval()
    for p in mv.parameters(): p.requires_grad_(False)
    m = MotionTrans(**ck["model_cfg"]).to(DEV); m.load_state_dict(ck["state_dict"]); m.eval()
    norms = {k: np.asarray(ck[k]) for k in ("zmean", "zstd", "cmean", "cstd", "mmean", "mstd")}
    return mv, m, norms


def seed_segment(min_len=CLIP * 4, session_glob="raw_*.npz"):
    """a real contiguous loco window (highest-planar-speed slice) from one cached session -> seed dict."""
    f = sorted(glob.glob(str(C.CACHE) + "/" + session_glob))
    if not f:
        raise FileNotFoundError(f"no {session_glob} under {C.CACHE}")
    z = np.load(f[0], allow_pickle=True); q = z["qpos"]; kp = z["kp"]                 # q(T,74) kp(T,3,23)
    xy = q[:, :2].astype(np.float64)
    spd = np.zeros(len(q)); spd[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1)     # planar step
    s = int(np.convolve(spd, np.ones(min_len) / min_len, "valid").argmax())           # fastest window
    sl = slice(s, s + min_len)
    feat = C.motion_features(q[sl], kp[sl])                                            # (min_len,281)
    return dict(feat=feat, xy=xy[sl] - xy[sl][0], yaw=quat2yaw(q[sl, 3:7]), sid=0,
                name=f[0].split("/")[-1].replace("raw_", "").replace(".npz", ""), start=s)


@torch.no_grad()
def roll(m, mv, norms, seg, n_steps, command="constant", cfg=1.5):
    """autoregressively roll the motion latent under `command`, decode via the frozen tokenizer -> qpos(T,74)."""
    feat, xy, yaw, sid = seg["feat"], seg["xy"], seg["yaw"], seg["sid"]
    mm, ms = norms["mmean"], norms["mstd"]; Ls = len(feat)
    zmean, zstd = torch.tensor(norms["zmean"], device=DEV), torch.tensor(norms["zstd"], device=DEV)
    cmean, cstd = torch.tensor(norms["cmean"], device=DEV), torch.tensor(norms["cstd"], device=DEV)
    sidx = torch.full((1,), sid, device=DEV)
    zm0 = mv.encode(torch.tensor((feat[:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    z = ((zm0 - zmean[0]) / zstd[0])[:H]; stream = [z]
    c_const = np.mean([_cmd_at(xy, yaw, f0) for f0 in range(0, Ls - 32, 16)], 0)       # mean hindsight command
    for s in range(n_steps):
        hist = torch.cat(stream, 0)[-H:][None]
        if command == "hindsight":
            c = _cmd_at(xy, yaw, min(4 * H + 4 * K * s, Ls - 32))
        else:
            c = c_const
        cmd = (torch.tensor(np.asarray([c], np.float32), device=DEV) - cmean) / cstd
        stream.append(ddim(m, m.context(hist), cmd, sidx, cfg=cfg)[0])
    Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
    gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
    return reconstruct_qpos(gfeat, _q0(gfeat[0], xy[0], yaw[0]))


def _render_frames(r, model, d, gq, label):
    """render a qpos(T,74) trajectory to a list of RGB frames, captioned `label`."""
    import mujoco
    from PIL import Image, ImageDraw
    spd = float(np.linalg.norm(np.diff(gq[:, :2], axis=0), axis=1).mean() * C.FPS); frames = []
    for t in range(len(gq)):
        d.qpos[:] = gq[t]; mujoco.mj_forward(model, d); r.update_scene(d, camera="close_profile-rodent")
        im = Image.fromarray(r.render().copy())
        ImageDraw.Draw(im).text((6, 4), f"{label} | {len(gq)/C.FPS:.1f}s | {spd:.2f} m/s", fill=(255, 255, 255))
        frames.append(np.array(im))
    return frames


def render(gq, name, out):
    """MuJoCo render of a single qpos(T,74) trajectory to `out` (mp4)."""
    import mujoco, mediapy as media, imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    model = U.build_model(); d = mujoco.MjData(model); r = mujoco.Renderer(model, height=U.RH, width=U.RW)
    fr = _render_frames(r, model, d, gq, name); r.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out), np.stack(fr), fps=C.FPS); print(f"  [video] {out}")


def render_compare(gq, gq_ff, name, out):
    """side-by-side MuJoCo render (raw | +fix_floor) of the same rollout to one mp4."""
    import mujoco, mediapy as media, imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    model = U.build_model(); d = mujoco.MjData(model); r = mujoco.Renderer(model, height=U.RH, width=U.RW)
    A = _render_frames(r, model, d, gq, f"{name} RAW")
    B = _render_frames(r, model, d, gq_ff, f"{name} +fix_floor"); r.close()
    frames = np.stack([np.concatenate([a, b], 1) for a, b in zip(A, B)])   # left=raw, right=snapped
    out.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(out), frames, fps=C.FPS); print(f"  [video] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    mv, m, norms = load_motor()
    seg = seed_segment(); n_steps = U.n_steps_for_seconds(a.seconds)
    print(f"seed: {seg['name']} @frame {seg['start']} ({len(seg['feat'])}f); rollout {n_steps} steps -> ~{a.seconds:.0f}s\n")
    outdir = C.HERE.parent / "demo_b" / "out"
    for command in ("hindsight", "constant"):
        gq = roll(m, mv, norms, seg, n_steps, command=command)
        FM_.report(gq, label=f"gen/{command}")
        FM_.report(FM_.fix_floor(gq), label=f"gen/{command}+ff")
        if a.render:
            render(gq, f"motor {command}", outdir / f"baseline_{command}.mp4")
            render(FM_.fix_floor(gq), f"motor {command}+fixfloor", outdir / f"baseline_{command}_fixfloor.mp4")
    print("\nCompare to REAL reference (uv run python -m demo_b.foot_metrics): skate ~0.003, pen ~1mm, jerk ~570.")


if __name__ == "__main__":
    main()
