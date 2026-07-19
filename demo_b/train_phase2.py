"""demo_b/train_phase2.py -- Phase 2: restore MotionStreamer's Two-Forward to the motor transition.

The demo_b/ motor model (motor_rollout.train) is a single-forward, teacher-forced reimplementation -- it dropped
the Two-Forward that CANVAS's full Stage-C Transition (train.py) already uses. Teacher-forcing -> the model
never sees its own predictions as history -> autoregressive drift (paws sink, trunk pitches nose-up / 'flies'
past the ~2.5 s horizon). Two-Forward (= scheduled sampling, same idea as DART's): forward once to predict
block-1, mix a cosine-scheduled fraction of the PREDICTED block-1 back in as history, forward again to predict
block-2. This exposes the model to 1 step of its own error during training.

Ported verbatim from canvas/train.py Stage C (rho cosine schedule, b1mix), on the motor recipe (lr 1e-3 warmup,
EMA 0.99, MSE latent loss) for a clean A/B vs the baseline. Step-matched. Reports the 16 s drift WITHOUT the
cosmetic ground() -- success = the model itself stays level. No neural.

Run:  uv run python -m demo_b.train_phase2 [--steps 14500] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse, math, time
import numpy as np, torch
import canvas as C
from canvas.train import abar, CLIP, DM, H, K, DEV
from canvas.prepare import quat2mat
from canvas.autoresearch.stage3.motor_rollout import MotionTrans, _zm_stream, _cmd_at, rollout
from canvas import utils as U
from demo_b import foot_metrics as FM_
from demo_b.rollout import seed_segment, render_compare
from demo_b.reproduce import load_tokenizer
from demo_b.drive import roll_cmd

WIN = H + 2 * K


def build_windows_2f(D, mv):
    """H+2K two-forward windows over the loco z^m streams, with per-block egocentric commands + latent norm."""
    mm, ms = D["mmean"], D["mstd"]; Wz, Wc1, Wc2, Ws = [], [], [], []
    for seg in D["train"]["segs"]:
        feat, xy, yaw = np.asarray(seg["feat"]), np.asarray(seg["xy"]), np.asarray(seg["yaw"])
        if len(feat) < CLIP * 2: continue
        zm = _zm_stream(mv, feat, mm, ms).cpu(); Nt = zm.shape[0]
        if Nt < WIN: continue
        for s0 in range(0, Nt - WIN + 1, 4):
            f1, f2 = 4 * (s0 + H), 4 * (s0 + H + K)
            if f2 + 31 >= len(xy): break
            Wz.append(zm[s0:s0 + WIN]); Ws.append(seg["sid"])
            Wc1.append(_cmd_at(xy, yaw, f1)); Wc2.append(_cmd_at(xy, yaw, f2))
    Wz = torch.stack(Wz); allz = Wz.reshape(-1, DM)
    zmean = allz.mean(0, keepdim=True); zstd = allz.std(0, keepdim=True) + 1e-6
    cm = torch.tensor(D["cmean"], dtype=torch.float32); cs = torch.tensor(D["cstd"], dtype=torch.float32)
    c1 = (torch.tensor(np.array(Wc1), dtype=torch.float32) - cm) / cs
    c2 = (torch.tensor(np.array(Wc2), dtype=torch.float32) - cm) / cs
    return (Wz.to(DEV), c1.to(DEV), c2.to(DEV), torch.tensor(Ws, device=DEV).long(),
            zmean.to(DEV), zstd.to(DEV))


def train_2f(D, mv, steps):
    Wz, Wc1, Wc2, Ws, zmean, zstd = build_windows_2f(D, mv)
    N = Wz.shape[0]; print(f"[P2] {N} two-forward windows (WIN={WIN}); {D['n_sess']} sess; steps {steps}", flush=True)
    Wzn = (Wz - zmean) / zstd                                            # normalized windows
    m = MotionTrans(D["n_sess"]).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01, fused=True)
    plist = list(m.parameters()); ema = [p.detach().clone() for p in plist]; B = 256; t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups: g["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        rho = 0.5 * (1 - math.cos(math.pi * step / max(steps - 1, 1)))    # cosine replacement schedule 0->1
        i = torch.randint(0, N, (B,), device=DEV)
        win = Wzn[i]; sidx = Ws[i]; c1 = Wc1[i]; c2 = Wc2[i]
        hist, b1, b2 = win[:, :H], win[:, H:H + K], win[:, H + K:H + 2 * K]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lam = torch.rand(B, device=DEV); ab = abar(lam)[:, None, None]
            x0_1 = m.denoise(ab.sqrt() * b1 + (1 - ab).sqrt() * torch.randn_like(b1), lam, m.context(hist), c1, sidx)
            loss1 = ((x0_1 - b1) ** 2).mean()
            repl = (torch.rand(B, K, 1, device=DEV) < rho).float()        # mix predicted block-1 into the history
            b1mix = repl * x0_1.detach() + (1 - repl) * b1
            lam2 = torch.rand(B, device=DEV); ab2 = abar(lam2)[:, None, None]
            x0_2 = m.denoise(ab2.sqrt() * b2 + (1 - ab2).sqrt() * torch.randn_like(b2), lam2, m.context(b1mix), c2, sidx)
            loss = loss1 + ((x0_2 - b2) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(plist, 1.0); opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ema, 0.99); torch._foreach_add_(ema, plist, alpha=0.01)
        if step % 500 == 0:
            print(f"    step={step:5d} ({time.time()-t0:4.0f}s) loss={float(loss):.4f} rho={rho:.2f}", flush=True)
    with torch.no_grad():
        for e, p in zip(ema, plist): p.copy_(e)
    m.eval(); return m, zmean, zstd


def drift_report(gq, label):
    """raw open-loop drift (NO ground()): final trunk pitch + max head height. Baseline: +29 deg, 0.127 m."""
    import mujoco
    R = quat2mat(gq[:, 3:7]); pitch = np.degrees(np.arcsin(np.clip(R[:, 2, 0], -1, 1)))
    mdl = FM_._model(); d = mujoco.MjData(mdl); hid = mdl.site("head-rodent").id; zs = []
    for t in range(0, len(gq), 25):
        d.qpos[:] = gq[t]; mujoco.mj_forward(mdl, d); zs.append(d.site_xpos[hid][2])
    print(f"  [{label}] final pitch {pitch[-1]:+.0f} deg  |  head z max {max(zs):.3f} m   (baseline +29 deg / 0.127 m)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=14500)
    ap.add_argument("--render", action="store_true"); a = ap.parse_args()
    t0 = time.time(); D = C.build(motion_only=True); mv = load_tokenizer()
    m, zmean, zstd = train_2f(D, mv, a.steps)
    out = C.HERE.parent / "demo_b" / "out" / "motor_phase2.pt"
    torch.save(dict(model_cfg=dict(n_sess=int(D["n_sess"]), d=192, heads=4, layers=6, se_d=16),
                    state_dict=m.state_dict(), zmean=zmean.detach().cpu().numpy(), zstd=zstd.detach().cpu().numpy(),
                    cmean=np.asarray(D["cmean"], np.float32), cstd=np.asarray(D["cstd"], np.float32),
                    mmean=np.asarray(D["mmean"], np.float32), mstd=np.asarray(D["mstd"], np.float32),
                    provenance=dict(steps=a.steps, method="two-forward", git=C.git_provenance())), out)
    print(f"[P2] saved -> {out}", flush=True)
    norms = {k: np.asarray(v) for k, v in
             dict(zmean=zmean.detach().cpu().numpy(), zstd=zstd.detach().cpu().numpy(),
                  cmean=D["cmean"], cstd=D["cstd"], mmean=D["mmean"], mstd=D["mstd"]).items()}
    # foot quality on the 2 loco seeds (hindsight, 8 s) -- baseline jerk 754 / 1095
    segs = sorted([s for s in D["train"]["segs"] if len(s["feat"]) >= CLIP * 2], key=lambda s: -len(s["feat"]))
    n8 = U.n_steps_for_seconds(8); outdir = C.HERE.parent / "demo_b" / "out"
    print("\n[P2] foot quality (hindsight 8 s):", flush=True)
    for seg in segs[:2]:
        nm = next(mm["name"] for mm in D["meta"] if mm["sid"] == seg["sid"]).replace("/", "_")
        gq = rollout(D, mv, m, torch.tensor(norms["zmean"], device=DEV), torch.tensor(norms["zstd"], device=DEV), seg, n8)
        FM_.report(gq, label=f"P2/{nm[:10]}")
        if a.render:
            render_compare(gq, FM_.ground(gq), f"P2 {nm}", outdir / f"phase2_{nm}.mp4")
    # THE TEST: 16 s straight, raw drift (no ground)
    seg = seed_segment(min_len=CLIP * 4)
    cs = np.array([_cmd_at(seg["xy"], seg["yaw"], f0) for f0 in range(0, len(seg["feat"]) - 32, 16)])
    fwd = float(np.linalg.norm(cs[:, :2], axis=1).mean()); n16 = U.n_steps_for_seconds(16)
    print("\n[P2] 16 s straight -- open-loop drift (the flying test):", flush=True)
    gq16 = roll_cmd(m, mv, norms, seg, n16, np.array([fwd, 0, 0], np.float32))
    drift_report(gq16, "P2 straight 16s")
    if a.render:
        from demo_b.rollout import render
        render(FM_.fix_floor(gq16), "P2 straight 16s (floor only, NO upright)", outdir / "phase2_drift_straight.mp4")
    print(f"\n[P2] done in {time.time()-t0:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
