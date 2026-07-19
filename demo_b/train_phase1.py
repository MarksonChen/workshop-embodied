"""demo_b/train_phase1.py -- Phase 1: add DART-style decode-space losses to the motor transition training.

Baseline (demo_b.reproduce / stage3.motor_rollout.train) trains the MotionTrans with an x0 MSE in LATENT space
only -- nothing tells the generator its DECODED motion must be velocity-consistent, so the rollout jitters
(jerk ~750-1090 vs real ~570). DART's fix: decode the predicted future through the frozen tokenizer and add
  - feature-recon : SmoothL1(decode(pred), decode(true))                       [DART feature_rec]
  - velocity self-consistency : predicted qd/kpd == finite-diff of predicted q/kp   [DART delta-consistency]
The velocity terms are exact for real data (prepare.motion_features defines qd[t]=(q[t]-q[t-1])*FPS,
kpd[t]=(kp[t]-kp[t-1])*FPS), so enforcing them on the decoded prediction pulls it onto the data manifold.

STEP-matched to the baseline (~14500 steps) for a fair A/B on jerk. Saves demo_b/out/motor_phase1.pt, scores the
same 2 loco seeds, and renders raw-vs-fix_floor. No neural.

Run:  uv run python -m demo_b.train_phase1 [--steps 14500] [--w_feat 1.0 --w_vc 1.0] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse, time
import numpy as np, torch, torch.nn.functional as F
import canvas as C
from canvas.train import abar, CLIP, FPS, DEV
from canvas.autoresearch.stage3.motor_rollout import MotionTrans, build_windows, rollout
from canvas import utils as U
from demo_b import foot_metrics as FM_
from demo_b.rollout import render_compare
from demo_b.reproduce import load_tokenizer

# raw-feature channel slices (prepare.SL): q 9:76, qd 76:143, kp 143:212, kpd 212:281
QA, QB, QDA, QDB = 9, 76, 76, 143
KA, KB, KDA, KDB = 143, 212, 212, 281


def vc_loss(fp, ms_t, mm_t):
    """velocity self-consistency on decoded (normalized) features fp (B,T,281): predicted velocity channels must
    equal the finite-difference of the predicted position channels, per-channel-std-normalized (dimensionless)."""
    fpr = fp * ms_t + mm_t                                          # -> raw units (m, rad, m/s, rad/s)
    q, qd = fpr[..., QA:QB], fpr[..., QDA:QDB]
    kp, kpd = fpr[..., KA:KB], fpr[..., KDA:KDB]
    rq = (qd[:, 1:] - (q[:, 1:] - q[:, :-1]) * FPS) / ms_t[QDA:QDB]
    rk = (kpd[:, 1:] - (kp[:, 1:] - kp[:, :-1]) * FPS) / ms_t[KDA:KDB]
    return rq.pow(2).mean() + rk.pow(2).mean()


def smooth_loss(fp, ms_t, mm_t):
    """jerk penalty on the RENDERED channels only -- root (vxy,h,d6: 0:9) + joint angles (jq: 9:76), the exact
    channels reconstruct_qpos uses. 3rd time-difference (jerk), per-channel-std-normalized. This is DART's
    calc_jerk adapted to CANVAS's direct-readout rep (qd/kpd are auxiliary and never rendered)."""
    ch = (fp * ms_t + mm_t)[..., 0:76]                             # vxy,h,d6,jq in raw units
    j = (ch[:, 3:] - 3 * ch[:, 2:-1] + 3 * ch[:, 1:-2] - ch[:, :-3]) / ms_t[0:76]
    return j.pow(2).mean()


def train_vc(D, mv, steps, w_feat, w_vc, w_sm, warmup=2000):
    """MotionTrans training = baseline latent x0 MSE + decode-space feature-recon + velocity-consistency (vc,
    aux channels) + rendered-channel jerk penalty (sm, the direct anti-jitter lever)."""
    n_sess = D["n_sess"]; zh, zf, cmd, sid, zmean, zstd = build_windows(D, mv)
    N = zh.shape[0]; print(f"[P1] {N} motion windows; {n_sess} sess; steps {steps}", flush=True)
    ms_t = torch.tensor(D["mstd"], device=DEV); mm_t = torch.tensor(D["mmean"], device=DEV)
    m = MotionTrans(n_sess).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01, fused=True)
    plist = list(m.parameters()); ema = [p.detach().clone() for p in plist]
    zhn = (zh - zmean) / zstd; zfn = (zf - zmean) / zstd; B = 256; t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups: g["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        aw = min(step / warmup, 1.0)                                # ramp the aux losses after the latent fit starts
        i = torch.randint(0, N, (B,), device=DEV)
        x0 = zfn[i]; lam = torch.rand(B, device=DEV); ab = abar(lam)[:, None, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = m.context(zhn[i]); pred = m.denoise(xt, lam, ctx, cmd[i], sid[i])
            l_lat = ((pred - x0) ** 2).mean()
        # decode-space losses in fp32 (finite differences want the precision); grad flows through frozen mv
        zfull_pred = torch.cat([zh[i], pred.float() * zstd + zmean], 1)          # (B,16,DM) raw latent
        fp = mv.decode(zfull_pred).float()                                       # (B,64,281) normalized feat
        with torch.no_grad():
            ft = mv.decode(torch.cat([zh[i], zf[i]], 1)).float()
        l_feat = F.smooth_l1_loss(fp, ft)
        l_vc = vc_loss(fp, ms_t, mm_t) if w_vc else torch.zeros((), device=DEV)
        l_sm = smooth_loss(fp, ms_t, mm_t) if w_sm else torch.zeros((), device=DEV)
        loss = l_lat + aw * (w_feat * l_feat + w_vc * l_vc + w_sm * l_sm)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(plist, 1.0); opt.step()
        with torch.no_grad():
            torch._foreach_mul_(ema, 0.99); torch._foreach_add_(ema, plist, alpha=0.01)
        if step % 500 == 0:
            print(f"    step={step:5d} ({time.time()-t0:4.0f}s) lat={float(l_lat):.4f} feat={float(l_feat):.4f} "
                  f"vc={float(l_vc):.4f} sm={float(l_sm):.4f} aw={aw:.2f}", flush=True)
    with torch.no_grad():
        for e, p in zip(ema, plist): p.copy_(e)
    m.eval(); return m, zmean, zstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=14500)        # step-matched to the reproduce baseline
    ap.add_argument("--w_feat", type=float, default=1.0)
    ap.add_argument("--w_vc", type=float, default=0.0)         # velocity-consistency on aux channels
    ap.add_argument("--w_sm", type=float, default=1.0)         # jerk penalty on rendered channels (the direct lever)
    ap.add_argument("--tag", type=str, default="sm")
    ap.add_argument("--nvid", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    D = C.build(motion_only=True); mv = load_tokenizer()
    print(f"[P1:{a.tag}] {D['n_sess']} loco sess; frozen tok; w_feat={a.w_feat} w_vc={a.w_vc} w_sm={a.w_sm}", flush=True)
    m, zmean, zstd = train_vc(D, mv, a.steps, a.w_feat, a.w_vc, a.w_sm)
    out = C.HERE.parent / "demo_b" / "out" / f"motor_phase1_{a.tag}.pt"
    torch.save(dict(model_cfg=dict(n_sess=int(D["n_sess"]), d=192, heads=4, layers=6, se_d=16),
                    state_dict=m.state_dict(), zmean=zmean.detach().cpu().numpy(), zstd=zstd.detach().cpu().numpy(),
                    cmean=np.asarray(D["cmean"], np.float32), cstd=np.asarray(D["cstd"], np.float32),
                    mmean=np.asarray(D["mmean"], np.float32), mstd=np.asarray(D["mstd"], np.float32),
                    provenance=dict(steps=a.steps, w_feat=a.w_feat, w_vc=a.w_vc, w_sm=a.w_sm,
                                    git=C.git_provenance())), out)
    print(f"[P1:{a.tag}] saved -> {out}", flush=True)
    segs = sorted([s for s in D["train"]["segs"] if len(s["feat"]) >= CLIP * 2], key=lambda s: -len(s["feat"]))
    n_steps = U.n_steps_for_seconds(a.seconds); outdir = C.HERE.parent / "demo_b" / "out"
    print("\n[P1] scores (baseline was 07_29 jerk 754 / 07_30 jerk 1095):", flush=True)
    for seg in segs[:a.nvid]:
        nm = next(mm["name"] for mm in D["meta"] if mm["sid"] == seg["sid"]).replace("/", "_")
        gq = rollout(D, mv, m, zmean, zstd, seg, n_steps); gq_ff = FM_.fix_floor(gq)
        FM_.report(gq, label=f"P1:{a.tag}/{nm[:10]}")
        FM_.report(gq_ff, label=f"P1:{a.tag}/{nm[:10]}+ff")
        if a.render:
            render_compare(gq, gq_ff, f"P1:{a.tag} {nm}", outdir / f"phase1_{a.tag}_{nm}.mp4")
    print(f"\n[P1] done in {time.time()-t0:.0f}s.", flush=True)


if __name__ == "__main__":
    main()
