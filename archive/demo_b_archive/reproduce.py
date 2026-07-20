"""demo_b/reproduce.py -- reproduce the AUTORESEARCH motor rollout (stage3/motor_rollout.py) and SCORE it.

The reference behind canvas/out/motor_rollout_*.mp4 was produced by stage3.motor_rollout, which trains a fresh
MotionTrans (300 s, x0 diffusion + EMA) on the LOCO subset and rolls out real loco-segment seeds under the
segment's hindsight command. That harness cannot run as-is here (s3.load_motion_tokenizer wants the missing
out/stage1_motion_tokenizer.pt), so we load the SAME frozen tokenizer from ms_ckpt.pt['motion'] (verified:
identical mmean, ~c1 reconstruction) and otherwise reuse the autoresearch train()/rollout() verbatim.

This replaces the live_link motor_ckpt.pt baseline (a different artifact). Scores each rollout with
demo_b.foot_metrics (+ fix_floor) and renders to demo_b/out/repro_*.mp4 (does NOT touch canvas/out/motor_rollout_*).

Run:  uv run python -m demo_b.reproduce [--secs 300] [--nvid 2] [--seconds 8] [--render]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse, time
import numpy as np, torch
import canvas as C
from canvas.train import CLIP, DEV
from canvas.autoresearch.stage3.motor_rollout import train, rollout
from canvas import utils as U
from demo_b import foot_metrics as FM_
from demo_b.rollout import render_compare, REPRO_CKPT


def load_tokenizer(tok_ckpt=C.OUT / "ms_ckpt.pt"):
    """the FROZEN Stage-1 production tokenizer (from ms_ckpt.pt['motion'] -- the mint motor_rollout would load)."""
    tck = torch.load(tok_ckpt, map_location=DEV, weights_only=False)
    mv = C.MotionVAE().to(DEV); mv.load_state_dict(tck["motion"]); mv.eval()
    for p in mv.parameters(): p.requires_grad_(False)
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=300.0)       # transition training wall-clock (reference default)
    ap.add_argument("--nvid", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    D = C.build(motion_only=True); mv = load_tokenizer()
    print(f"[repro] {D['n_sess']} loco sess; frozen tokenizer; build {time.time()-t0:.0f}s; train {a.secs:.0f}s", flush=True)
    m, zmean, zstd = train(D, mv, a.secs)                      # autoresearch training, verbatim
    # persist the faithful model as the canonical demo_b/ baseline (schema matches demo_b.rollout.load_motor)
    REPRO_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_cfg=dict(n_sess=int(D["n_sess"]), d=192, heads=4, layers=6, se_d=16),
                    state_dict=m.state_dict(),
                    zmean=zmean.detach().cpu().numpy(), zstd=zstd.detach().cpu().numpy(),
                    cmean=np.asarray(D["cmean"], np.float32), cstd=np.asarray(D["cstd"], np.float32),
                    mmean=np.asarray(D["mmean"], np.float32), mstd=np.asarray(D["mstd"], np.float32),
                    provenance=dict(secs=a.secs, source="demo_b.reproduce (autoresearch motor_rollout)",
                                    git=C.git_provenance())), REPRO_CKPT)
    print(f"[repro] saved faithful model -> {REPRO_CKPT}", flush=True)
    segs = sorted([s for s in D["train"]["segs"] if len(s["feat"]) >= CLIP * 2], key=lambda s: -len(s["feat"]))
    seeds = segs[:a.nvid]; n_steps = U.n_steps_for_seconds(a.seconds)
    outdir = C.HERE.parent / "demo_b" / "out"
    print(f"[repro] {len(seeds)} loco seeds, {n_steps} steps -> ~{a.seconds:.0f}s each\n", flush=True)
    for seg in seeds:
        nm = next(mm["name"] for mm in D["meta"] if mm["sid"] == seg["sid"]); safe = nm.replace("/", "_")
        gq = rollout(D, mv, m, zmean, zstd, seg, n_steps)     # autoresearch rollout (hindsight command)
        gq_ff = FM_.fix_floor(gq)
        FM_.report(gq, label=f"repro/{safe[:12]}")
        FM_.report(gq_ff, label=f"repro/{safe[:12]}+ff")
        if a.render:
            render_compare(gq, gq_ff, safe, outdir / f"repro_{safe}.mp4")
    print(f"\n[repro] done in {time.time()-t0:.0f}s. REAL reference: skate ~0.003, pen ~1mm, jerk ~570.", flush=True)


if __name__ == "__main__":
    main()
