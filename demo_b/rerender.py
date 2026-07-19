"""demo_b/rerender.py -- re-render the comparison videos as raw | grounded (floor + upright).

The original renders corrected only floor (fix_floor); over an 8-9 s rollout the open-loop orientation
integration also drifts the trunk nose-up ('flying'). ground() = fix_upright + fix_floor cleans both. This
re-renders the baseline and Phase-1 comparison clips for the same two loco seeds. No retrain.

Run:  uv run python -m demo_b.rerender [--seconds 8]
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import argparse
import torch
import canvas as C
from canvas.train import CLIP, DEV
from canvas.autoresearch.stage3.motor_rollout import rollout
from canvas import utils as U
from demo_b.rollout import load_motor, render_compare, REPRO_CKPT, PHASE1_CKPT
from demo_b import foot_metrics as FM_


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, default=8.0); a = ap.parse_args()
    D = C.build(motion_only=True)
    segs = sorted([s for s in D["train"]["segs"] if len(s["feat"]) >= CLIP * 2], key=lambda s: -len(s["feat"]))[:2]
    n_steps = U.n_steps_for_seconds(a.seconds); outdir = C.HERE.parent / "demo_b" / "out"
    for prefix, ckpt in [("repro", REPRO_CKPT), ("phase1", PHASE1_CKPT)]:
        mv, m, norms = load_motor(path=ckpt)
        zmean = torch.tensor(norms["zmean"], device=DEV); zstd = torch.tensor(norms["zstd"], device=DEV)
        for seg in segs:
            nm = next(mm["name"] for mm in D["meta"] if mm["sid"] == seg["sid"]).replace("/", "_")
            gq = rollout(D, mv, m, zmean, zstd, seg, n_steps)
            render_compare(gq, FM_.ground(gq), f"{prefix} {nm}", outdir / f"{prefix}_{nm}.mp4")


if __name__ == "__main__":
    main()
