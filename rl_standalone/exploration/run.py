"""Run ONE transition-simplification experiment: train a variant, score it with the frozen objective, append a
row to results.tsv. The agent calls this once per experiment and judges accept/reject by hand (see DECISIONS.md).

    uv run python rl_standalone/exploration/run.py --attn std --head regression --steps 8000 --seed 0
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib, objective

TSV = Path(__file__).resolve().parent / "results.tsv"
COLS = "id attn head d layers sess steps seed params jerk disp skate status train_loss secs note".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="rope"); ap.add_argument("--head", default="diffusion")
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--sess", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    data = lib.build_data(); mv = lib.load_tokenizer()
    v, cfg = lib.make_variant(dict(attn=a.attn, head=a.head, d=a.d, layers=a.layers, heads=a.heads, use_sess=a.sess),
                              data["n_sess"])
    tl, dt = lib.train(v, data, a.steps, a.seed)
    r = objective.evaluate(v, mv, data)
    params = sum(p.numel() for p in v.parameters())
    cid = f"{a.attn}-{a.head}-d{a.d}-l{a.layers}-s{a.sess}"
    row = dict(id=cid, attn=a.attn, head=a.head, d=a.d, layers=a.layers, sess=a.sess, steps=a.steps, seed=a.seed,
               params=params, jerk=f"{r['jerk']:.1f}", disp=f"{r['disp']:.2f}", skate=f"{r['skate']:.4f}",
               status=r["status"], train_loss=f"{tl:.4f}", secs=f"{dt:.0f}", note=a.note)
    new = not TSV.exists()
    with open(TSV, "a") as f:
        if new: f.write("\t".join(COLS) + "\n")
        f.write("\t".join(str(row[c]) for c in COLS) + "\n")
    print(f"[{cid} seed{a.seed} {a.steps}st] jerk {r['jerk']:.0f}  disp {r['disp']:.2f}  skate {r['skate']:.4f}  "
          f"{r['status']}  | {params/1e6:.2f}M | {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
