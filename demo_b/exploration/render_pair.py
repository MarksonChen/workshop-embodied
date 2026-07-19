"""Train two transition variants and render their rollouts SIDE-BY-SIDE (left = A, right = B) under the same
constant-forward command, so you can see that a simplified design walks like the shipped one.

    uv run python demo_b/exploration/render_pair.py --b_attn std --b_head regression   # anchor | simplified
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import lib
from constants import DEV, CLIP, H, NLAT, DM, FM, n_steps_for_seconds
from geometry import reconstruct_qpos, q0_from
from rollout import render_compare
import foot_metrics as met

OUT = Path(__file__).resolve().parent / "out"


@torch.no_grad()
def roll_gq(v, mv, data, seconds=8):
    seed = data["seed"]; zmean, zstd = data["zmean_t"], data["zstd_t"]; cmn, csd = data["cmean_t"], data["cstd_t"]
    mm, ms, fwd = data["mm"], data["ms"], data["fwd"]
    sidx = torch.full((1,), seed["sid"], device=DEV)
    zm0 = mv.encode(torch.tensor((seed["feat"][:CLIP] - mm) / ms, dtype=torch.float32, device=DEV)[None])[0][0]
    stream = [((zm0 - zmean[0]) / zstd[0])[:H]]
    cmdn = (torch.tensor([[fwd, 0.0, 0.0]], device=DEV) - cmn) / csd
    for _ in range(n_steps_for_seconds(seconds)):
        stream.append(v.predict(torch.cat(stream, 0)[-H:][None], cmdn, sidx)[0])
    Z = torch.cat(stream, 0) * zstd[0] + zmean[0]; nn_ = (Z.shape[0] // NLAT) * NLAT
    gfeat = mv.decode(Z[:nn_].reshape(-1, NLAT, DM)).reshape(-1, FM).cpu().numpy() * ms + mm
    return reconstruct_qpos(gfeat, q0_from(gfeat[0], seed["xy"][0], seed["yaw"][0]))


def one(over, data, mv, steps, seed):
    v, cfg = lib.make_variant(over, data["n_sess"]); lib.train(v, data, steps, seed)
    gq = roll_gq(v, mv, data); p = sum(pp.numel() for pp in v.parameters())
    print(f"  {over}: jerk {met.jerk(met.paw_positions(gq)):.0f}  {p/1e6:.2f}M", flush=True)
    return gq


def cfg(prefix, a):
    return dict(attn=getattr(a, prefix + "attn"), head=getattr(a, prefix + "head"),
                layers=getattr(a, prefix + "layers"), use_sess=getattr(a, prefix + "sess"))


def main():
    ap = argparse.ArgumentParser()
    for p, da, dh, dl in [("a_", "rope", "diffusion", 6), ("b_", "std", "regression", 6)]:
        ap.add_argument(f"--{p}attn", default=da); ap.add_argument(f"--{p}head", default=dh)
        ap.add_argument(f"--{p}layers", type=int, default=dl); ap.add_argument(f"--{p}sess", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="simplify_pair.mp4")
    a = ap.parse_args()
    data = lib.build_data(); mv = lib.load_tokenizer()
    ca, cb = cfg("a_", a), cfg("b_", a)
    print(f"A (left)  = {ca}\nB (right) = {cb}", flush=True)
    ga = one(ca, data, mv, a.steps, a.seed); gb = one(cb, data, mv, a.steps, a.seed)
    name = f"{ca['attn']}+{ca['head']}  |  {cb['attn']}+{cb['head']}"
    render_compare(met.ground(ga, alpha=0.03), met.ground(gb, alpha=0.03), name, OUT / a.out)


if __name__ == "__main__":
    main()
