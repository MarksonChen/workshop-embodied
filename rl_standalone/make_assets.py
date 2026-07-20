"""BOOTSTRAP ONLY -- train the simplified transition (models.SimpleTrans) on the parent loco data and bundle it,
with the frozen tokenizer + norms + a real seed, into assets/motor_standalone.pt (the standalone model).

This is the ONE file here that reaches outside rl_standalone: it imports `canvas` (via the exploration harness)
for the training data + frozen tokenizer. The runtime (models/rollout/drive/waypoint/foot_metrics) imports nothing
outside rl_standalone. Run once from the CANVAS repo root:

    uv run python rl_standalone/make_assets.py [--steps 16000]

The transition it bundles is the graduated design from the autoresearch loop (exploration/DECISIONS.md): a standard
Transformer + regression head, no rotary attention / diffusion / session table.
"""
import sys, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # models, constants
sys.path.insert(0, str(HERE / "exploration"))          # lib: build_data + train (imports canvas)
import torch
import lib
from models import SimpleTrans
from constants import DEV

ROOT = HERE.parent
DST = HERE / "assets" / "motor_standalone.pt"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=16000); a = ap.parse_args()
    data = lib.build_data()
    m = SimpleTrans(d=192, layers=6, heads=4).to(DEV)
    fl, dt = lib.train(m, data, a.steps, seed=0)
    motion = torch.load(ROOT / "canvas" / "out" / "ms_ckpt.pt", map_location="cpu", weights_only=False)["motion"]
    seed = data["seed"]
    torch.save(dict(
        arch="simple", motion=motion, trans=m.state_dict(), model_cfg=dict(d=192, layers=6, heads=4),
        zmean=data["zmean"].cpu().numpy(), zstd=data["zstd"].cpu().numpy(),
        cmean=data["cmean"], cstd=data["cstd"], mmean=data["mm"], mstd=data["ms"],
        seed_feat=seed["feat"], seed_xy=seed["xy"], seed_yaw=seed["yaw"], seed_name=data.get("seed_name", ""),
    ), DST)
    p = sum(pp.numel() for pp in m.parameters())
    print(f"trained SimpleTrans {a.steps} steps ({dt:.0f}s, final mse {fl:.4f}); {p/1e6:.2f}M params", flush=True)
    print(f"wrote {DST} ({DST.stat().st_size/1e6:.1f} MB); seed={data.get('seed_name','')}", flush=True)


if __name__ == "__main__":
    main()
