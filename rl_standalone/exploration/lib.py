"""Tunable surface for the transition-simplification loop: the data cache, the variant factory, the trainer.

The variant is a 2-axis architecture the autoresearch loop walks over, plus minimization knobs:
  attn : rope (RoPE+QK-norm blocks, shipped) | std (nn.TransformerEncoder+sinusoidal PE) | gru | mlp
  head : diffusion (10-step DDIM x0) | regression (predict K latents directly, MSE)
  d, layers, heads, use_sess : size + whether to keep the per-session embedding

Imports the parent `canvas` ONLY to build the loco latent dataset + frozen tokenizer (cached to logs/data.pt);
everything scored downstream uses the frozen `objective.py`.
"""
import sys, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from constants import DEV, CLIP, H, K, DM
from shipped import Block, ddim, abar        # shipped baseline pieces (ablation only)

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "logs" / "data.pt"


# --------------------------------------------------------------------------- data (built once, cached)
def build_data():
    if CACHE.exists():
        d = torch.load(CACHE, map_location="cpu", weights_only=False)
    else:
        import canvas as C
        from canvas.autoresearch.stage3.motor_rollout import build_windows, _cmd_at
        D = C.build(motion_only=True)
        tck = torch.load(ROOT / "canvas" / "out" / "ms_ckpt.pt", map_location="cpu", weights_only=False)
        mv = C.MotionVAE(); mv.load_state_dict(tck["motion"]); mv = mv.to(DEV).eval()
        zh, zf, cmd, sid, zmean, zstd = build_windows(D, mv)
        segs = sorted([s for s in D["train"]["segs"] if len(s["feat"]) >= CLIP * 2], key=lambda s: -len(s["feat"]))
        seg = segs[0]
        cs = np.array([_cmd_at(seg["xy"], seg["yaw"], f0) for f0 in range(0, len(seg["feat"]) - 32, 16)])
        d = dict(zhn=((zh - zmean) / zstd).cpu(), zfn=((zf - zmean) / zstd).cpu(), cmd=cmd.cpu(), sid=sid.cpu(),
                 zmean=zmean.cpu(), zstd=zstd.cpu(), cmean=np.asarray(D["cmean"], np.float32),
                 cstd=np.asarray(D["cstd"], np.float32), mm=np.asarray(D["mmean"], np.float32),
                 ms=np.asarray(D["mstd"], np.float32), n_sess=int(D["n_sess"]), fwd=float(np.linalg.norm(cs[:, :2], axis=1).mean()),
                 seed=dict(feat=seg["feat"].astype(np.float32), xy=np.asarray(seg["xy"], np.float32),
                           yaw=np.asarray(seg["yaw"], np.float32), sid=int(seg["sid"])),
                 seed_name=next(m["name"] for m in D["meta"] if m["sid"] == seg["sid"]))
        CACHE.parent.mkdir(parents=True, exist_ok=True); torch.save(d, CACHE)
    for k in ("zhn", "zfn", "cmd", "sid", "zmean", "zstd"):
        d[k] = d[k].to(DEV)
    d["zmean_t"], d["zstd_t"] = d["zmean"], d["zstd"]
    d["cmean_t"] = torch.tensor(d["cmean"], device=DEV); d["cstd_t"] = torch.tensor(d["cstd"], device=DEV)
    return d


def load_tokenizer():
    import canvas as C
    tck = torch.load(ROOT / "canvas" / "out" / "ms_ckpt.pt", map_location="cpu", weights_only=False)
    mv = C.MotionVAE(); mv.load_state_dict(tck["motion"]); return mv.to(DEV).eval()


# --------------------------------------------------------------------------- backbones + variants
def pos_enc(T, d, device):
    pe = torch.zeros(T, d, device=device); pos = torch.arange(T, device=device)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); return pe


class Enc(nn.Module):
    """encode H history token-embeddings (B,H,d) -> one context vector (B,d)."""
    def __init__(self, attn, d, layers, heads, ff=768):
        super().__init__(); self.attn = attn; self.nf = nn.LayerNorm(d)
        if attn == "rope":
            self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        elif attn == "std":
            self.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, norm_first=True, activation="gelu"), layers)
        elif attn == "gru":
            self.gru = nn.GRU(d, d, layers, batch_first=True)
        elif attn == "mlp":
            self.mlp = nn.Sequential(nn.Linear(H * d, ff), nn.SiLU(), nn.Linear(ff, d))

    def forward(self, x):
        if self.attn == "rope":
            for b in self.blocks: x = b(x)
            return self.nf(x)[:, -1]
        if self.attn == "std":
            return self.nf(self.enc(x + pos_enc(x.shape[1], x.shape[2], x.device)[None]))[:, -1]
        if self.attn == "gru":
            return self.nf(self.gru(x)[1][-1])
        return self.nf(self.mlp(x.reshape(x.shape[0], -1)))


class _Base(nn.Module):
    def __init__(self, cfg):
        super().__init__(); d = cfg["d"]; self.use_sess = cfg["use_sess"]; self.dz = DM
        self.inp = nn.Linear(DM, d); self.enc = Enc(cfg["attn"], d, cfg["layers"], cfg["heads"])
        se = cfg["se_d"] if self.use_sess else 0
        if self.use_sess:
            self.sess = nn.Embedding(cfg["n_sess"], se)
        self.cmd = nn.Sequential(nn.Linear(3 + se, 64), nn.SiLU(), nn.Linear(64, 64))

    def context(self, hist):
        return self.enc(self.inp(hist))

    def cmd_emb(self, cmd, sid):
        return self.cmd(torch.cat([cmd, self.sess(sid)], -1) if self.use_sess else cmd)


class DiffTrans(_Base):
    head = "diffusion"

    def __init__(self, cfg):
        super().__init__(cfg); d = cfg["d"]
        self.fpos = nn.Parameter(torch.randn(K, 32) * 0.02); cond = d + 64 + 64
        self.den = nn.Sequential(nn.Linear(DM + 32 + cond, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, DM))

    def time_emb(self, lam):
        f = torch.exp(torch.linspace(0, 4, 32, device=lam.device)); a = lam[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def denoise(self, xt, lam, ctx, cmd, sid):
        B = xt.shape[0]; c = torch.cat([ctx, self.cmd_emb(cmd, sid), self.time_emb(lam)], -1)
        fp = self.fpos[None].expand(B, K, 32)
        return self.den(torch.cat([xt, fp, c[:, None].expand(B, K, c.shape[-1])], -1))

    def loss(self, hist, fut, cmd, sid):
        B = hist.shape[0]; lam = torch.rand(B, device=hist.device); ab = abar(lam)[:, None, None]
        xt = ab.sqrt() * fut + (1 - ab).sqrt() * torch.randn_like(fut)
        return F.mse_loss(self.denoise(xt, lam, self.context(hist), cmd, sid), fut)

    def predict(self, hist, cmd, sid): return ddim(self, self.context(hist), cmd, sid, cfg=1.0)


class RegTrans(_Base):
    head = "regression"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Sequential(nn.Linear(cfg["d"] + 64, 512), nn.SiLU(), nn.Linear(512, K * DM))

    def predict(self, hist, cmd, sid):
        return self.out(torch.cat([self.context(hist), self.cmd_emb(cmd, sid)], -1)).view(-1, K, DM)

    def loss(self, hist, fut, cmd, sid): return F.mse_loss(self.predict(hist, cmd, sid), fut)


DEFAULTS = dict(attn="rope", head="diffusion", d=192, layers=6, heads=4, se_d=16, use_sess=1)


def make_variant(overrides, n_sess):
    cfg = {**DEFAULTS, **overrides, "n_sess": n_sess, "use_sess": bool(overrides.get("use_sess", 1))}
    v = (DiffTrans if cfg["head"] == "diffusion" else RegTrans)(cfg)
    return v.to(DEV), cfg


def train(v, data, steps, seed=0):
    torch.manual_seed(seed)
    zhn, zfn, cmd, sid = data["zhn"], data["zfn"], data["cmd"], data["sid"]
    opt = torch.optim.AdamW(v.parameters(), 1e-3, weight_decay=0.01); N = zhn.shape[0]; B = 256; t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups: g["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        i = torch.randint(0, N, (B,), device=DEV)
        loss = v.loss(zhn[i], zfn[i], cmd[i], sid[i])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(v.parameters(), 1.0); opt.step()
    v.eval(); return float(loss), time.time() - t0
