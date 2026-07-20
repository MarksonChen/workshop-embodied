"""Motor models. Two pieces:

  MotionVAE   -- the frozen motion tokenizer (causal conv VAE): 64-frame crop <-> 16x16 latent. Copied verbatim
                 from CANVAS so its saved state_dict loads.
  SimpleTrans -- the transition, SIMPLIFIED to a workshop model: a standard Transformer over the last H motion
                 tokens that predicts the next K tokens directly (MSE). The shipped model used rotary/QK-norm
                 attention + a 10-step diffusion head + a per-session embedding; an autoresearch ablation
                 (exploration/DECISIONS.md) showed none of that changes the walk beyond the noise floor, so the
                 graduated model drops all of it. Pure torch; loads assets/motor_standalone.pt.
"""
import math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try:  # package import (``python -m demo_b...``) and workshop script import.
    from .constants import FM, DM, K, DEV
except ImportError:  # pragma: no cover - exercised by direct script entry points.
    from constants import FM, DM, K, DEV

ASSETS = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------- frozen tokenizer: causal conv VAE
class CausalConv1d(nn.Module):
    def __init__(self, ci, co, k, stride=1, dilation=1):
        super().__init__(); self.pad = (k - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(ci, co, k, stride=stride, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, ci, co, k, stride):
        super().__init__(); self.trim = k - stride
        self.conv = nn.ConvTranspose1d(ci, co, k, stride=stride)

    def forward(self, x):
        y = self.conv(x); return y[..., :-self.trim] if self.trim > 0 else y


class ChannelLN(nn.Module):
    def __init__(self, c):
        super().__init__(); self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalResBlock(nn.Module):
    def __init__(self, c, dilation=2, norm=False):
        super().__init__()
        self.n1 = ChannelLN(c) if norm else nn.Identity(); self.c1 = CausalConv1d(c, c, 3, dilation=dilation)
        self.n2 = ChannelLN(c) if norm else nn.Identity(); self.c2 = CausalConv1d(c, c, 3, dilation=dilation)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class MotionVAE(nn.Module):
    def __init__(self, fm=FM, hid=256, dm=DM, nblk=0, norm=False, film=False):
        super().__init__()
        enc = [CausalConv1d(fm, hid, 5), nn.SiLU(), CausalConv1d(hid, hid, 4, stride=2), nn.SiLU(),
               CausalConv1d(hid, hid, 4, stride=2), nn.SiLU()]
        enc += [CausalResBlock(hid, norm=norm) for _ in range(nblk)]
        self.enc = nn.Sequential(*enc)
        self.film = nn.Sequential(nn.Linear(fm, hid), nn.SiLU(), nn.Linear(hid, 2 * hid)) if film else None
        self.to_mu = CausalConv1d(hid, dm, 3); self.to_lv = CausalConv1d(hid, dm, 3)
        dec = [CausalConv1d(dm, hid, 3), nn.SiLU()] + [CausalResBlock(hid, norm=norm) for _ in range(nblk)]
        dec += [CausalConvTranspose1d(hid, hid, 4, stride=2), nn.SiLU(),
                CausalConvTranspose1d(hid, hid, 4, stride=2), nn.SiLU(), CausalConv1d(hid, fm, 5)]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):                                    # (B,64,FM) -> mu,lv (B,16,DM)
        e = self.enc(x.transpose(1, 2))
        if self.film is not None:
            s, b = self.film(x.mean(1)).chunk(2, -1); e = e * (1 + s.unsqueeze(-1)) + b.unsqueeze(-1)
        return self.to_mu(e).transpose(1, 2), self.to_lv(e).transpose(1, 2)

    def decode(self, z):                                    # (B,16,DM) -> (B,64,FM)
        return self.dec(z.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(latent), mu, logvar


# ---------------------------------------------------------------- the transition: a standard Transformer
def pos_enc(T, d, device):
    pe = torch.zeros(T, d, device=device); pos = torch.arange(T, device=device)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); return pe


class SimpleTrans(nn.Module):
    """Predict the next K motion latents from the last H, conditioned on the egocentric command.
    A `nn.TransformerEncoder` encodes the history; an MLP head reads the last token + the command and emits the
    next K latents. Trained with plain MSE. `predict(hist, cmd)` -> (B, K, DM)."""
    def __init__(self, d=192, layers=6, heads=4, ff=768):
        super().__init__(); self.dz = DM
        self.inp = nn.Linear(DM, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, norm_first=True, activation="gelu"), layers)
        self.nf = nn.LayerNorm(d)
        self.cmd = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        self.out = nn.Sequential(nn.Linear(d + 64, 512), nn.SiLU(), nn.Linear(512, K * DM))

    def context(self, hist):
        x = self.inp(hist); x = x + pos_enc(x.shape[1], x.shape[2], x.device)[None]
        return self.nf(self.enc(x))[:, -1]

    def predict_from_context(self, context, cmd):
        """Predict from a context already computed by :meth:`context`.

        This is algebraically identical to ``predict`` and lets downstream closed-loop
        controllers reuse the context they expose to a policy instead of running the
        Transformer twice.
        """
        return self.out(torch.cat([context, self.cmd(cmd)], -1)).view(-1, K, DM)

    def predict(self, hist, cmd, sid=None):                 # sid is ignored (no per-session embedding)
        return self.predict_from_context(self.context(hist), cmd)

    def predict_next(self, hist, cmd):
        """Conditional mean of the next normalized motion token."""
        return self.predict(hist, cmd)[:, 0]

    def log_prob_next(self, hist, next_token, cmd, sigma):
        """Mean log likelihood per latent dimension under a fixed Gaussian.

        ``SimpleTrans`` is still trained with MSE.  This method merely exposes
        the equivalent probabilistic interpretation used as Demo E's frozen
        reward; it does not introduce another learned head.
        """
        sigma = torch.as_tensor(sigma, dtype=hist.dtype, device=hist.device)
        error = (next_token - self.predict_next(hist, cmd)) / sigma
        return -0.5 * (error.square() + 2 * sigma.log() + math.log(2 * math.pi)).mean(-1)

    def loss(self, hist, fut, cmd, sid=None):
        return F.mse_loss(self.predict(hist, cmd), fut)


def load_motor(ckpt=ASSETS / "motor_standalone.pt"):
    """frozen tokenizer + the simplified transition + norm constants + a bundled real seed window."""
    d = torch.load(ckpt, map_location=DEV, weights_only=False)
    # ``rl_standalone`` is the known-good pre-Demo-E checkpoint.  It predates
    # explicit metadata, so infer its tokenizer shape from the saved weights.
    # Newer bundles carry the same values in ``motion_cfg``.  Supporting both
    # here makes regression comparisons exact rather than approximate.
    motion_cfg = dict(d.get("motion_cfg", {}))
    first_weight = d["motion"]["enc.0.conv.weight"]
    motion_cfg.setdefault("fm", int(first_weight.shape[1]))
    motion_cfg.setdefault("hid", int(first_weight.shape[0]))
    motion_cfg.setdefault("dm", int(d["motion"]["to_mu.conv.weight"].shape[0]))
    if motion_cfg["fm"] != int(np.asarray(d["mmean"]).shape[0]):
        raise ValueError("motion tokenizer and normalization feature dimensions disagree")
    mv = MotionVAE(**motion_cfg).to(DEV); mv.load_state_dict(d["motion"]); mv.eval()
    for p in mv.parameters(): p.requires_grad_(False)
    m = SimpleTrans(**d["model_cfg"]).to(DEV); m.load_state_dict(d["trans"]); m.eval()
    norms = {k: np.asarray(d[k], np.float32) for k in ("zmean", "zstd", "cmean", "cstd", "mmean", "mstd")}
    norms["sigma"] = np.asarray(d.get("sigma", 1.0), np.float32)
    norms["logp_clip"] = np.asarray(d.get("logp_clip", [-20.0, 0.0]), np.float32)
    seed = dict(feat=np.asarray(d["seed_feat"], np.float32), xy=np.asarray(d["seed_xy"], np.float32),
                yaw=np.asarray(d["seed_yaw"], np.float32), sid=0, name=d.get("seed_name", ""))
    return mv, m, norms, seed
