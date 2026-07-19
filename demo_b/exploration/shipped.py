"""Shipped transition primitives (RoPE + QK-norm attention, cosine schedule + DDIM sampler) -- kept HERE only so
the ablation can still build the shipped baseline. The graduated standalone model (models.SimpleTrans) no longer
uses any of these; the loop showed they change the walk by less than the noise floor. See DECISIONS.md."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F
from constants import K


def abar(lam):
    return torch.cos((lam + 0.008) / 1.008 * math.pi / 2).clamp(1e-4, 0.9999) ** 2


def _rope(T, dh, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    a = torch.arange(T, device=device).float()[:, None] * inv[None]
    return a.cos(), a.sin()


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]; cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class RoPEAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.nh = heads; self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(math.sqrt(self.dh))))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = (t.view(B, T, self.nh, self.dh) for t in self.qkv(x).chunk(3, -1))
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        cos, sin = _rope(T, self.dh, x.device); q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) * self.scale
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf"))
        return self.o((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__(); self.n1 = nn.LayerNorm(d); self.attn = RoPEAttn(d, heads)
        self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x)); return x + self.mlp(self.n2(x))


@torch.no_grad()
def ddim(m, ctx, cmd, sidx, steps=10, cfg=1.0):
    B = ctx.shape[0]; x = torch.randn(B, K, m.dz, device=ctx.device); zc = torch.zeros_like(cmd)
    lams = torch.linspace(1, 0, steps + 1, device=ctx.device)
    for i in range(steps):
        x0 = m.denoise(x, lams[i].expand(B), ctx, cmd, sidx)
        if cfg != 1.0:
            x0u = m.denoise(x, lams[i].expand(B), ctx, zc, sidx); x0 = x0u + cfg * (x0 - x0u)
        ab, abn = abar(lams[i]), abar(lams[i + 1])
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp(min=1e-4)
        x = abn.sqrt() * x0 + (1 - abn).sqrt() * eps
    return x
