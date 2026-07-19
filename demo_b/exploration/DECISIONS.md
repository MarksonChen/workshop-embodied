# Transition simplification — autoresearch decision log

**Goal.** Find the *simplest, most standard* transition architecture whose rollout is not meaningfully worse than
the shipped design — so the workshop model is easy to explain. Sequential accept/reject loop per
`ref/docs/method/autoresearch.md`: freeze the objective + budget, calibrate a noise floor, then remove one trick at
a time against a single frozen anchor, logging every decision here.

## Frozen (do not edit mid-loop)
- **Objective** (`objective.py`): scalar = **jerk** of an 8 s constant-forward rollout (lower = smoother; real ~570).
  Gates: walks (disp 0.4–3.0 m), alive (skate 0.001–0.012), sane (finite, jerk < 3000). Gated ⇒ rejected.
- **Anchor** = the shipped design **`rope-diffusion-d192-l6-s1`** (RoPE+QK-norm attention + 10-step DDIM head).
- **Budget** = fixed **STEPS** (set by the convergence probe below), not wall-clock. Frozen tokenizer + frozen seed.
- **Data** = loco latent windows from the parent repo (cached `logs/data.pt`); tunable surface = `lib.py` only.

## The two tricks under test (biggest, least-standard)
1. **attention**: `rope` (RoPE + QK-norm, MotionStreamer) → `std` (`nn.TransformerEncoder` + sinusoidal PE).
2. **head**: `diffusion` (10-step DDIM x0 + `time_emb`/`fpos`/CFG) → `regression` (predict the K latents, MSE).
Removing both would delete `RoPEAttn`, `_rope`, `_apply_rope`, `ddim`, `abar`, `time_emb`, `fpos`, CFG — collapsing
the transition to *"a standard Transformer that predicts the next 8 motion tokens."*

## Log

### 0. Convergence probe (is the step budget fair?) — DONE
Anchor (rope-diffusion), seed 0: jerk **744 (4k) → 739 (8k) → 725 (16k)**. Essentially flat (spread ~19 over 4×
compute — within the seed noise measured below). **Budget = 8000 steps.** All three walk (disp 0.82–0.95, skate ~0.0035).

### 1. Noise floor (eta = 2σ) — DONE
Anchor @ 8k, seeds 0/1/2: jerk **[739, 708, 727]** → mean **724.7**, σ **15.6**, **eta = 2σ = 31**.
⇒ **Accept a simplification iff jerk ≤ 725 + 31 = 756 and it passes the gates.** (Lower jerk = smoother = better,
so "not worse than the anchor by more than noise" is the bar. The probe's 744→725 spread sits inside ±31, fair budget.)

### Block A — attention (rope → std) — **ACCEPT**
`std-diffusion` jerk **710** ≤ 756, walks (disp 0.92, skate 0.0033). RoPE + QK-norm buy nothing here → drop them
(`RoPEAttn`, `_rope`, `_apply_rope`). Use a plain `nn.TransformerEncoder` + sinusoidal PE. Running design → std attn.

### Block B — head (diffusion → regression) — **ACCEPT** (and better)
`std-regression` jerk **659** ≤ 756 — *below* the anchor (smoother, toward real 570) — walks (disp 0.85, skate 0.0033),
and **2.88M vs 3.13M params**. The 10-step diffusion head buys nothing for this near-deterministic task → drop it
(`ddim`, `abar`, `time_emb`, `fpos`, CFG). **Running design = `std-regression`: a standard Transformer that directly
predicts the next 8 motion tokens.** This is the pedagogical target — one attention class, one MLP head, MSE loss.

### Block C — minimize the survivor (`std-regression`)
| candidate | jerk | verdict |
|---|---|---|
| drop session embedding | **695** | **ACCEPT** — session emb is free to remove |
| 2 layers (from 6) | 770 | REJECT (>756) — depth matters; keep ~6 |
| GRU backbone | 858 | REJECT — attention beats recurrence |
| MLP backbone | 1034 | REJECT — needs a sequence model |
The **Transformer earns its place** (GRU/MLP degrade); depth ~6 matters. Session embedding does not.

## Verdict — the workshop model
**`std + regression` (+ no session embedding), ~6 layers.** A standard `nn.TransformerEncoder` over the 8 history
tokens → one MLP head → predict the next 8 latents with MSE. jerk **659–695 ≤ anchor 725 ± 31** (actually *smoother*),
and it trains ~2× faster. **Deleted vs shipped:** `RoPEAttn`, `_rope`, `_apply_rope`, QK-norm, `ddim`, `abar`,
`time_emb`, `fpos`, classifier-free guidance, the session embedding — from ~8 custom classes + a diffusion sampler
down to *one torch built-in transformer + a small head + MSE*. That is the model to present.

All runs in `results.tsv`. Not tried (out of scope, would need retraining the frozen tokenizer): simplifying the
causal-conv-VAE tokenizer — a candidate second loop.

### Block A — attention (rope → std)
_pending._

### Block B — head (diffusion → regression)
_pending._

### Block C — minimize the survivor (drop session emb / fewer layers / gru / mlp)
_pending._
