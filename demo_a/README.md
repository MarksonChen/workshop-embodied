# demo_a/ — RL-only from-scratch walker (Demo A)

**Plan:** [ref/docs/demo_a.md](../ref/docs/demo_a.md). **Thesis:** [ref/docs/WORKSHOP_PLAN.md](../ref/docs/WORKSHOP_PLAN.md).

Demo A is the **RL-only** corner of the SSL+RL 2×2: **functional realism, ~zero
distributional realism** — a *competent-but-unnatural* walker. Pure RL on the raw
38-DoF rodent body (**no decoder, no data prior**), so its gait and internal
representation don't match the real rat. It reuses the track-mjx / vnl-playground
stack as a library; nothing here touches `rl/` or `demo_b/`.

## Substrate
- **Env: `RodentMaintainVelocity`** (vnl-playground) — `torque_actuators=True` (raw 38-D
  torque), a forward-velocity reward, and **no upright reward** (staying up must be
  learned). Termination on fall / below-ground / NaN.
- **Trainer:** brax PPO via track-mjx `scripts/train_task.py`, launched through
  `demo_a/train.py` (applies this setup's shims: typed-key, `device_put_replicated`,
  checkpoint-only eval callback — see `rl/train_joystick.py`).

## Layout
```
demo_a/
  README.md    this file
  train.py     direct-task RL launcher (no decoder); --smoke for a fast loop test
  render.py    roll out a checkpoint, measure ACTUAL forward speed (m/s), render mp4
  runs/        training outputs (git-ignored)
  out/         rendered videos (git-ignored)
```

## Run
```bash
# true smoke test (seconds+compile): confirm the loop + checkpointing
uv run python demo_a/train.py --smoke

# convergence probe / full walker (H100)
env -u LD_LIBRARY_PATH uv run python demo_a/train.py \
    --num_envs 8192 --batch_size 2048 --num_timesteps 1e8 --eval_every 5000000 \
    2>&1 | tee demo_a/runs/train.log
uv run python rl/watch_health.py demo_a/runs/train.log   # reward=fwd-vel, len, fallen

# render + measure actual velocity (the eval log only has the *reward*, not m/s)
env -u LD_LIBRARY_PATH uv run python demo_a/render.py --ckpt demo_a/runs/<run>/<step>
```

## Status (2026-07-19, convergence probe COMPLETE — 53 M steps, ~50 min on one H100)
A from-scratch walker emerges. The forward-velocity **reward** climbs monotonically
(1.2→528) and *never plateaus* — but that is misleading, because `reward ≈ speed ×
episode-length` conflates two things that separate cleanly once you render for the actual
speed (m/s) and read `fallen%` on its own:

| step | fwd speed (m/s, render) | survives (deterministic) | eval fallen% | eval len | reward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 M | 0.19 | fell 0.4 s | 100% | 31 | 18.6 |
| 24 M | 0.35 | fell 0.65 s | 100% | 88 | 47.6 |
| **35 M** | **0.46** | fell 2.7 s | 100% | 201 | 114 |
| 47 M | 0.36 | full 5 s | 91% | 612 | 388 |
| 53 M | 0.40 | full 5 s | 50% | 786 | 528 |

**Findings:**
- **Forward speed plateaus at ~0.4 m/s by ~35 M (~35 min).** What improves *after* is
  **survival** (`fallen` 100%→50%, `len` 200→786), not speed. So the raw **reward is a
  poor accept/reject metric** (autoresearch §3) — track **speed + fallen% separately**.
- It **veers** (lateral drift up to ~1 m per 5 s): the unshaped, forward-velocity-only
  reward doesn't penalize sideways drift → a curving, unnatural gait. **On-thesis** for the
  RL-only corner (functional, not distributional).
- Training is stable (KL settles to ~0.015 after warmup). Throughput ~20 k sps
  (~1 M steps/min wall-clock); **evals ate ~⅓ of the wall-clock** — the cheapest speedup.
- Videos: `demo_a/out/demoa_<step>.mp4` (gait timeline; render **sequentially**, never
  concurrent with training — that silently OOM-kills, autoresearch §12).

**Budget for autoresearch (§4):** the *functional* signal (forward speed) plateaus by
**~35 M**, so ~30 M is a reasonable per-experiment budget for ranking reward/arch variants
on speed; survival needs longer. Cutting eval overhead + a shaped-reward harness would lower
it further. Not yet run: the multi-seed noise floor (§5).

## Not yet built
- Kinematic-realism metric (gait stats vs the real-rat mocap distribution) — the
  distributional axis; would reuse `demo_b/foot_metrics.py` + a gait-stats module.
- Reward-shaping variants (add alive/upright + action-rate smoothness) if the gait is a
  lunge rather than a stable walk.
