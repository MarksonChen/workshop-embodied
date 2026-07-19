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

## Status (2026-07-19, convergence probe in progress)
A from-scratch walker **is emerging**. Forward-velocity reward and episode length climb
(and *accelerate*) with no plateau through 35 M steps on an H100 (~1 M steps/min):

| step | fwd-vel reward | episode_len | fallen |
| ---: | ---: | ---: | ---: |
| 0 | 1.2 | 7.6 | 100% |
| 12 M | 18.6 | 31 | 100% |
| 24 M | 47.6 | 88 | 100% |
| 35 M | 114.0 | 201 | 100% |

Training is stable (KL settles to ~0.015 after warmup). **`fallen` stays 100%** — it always
falls *eventually*, surviving longer each time; whether this is a real gait or lunge-and-fall
needs `render.py` (see the "monitoring is insufficient on its own" note in
[ref/docs/demo_a.md](../ref/docs/demo_a.md)). The metric is **not yet plateaued**, so the
convergence-probe budget (autoresearch §4) is > 35 M — see `canvas/misc/autoresearch.md`.

## Not yet built
- Kinematic-realism metric (gait stats vs the real-rat mocap distribution) — the
  distributional axis; would reuse `demo_b/foot_metrics.py` + a gait-stats module.
- Reward-shaping variants (add alive/upright + action-rate smoothness) if the gait is a
  lunge rather than a stable walk.
