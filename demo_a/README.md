# Demo A — Fetch PPO from scratch

Demo A is the workshop’s reinforcement-learning baseline. Standard PPO trains
the unmodified ten-actuator Brax v1 Fetch body from random initialization. No
motion clips, action labels, imitation decoder, or learned prior are used.

Use `FetchRun` for the current workshop: it rewards sustained 3.0-unit/s forward
motion, upright posture, and low action magnitude, and terminates on a fall. The
original target-reaching Fetch task remains available with `--env fetch` as a
reward-design contrast.

```bash
env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  python -m demo_a.train_fetch --env run --num_timesteps 3e7
```

Add `--smoke` for a 2M-transition wiring run or `--save_deciles` to save
learning snapshots. A saved 500M reference reached 100M transitions in about
106 seconds and completed in 423 seconds on this machine. The matched Demo G
task-only runs show that the 30M live budget already works in 58–59 seconds
with three evaluations; use the longer reference only for a polished replay.

```text
fetch_run.py       task reward, fall termination, and environment selection
train_fetch.py     v1-to-v2 adapter and unchanged Brax PPO call
render_deciles.py checkpoint videos and trajectory statistics
analyze_gait.py    stride-band and high-frequency foot-motion analysis
render_fetch.py    original target-reaching Fetch renderer
```

See [`ref/docs/demo_a.md`](../ref/docs/demo_a.md) for the teaching sequence,
measurements, and the exact bridge to Demo F/G.
