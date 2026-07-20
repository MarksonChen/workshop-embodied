# Demo A — PPO locomotion from scratch on Fetch

_Updated 2026-07-20. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) and the
operational [`demo_a/README.md`](../../demo_a/README.md)._

## Purpose

Demo A introduces reinforcement learning before any motion data or learned
prior appears. A policy controls the unmodified ten-actuator Brax v1 Fetch body
and learns only from consequences in simulation.

The canonical workshop task is `FetchRun`, not the original one-shot target
reach. The body must sustain forward speed, remain upright, and avoid excessive
control effort:

\[
r_t=\exp\left(-\frac{(v_x-3)^2}{2}\right)
+0.1\,u_z-10^{-3}\lVert a_t\rVert^2.
\]

The episode ends if torso height falls below half the standing height or torso
up points below the horizon.

## What students should learn

- A policy maps a current observation to an action distribution.
- The environment, not a labeled dataset, returns the next observation and one
  scalar reward.
- PPO changes the policy so sampled actions with better-than-expected returns
  become more likely.
- No target torque or target joint trajectory is supplied.
- A policy can optimize the task without matching the distribution of recorded
  animal motion.

Use the original Fetch reach task only as an optional reward-design contrast:
reaching a point can be solved by lunging or scrambling, while holding speed
requires sustained cyclic behavior.

## Code path

```text
fetch_run.py       FetchRun reward and termination; original Fetch is also available
train_fetch.py     thin Brax v1-to-v2 adapter plus standard PPO call
render_deciles.py deterministic checkpoint rollouts and simple gait statistics
analyze_gait.py    stride-band versus high-frequency spectral diagnostics
render_fetch.py    single-checkpoint original-Fetch renderer
```

The 101-D observation and ten actions are inherited unchanged from Brax Fetch.
`FetchV2` carries the full v1 state through Brax’s v2 PPO wrappers; it does not
change the task.

## Run

From the repository root:

```bash
env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  python -m demo_a.train_fetch --env run --num_timesteps 3e7
```

Use `--smoke` for a 2M-transition wiring check and `--save_deciles` to preserve
learning snapshots. Render and analyze the saved deciles with:

```bash
env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'imageio[ffmpeg]' python demo_a/render_deciles.py --env run

env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  python demo_a/analyze_gait.py --env run
```

## Measured evidence

The saved 500M-transition run used 2,048 environments:

- initial compilation/evaluation completed at about 15 seconds;
- the 50M checkpoint appeared at about 66 seconds;
- the 100M checkpoint appeared at about 106 seconds;
- the full 500M run completed at about 423 seconds.

By 50M, evaluation return was about 1,060 over a 1,000-step episode. Fixed-seed
300-step rollouts of later checkpoints sustain roughly 3.0 Fetch units/s with
path straightness around 0.99. Spectral analysis places most foot-height power
in the 1–6 Hz gait band rather than above 8 Hz.

The transition-matched task-only arms in Demo G establish the shorter live
budget: 30M transitions with three evaluations takes 57.8–59.8 seconds across three
seeds and already supplies the functional G0 behavior. Use that budget for the
live A-to-G arc. Keep the 100M or 500M checkpoint only as an optional polished
standalone Demo A asset.

## Bridge to Demo F and Demo G

The original spatial retarget happened to map 0.20 m/s source motion to about
3.06 Fetch units/s, but it enlarged the body by 21.3789x without changing time.
That is a kinematic coincidence rather than a dynamically valid bridge. Demo F
now applies the 4.6237x Froude time scale, mapping the same source command to
0.924747 Fetch units/s. Demo G reuses `FetchRun`'s reward form and PPO plumbing,
sets this declared target, and preserves Demo A's `sigma / target = 1/3` ratio.

`demo_a/train_fetch.py` also exports the adapter and wrappers reused by Demo G.
The beta-zero Demo G arm must remain numerically equivalent to this task at the
declared slower target.

## Claim boundary

Demo A demonstrates functional learning from interaction. It does not establish
that the learned gait resembles a rat, and it is not a same-body baseline for
native skeletal-rodent experiments. The controlled workshop comparison is
between Demo G’s G0 and G1 arms on this exact Fetch body.
