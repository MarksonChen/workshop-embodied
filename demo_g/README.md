# Demo G — Demo A task PPO plus Demo F motion prior

Demo G is the workshop's controlled SSL + RL comparison on one physical body:

```text
G0: Demo A task reward
G1: Demo A task reward + frozen Demo F motion score
```

The best-performing accepted model is dynamic seed 0. It preserves the task and
substantially reduces the full held-out gait-distance score. The three-seed
result is intentionally narrower: learned likelihood improves in every seed,
but the full direct gait composite improves in only two. This is evidence that
a data prior can shape physical RL, not that it solves locomotion realism.

## Objective

\[
\max_\pi\;\mathbb E_{\tau\sim\pi}\sum_t\gamma^t
\left[r_t^{\rm task}+\beta\,
\bar\ell_\phi(w_t\mid h_t,c_t)\right].
\]

`w_t` is the motion Fetch just produced, `h_t` is its causal motion history,
and `c_t` is Demo F's fixed hindsight displacement command. The Demo F model is
frozen. PPO receives no target torque or action and still discovers actions
from return.

| arm | task reward | frozen prior score |
|---|---:|---:|
| G0 | yes | no (`beta=0`) |
| G1 | yes | yes (`beta=0.1`) |

Matched arms share policy initialization, environment count, PPO settings,
transition budget, reset distribution, and paired evaluation seeds.

## Dynamic alignment

The original retargeting enlarged length by 21.3789x but retained a 50 Hz clock,
making the old 3-unit/s target look like low-gravity motion. Froude similarity
requires a 4.6237x time dilation. The accepted Demo F prior and Demo G task now
share the declared mapping:

- rodent source command: 0.20 m/s;
- physical Fetch target: 0.924747 units/s;
- 0.62-second prior command: `[0.573343, 0, 0]`;
- tracking width: `sigma = target / 3 = 0.308249`.

Preserving Demo A's dimensionless `sigma / target` ratio matters: keeping
`sigma=1` would give a stationary Fetch about 65% of maximum tracking reward.

## Frozen scorer and reward calibration

Each environment collects the same 60-D feature contract used by Demo F. A
32-frame causal buffer warms up before scoring. One batch-level JAX call scores
all 2,048 environments every four 50 Hz control frames (12.5 Hz), and the most
recent score is held between calls. The task reward is still computed every
frame. This is what “scoring every four frames” means; it does not change the
physics rate.

Retargeted root motion is yaw-only, so four unsupported roll/pitch channels are
projected to zero before scoring. The remaining planar root, joint, foot, and
contact contract matches offline features to `6e-7`. The JAX export matches the
PyTorch prior at `5e-4` tolerance.

Calibration was frozen before the dynamic G1 runs. Retimed validation windows
have median raw log likelihood `-0.17`; the matched task-only seed-0 policy has
median `-21.2` with 5th/95th percentiles `-34.3/-12.0`. The bounded reward is

\[
\bar\ell=\operatorname{sigmoid}\left(\frac{\ell+20}{5}\right),
\qquad \beta=0.1.
\]

Always report raw held-out `prior_logp` separately from this transformed
training reward.

## Training

The frozen live budget is 30M transitions, 2,048 environments, and three PPO
evaluations:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with scipy \
  python -m demo_g.train --arm g0 --seed 0

env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with scipy \
  python -m demo_g.train --arm g1 --seed 0
```

Measured `ppo.train` time includes compilation and three evaluations, but not
Python/uv startup:

| training seed | G0 task only | G1 + prior | sequential pair |
|---:|---:|---:|---:|
| 0 | 58.4 s | 68.8 s | 127.2 s |
| 1 | 59.8 s | 69.5 s | 129.3 s |
| 2 | 57.8 s | 68.0 s | 125.8 s |

Every arm is below two minutes; a matched pair is about 2.1 minutes on the
current workshop GPU.

## Shaping-disabled evaluation

Evaluation runs both policies with identical random keys, disables shaping,
and scores only the motion they actually produce. Five rollout seeds are paired
for each of three policy-training seeds. CPU evaluation is intentional because
the identity-checkpoint control is exactly zero there.

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' --with scipy \
  python -m demo_g.evaluate \
  --g0 demo_g/out/g0_seed0_20260720-192221.pkl \
  --g1 demo_g/out/g1_seed0_20260720-192952.pkl \
  --output demo_g/out/evaluation_dynamic_seed0.json

uv run --extra workshop python -m demo_g.summarize
```

The multiseed result is:

| training seed | raw log-p improvement | paired wins | direct composite | tracking retained | survival |
|---:|---:|---:|---:|---:|---:|
| 0 | +18.11 | 5/5 | +5.78 (5/5) | 100.15% | 100% |
| 1 | +32.76 | 5/5 | +3.66 (5/5) | 100.58% | 100% |
| 2 | +17.28 | 5/5 | -0.30 (0/5) | 100.15% | 100% |

Across training seeds, raw likelihood improves by `22.72 ± 7.11`, is positive
in 3/3 seeds, and exceeds two between-seed standard deviations. Mean task
return changes by +0.58%; task tracking and survival are retained in every
seed. The nine-measure direct composite improves by `3.05 ± 2.52`, but only in
2/3 seeds.

Direct distance-to-reference improvements are:

| measure | mean improvement | seeds improved |
|---|---:|---:|
| airborne fraction | +0.036 | 3/3 |
| stance-foot speed | +0.673 | 3/3 |
| approximate stance-world foot speed | +0.701 | 3/3 |
| joint-speed RMS | +0.200 | 3/3 |
| duty factor | +0.087 | 2/3 |
| maximum flight duration | +0.004 | 2/3 |
| contact-switch frequency | +3.757 | 2/3 |
| vertical-acceleration RMS | +0.435 | 2/3 |
| cyclicity | -0.045 | 0/3 |

The accepted claim is therefore limited: the frozen data score improves
robustly while function is preserved, and four direct measures support a
behavioral shift. The complete gait metric is not robust, and cyclicity moves
away from the reference in every seed.

## Best-performing checkpoint and failure inspection

Seed 0 is selected for presentation because it passes every single-seed gate
and has the largest direct-composite improvement, not because it has the largest
learned-likelihood gain. G1 changes these representative means:

| measure | G0 | G1 | held-out motion |
|---|---:|---:|---:|
| speed RMSE | 0.0299 | 0.0243 | 0 |
| action energy | 0.120 | 0.041 | — |
| duty factor | 0.358 | 0.491 | 0.694 |
| airborne fraction | 10.0% | 5.4% | 0.47% |
| contact switches | 11.48 Hz | 3.11 Hz | 0.77 Hz |
| approximate stance-world foot speed | 1.92 | 0.75 | 0.36 |
| vertical acceleration | 1.78 g | 0.91 g | 0.053 g |
| cyclicity | 0.877 | 0.759 | 0.913 |

The largest remaining failures are physically meaningful: maximum continuous
flight grows from 0.032 to 0.060 s, vertical acceleration remains far above the
retargeted reference, and the policy is visibly crouched. Do not describe it as
literal rodent biomechanics.

Local, gitignored inspection artifacts are:

- `demo_g/out/dynamic_videos/seed0_g0_task_only.mp4`;
- `demo_g/out/dynamic_videos/seed0_g1_motion_prior.mp4`;
- `demo_g/out/dynamic_videos/seed0_g0_vs_g1.mp4`;
- `demo_g/out/dynamic_videos/seed0_trace.png`.

Recreate them with `python -m demo_g.render` and `python -m demo_g.diagnose`
using the seed-0 checkpoints shown above.

## Permanent checks

```bash
uv run --extra workshop pytest -q demo_f/tests \
  demo_g/tests/test_features.py demo_g/tests/test_prior.py \
  demo_g/tests/test_evaluate.py

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'pytest>=8' --with scipy \
  python -m pytest -q demo_g/tests/test_env.py
```

## Layout

```text
config.py       frozen beta, calibration, cadence, and PPO budget
features.py     pure-JAX form of Demo F's 60-D contract
prior.py        frozen encoder, Transformer, and Gaussian score
env.py          Demo A task plus causal feature collection
wrappers.py     batch-level scoring and reset-safe history
train.py        shared matched-arm PPO entry point
metrics.py      direct gait/contact/acceleration statistics
evaluate.py     paired shaping-disabled evaluation
summarize.py    three-training-seed acceptance report
render.py       individual and side-by-side videos
diagnose.py     time-varying speed/contact/likelihood traces
experiment/     append-only decisions and measured gates
```

See [`ref/docs/demo_g.md`](../ref/docs/demo_g.md) for the workshop-facing
version and [`ref/docs/WORKSHOP_PLAN.md`](../ref/docs/WORKSHOP_PLAN.md) for the
full teaching arc.
