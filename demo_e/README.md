# Demo E — task RL with a frozen motion prior

> **Research reference (2026-07-20).** Demo E preserves the full-skeletal-rodent
> attempt and its negative ten-minute result. It is no longer the live workshop
> synthesis; the current fast same-body design is Demo G. See
> [`ref/docs/WORKSHOP_PLAN.md`](../ref/docs/WORKSHOP_PLAN.md).

Demo E was the workshop's first literal composition of Demos A and B:

```text
Demo A: PPO learns what works from task reward.
Demo B: conditional prediction learns what rat motion looks like from data.
Demo E: PPO receives both signals while controlling the virtual rodent.
```

The controlled comparison is

```text
E0: task reward
E1: task reward + beta × frozen Demo B motion score
```

Both arms use the same MIMIC skeletal rodent, 100-Hz torque physics, native
RodentJoystick reset, observations, PPO architecture, and frozen 16-D TRACK-MJX imitation
decoder.  PPO trains a new high-level intention policy from scratch.  The
decoder is shared motor infrastructure, not the experimental SSL manipulation;
only E1 receives Demo B's frozen conditional Gaussian likelihood as reward.
For throughput, the scorer is compile-time pruned from E0 training; the paired
evaluator re-enables it with `beta=0` and scores both saved policies identically.

The likelihood history is initialized from the native pose and withheld for
the first 0.64 s while real policy motion replaces the synthetic context. The
predictor keeps its native eight-token block contract. PPO receives a frozen
monotone `[0, 1]` normalization of physical-controller log likelihood; raw log
likelihood is always evaluated separately.

This hierarchy is deliberate.  A direct 38-actuator policy repeatedly learned
to sit or fall within the practical budget.  Reproducing the upstream hierarchy
first produced steady locomotion, so Demo E keeps the part of the original
implementation that solves low-level coordination and asks a clean question at
the task-policy level: does a learned motion likelihood improve distributional
realism while task reward preserves function?

## Run

The format-v6 Demo E scorer is committed at
`demo_b/assets/motor_prior_demo_e_jax.npz`. Validate its contracts:

```bash
uv run --extra dev --extra workshop pytest -q demo_e/tests
```

Run a wiring check for each arm:

```bash
uv run python -m demo_e.train --arm e0 --smoke
uv run python -m demo_e.train --arm e1 --smoke
```

`--smoke` is only an integration test.  It is not expected to learn
locomotion. The measured report-geometry live profile runs 2.62M transitions
in under five minutes on the H100:

```bash
uv run python -m demo_e.train --arm e0 --workshop
uv run python -m demo_e.train --arm e1 --workshop
```

This profile demonstrates optimization, not convergence. Full controlled runs
use identical seeds:

```bash
uv run python -m demo_e.train --arm e0 --seed 0
uv run python -m demo_e.train --arm e1 --seed 0
```

Evaluate the final pair, or every saved checkpoint to expose the locomotion
onset rather than judging only an early video:

```bash
uv run python -m demo_e.evaluate
uv run python -m demo_e.evaluate --all-checkpoints
uv run python -m demo_e.render demo_e/out/evaluation-<stamp>.npz \
  --command 3 --seed 0
```

The evaluator reports velocity tracking, survival, task reward, and Demo B
motion likelihood separately.  It evaluates both arms with `beta=0`, so the
score cannot alter the measured trajectories.

## Budget

The reproduced upstream joystick reference requested 50,000,000 environment
steps and reached 52,428,800 after PPO batch rounding.  On the repository's
H100 this was about 40 minutes of optimizer/environment work and 7 minutes of
evaluation, or roughly 45–47 minutes end to end. A three-seed checkpoint curve
at a fixed 0.30 m/s command under the literal native reset shows why the full
budget matters:

| transitions | elapsed after initial save | survival | mean forward speed |
|---:|---:|---:|---:|
| 13.1M | ~11 min | 1.00 | 0.000 m/s |
| 26.2M | ~22 min | 1.00 | -0.002 m/s |
| 39.3M | ~34 min | 1.00 | 0.002 m/s |
| 52.4M | ~45 min | 1.00 | 0.240 m/s |

The saved evidence therefore places locomotion onset somewhere after 39.3M and
at or before 52.4M transitions—roughly 34–45 minutes into this run. The final
0.30 m/s video is the user-confirmed steady-locomotion reference. A prior v4
audit that appeared to show gait at 13M had silently forwarded the reset; it is
archived under `out/reproduction/pipeline_v4_forwarded_reset/`. The five-minute
profile must be described only as a live optimization/wiring exercise.

Pipeline v6 repairs the prior bridge and the validation-selected scorer passes
the frozen source and physical-transfer gates on two training seeds. Its
physically calibrated reward range is `[-1.5, -0.75]` nats per latent
dimension, with `beta=1`.

The first pipeline-v6 E1 diagnostic saved 9.83M transitions in 9.71 minutes.
At a 0.30 m/s command it learns to stand almost upright on two legs, but moves
only about 0.013 m/s and reaches the torso-angle termination threshold around
3.04 s. Across the full command grid its final survival is 0.450 and functional
score 0.055. This is meaningful posture learning, not locomotion, and no
intermediate checkpoint hides a gait. The tracked analysis is
`experiment/TEN_MINUTE_E1.md`.

The next valid comparison is paired E0/E1 at the same transition count. A
ten-minute wall-clock run is not a convergence budget: scorer overhead limited
E1 to fewer transitions than the upstream 13.1M standing checkpoint, while the
confirmed native-reset gait required 52.4M transitions.

## Canonical layout

```text
config.py       one time grid, task configuration, PPO budget, command grid
features.py     exact 281-D MJX side of Demo B's feature contract
prior.py        frozen pure-JAX encoder, causal predictor, Gaussian score
env.py          joystick task, likelihood reward, decoder, reset wrapper
train.py        the sole PPO entry point for E0 and E1
evaluate.py     the sole functional/likelihood evaluator and learning curve
render.py       single or paired rollout renderer
runtime.py      checkpoint loading shared by evaluation
provenance.py   fail-closed artifact/controller metadata
experiment/     offline prior audit and tracked diagnostic results
tests/          split, time, feature, parity, ablation, and contract tests
```

Rejected direct-actuator/Freddie-era scripts were removed from the canonical
package.  Their old files under `demo_e/out/` remain historical artifacts and
are rejected by pipeline-version metadata; they are not workshop results.
