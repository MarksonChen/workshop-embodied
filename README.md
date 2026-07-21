# embodied

Neuromechanical rodent control experiments built on
[MIMIC-MJX](https://mimic-mjx.talmolab.org) (stac-mjx + track-mjx).

The repository has two connected tracks:

- `rl/` drives the virtual rodent from point A to point B with a high-level policy
  over the published frozen imitation decoder (the paper's §2.5 transfer).
- Demos A, B, F, and H form the core graduate workshop on reinforcement
  learning, self-supervised learning, and generative pretraining followed by
  RL post-training. Start with
  [`ref/docs/WORKSHOP_PLAN.md`](ref/docs/WORKSHOP_PLAN.md).

## Layout

- `demo_a/` — PPO and quadruped locomotion from scratch.
- `demo_b/` — conditional self-supervised motion modelling on the validated
  strict-locomotion subset of Coltrane recordings.
- `demo_f/` — Coltrane locomotion retargeted to Fetch plus a conditional
  generative motion model; canonical Demo F uses Froude timing, while Demo H
  owns a separate 1.75x derivative.
- `demo_h/` — accepted body-centric world–action prior and β=0.10 residual PPO
  capstone in exact Fetch physics.
- `demo_g/` — measured research comparison that uses Demo F likelihood as a
  reward-side prior for scratch PPO.
- `demo_c/` — research reference: frozen rodent world model, PPO, and aligned
  neural comparison.
- `demo_d/` — research reference: one-stage hindsight-command imitation PPO and
  its measured negative command-control result.
- `demo_e/` — research reference: Demo B likelihood with MIMIC-MJX
  `RodentJoystick`; its long scratch-PPO diagnostic learned standing, not gait.
- `rl/` — published-decoder joystick experiment. Start with `rl/README.md`.
- `ref/repos/` — upstream repos as submodules (track-mjx, stac-mjx, DART,
  MotionStreamer).
- `ref/papers/` — local reference papers used by the design documents.

## Setup

```bash
uv sync --extra cuda12 --extra workshop --extra dev
# Do not use cuda13 on WSL2; see rl/README.md.
```
