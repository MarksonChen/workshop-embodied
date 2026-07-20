# embodied

Neuromechanical rodent control experiments built on
[MIMIC-MJX](https://mimic-mjx.talmolab.org) (stac-mjx + track-mjx).

The repository now has two connected tracks:

- `rl/` drives the virtual rodent from point A to point B with a high-level policy
  over the published frozen imitation decoder (the paper's §2.5 transfer).
- Demos A, B, and E form the core graduate workshop on
  self-supervised learning, reinforcement learning, and their combination in
  physical rodent control. Start with
  [`ref/docs/WORKSHOP_PLAN.md`](ref/docs/WORKSHOP_PLAN.md).

## Layout

- `demo_a/` — PPO and quadruped locomotion from scratch.
- `demo_b/` — conditional self-supervised motion modelling on the validated
  strict-locomotion subset of Coltrane recordings.
- `demo_e/` — paired high-level PPO on MIMIC-MJX `RodentJoystick`, with and
  without Demo B's frozen conditional motion likelihood; both arms share the
  published frozen imitation decoder. Pipeline v6 passes the prior-transfer
  gates; its ten-minute E1 diagnostic learns upright standing but not gait.
- `ref/docs/demo_e.md` — design, measured throughput, evaluation protocol, and
  workshop interpretation for Demo E.
- `demo_c/` — research reference: frozen rodent world model, PPO, and aligned
  neural comparison.
- `demo_d/` — research reference: one-stage hindsight-command imitation PPO and
  its measured negative command-control result.
- `rl/` — published-decoder joystick experiment. Start with `rl/README.md`.
- `ref/repos/` — upstream repos as submodules (track-mjx, stac-mjx, DART,
  MotionStreamer).
- `ref/papers/` — reference PDFs (gitignored).

## Setup

```bash
uv sync --extra cuda12 --extra workshop --extra dev
# Do not use cuda13 on WSL2; see rl/README.md.
```
