# Demo D — one-stage hindsight-command imitation RL

> **Research reference (2026-07-19).** Preserve this implementation and its
> negative command-control evidence, but do not include it in the core live
> workshop. Demo G is now the planned same-body synthesis; see
> [`ref/docs/demo_g.md`](../ref/docs/demo_g.md).

The canonical design, measured v3/v4 results, diagnosis, claim boundaries, and next
steps are in [`ref/docs/demo_d.md`](../ref/docs/demo_d.md). The runnable package here
is currently pipeline v5: implemented and tested, but not yet trained.

Demo D trains a physical virtual rodent **from random initialization**.  It does not
load MIMIC's published imitation decoder and it does not add a second
`RodentJoystick` policy.

The central move is the same one used to train Demo B: take two poses from an
unlabelled motion trajectory and turn their difference into an egocentric command.
Here, however, one PPO policy must turn that compact command plus proprioception
directly into 38 joint torques:

```text
recorded pose at t ───────────────────────────────────────┐
recorded pose at t + 0.62 s ── hindsight relabel ──> γt  │
                                               [dx,dy,dψ] │
                                                         ▼
                                      γt + proprio ──> PPO policy ──> 38 torques
                                                         │                 │
hidden recorded full pose at t ── imitation reward <─────┴── MJX physics ──┘
                          measured command velocity reward <───────────────┘
```

At deployment, `γt` is overwritten by a user command.  The simulator then advances
using only that command, proprioception, the policy, and physics; it no longer reads a
future reference frame or imitation reward.

## What kind of learning is this?

The precise name is **goal-conditioned imitation RL with hindsight relabelling**.

- The three-number goal is self-generated from an unlabelled recording, just as in
  Demo B.  No human annotates “walk forward” or “turn left.”
- PPO still optimizes a scalar imitation reward from interaction with physics, so the
  optimizer is reinforcement learning—not a pure self-supervised loss.
- The reference action is never supplied.  The policy discovers torques that keep the
  simulated body close to the recorded motion.

It is fair to call this an SSL-like or data-grounded RL example in the workshop, but
not fair to call PPO itself self-supervised learning.

## Why one stage is pedagogically useful

The previous two-stage idea—first train a full-motion decoder, freeze it, then train a
joystick policy—matches the published MIMIC stack more closely.  This one-stage version
makes the relationship to Demos A and B much easier to see:

| demo | target supplied by | optimizer | output |
|---|---|---|---|
| A | task reward | PPO | quadruped actions |
| B | shifted recorded data | reconstruction/prediction loss | kinematic rat motion |
| D | shifted data makes the command; physics gives imitation reward | PPO | physical rat torques |

The tradeoff is intentional: `[dx, dy, dψ]` does not uniquely describe every joint
motion.  Demo D restricts training to locomotion and supplies current proprioception,
which carries gait phase, but it cannot reproduce arbitrary behaviours from that
three-number command.

## Frozen experiment

- Public source: `talmolab/MIMIC-MJX`, exact
  `rodent_reference_clips.h5` SHA-256 recorded in `config.py`.
- Training set: 48 explicit clips, balanced between `Walk` and `FastWalk`.
- Validation set: 16 disjoint explicit clips, also balanced.
- Command: root-local displacement and wrapped yaw change from frame `t` to `t+31`
  (0.62 s at 50 Hz), exactly Demo B's definition.
- Observation: 3 command values + 277 proprioceptive values; no future joint or body
  target is visible to the policy.
- Action: 38 bounded torque controls at 100 Hz.
- Reward: root position/orientation, joint pose, end effectors, healthy height, small
  control/energy costs, and a causal physical command-tracking term.
- Training termination in pipeline v5: physical fall/non-finite state, while the
  unchanged held-out imitation evaluator retains strict reference-drift termination.
- Optimizer: standard Brax PPO, one MLP actor-critic, random initialization.
- Report budget: requested 25 M physics steps (26,214,400 realized because PPO updates
  in fixed blocks), with checkpoints near every 5 M steps.

`config.py` is the frozen source of truth.  Every checkpoint has a provenance marker;
runtime loading rejects a parent checkpoint, a changed data hash, or the known
published MIMIC checkpoint identifier.

## Current evidence

Pipeline v3 (imitation only) and pipeline v4 (plus command grounding) both completed
26,214,400 steps. They improved natural imitation over checkpoint 0 but failed direct
command control: command scores were 0.041 and 0.044, survival was 0.333, and the only
surviving initial pose remained nearly stationary under every command. Pipeline v5's
physical-termination correction is ready but unrun; the first attempt was stopped
during compilation before checkpoint 0 or any PPO update. See the canonical plan for
the full table and the empirical-command-replay fallback.

## Setup and run

From the repository root:

```bash
uv sync --extra cuda12 --extra workshop --extra dev

# Optional short wiring run. Smoke artifacts never become report checkpoints.
uv run python -m demo_d.train --smoke

# Fixed pipeline-v5 report run from scratch. A completed run atomically updates the
# latest pointer; do not try to resume the interrupted metadata-only directory.
uv run python -m demo_d.train --seed 0

# Compare the saved random policy at step 0 with the latest trained policy.
uv run python -m demo_d.evaluate
uv run python -m demo_d.plot_learning

# Natural held-out imitation (white reference ghost) and intervention on γ.
uv run python -m demo_d.render --mode imitation
uv run python -m demo_d.render --mode command --command 0.08 0.00 0.00

# Closed-loop waypoint commands; the waypoint steering rule is not learned.
uv run python -m demo_d.waypoint --shape square --render

# Fast CPU tests.
JAX_PLATFORMS=cpu uv run pytest -q demo_d/tests
```

The reference file is downloaded on first use and verified byte-for-byte.  Generated
checkpoints, JSON reports, plots, and videos are written under `demo_d/out/`, which is
git-ignored.

## Evaluation and claim gate

The evaluator always compares checkpoint 0 with the requested trained checkpoint on
the same validation resets.

1. **Natural imitation:** normalized reward, physical/reference survival, joint error,
   and root distance on all 16 held-out clips.
2. **Direct commands:** six fixed interventions spanning forward speed, turning, and
   lateral motion, each from three fixed initial states.  Only physical fall/NaN
   termination applies; no future reference is consulted.
3. **Waypoint probe:** a transparent geometric controller updates the same command
   from current position and bearing.  Success tests the learned motor policy, not a
   second RL stage.

The predeclared workshop gate requires both natural-imitation and direct-command score
to improve by at least 0.08 over random initialization, direct-command score to reach
at least 0.45, and both survival scores to be at least 0.70.  Raw per-command and
per-clip rows remain in the JSON even when a gate fails.

## Neuroscience scope

Demo D is a useful same-body physical model, so its hidden units are in principle a
better candidate for a rat-neural comparison than Demo A's different quadruped.
However, the curated 842 MIMIC clips do not ship frame identities linking them to the
simultaneous Aldarondo neural recordings, and the torque policy's full proprioception
also depends on its own simulated action history.  Pretending those clips align would
be leakage.

The completed Demo C analysis remains the defensible workshop neural comparison: it
uses genuinely aligned continuous recordings, fixed temporal splits, a matched
rodent RL-only baseline, Poisson encoding, and crossvalidated RSA.  A future Demo D
analysis could teacher-force the one known frame-aligned smoothed session, but it must
be labelled as a one-session diagnostic rather than evidence that Demo D is “more
brain-like.”

## File map

```text
demo_d/
  config.py             fixed clips, command horizon, PPO budget, evaluation gates
  reference.py          download/hash check and immutable train/validation split
  env.py                hindsight relabel, hidden imitation reward, causal command step
  train.py              one standard PPO run from random initialization
  runtime.py            checkpoint reconstruction and provenance validation
  metrics.py            frozen command score and reportability gate
  evaluate.py           checkpoint-0 vs trained held-out comparison
  plot_learning.py      episode-length/return/KL/throughput audit figure
  render.py             natural imitation or fixed-command videos
  waypoint.py           transparent closed-loop command steering
  provenance.py         guards against accidental published/parent weights
  experiment/DECISIONS.md
  tests/test_core.py
```

The sentence to use during the live workshop is:

> **The future recording supplied the command; reward taught the policy which torques
> make that command real in physics.**
