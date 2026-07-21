# Demo D — one-stage hindsight-command reinforcement learning

_Drafted 2026-07-19. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md),
[dataset.md](dataset.md), the runnable [`demo_d/` package](../../demo_d/), and the
append-only [experiment log](../../demo_d/experiment/DECISIONS.md)._

> **Presentation status.** Retained as a measured research reference. The core
> workshop now presents Demos A, B, F, and H; see
> [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) and [demo_h.md](demo_h.md).

> **Current status.** The one-stage system is implemented end to end. Two complete
> fixed-budget runs establish that the present policy learns some held-out imitation
> but does **not** yet learn useful command control. A third, training-termination
> correction is implemented and tested as pipeline v5, but its run was stopped before
> checkpoint 0 at the user's request. It has no result yet.

## 1. Executive decision

The proposed one-stage design is possible and is the clearest pedagogical successor
to Demos A and B:

1. Shift an unlabelled locomotion trajectory into the future.
2. Convert that future displacement into Demo B's compact egocentric command.
3. Give the command and current proprioception to one policy.
4. Use PPO and MJX physics to discover the 38 torques that realize the motion.

There is no published MIMIC policy, pretrained imitation decoder, second joystick
policy, or target action. The precise name is **goal-conditioned imitation RL with
hindsight relabelling**. It is SSL-like because data construct the goal without human
labels, but PPO remains RL because scalar reward—not a target action or prediction
loss—updates the policy.

The implementation has also exposed an important negative result: a hindsight label
that is always paired with its own demonstrated body state can be statistically
redundant. A policy may infer gait phase from proprioception and largely ignore the
three command values. Hindsight relabelling supplies labels; it does not automatically
supply the counterfactual interventions needed for controllability.

## 2. Historical place in the workshop

When Demo D was designed, the four demos answered different questions:

| demo | signal | body | lesson |
|---|---|---|---|
| A | task reward | Fetch quadruped | PPO learns actions from consequences |
| B | reconstruction and shifted-future targets | recorded rat | data can supervise itself |
| C | future-latent loss plus food reward | recorded/virtual rat | an SSL world factor can be frozen inside an RL loop |
| D | data-derived command plus physical reward | MJX rat | self-generated goals can condition a torque policy |

Demo D should not be sold as a state-of-the-art MIMIC reproduction. Its value is that
a new graduate can see exactly where the self-generated target ends and where RL
begins:

```text
unlabelled future motion ──> hindsight command             data supplies the goal
command + proprioception ──> policy ──> torque ──> physics
                                      ▲              │
                                      └──── reward ──┘  PPO learns the action
```

The live definition is:

> **The future recording supplied the command; reward taught the policy which
> torques make that command real in physics.**

## 3. The one-stage task

### 3.1 Data and frozen split

Demo D uses the public MIMIC-MJX `rodent_reference_clips.h5` file, verified against
SHA-256
`c7b02c16d6796f70e62169b5a5aeb65381ea5d42d8e9c75af95cd26b31fb638e`.
The curated file contains 842 five-second clips. The frozen workshop subset is:

- 48 training clips: 24 `Walk` and 24 `FastWalk`;
- 16 disjoint validation clips: 8 `Walk` and 8 `FastWalk`;
- explicit clip IDs stored in [`demo_d/config.py`](../../demo_d/config.py), never
  recomputed from a mutable catalogue.

The relabel audit contains 10,512 training windows and 3,504 validation windows. All
commands are finite. Train and validation means/stds are close:

| split | mean `[dx, dy, dψ]` | standard deviation |
|---|---|---|
| train | `[0.0641, 0.0073, 0.1715]` | `[0.0733, 0.0372, 0.6823]` |
| validation | `[0.0666, 0.0074, 0.1674]` | `[0.0730, 0.0377, 0.6157]` |

### 3.2 Exact Demo B hindsight command

At recorded frame `t`, take the root pose 31 frames later. At 50 Hz this is a
0.62-second horizon. Rotate planar displacement into the current root-yaw frame and
wrap the yaw difference:

```text
γt = [dxego, dyego, wrap(ψt+31 − ψt)]
```

The implementation deliberately uses Demo B's yaw-only planar rotation. Root pitch,
roll, and height cannot leak into `dxego` or `dyego`. Tests cover heading, yaw sign,
nonzero pitch/roll, and measured-velocity geometry.

### 3.3 Policy and physics

The actor observes:

```text
3-D hindsight command + 277-D current proprioception = 280 values
```

It outputs all 38 bounded torques at 100 Hz. The actor and critic are standard
512–512–256 MLPs trained from random initialization with Brax PPO. No reference joint,
body, or future-pose target enters the observation.

During direct deployment, an external command replaces the first three observation
values. `command_step` advances physics and rebuilds proprioception without reading a
future clip frame, imitation reward, or hidden target. A synthetic-time test has run
this path beyond the end of the source clip and rendered it headlessly, establishing
causal independence from mocap after initialization.

### 3.4 Training reward

The current pipeline combines two readable terms:

```text
reward = MIMIC pose-imitation terms
       + 2 / (1 + Σ ((measured velocity − γt / 0.62 s) / scale)²)
```

The physical scales are 0.06 m/s forward, 0.04 m/s lateral, and 0.35 rad/s yaw.
Velocity is estimated causally from successive simulated root poses in the same
yaw-only frame and averaged with a 0.62-second exponential time constant. Recorded
`qvel` is not used: an audit found reset values inconsistent with displacement over
the command horizon.

Pipeline v5 changes training termination only. Reference drift no longer ends a
training episode; only torso height at or below 0.03 m, torso tilt beyond 60 degrees,
or a non-finite simulation does. The frozen natural-imitation evaluator still applies
the original reference-relative termination, so v5 cannot pass merely by weakening
the held-out test.

## 4. What is implemented

```text
demo_d/
  config.py             immutable clip IDs, horizon, budget, gates, pipeline version
  reference.py          download/hash verification and split construction
  env.py                exact relabel, reward, training and causal deployment steps
  train.py              standard from-scratch Brax PPO and atomic latest-run pointer
  runtime.py            environment/checkpoint reconstruction and provenance checks
  metrics.py            frozen bounded scores and reportability gates
  evaluate.py           paired checkpoint-0 and trained evaluation
  plot_learning.py      return, episode-length, KL, and throughput curves
  render.py             reference-ghost and reference-free command videos
  waypoint.py           transparent geometric command steering, not a learned policy
  provenance.py         rejects parents, wrong data, and published MIMIC weights
  experiment/DECISIONS.md
  tests/test_core.py
```

Thirteen fast tests currently pass. They cover data split integrity, exact command
geometry, causal velocity measurement, reward/null behavior, paired evaluation starts,
physical termination, provenance, report gates, and waypoint geometry. Modules compile
and the repository diff passes whitespace validation.

## 5. Frozen evaluation contract

Every completed report run saves checkpoint 0 before learning. The evaluator compares
it with the final checkpoint on identical held-out starts.

### Natural imitation

- all 16 validation clips;
- 300 control steps;
- normalized pose-imitation reward with the explicit command reward removed;
- reference-relative survival, joint error, and root distance.

### Reference-free command intervention

- commands `(0.04, 0, 0)`, `(0.08, 0, 0)`, `(0.12, 0, 0)`,
  `(0.08, 0, −0.30)`, `(0.08, 0, +0.30)`, and `(0.08, 0.02, 0)`;
- each command from the same three paired initial poses;
- 400 controls with a 50-control warm-up;
- only physical fall/non-finite termination;
- score 1 for exact joint forward/lateral/yaw tracking and approximately 0.14 for
  a stationary survivor averaged across commands.

The predeclared report gates are:

| gate | threshold |
|---|---:|
| natural-imitation gain over checkpoint 0 | at least 0.08 |
| natural-imitation survival | at least 0.70 |
| direct-command gain over checkpoint 0 | at least 0.08 |
| absolute direct-command score | at least 0.45 |
| direct-command survival | at least 0.70 |

The waypoint probe is downstream of these gates. It may demonstrate a passing motor
policy, but it cannot turn a failed command policy into a successful result.

## 6. Results so far

The fixed random checkpoint has natural imitation score 0.0026, natural survival
0.0187, and zero command score/survival. Two complete 26,214,400-step runs give:

| pipeline | single controlled design | imitation score | imitation survival | command score | command survival | verdict |
|---|---|---:|---:|---:|---:|---|
| v3 | full-pose imitation only | 0.1045 | 0.2037 | 0.0411 | 0.3333 | reject |
| v4 | add causal command-grounding reward | 0.0835 | 0.1687 | 0.0439 | 0.3333 | reject |
| v5 | train through reference drift | — | — | — | — | implemented, not run |

Both completed runs pass only the imitation-gain gate. In both, initial poses 0 and 1
fall for every external command. Pose 2 survives but remains almost stationary:
approximately 0.002–0.003 m/s under every requested speed, turn, and lateral command.
For v3, a same-state intervention found across-command action standard deviation
0.0073 against action RMS 0.2089. The command is connected to the actor, but its effect
is weak.

The v4 reward produced a temporary held-out episode-length advantage at intermediate
checkpoints, but it disappeared at the frozen endpoint:

| realized steps | v3 episode length | v4 episode length |
|---:|---:|---:|
| 5.24 M | 43.5 | 44.1 |
| 10.49 M | 87.1 | 99.9 |
| 15.73 M | 75.0 | 81.2 |
| 20.97 M | 95.3 | 102.5 |
| 26.21 M | 93.7 | 89.5 |

This is why v4 is rejected rather than rescued by selecting its best-looking
intermediate checkpoint after seeing the test behavior.

## 7. Diagnosis

The core problem is conditional identifiability, not a missing tensor connection.
Training only presents congruent triples:

```text
(current demonstrated body state, its own future command, its own future pose)
```

Proprioception already carries gait phase. Because the future command is correlated
with that state and full-pose reward follows the same hidden clip, the actor can obtain
imitation reward with little causal dependence on the command. Adding a command reward
to those same congruent samples does not create experience of “this body state under a
different command.” External command replacement is therefore an intervention outside
the training joint distribution.

Reference-relative termination compounds the problem: in v4's final held-out PPO
evaluation, nearly all episodes ended for root distance or rotation from the hidden
reference, not a physical fall. Pipeline v5 isolates this second issue by allowing
training recovery while retaining the strict evaluator. It does not, by itself,
remove the command/state correlation.

## 8. Next steps

### Step 1 — finish the frozen v5 experiment

Run pipeline v5 once at the unchanged seed-0 budget, then apply the unchanged paired
evaluator. Do not resume or score the interrupted directory
`report-seed0-20260719-211602`; it contains only configuration metadata and no
checkpoint. A fresh run is required.

```bash
uv run python -m demo_d.train --seed 0
uv run python -m demo_d.evaluate
uv run python -m demo_d.plot_learning
```

Accept v5 only if all five existing gates pass. Otherwise append the result and move
to the next architecture block; do not tune termination thresholds on validation.

### Step 2 — if v5 fails, use empirical command replay

The next principled design is still one stage and still uses Demo B-style hindsight:

1. Build a train-only bank of the 10,512 measured `[dx, dy, dψ]` labels.
2. Sample commands independently of the current mocap clip/phase, optionally holding
   each for 0.62 s.
3. Initialize the rat from valid training locomotion poses, but reward the requested
   physical displacement/turn rather than an incompatible time-indexed root target.
4. Retain physical fall, energy/action smoothness, joint limits, and healthy posture.
5. Keep the validation command set and all report gates frozen.

This creates the counterfactual support missing from v3/v4: the same kind of body state
must respond differently when the empirical command changes. It is best described as
**goal-conditioned RL with a self-supervised command distribution**, not full-pose
imitation. Functionally it resembles velocity-command locomotion, but it is not the
repository's `RodentJoystick` task: commands come from unlabelled future rat motion,
include lateral displacement, and one torque policy is trained from scratch.

For pedagogical simplicity, do not add a discriminator, latent decoder, planner,
second policy, or learned reward in this block. If a rat-like gait prior is later
needed, test it only after command control works.

### Step 3 — explicitly guard against standing

The earlier `rl/` experiment showed that a broad velocity reward makes standing a
strong local optimum. Before any empirical-replay run:

- calculate the exact stationary return under the proposed training reward;
- preserve the already-frozen evaluation null and absolute score gate;
- log achieved versus commanded forward/lateral/yaw rates separately;
- include a real fall penalty or lost-return consequence;
- change reward sharpness/weight one axis at a time.

Do not accept a run because it survives. Survival and command tracking are separate
gates.

### Step 4 — confirm a passing design

After one configuration passes seed 0:

- train at least seeds 1 and 2 at the same budget;
- report mean, standard deviation, and every per-command row;
- require any claimed improvement to exceed the measured seed noise;
- render fixed speed, left/right turn, lateral, and waypoint trials;
- keep failed and interrupted runs in the append-only decision log.

### Step 5 — defer strong neural claims

The curated 842 clips do not contain frame identities linking them back to simultaneous
Aldarondo spikes. Most MIMIC-MJX clip files therefore cannot support a leak-free neural
comparison. The one genuinely frame-aligned smoothed coltrane session could support a
teacher-forced diagnostic: replay recorded poses through the frozen policy, extract
hidden activity, and compare it with active-unit population activity under strict time
blocks and shift controls. With one session it must be labelled a feasibility plot,
not evidence that Demo D is more neural-like than Demos A–C.

Demo C remains the defensible neural comparison because it uses four continuous,
aligned DLS/MC sessions, matched rodent policies, Poisson encoding, and crossvalidated
RSA.

## 9. Workshop presentation now

Until a command policy passes, Demo D is best presented as a short scientific lesson
rather than a completed locomotion triumph:

1. Shift one real trajectory by 31 frames and compute the command live.
2. Point out that no human supplied “forward” or “turn left.”
3. Show the single policy and distinguish the data-derived goal from PPO reward.
4. Show that held-out imitation improves over random initialization.
5. Intervene on the command and show the failed control result honestly.
6. Explain why observation is not causation: correlated hindsight commands can be
   ignored without counterfactual training support.
7. End with empirical command replay as the minimal next experiment.

That negative result is pedagogically legitimate. It teaches both definitions and a
deeper point:

> **Self-generated labels tell a model what varied in the data; controllable behavior
> also requires learning what changes when we intervene.**

## 10. Claim boundaries

- Do call Demo D goal-conditioned RL with self-generated hindsight commands.
- Do not call PPO self-supervised learning.
- Do not claim external command following from natural imitation reward.
- Do not select a checkpoint after inspecting the fixed held-out interventions.
- Do not describe v5 as evaluated; it has not taken one training update.
- Do not compare Demo D activity with unaligned neural recordings.
- Do not imply that Demo D reproduces MIMIC's published two-stage decoder stack.

The implementation and negative results already make a coherent workshop chapter.
The next milestone is not more architectural sophistication; it is a torque policy
whose physical velocity changes reliably when the empirical hindsight command changes.
