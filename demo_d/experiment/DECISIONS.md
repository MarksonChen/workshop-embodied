# Demo D experiment decisions

This is an append-only implementation log following the useful practices in
`canvas/misc/autoresearch.md`: freeze the split/budget/metric, keep a random baseline,
change one thing at a time, and record rejected runs rather than narrating only the
winner.

## D0 — reject the inherited two-stage transfer plan

The initial proposal mirrored `rl/`: load a published MIMIC imitation decoder and
train a new `RodentJoystick` task policy.  That does not satisfy the workshop goal of
learning the physical motor network from scratch.  A variant that trained the decoder
first and then froze it would satisfy provenance, but would still create two concepts
and two training stages.

Decision: use one goal-conditioned torque policy.  Future mocap supplies Demo B's
three-number hindsight command; the hidden full reference supplies imitation reward.
At test time replace the command without a second learned policy.

## D1 — freeze the smallest honest problem

- 48 training clips and 16 validation clips, explicit IDs and balanced `Walk` /
  `FastWalk` membership;
- exact public reference SHA-256;
- 31-frame / 0.62-s command horizon copied from Demo B;
- one 512-512-256 MLP actor and value network;
- one standard Brax PPO run, seed 0, requested budget 25 M steps;
- random checkpoint 0 as the paired baseline;
- natural imitation and externally commanded physics as co-primary report gates.

No published policy, parent checkpoint, imitation action, future joint target, or
joystick reward enters the actor input.

## D2 — smoke run, keep the wiring

Run: `smoke-seed0-20260719-181518`, 327,680 realized physics steps.

- held-out Brax episode length rose from 7.25 to 38.56 control steps;
- a deliberately short natural-imitation probe improved, showing reward and gradients
  were connected;
- direct-command survival remained near zero at this tiny budget.

Decision: keep the implementation, not the checkpoint.  The result is a wiring test,
not evidence for command control.

## D3 — reject an interrupted report run after a geometry test

Rejected run: `report-seed0-20260719-184314`, interrupted after its 10,485,760-step
checkpoint.

A focused invariant test compared the JAX hindsight relabel with Demo B's NumPy
definition under a 90-degree heading.  Translation agreed, but yaw had the opposite
sign.  Brax defines `relative_quat(q1, q2)` as the rotation from `q1` to `q2`; the
arguments had been reversed.  Natural imitation can still learn with that internally
consistent label, so reward curves alone would not reveal the semantic error.

Decision: reject the run, swap the quaternion arguments, add the invariant test, and
restart from random initialization.  This is a correctness correction, not a tuned
hyperparameter.

## D3b — reject the first restart after an exact Demo B audit

Rejected run: `report-seed0-20260719-190348`, stopped after checkpoint 0 and before
the first trained report checkpoint.

The corrected relative-yaw sign passed, but a source-to-source audit found that the
translation still used a full 3-D inverse quaternion rotation. Demo B deliberately
rotates planar `dxy` by root yaw only. During a gait, the full rotation lets pitch and
roll alter the nominal forward/lateral command.

Decision: implement Demo B's formula literally—yaw-only planar rotation plus wrapped
future-yaw minus current-yaw—and add an invariant with nonzero pitch, roll, and root
height change. Restart again from random initialization. No trained result from this
attempt is retained.

## D3c — real-clip relabel and split audit, keep

Before restarting, calculate every 31-frame command window in the frozen clips using
the literal Demo B NumPy formula (10,512 train and 3,504 validation windows). All were
finite. Train mean/std were `[0.0641, 0.0073, 0.1715]` /
`[0.0733, 0.0372, 0.6823]`; validation mean/std were
`[0.0666, 0.0074, 0.1674]` / `[0.0730, 0.0377, 0.6157]`. The six direct evaluation
commands lie in the dense central region rather than probing extreme tails.

Load the actual clip metadata, not only the declared ID lists: training is exactly 24
`Walk` + 24 `FastWalk`; validation is exactly 8 + 8. Keep the frozen split and command
set.

## D3d — reject an over-generous command metric before evaluation

Evaluate two synthetic invariants before applying the metric to any trained
checkpoint: perfect target velocity scored 1.0, but a stationary, upright rat scored
0.81 averaged across the six commands. The additive translation/turn components and
wide error scales made survival dominate, recreating the same stand-still failure mode
diagnosed in the earlier `rl/` project.

Decision: require translation and turning jointly with one exponential and use explicit
error scales of 0.06 m/s forward, 0.04 m/s lateral, and 0.35 rad/s yaw. Perfect tracking
remains 1.0; the stationary null is 0.14. Add an absolute command-score gate of 0.45 in
addition to the improvement and survival gates, so even an improved stationary policy
cannot pass. Freeze this metric before evaluating any trained report checkpoint. PPO
training and its reward are unchanged.

## D3e — restart to bind the frozen gate into metadata

Rejected run: `report-seed0-20260719-191324`, stopped during its initial compilation,
before a trained checkpoint or held-out comparison.

The absolute 0.45 command gate was added after this process had serialized its resolved
configuration. Even though the gate does not affect PPO, retaining the process would
make the run metadata disagree with the report criterion.

Decision: stop and restart after freezing the complete evaluator. Increment the Demo D
pipeline version to 2 and use that constant in checkpoint metadata, which makes runtime
loading reject the earlier wrong-geometry version-1 runs automatically.

## D3f — finish the causal scoring/render audit

Rejected run: `report-seed0-20260719-191816`, stopped after checkpoint 0 and before
the first trained checkpoint.

The last source audit found that direct-command scoring still measured local velocity
with a full 3-D quaternion, while the frozen command uses yaw-only planar geometry.
The upstream `Imitation.render(..., render_ghost=False)` also indexes a reference frame
unconditionally, which is harmless to actions but contradicts an indefinitely
reference-free deployment artifact.

Decision: measure velocity from successive root positions in the same yaw-only frame;
exclude terminal steps consistently; add a reference-free physics renderer; and test
rendering at a synthetic time beyond the source clip. Increment the complete audited
contract to pipeline version 3. Version-2 checkpoints are not report candidates.

## D3g — headless causal-deployment integration, keep

The first render probe selected GLFW because checkpoint-runtime imports occurred before
the environment module set its EGL default. Move the backend defaults to
`demo_d/__init__.py`, which runs before every Demo D submodule. Repeat in a fresh
process on CPU JAX plus EGL rendering.

Result: a reference-free command step produced observation `(280,)`, action size 38,
reward 0, done 0, and preserved the external command. Set simulator time to 100 s—far
beyond the 5-s source clip—and render successfully to `(120, 160, 3)`. Keep. This is
the final pre-report integration gate.

## D4 — make command deployment causally independent of mocap

The first evaluator overwrote the command observation but called the inherited
imitation step, which continued calculating hidden targets and reference termination.
Those values did not enter the action, but retaining them made the causal story muddy
and limited rollouts to the clip duration.

Decision: add `command_step`.  It advances MJX physics, rebuilds the observation from
the external command and proprioception, returns zero reward, and terminates only on a
physical fall or non-finite state.  Direct-command and waypoint evaluations use this
path; training remains unchanged.

## D5 — corrected fixed-budget report

Run: `report-seed0-20260719-192712`, pipeline v3, 26,214,400 realized steps.

The natural imitation score improved from 0.0026 to 0.1045, passing the +0.08 gain
gate. Natural survival was 0.204, however. In reference-free deployment, command score
was 0.041 and survival 0.333; only one of the three paired initial poses survived, and
it moved at about 0.003 m/s under every requested forward speed. The report verdict is
**not reportable**: only imitation gain passed.

A same-state intervention confirmed that the actor is wired to the command but weakly:
mean across-command action SD was 0.0073 against action RMS 0.2089. The compact label
is correlated with the demonstrated future, but the full-pose imitation reward gives
no direct consequence for obeying a counterfactual command. More training alone is a
poor first response to that causal failure.

Decision: keep this as the frozen negative baseline and make one controlled change.

## D6 — add one command-grounding reward

Pipeline v4 keeps the exact split, actor, PPO settings, budget, imitation terms,
reference termination, evaluator, and gates from v3. Add one reward of weight 2.0:

```text
2 / (1 + Σ ((physical [vx, vy, yaw_rate] - command / 0.62 s) / scale)²)
```

The scales are the already-frozen evaluation scales (0.06 m/s, 0.04 m/s,
0.35 rad/s). The target is still self-generated from future mocap; there is no second
policy, joystick environment, or target action. Physical velocity is measured causally
from successive simulated root positions/yaw; the reference file's instantaneous
`qvel` is deliberately not used because a reset audit found multi-m/s root values
inconsistent with the 0.62-s displacement. Average causal physical velocity with an
exponential time constant of 0.62 s so 10-ms contact impulses do not dominate a
long-horizon command. The average is initialized to the hindsight target at reset and
thereafter updated only by simulated root motion. The Cauchy shape keeps a learning
signal in the tails instead of underflowing like a narrow exponential.
Natural-imitation scoring subtracts this new term so v3/v4 remain comparable. Run the
same 25 M requested steps from random initialization and accept/reject on the unchanged
held-out gates.

Result: `report-seed0-20260719-203009`, 26,214,400 realized steps. The final
natural imitation score/survival were 0.0835/0.1687 and direct-command
score/survival were 0.0439/0.3333. Compared with v3, the command score changed by
only +0.0028 while natural imitation and survival both fell. Seeds 0 and 1 fell
under every direct command; seed 2 survived but moved at roughly 0.002 m/s under
all requested speeds and turns. Only the imitation-gain gate passed. Reject v4:
reward grounding by itself did not remove command/state confounding.

## D7 — train through reference drift, pending

Pipeline v5 changes one training-only condition. Replace the three hidden-reference
termination criteria (root distance, root rotation, and joint pose error) with the
standard physical rodent fall criterion: torso height above 0.03 m, torso up-axis
cosine above 0.5 (60 degrees), and finite simulator state. Keep clip truncation,
all imitation and command rewards, split, architecture, PPO settings, and budget
unchanged. The fixed natural evaluator deliberately retains the original reference
termination; direct deployment already uses the same physical fall criterion. This
tests the concrete v4 failure mode—PPO was cut off for reference drift before it
could learn recoverable command-directed locomotion—without weakening the held-out
gate.

The first attempted v5 run, `report-seed0-20260719-211602`, was stopped at the user's
request during the initial checkpoint-0 evaluation compile. It contains only
`config.json`: no checkpoint, progress row, or PPO update exists. This is an
interrupted pre-run, not an experiment result. Pipeline v5 remains pending and must
start afresh if pursued.
