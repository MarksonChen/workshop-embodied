# Archived direct-actuator Demo E experiments

> Historical record only. These entries describe the rejected pipeline-v1/v2
> position-actuator and Freddie experiments. They are not the configuration,
> commands, or acceptance criteria of aligned pipeline v6; the canonical
> implementation and current pipeline-v6 contract are in `demo_e/README.md`
> and `demo_e/config.py`.

## 2026-07-19 — implementation baseline

- Keep the 143-D simulator-native Demo B feature bridge.  It removes the marker
  reconstruction domain shift while retaining the causal VAE + conditional
  Transformer lesson.
- Keep a fixed scalar Gaussian variance after mean-model training.  This makes
  the MSE-to-log-likelihood equivalence literal and testable.
- Start with native filtered position actuators and one 80 ms policy action held
  over eight 10 ms controls.  Raw torque is an optional stress test, not the
  workshop default.
- Measure environment throughput before any report PPO run.  Architecture or
  environment-count changes are accepted only with parity/quality gates intact.

## 2026-07-19 — throughput iteration 0: reject 512 environments

- Baseline (5 Newton iterations, reward sampled at all eight 10 ms controls):
  1,135 E0 and 1,144 E1 macro-transitions/s at 512 environments; the prior's
  overhead was below timing noise.  At 2,048 environments E1 reached 2,153/s.
- Reject 512: MJX is under-filled.  The projected environment-only time for the
  frozen 1,048,576-transition budget is about 15.4 minutes at 512 and 8.1
  minutes at 2,048, before PPO updates.
- Next iteration keeps all four required 50 Hz feature samples but evaluates
  the task reward four times (20 ms grid), and tests two Newton/line-search
  iterations.  This changes neither the 80 ms action hold nor the prior target.

## 2026-07-19 — throughput iteration 1: keep 20 ms reward sampling

- At 2,048 environments, E1 improved from 2,153 to 2,869
  macro-transitions/s (+33%).  The projected environment-only report budget is
  366 s, still above the workshop target before PPO updates.
- Keep the four-sample task-reward implementation: it matches the exact 50 Hz
  states consumed by the prior and removes redundant reward-tree work.
- Probe a 4 ms simulation step next (five physics steps per 20 ms feature), with
  finite-state and learning gates required before acceptance.

## 2026-07-19 — throughput iteration 2: reject 4 ms physics

- 4 ms reached 5,676 macro-transitions/s at 2,048 environments, but the mean
  zero-action reward exploded to -3.5e6 rather than the baseline's order-10
  fall penalty.  This fails the finite/stability gate and is rejected despite
  its speed.
- Probe 2.5 ms (eight physics steps per 20 ms) as the final integrator change.

## 2026-07-19 — throughput iteration 3: provisionally keep 2.5 ms

- At 2,048 environments, 2.5 ms reached 3,515 macro-transitions/s (+22.5%
  over the stable 2 ms variant) with a finite mean reward (-34.0 versus -34.0).
- Keep provisionally, subject to PPO smoke learning and rollout finiteness.  Test
  4,096 environments once to choose the final vectorization setting.

## 2026-07-19 — vectorization choice: keep 4,096

- 4,096 environments reached 4,285 macro-transitions/s versus 3,515/s at
  2,048 (+22%).  The mean zero-action diagnostic remained finite (-34.1).
- Keep 4,096 for report probes.  The frozen 1,048,576-transition budget now
  projects to 245 s of environment stepping alone; PPO/evaluation may still put
  the complete run over five minutes, so the time target remains a measured
  stretch goal rather than a promised result.

## 2026-07-19 — PPO stability audit: reject 2.5 ms

- The first actual PPO checkpoint-0 evaluation at 2.5 ms produced NaNs in
  velocity, energy, acceleration, torque, and task reward; 4.7% of evaluation
  episodes terminated explicitly for NaNs after an average 1.34 macrosteps.
- Stop the run before its first optimizer update and reject 2.5 ms.  The
  ten-step zero-action throughput diagnostic was too weak to expose aggressive
  random-policy instability.
- Restore 2 ms.  The earlier 2 ms/2-iteration benchmark was finite; an actual
  PPO smoke now decides whether two solver iterations are safe.

## 2026-07-19 — random-policy audit: reject fast-solver/full-range controls

- At 2 ms and two solver iterations, there were no explicit NaN terminations,
  but checkpoint-0 reward diagnostics still became non-finite after aggressive
  random position targets and episodes averaged only 1.28 macrosteps.
- Restore upstream's five Newton/line-search iterations and multiply policy
  output by 0.35 before the native position actuators.  The transform is fixed,
  visible, and identical for E0/E1; it is an actuator-range engineering bias,
  not a learned decoder or motion prior.

## 2026-07-19 — keep stable actuator candidate

- A 32-environment, 125-step random-action audit with 2 ms physics, five solver
  iterations, and 0.35 action scaling kept every reward and final metric finite.
  The unscaled/fast-solver candidate failed the same qualitative audit inside
  PPO.
- Keep this candidate for the learning smoke.  Falls are expected before
  learning; non-finite signals are not.

## 2026-07-19 — E0 learning smoke and evaluation batching

- E0 at 131,072 transitions improved evaluation return -51.9 -> +22.5 and
  average episode length 17.3 -> 59.6 macrosteps.  The task is learnable and all
  final diagnostics were finite.
- Training itself took 259 s at 506 transitions/s with only 512 training envs.
  Each 32-environment evaluation took ~222 s because MJX was severely
  under-filled, dominating the 921 s cold run.
- Keep the stable physics/action design.  Use 512 evaluation environments and
  only initial/final report evaluations; this is more statistically precise and
  expected to be faster through vectorization.

## 2026-07-19 — solver/action interaction probe

- The five-iteration stable E1 environment reaches 2,532 transitions/s at
  4,096 environments (414 s of physics for the one-million budget).
- The earlier two-iteration failure was confounded with full-range position
  targets.  Re-test two iterations while retaining the accepted 0.35 action
  scale; accept only after the same full-horizon random-action finiteness audit.

## 2026-07-19 — keep two iterations with bounded targets

- The full 32-environment, 125-step random-action audit passed: all rewards and
  final metrics were finite (mean reward -27.9).  Thus aggressive target range,
  not two solver iterations alone, caused the rejected numerical behavior.
- Keep 2 ms, two Newton/line-search iterations, and action scale 0.35.  Re-run
  the 4,096-environment benchmark for the final stable throughput number.

- Final stable E1 environment throughput at 4,096 environments is 3,459
  macro-transitions/s (27.7k equivalent 10 ms controls/s).  The one-million
  transition physics floor is ~303 s before PPO, so the original sub-five-minute
  full-run goal is not achievable with this exact scorer/physics stack.

## 2026-07-20 — prior-plumbing audit: residual native targets

- Exact recorded replay scores normally (example -0.41 nats/dim), proving the
  143-D feature/encoder alignment.  In contrast, a zero policy with absolute
  position targets scored -9.3e6 on its first macrostep and saturated the
  -3.85 training-quantile clip.
- The issue is actuator semantics, not the learned prior: zero control requests
  the XML neutral pose rather than holding the real reset pose.  Parameterize
  each action as a bounded residual around the native affine servo's current
  zero-force target.  This uses no learned weights or future/reference pose and
  is identical in E0/E1.

## 2026-07-20 — remove zero-information passive coordinates

- Residual targets alone did not fix the score.  The realized latent was huge
  because 29 of 67 fitted joint coordinates have exactly zero variance in all
  real clips (normalizer 1e-4) but move passively in MJX.  They contain no SSL
  signal and impose an impossible physical constraint.
- Keep the 38 actuator-linked/non-constant joints for position and velocity.
  The exact feature is now 2 + 1 + 6 + 38 + 38 = 85 dimensions.  This also
  makes the teaching symmetry explicit: 38 controls, 38 modeled joints.

- Follow-up: reject residual target centering.  In a 512-environment PPO
  checkpoint-0 audit it reduced average episode length to 2.81 and caused 0.34%
  explicit NaN terminations.  Restore the previously clean absolute 0.35-scaled
  targets and five solver iterations; retain the 85-D feature correction, which
  addresses the likelihood issue directly.

## 2026-07-20 — freeze null-calibrated likelihood scale

- Across 2,393 alive random-policy transitions, raw logp quantiles were q05
  -380, median -26.9, q95 -3.67; real training q99 is -0.316.  The earlier
  real-only lower clip (-3.83) therefore erased nearly all novice-policy
  variation.
- Freeze the lower bound at -400 (rounded random-null q05), retain the real q99
  upper bound, and set beta=0.0015.  The maximum prior span is ~0.60 reward,
  under 25% of the 2.5-point velocity-tracking range.  Raw logp remains logged.

## 2026-07-20 — E1 smoke and report budget

- E1 at 131,072 transitions improved task return -35.9 -> +12.9 and episode
  length 25.2 -> 50.6 with zero NaN terminations.  Prior reward remained
  variable (not floor-saturated) and small relative to task return.
- Brax built-in evaluation costs ~255 s per call even at 512 environments.  It
  duplicates the stricter paired evaluator, so disable it inside report
  training and retain checkpoint/progress callbacks only.
- Freeze 524,288 report transitions (4x the successful smoke).  The original
  1,048,576 budget has a >5 minute physics floor; expand back to it only if the
  half-budget paired gates fail.

## 2026-07-20 — learner throughput iteration: reject 256x256 / four passes

- Stop E0 after its first 131,072-transition report block: learner throughput
  was only 702 transitions/s and KL was ~2,460, so the recipe was both too slow
  and too aggressive.
- Probe a 128x128 actor/critic, two PPO passes, and 1e-4 learning rate.  Across
  524k samples this still supplies twice the total optimizer exposure of the
  successful 131k/four-pass smoke, while reducing per-sample learner work and
  update shock.

- The smaller learner's post-compile KL was 0.014 and throughput 659/s at 512
  environments (+30%), so keep its stability settings provisionally.  Optimizer
  work is no longer the only bottleneck.
- Re-audit the exact two-iteration + absolute 0.35-target combination at 512
  environments.  Prior failures used either full-range or residual targets and
  do not answer whether this bounded absolute form is stable.

- Reject it decisively: over the 512x125 audit, final metrics became non-finite
  and the mean `nan_to_num` reward exploded to -1.37e12.  Freeze upstream's
  five solver iterations.  Do not trade numerical validity for the five-minute
  target.

## 2026-07-20 — reject 0.35 action range after the first paired report

- The 524,288-transition E0/E1 report remained a stand-and-fall policy:
  independent fixed-command evaluation measured survival 0.489/0.444 and
  functional score 0.00218/0.00183.  Videos confirmed that neither arm had
  learned locomotion.  Do not present these checkpoints as a positive result.
- The 0.35 multiplier was introduced to stabilize a solver/action combination,
  but the XML's native filtered actuators are explicitly normalized over
  [-1, 1].  With the already-restored five-iteration Newton solver, a strict
  32-environment x 125-step random-action audit at scale 1.0 kept every reward
  and final metric finite (mean reward -42.69).
- Reject 0.35 and restore the native scale 1.0.  Require a short E0 probe to
  demonstrate sustained movement before spending another paired report budget.

## 2026-07-20 — reject full-range PPO; isolate a simpler task reward

- The native-range 131,072-transition E0 probe failed every fixed-command
  post-warmup survival trial (aggregate survival and functional score both 0).
  Numerical stability did not imply learnable exploration.  Revert to the
  behaviorally safer 0.35 range; an intermediate range remains a later,
  one-axis probe.
- The safe-range E0 could survive complete 0.2 m/s episodes yet stayed near
  zero forward speed.  The upstream reward combines seven costs with a -100
  fall event, making risk avoidance dominate a short workshop budget.
- For the next isolated probe, keep all physics/action settings and simplify
  `r_task`: forward tracking 2.0, yaw tracking 1.5, alive 0.1, termination -10,
  and lateral drift -0.1.  Retain energy, torque, action-rate, acceleration, and
  stand-still metrics at zero reward weight.  This is also a clearer from-scratch
  RL lesson; accept it only on fixed-command behavior, not optimizer loss.

## 2026-07-20 — reject reward simplification; test reset-centered residuals

- The simplified-reward E0 probe reduced critic loss from thousands to about
  6.5, but independent behavior worsened: survival 0.222 and functional score
  0.00067.  Revert to the upstream reward with only the established
  `tracking_sigma=0.05` change.  This confirms optimizer scale was not the gait
  bottleneck.
- Absolute control is mismatched to a real mid-stride reset: zero policy output
  requests every actuator's XML midpoint, not the reset pose.  Test a fixed
  reset-centered residual target.  Compute the native affine servo target once
  from reset actuator lengths, initialize filtered actuator state at that
  target, and apply `clip(neutral + 0.20 * action, -1, 1)` thereafter.
- Unlike the rejected moving-center residual, this baseline never follows the
  current or future pose.  It is a fixed per-episode coordinate choice, hidden
  from the actor, identical in E0/E1, and contains no learned decoder.

## 2026-07-20 — zero fitted velocity at reset

- Fixed reset-centered zero action remained finite and improved on aggressive
  control, but still produced 1,854 termination flags across 4,000 audited
  vector steps.  A kinematic mid-stride `qvel` assumes the unobserved forces
  that created it; a newly initialized servo cannot reconstruct those forces.
- Test the standard RL convention of zero physical `qvel` at reset while
  retaining the real pose and causal Demo B history.  This changes no training
  target or future information and is identical in E0/E1.  Keep it only if the
  zero-action posture audit improves materially.

## 2026-07-20 — replace the motion prior with all-session Freddie locomotion

- The workshop's neuroscience/behavior target is Freddie, and a source audit
  found Freddie's adjacent locomotion-frame mean planar speed to be 0.0290 m/s,
  versus Coltrane's 0.0209 m/s (38% higher). Replace the curated/Coltrane asset
  with contiguous locomotion crops from all 25 Freddie sessions.
- Freeze a 17/4/4 session split, never a crop-level random split. Accept crops
  with at least 25% `Amble`/`Walk`/`WalkFast` labels and at least 0.08 m/s mean
  measured planar speed; cap each session at 2,048 deterministic crops.
- The accepted seed-0 build contains 16,889/4,646/2,955 train/validation/test
  crops and takes 59.5 s of model optimization. On held-out sessions it beats
  persistence by 39.6% and real futures beat shuffled futures by 0.668
  nats/dimension.
- Keep the likelihood only after a counterfactual speed audit. All five
  actual-speed-bin rows peak under their matching conditioned bin and the
  per-example relative curve peaks at zero mismatch. Individual-crop top-1
  accuracy is only 25.7% versus 20% chance, so describe this as
  population-level speed calibration.
- Freddie's positive training-command q99 is 0.230 m/s. Round the Demo E
  primary range inward to 0–0.20 m/s. Label the requested Demo B 0.25 m/s
  videos as slight command extrapolations; they realize 0.13/0.16 m/s.
- Reject every earlier E0/E1 comparison as current evidence: those runs already
  failed locomotion and use an incompatible prior. Rebuild the independent
  realism reference, recalibrate the random-policy likelihood scale, and repeat
  reset/controller gates before another paired PPO run.

## 2026-07-20 — reject Freddie and restore the 281-D Coltrane prior

- User inspection rejected every Freddie rollout: the rat twitched rather than
  locomoting. Controlled experiments held the architecture and strict gait rule
  fixed. The raw Freddie seed was rougher than Coltrane (paw jerk 1,050 versus
  567), and the Coltrane tokenizer amplified Freddie reconstruction jerk to
  2,244. A Freddie-matched 281-D tokenizer improved reconstruction but its slow
  rollouts remained severely twitchy (jerk 1,236–1,755).
- Restore the exact `rl_standalone` Coltrane tokenizer and transition weights as
  Demo B. Adding a fixed-Gaussian score does not change those weights or their
  rollout. Calibration on the standalone windows gives sigma 0.0523, 5/5
  matching speed bins, 91.9% per-window top-1 speed-bin accuracy versus 20%
  chance, and a likelihood peak at zero speed mismatch.
- Reject all pipeline-v2 Demo E artifacts. They use the obsolete 85-D Freddie
  scorer and already failed behaviorally (paired report functional scores
  0.00218/0.00183; survival 0.489/0.444). Bump Demo E to pipeline version 3 and
  fail closed until a 281-D Coltrane JAX scorer and physical reset banks exist.
- Do not start another E1 run yet. First make E0 learn locomotion under the same
  MIMIC skeleton and workshop budget. Only after that gate passes should the
  Coltrane likelihood scale be calibrated and the paired E0/E1 experiment run.
