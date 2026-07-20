# Demo G decisions

Append-only record for the same-body SSL-guided PPO comparison. Do not tune on
final report seeds.

## Frozen comparison contract

- Body/task: unmodified Brax v1 Fetch in Demo A `FetchRun`, dynamically aligned
  target 0.924747 units/s and `sigma=0.308249`.
- Prior command: Demo F source-equivalent 0.20 m/s, or Fetch displacement
  `[0.573343, 0, 0]` over 0.62 s.
- Arms: G0 `beta=0`; G1 frozen correctly conditioned prior.
- Accepted budget: 30M transitions, 2,048 environments, three evaluations.
- Runtime gate: complete training including compilation in less than five
  minutes on the current workshop machine.
- Functional gate: G1 retains at least 95% of G0 tracking and survival.
- Distributional gate: G1 improves held-out raw likelihood and at least one
  direct gait/contact statistic over multiple seeds.

## P0 — export the accepted Demo F v2 prior

- Source checkpoint SHA-256:
  `8e8634d5f6b362a496799b1cb5259b05866ff489aa1ed42ad1c28d0e8eb26d12`.
- JAX archive SHA-256:
  `7a8b491c225243980ee05d07691365f569054482bc35e1103f91c816238a43ff`.
- Archive size: 7,847,531 bytes.
- Real validation-window parity: normalized token max error `8.1e-5`, next-token
  prediction max error `4.2e-4`, log-likelihood error `1.4e-4`.
- Verdict: accept the framework bridge.

## P1 — align physical and offline features

- Pure-JAX transition features match Demo F NumPy features to `6e-7` on a
  synthetic translating, turning, articulated trajectory.
- Live Fetch revealed four normalized channels in the thousands: retargeted
  orientation is yaw-only, making rotation-6D indices 7/8 and angular-velocity
  indices 9/10 numerical constants, while physical Fetch rolls and pitches.
- Project exactly these unsupported normalized channels to zero before the
  frozen encoder. No learned weight changes.
- With the projection, fixed seed-0 score means are -13.1 for Demo A’s stable
  run and -18.1 for its reach-task scramble; retargeted validation median is
  -0.88. Without it, stable-run scores are roughly -25,000 and meaningless.
- Verdict: accept as an explicit planar-representation boundary; never describe
  the prior as full 3-D motion likelihood.

## P2 — freeze reward calibration before PPO

- Reject validation-quantile clipping: physical policies both fall below the
  floor and receive a constant reward.
- Use `sigmoid((raw_logp + 15) / 5)` and `beta=0.1`. Rounded constants avoid
  false precision and preserve separation across retargeted, stable-run, and
  scramble trajectories.
- Continue to report raw likelihood separately.
- Verdict: freeze for the first paired experiment.

## P3 — beta-zero and throughput preflight

- Within G0, returned reward is exactly Demo A task reward.
- Two separately compiled identical-action graphs remain within `4.1e-6` body
  position, `4.6e-5` observation, and `6e-8` reward over 40 GPU steps.
- The first 512-environment, 2M-transition smoke reached initial evaluation at
  23 s, then was interrupted during compilation at the user’s request. It
  produced no checkpoint and does not establish throughput.
- Verdict: beta-zero arithmetic passes the current preflight; vectorized reset
  isolation and a completed under-five-minute throughput measurement remain
  mandatory.

## P4 — batched scorer and frozen iteration budget

- Move frozen-prior inference outside the per-environment vmap. Collect the
  60-D history inside each v1 environment, then score the full batch under one
  synchronized scalar conditional every four frames. This preserves the 12.5 Hz
  token likelihood while avoiding a replicated Transformer graph.
- G0 excludes the frozen network from its compiled graph; the held-out evaluator
  scores both arms afterward with the same prior and shaping disabled.
- Permanent CPU tests pass for offline/online features, PyTorch/JAX export,
  beta-zero task parity, and vectorized history clearing on auto-reset.
- Completed G1 throughput probes:
  - 2.01M transitions, 512 envs: 40.0 s;
  - 8.03M transitions, 2,048 envs: 45.2 s, with initial evaluation at 19 s.
- Reject the original 100M placeholder: even the favorable extrapolation is too
  close to or beyond five minutes and is unnecessary for a quick workshop loop.
- Freeze 30M transitions, 2,048 envs, and three evaluations as the iteration
  budget. Expected in-process wall time is approximately two minutes including
  compilation/evaluation. Reserve 50M only for a post-hoc convergence probe if
  the 30M curve is still rising steeply.
- Freeze CPU held-out evaluation because the GPU generalized-physics reduction
  gives tiny nonzero differences even in an identity-checkpoint control. The
  paired CPU graph returns exactly zero for that control.
- Verdict: accept the batched path and 30M iteration budget; begin matched G0/G1.

## P5 — beta block and contact-chatter failure

- At `beta=0.1`, training seeds 0 and 1 passed every held-out gate. Seed 2
  improved raw likelihood by `+9.63` on 5/5 rollout seeds and preserved tracking,
  but failed the direct composite: contact switching rose to `10.92 Hz` versus
  the held-out retargeted reference `3.52 Hz`.
- Do not waive or redefine the frozen metric. Treat this as learned-score
  exploitation localized to contact chatter; the other three direct statistics
  (duty factor, stance-foot speed, cyclicity) moved toward the reference.
- Seed-2 bounded beta sweep:
  - `0.2`: tracking `100.0%`, direct improvement `-0.05`, raw `+11.20`;
  - `0.3`: tracking `86.6%` (functional gate failure), direct `+1.60`;
  - `0.25`: tracking `99.0%`, direct `+1.43`, raw `+10.46`; all distributional
    comparisons win on 5/5 rollout seeds.
- Freeze `beta=0.25`. Seed 2 is tuning/validation and cannot confirm the selected
  value. Confirm on training seeds 0, 1, and a fresh seed 3; unchanged G0 seeds 0
  and 1 may be reused because beta cannot affect them.
- Verdict: accept `beta=0.25` provisionally, pending three post-selection pairs.

## P6 — reject beta-only fix; test temporal aliasing

- Post-selection seed-0 confirmation at `beta=0.25` fails the direct gate:
  likelihood improves by `+11.94` and function improves, but contact switching
  rises to `6.87 Hz` and direct distance worsens by `0.52` on 0/5 rollouts.
- Reject `beta=0.25` as a robust solution and restore `beta=0.1`. Do not run the
  remaining confirmation seeds for a rejected setting.
- Hypothesis: scoring only one token phase every four frames permits contact
  changes between reward updates. Add a tunable scorer cadence without changing
  the frozen model, features, reward transform, or held-out metric.
- Next experiment: known failure training seed 2, `beta=0.1`, score every frame.

## P7 — reject every-frame scoring; accept a limited workshop claim

- Scoring every physical frame for seed 2 increased G1 runtime from `68.6 s`
  to `84.4 s`, but did not fix the direct composite (`-0.28`). Reject this
  cadence change and restore the frozen four-frame stride.
- The accepted comparison is therefore the original `beta=0.1`, stride-4,
  30M-transition pair for training seeds 0, 1, and 2. G0 takes `58.1–59.3 s`;
  G1 takes `67.8–68.6 s`, including compilation and three evaluations. Every
  arm is below two minutes, and one sequential matched pair is about 127 s.
- Shaping-disabled CPU evaluation uses five paired rollout seeds per training
  seed. Raw held-out log likelihood improves by `+8.46 ± 0.89` across training
  seeds, wins all 15 paired rollouts, and exceeds two between-seed standard
  deviations. Tracking and survival retain at least 95% in every seed.
- Distance to held-out motion improves in all three training seeds for duty
  factor (`+0.060`), stance-foot speed (`+0.257`), and cyclicity (`+0.111`).
  The aggregate four-measure composite improves in only 2/3 seeds because
  contact-switch frequency is not robust (`+0.119 ± 2.213`, only 1/3 seeds).
- The representative seed-0 rendering and time trace agree with the numerical
  result: G1 remains upright with smoother near-target speed and much higher
  raw likelihood; the video is illustrative, not an additional test set.
- Verdict: accept Demo G for the workshop with the limited claim that the
  frozen data prior improves its learned distributional score while preserving
  function, with three direct kinematic/contact-adjacent measures improving
  across seeds. Explicitly reject the broader claim that every aspect of gait
  realism improves; contact timing remains a visible limitation.

## P8 — reopen the physical scaling contract

- The kinematic retarget enlarged the rodent trunk from 0.09355 m to Fetch's
  2.0 units, a 21.379x length scale, while preserving a 50 Hz clock. Dynamic
  similarity instead requires a `sqrt(21.379)=4.624x` time dilation and maps
  0.20 m/s rodent locomotion to 0.9247 Fetch units/s, not 3.0 units/s.
- Build the time-dilated Demo F v2 release without joining clips and train an
  accepted prior on it. Set Demo G's target to 0.9247 and preserve Demo A's
  dimensionless reward width with `sigma=target/3=0.3082`.
- A 30M-transition G0 trains in 58.4 s, survives every held-out episode, and
  tracks 0.9247 at 0.0299-unit/s RMSE. It still exposes the physical failure:
  airborne fraction is 10.0% versus 0.47% in data, contact switching 11.48 Hz
  versus 0.77 Hz, vertical acceleration 1.78 g versus 0.053 g, and approximate
  stance-foot world slip 1.92 versus 0.36 units/s.
- Raw G0 prior scores have median -21.2 and 5th/95th percentiles -34.3/-12.0;
  retimed validation has median -0.17. Freeze the rounded shaping transform
  `sigmoid((logp + 20) / 5)` and `beta=0.1` before training G1.
- Verdict: target-speed alignment fixes the moon-like command mismatch, but
  task-only PPO still finds a high-frequency hopping gait. Train the matched G1
  and require improvements in flight, acceleration, slip, and contact metrics—not
  only learned likelihood.

## P9 — first dynamically aligned G0/G1 pair

- Matched seed-0 G1 completes 30M transitions in 68.8 s. Shaping-disabled CPU
  evaluation retains survival (100%) and tracking (100.15% of G0), while raw
  likelihood improves by 18.11 on all 5/5 paired rollout seeds.
- The nine-measure standardized distance to held-out motion falls from 9.70 to
  3.92 and improves on all 5/5 seeds. Key physical changes are: contact switching
  11.48→3.11 Hz (reference 0.77), vertical acceleration 1.78→0.91 g (0.053),
  approximate stance-world foot slip 1.92→0.75 (0.36), duty factor 0.36→0.49
  (0.69), and airborne fraction 10.0%→5.4% (0.47%). Action energy also falls
  from 0.120 to 0.041.
- Retain two failures: maximum continuous flight grows from 0.032 to 0.060 s
  (reference 0.0056 s), and the current foot-height cyclicity statistic moves
  from 0.877 to 0.759 (reference 0.913). The learned prior improves the gait
  substantially but does not reproduce the retargeted distribution.
- Verdict: accept this as a successful single-training-seed dynamic-alignment
  result and render it. A workshop-level robustness claim still requires
  matched training seeds 1 and 2; do not imply that one seed establishes it.

## P10 — dynamic three-seed confirmation and stopping point

- Complete the predeclared seeds without changing beta, score cadence, target,
  PPO budget, or held-out metrics. G0 takes 57.8–59.8 s and G1 takes 68.0–69.5 s
  inside `ppo.train`; all six arms remain below two minutes.
- Raw held-out log likelihood improves by `+18.11/+32.76/+17.28` for training
  seeds 0/1/2, wins every one of 15 paired rollouts, and averages
  `+22.72 ± 7.11`. Tracking retention is 100.15%/100.58%/100.15%; survival is
  100% in every arm.
- The nine-measure direct gait-distance improvement is
  `+5.78/+3.66/-0.30`, so its every-seed gate fails. Four individual distances
  improve in all three seeds: airborne fraction, stance-foot speed,
  approximate stance-world foot speed, and joint-speed RMS. Cyclicity improves
  in no seed.
- Select seed 0 for the workshop because it passes every single-seed gate and
  has the largest direct-composite gain. Do not select seed 1 merely for its
  larger learned-likelihood gain. Retain seed 2 as the robustness counterexample.
- Seed-0 videos and traces show a lower-energy, less chattering, less airborne
  but crouched gait. Maximum continuous flight and cyclicity worsen, and
  acceleration remains far from the retargeted reference.
- Verdict: this is the next best-performing accepted version and the declared
  stopping point. Freeze the checkpoints and document the limited claim; do not
  tune further on these report seeds.
