# Demo H experiment decisions

This log is append-only once full model training begins.

## 2026-07-20 — physics pseudo-label probe

- Keep Brax v1 `0.12.3`, 50 Hz control, four substeps, and the unmodified Fetch
  actuator/config used by Demo A and Demo G.
- Reject naive finite-difference torque labels because Demo F poses are
  kinematic and floating-base contact forces are underdetermined.
- Use a transparent bounded joint-reference feedback controller as the first
  exact-physics projection, and store only its simulator-realized trajectory.
- Correct the legacy sign convention empirically: positive normalized Fetch
  control increases the reported joint angle even though actuator-axis torque
  is `-300 * control`.
- Select `kp=400`, `kd=10` on one fixed 128-clip validation audit. It yielded
  99.2% stable clips, median joint RMSE 0.044 rad, median minimum upright 0.992,
  and 0.09% saturated controls. The realized forward-progress ratio was 0.66,
  so all Demo H commands must be recomputed from realized physics rather than
  copied from the kinematic reference.
- A full train/validation scale audit exposed a 0.1%-level tail of launched or
  nearly horizontal bodies that the coarse fall gate missed. Before using the
  final-test split for any model decision, strengthen the physical gates to
  torso height `>=0.90`, upright `>=0.50`, saturation fraction `<=5%`, joint
  RMSE `<=0.20 rad`, maximum instantaneous planar speed `<=4.0`, maximum yaw
  rate `<=3.0 rad/s`, and 0.62-second command speed `<=2.5`. These broad limits
  were set from train/validation diagnostics and do not select for the 0.925
  task speed.

## 2026-07-20 — restore the direct-scale speed regime

- Reject the Froude-retimed Demo F derivative as Demo H's parent. Its training
  median hindsight speed is `0.584` Fetch units/s and only `4.3%` of clips lie
  in `1.5–4.0`; it made a low crawl look successful under the RL reward.
- Restore Demo F's original direct-spatial release. Its training median
  hindsight speed is `2.530`, median instantaneous path speed is `2.855`, and
  `60.5%` of hindsight commands lie in `1.5–4.0`.
- On all 335 validation clips, the unchanged `kp=400`, `kd=10` exact-physics
  projection keeps `94.3%` physically stable. Realized command speed has
  median `1.491` and 90th percentile `3.751` Fetch units/s.
- Version this as `exact-fetch-feedback-projection-direct-v2`; do not silently
  reinterpret the old generated release. Broaden only the validation-audited
  rejection tails for the faster source: saturation `<=15%`, joint RMSE
  `<=0.30 rad`, instantaneous planar speed `<=12`, yaw rate `<=6 rad/s`, and
  0.62-second command speed `<=8`. Height `>=0.90` and upright `>=0.50` remain
  unchanged.
- Preserve the exact saved boundary controls, but fit the tanh-Gaussian action
  model and feed its recurrence inside `[-0.98, 0.98]`. Exact `+/-1` has no
  finite pre-tanh Gaussian mean, and allowing it into the recurrence lets the
  reference chase saturation. Select action checkpoints with four times more
  weight on causal multi-step rollout than on one-step persistence.

## 2026-07-21 — reject three-limb locomotion

- Reject the first 30M-transition variable-speed checkpoint despite good
  scalar speed tracking. Its back-right foot switches contact only `1–3` times
  in five seconds from 1.5 through 3.5 m/s, remaining unchanged for as long as
  4.9 seconds. Aggregating contact statistics across feet hid this defect.
- Freeze a four-limb audit that reports per-foot duty factor, switch rate,
  contact entropy, longest constant-contact interval, active-window coverage,
  and inter-foot switch-rate variation. All inspection speeds must pass.
- Add a gait-agnostic online participation bonus (`weight=0.25`) from one-second
  exponential contact/switch histories. The minimum across feet is used, so
  three active limbs cannot hide one unused limb. Add a mild normalized-action
  rate cost (`weight=0.01`) to make rapid contact tapping expensive. This does
  not prescribe a walk, trot, or gallop phase pattern.

## 2026-07-21 — keep naturalness gates out of the objective

- Reject the participation-shaped checkpoint even though all four feet pass
  the contact gate. Optimizing a hand-written gait validator would undermine
  the experiment's claim that naturalness comes from generative pretraining.
- Remove both the contact-participation bonus and its accompanying action-rate
  penalty from the environment. Keep per-foot contact, stride-band, and
  high-frequency diagnostics strictly as held-out model-selection gates.
- Train task reward plus the frozen action-prior KL only. This leaves speed and
  uprightness as functional task terms inherited from Demo A, while Demo H's
  data prior is the sole source of additional locomotor naturalness.

## 2026-07-21 — replace direct timing with the selected 1.75x release

- A distribution audit showed that direct spatial retargeting enlarges a
  0.09355 m rodent trunk to Fetch length 2.0 while retaining the source clock.
  The resulting local limb motion is visibly accelerated. The theoretical
  Froude factor `sqrt(21.3789)=4.6237` is dynamically principled but looked too
  slow for this workshop body and dataset.
- Render exact 50 Hz interpolations at factors 1.1, 1.25, 1.4, 1.6, 1.8, and
  2.0 on clips from three speed quantiles. Select `1.75x` after visual
  inspection. Record it explicitly as an empirical temporal dilation, not as
  Froude similarity.
- Use one centered crop per parent clip so moderate dilation does not create
  overlapping examples. Version the parent as `temporal-dilation-1p75-v1`.
  It contains 1,804/278/344 train/validation/test clips.
- Re-run the unchanged `kp=400`, `kd=10` controller in exact Fetch physics and
  version the result as
  `exact-fetch-feedback-projection-retime-1p75-v1`. It accepts
  1,784/278/342 clips (99.09%), builds in 84.4 seconds, has median-across-shard
  joint RMSE 0.103 rad, mean-across-shard saturation 1.36%, minimum torso
  height 1.133, and minimum upright 0.514.
- Independent paired-control replay agrees with the stored physical state to
  approximately `1e-5`; shuffled controls are materially worse. This release
  supersedes the direct-timing candidate for Demo H without changing canonical
  Demo F's separately versioned Froude release.

## 2026-07-21 — accept the 1.75x frozen state/action prior

- Train tokenizer (1,000 updates), state predictor (1,500), and feedback action
  decoder (2,000) from scratch in 70.8 seconds. Do not initialize this timing
  variant from canonical Demo F.
- On 342 held-out test clips, next-state prediction improves 49.8% over
  persistence and the matching command beats a shuffled command in 82.4% of
  windows. Shuffling the predicted motion plan raises action MSE from 0.0102 to
  0.0187.
- One-step action MSE is 6.8% worse than copying the previous 50 Hz control, a
  strong smooth-control baseline. Retain this result. The causal 20-step
  rollout is the relevant gate and improves 86.9% over repeating the initial
  control.
- In exact physics, the frozen prior alone survives five seconds from both an
  in-support state and the ordinary standing reset. From standing at a 1.5
  command it travels 5.16 units, keeps minimum upright 0.982, switches every
  foot contact, and has zero saturated actions.
- Accepted artifact hashes:
  - parent manifest:
    `85fe54ee9730fe3c79871c6739197496e92b726f5072d93c4322bd001df82b3f`;
  - physical manifest:
    `c02c0cc43775dc28ee33106b4841f7dc7a06696c20e956e7d21aeb36dfd76847`;
  - PyTorch prior:
    `181394fe81eba60aeb67a38d2cac229f2c26e7ea844d8701b68d648ff3d4f903`;
  - JAX prior:
    `fc4f5797844c2b2426d7c5f92ed093cb5d8d6ead8113ee2b1dc46cf649203382`.

## 2026-07-21 — accept H2 with beta=0.10

- Freeze 30M transitions, 2,048 environments, three PPO evaluations, seed 0,
  and a uniform 1.5–4.0 Fetch-units/s task distribution. Keep the objective to
  task reward minus mean per-action-dimension reference KL. Naturalness metrics
  remain validation-only.
- Compare `beta=0.075` and `beta=0.10` with the same prior, seed, task range,
  training budget, six inspection commands, batched rollout path, and renderer.
  Both runs take approximately 95.2 seconds.
- Accept `beta=0.10` after the user's direct video inspection. Its mean absolute
  speed error over 1.5/2.0/2.5/3.0/3.5/4.0 is 0.079 units/s and it survives all
  six five-second rollouts. The accepted checkpoint SHA-256 is
  `e876bf800b17b48d602f28f067033fd4bb48246cd7e8fd7420cfe7cb5357cb44`.
- Preserve the counter-evidence. The strict four-limb stride validator passes
  only 4/6 commands for `beta=0.10` (failures at 2.5 and 4.0), compared with 5/6
  for `beta=0.075`; mean joint-speed RMS is also higher (4.998 versus 3.165).
  The 4.0 command reaches only 3.647 and underuses one foot. These diagnostics
  do not enter the reward and remain visible in the workshop video.
- This is a pedagogical, single-training-seed acceptance. Do not claim a
  multiseed algorithm advantage, biological torque recovery, or neural
  similarity.

## 2026-07-21 — accelerate the inspection loop

- Evaluate all requested speeds in one vectorized rollout compilation instead
  of compiling one environment per speed. Six-speed evaluation falls from
  roughly 45 seconds to about 20 seconds including isolated-environment
  startup. Batched floating-point physics is not bit-identical over long chaotic
  rollouts, so compare candidates through the same path.
- Reuse each panel's static TinyRenderer scene and default to every second
  physics frame at 25 fps. This preserves five-second playback while reducing
  rendered frames from 251 to 126. Checked frames are pixel-identical to Brax's
  original per-frame renderer before temporal subsampling.
- Retain 30M PPO transitions for the 1.5–4.0 task: the 15M midpoint remained
  materially below the final return in both accepted-weight candidates.

## 2026-07-21 — bind feature, observation, and replay contracts explicitly

- Preserve the accepted version-1 frame-zero feature fill and Fetch-native
  online contact observation. The frozen prior and H2 checkpoint depend on
  those exact semantics; future changes require a new contract version.
- Qualify the earlier `1e-5` replay statement as same-H100/CUDA-backend
  agreement. CPU and GPU executions of legacy contact-rich PBD agree initially
  but diverge after contact, despite matching Brax/JAX versions.
- Record the JAX backend, device kind, jaxlib version, and x64 mode in future
  derived releases. Reject mismatched replay backends instead of weakening the
  exact-replay threshold.
- Correct the P0 in-support selector to compare the same frame-15→46 command
  used by rollout, and initialize standing contacts from Fetch's actual reset
  observation. This supersedes the earlier 5.16-unit standing figure: the
  corrected five-second standing rollout travels 4.36 Fetch units with minimum
  uprightness 0.975 and no saturated actions; both reset gates still pass.
