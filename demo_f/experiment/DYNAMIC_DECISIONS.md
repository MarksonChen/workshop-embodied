# Demo F dynamic-scaling decisions

Append-only record for the Froude-aligned successor to the original kinematic
release. The original release and accepted prior remain immutable baselines.

## Frozen contract

- Parent data: accepted `dynamic-similarity-v1` derivation is built only from
  the session-split `release/`; no parent clips are joined.
- Geometry scale: `2.0 / 0.09355 = 21.3789`.
- Time dilation: `sqrt(21.3789) = 4.6237`; output remains 50 Hz.
- Reference task: source 0.20 m/s, Fetch 0.9247 units/s, 0.62-second command
  `[0.5733, 0, 0]`.
- Data gates: retain the parent IK, foot-height, and 5% joint-limit gates.
- Prior gates: beat persistence and shuffled futures, command win rate at least
  60%, 5/5 speed-bin diagonal, local likelihood peak at matched speed, finite
  continuous rollouts, root height 1.0–1.8, and at most 1% generated joint-limit
  saturation.
- PPO is blocked until the derived release and prior pass their own gates.

## D0 — derived release

- Four disjoint 64-frame target-time crops are drawn from each independently
  retimed 64-frame parent; session splits and source provenance are preserved.
- The first build exposed concentrated joint saturation in about 2% of crops.
  Reject those crops under the existing gate rather than relaxing it.
- Accepted counts are 8,420 train, 1,307 validation, and 1,617 test; the complete
  release passes checksum, dtype, geometry, and continuity validation.
- At the fixed command, 144 test windows average 0.899 Fetch units/s, duty factor
  0.557, zero fully-airborne frames, contact switching 0.857 Hz, and joint-speed
  RMS 1.244 rad/s.
- Verdict: accept the derived dataset contract and begin prior training.

## D1 — unchanged prior, seed 0

- Training time is about 20 seconds. Validation skill over persistence is
  30.7%; command-versus-reversed win rate is 75.4%; real futures beat shuffled
  by 7.11 log-likelihood units; all five speed bins select their diagonal.
- Four rollout commands realize 0.091/0.145/0.199/0.242 m/s equivalents with
  no low-speed dwell; the rollout objective is 0.0489.
- Two gates fail: the local likelihood peak is one 0.0198 m/s bin below the
  matched command, and only the 0.25 m/s generation reaches 1.5–1.8% joint-limit
  saturation. The reference 0.20 m/s trajectory has zero saturation.
- Verdict: retain as the baseline, not yet accepted. Next test initialization
  noise with the unchanged architecture and fixed training budget.

## D2 — unchanged prior, seed 1

- Validation skill improves slightly to 32.2%, the local likelihood peak moves
  exactly to the matched command, and the 5/5 diagonal remains intact.
- Rollout objective worsens from 0.0489 to 0.0883, tracking MAE rises to 0.0133
  m/s, and maximum high-speed joint saturation rises from 1.76% to 3.24%.
- Verdict: reject seed 1. The likelihood offset is seed-sensitive, but generated
  saturation is not fixed by initialization. Because time dilation lengthened
  the gait period, next test a 0.64-second history with all other blocks fixed.

## D3 — double causal history

- Eight history tokens reduce the number of honest training windows from 42,100
  to 8,420 and validation skill from 30.7% to 22.1%.
- Rollout objective worsens to 0.1130 and maximum joint saturation grows to
  11.3%; the local likelihood peak remains one bin slow.
- Verdict: reject. A longer raw history is data-inefficient in the current
  64-frame contract. Restore four tokens and test a two-token training target,
  while generation continues to execute only the first receding-horizon token.

## D4 — two-token training target

- Two-token prediction raises skill over persistence to 47.3% and keeps the 5/5
  speed diagonal, but rollout objective is 0.0607 and maximum saturation is
  2.61%; the local likelihood peak is still one bin slow.
- Tokenizer reconstruction itself exceeds a joint limit in 0.17–0.20% of
  validation values, and autoregressive rollout amplifies that error.
- Verdict: reject the longer target. Restore next-token prediction and test a
  32-D latent so the tokenizer need not compress smooth joint trajectories as
  aggressively. Keep width, training steps, and every evaluation gate fixed.

## D5 — 32-D tokenizer latent, seed 0

- Skill over persistence rises to 52.8%, command win rate is 77.1%, and speed
  likelihood remains 5/5 diagonal. All generated speeds now have zero joint-
  limit saturation.
- Rollout objective is 0.0668 versus the 16-D baseline's 0.0489. The only failed
  gate is the shallow local likelihood optimum, still one 0.0198 m/s bin slow.
- Verdict: provisionally accept the safer representation and confirm one fresh
  initialization. Earlier seed variation moved this likelihood offset to zero,
  so do not change the metric or command calibration.

## D6 — 32-D tokenizer latent, seed 1

- Validation passes every gate: objective 0.0508, 48.5% skill over persistence,
  80.6% command win rate, 5/5 likelihood diagonal, exact local peak, and zero
  joint saturation.
- The untouched test split preserves likelihood gates but one of three rollout
  histories drifts into 1.5–5.9% saturation and under-tracks faster commands;
  therefore the full test gate fails.
- Seed-0 32-D diagnostics have zero saturation and better test rollout objective
  but retain the shallow one-bin likelihood offset. Neither seed clears both
  criteria on validation and test.
- Verdict: do not select between complementary failures. Run the predeclared
  third initialization of the unchanged 32-D architecture; accept only if its
  validation gates pass before opening its test report.

## D7 — 32-D tokenizer latent, seed 2

- Likelihood gates pass, including the exact local peak, but validation rollout
  saturation reaches 10%; objective is 0.0911. Reject.
- Across three seeds the high-rate predictor exhibits visibly spiky late losses
  and complementary rollout failures. Do not select the luckiest seed.
- Next and final block-level adjustment: halve the shared learning rate to
  `1e-3` at the unchanged 1,000/2,000-step budget. This targets optimization
  stability without increasing model size or compute steps.

## D8 — 32-D latent at half learning rate

- The lower rate reaches the best validation rollout objective so far, 0.0454,
  while retaining 50.2% skill over persistence, 75.9% command win rate, and a
  5/5 speed-bin diagonal.
- It does not resolve the two outstanding failures: the local optimum remains
  one 0.0198 m/s bin slow and generated joint-limit saturation reaches 1.88%.
- Verdict: reject. Optimization noise is not the root cause. The decoder can
  produce out-of-range angles and the generator currently repairs them only by
  hard clipping. Add an explicit joint-limit loss to tokenizer reconstruction
  and predicted-token decoding; keep the evaluation and data gates unchanged.

## D9 — one-step joint-limit loss

- With weight 10, one-step decoded predictions have essentially zero safety
  loss, yet four-second rollouts still reach 3.07% saturation. Rollout objective
  is 0.0809 and the local likelihood peak is one bin slow.
- The failure is recursive: safe individual predictions drift after their own
  outputs become history. Decoding a predicted token without its causal history
  also understates its physical angle excursion.
- Verdict: reject. Train the same one-token predictor through a four-token
  autoregressive unroll (0.32 seconds), decode with its causal history, and
  apply the unchanged physical loss across that unroll.

## D8b — lower learning rate and contract audit

- Halving the learning rate does not stabilize the candidate. Validation
  objective is 0.0454 and skill over persistence is 50.2%, but the local
  likelihood peak remains one bin slow and generated saturation reaches 1.88%.
- Audit the upstream targets before adding model machinery: 11.1% of training,
  10.8% of validation, and 11.9% of test crops exceed the prior's frozen 1%
  saturation gate, because the dynamic release inherited a 5% inspection gate
  from its kinematic parent.
- Verdict: reject the lower-rate model. This is a problem-contract defect, not
  evidence for a larger network. Rebuild a v2 dynamic release with the same
  retiming and provenance but a 1% data gate, then restart prior selection.

## D10 — four-token closed-loop training on the v2 release

- Rebuild the derived release with the aligned 1% data gate. The final complete
  session-split counts are 7,483 train, 1,166 validation, and 1,425 test clips;
  the validator passes checksums, schema, continuity, kinematics, and limits.
- Keep the predictor output at one token, but unroll its own predictions for
  four tokens (0.32 seconds) during training. Decode the prediction together
  with its causal history and apply the joint-limit loss to this closed loop.
- Seed 1 passes every validation gate with 0.0038 m/s tracking MAE and zero
  saturation; test preserves speed and safety but has a shallow one-bin local
  likelihood offset.
- The predeclared confirmation seed 0 passes every validation and test gate:
  validation/test tracking MAE is 0.0080/0.0129 m/s, both splits have a 5/5
  likelihood diagonal and exact local optimum, and all evaluated rollouts have
  zero joint-limit saturation.
- Verdict: accept seed 0 as the dynamic Demo F prior. Its 2,000 predictor steps
  still train in well under one minute; promote it to `demo_f/out/prior.pt` and
  export the corresponding pure-JAX archive for Demo G.
