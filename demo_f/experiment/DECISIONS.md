# Demo F predictor decisions

This log records the manual accept/reject loop for the conditional Fetch motion
prior. The retargeted dataset, session split, 60-D feature contract, tokenizer
architecture, fixed validation histories, speed grid, rollout duration, and
evaluation objective remain frozen throughout this block.

## Frozen evaluation contract

- Use three nearly straight 0.15 m/s histories from distinct validation sessions.
- Intervene with 0.10, 0.15, 0.20, and 0.25 m/s commands for four seconds.
- Minimize normalized forward-speed tracking error, low-speed dwell, and
  command-order reversals through `demo_f.evaluate` contract
  `demo-f-rollout-v1`.
- Reject non-finite output, worse-than-persistence prediction, command win rate
  below 60%, root height outside 1.0--1.8 Fetch units, or more than 1% joint
  saturation.
- Require real futures to beat shuffled futures, every actual-speed bin to
  prefer its matching conditioned-speed bin on average, and the local
  likelihood curve to peak at zero speed mismatch.
- Reserve test sessions for final confirmation after model selection.

## B0 — shipped eight-token predictor

- Checkpoint SHA-256:
  `6e51e9b5ab35460859a37a365982a602c96592cbe503df3dcbb54c668f71e13d`.
- Evaluation SHA-256:
  `efdfbdcc3ba6870518290025d79c62810685d083f9343de42bbc53e005e554ac`.
- Objective: 0.4374; tracking MAE: 0.0569 m/s; low-speed dwell: 11.7%;
  monotonic violation: 0.0000 m/s; gates: pass.
- Diagnosis: all three histories order the four commands, but the 0.10 m/s
  rollout approaches zero or negative forward velocity. Training exposes only
  one command window per clip, leaves commands unnormalized, predicts eight
  tokens while inference advances one, and selects by raw latent MSE.
- Verdict: freeze as the baseline; replace only the predictor/data-window block.

The same frozen contract on the final test split gives objective 0.4146,
tracking MAE 0.0532 m/s, and low-speed dwell 11.9%.

## N1 — align the supervised target with autoregressive inference

- Predict one next token from four history tokens instead of predicting eight
  tokens and retaining only the first one at inference.
- Extract every valid 31-frame hindsight command from each stored clip, yielding
  five anchors at token indices 4--8.
- Normalize the command from training data, just like the motion features and
  latent tokens.
- Retain latent validation-MSE checkpoint selection for this ablation.
- Across seeds 0, 1, and 2, objectives are 0.1798, 0.4728, and 0.2210.
- Diagnosis: two seeds improve clearly, but latent MSE selects a seed-1 model
  that overshoots 0.25 m/s. One-step prediction error is not a reliable proxy
  for the closed-loop quantity shown in the workshop.
- Verdict: keep the aligned target and additional windows; replace checkpoint
  selection only.

## N2 — select by the frozen rollout objective

- Every 100 predictor updates, run the unchanged three-history validation
  contract and retain the lowest-objective checkpoint that also beats latent
  persistence.
- Across seeds 0, 1, and 2, validation objectives are 0.1217, 0.1030, and
  0.0388 (mean 0.0878); every seed beats B0 by a wide margin.
- Select seed 2 at predictor step 1,100. Training takes 20.5 seconds, validation
  MSE is 0.4677 versus persistence at 1.1094, and command win rate is 65.4%.
- Canonical checkpoint SHA-256:
  `8e8634d5f6b362a496799b1cb5259b05866ff489aa1ed42ad1c28d0e8eb26d12`.
- Validation objective is 0.0388 with 0.0058 m/s tracking MAE, no command-order
  violation, zero low-speed dwell, and all gates passing.
- Final test objective is 0.1580 versus B0's 0.4146, with 0.0158 m/s tracking
  MAE, no command-order violation, and all gates passing. Two test histories
  track all commands closely; the third stalls at 0.10 m/s and under-tracks
  0.15 m/s. No hyperparameter was changed after inspecting this split.
- The frozen likelihood audit passes without further tuning. Validation and
  test each have 5/5 matching-bin likelihood maxima and a local peak at zero
  speed mismatch. Per-window top-1 speed-bin accuracy is 27.2% and 30.3%
  versus 20% chance; real-minus-shuffled-future mean log likelihood is 0.934
  and 0.924.
- A fixed training-history visualization realizes path-speed equivalents of
  0.106, 0.162, 0.204, and 0.274 m/s for the four requests, with zero low-speed
  dwell and no systematic slowdown at the old 32-frame boundaries.
- Verdict: accept N2 as the canonical Demo F prior and carry the explicitly
  documented low-speed cross-session limitation into Demo G testing.
