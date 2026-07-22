# Demo J PPO iteration log

This append-only log records the bounded PPO follow-up performed on 2026-07-21.
It follows [`canvas/misc/autoresearch.md`](../canvas/misc/autoresearch.md): the
data, objective, validation subset, promotion gates, and test split stay fixed
while one controller choice is changed at a time.

## Frozen contract

- Train on the projected Demo J training references in modern MJX.
- Select on the first 128 validation clips using median reference-tracking
  return, with completion at least 95% and action saturation below 2%.
- Repeat validation three times because GPU contact solving has a measurable
  numerical noise floor.
- Keep each 58-step recurrent trajectory intact; minibatch only environments.
- Do not use gait, neural, RSA, or test metrics for checkpoint selection.
- Evaluate all 342 test clips once after promotion.

## Experiments

| Run | Change | Budget | Outcome | Decision |
|---|---|---:|---|---|
| Existing ANN probe | Ordinary feed-forward PPO, including one-clip probe | 10.2M transitions | Did not learn close tracking | Keep as environment/optimizer diagnostic; do not blame the SNN |
| `smoke` | First full-sequence recurrent SNN PPO | 3,712 transitions | End-to-end actor/critic update worked, but stale-minibatch approximate KL was about 0.18 | Add exact full-rollout pre/post KL |
| `warm-lr1e6` | Whole distilled actor, actor LR `1e-6` | 445,440 transitions | Preserved completion; last-minibatch KL remained misleading and validation gain was within noise | Replace diagnostic and add a trust guard |
| `warm-trust` | Exact recurrent KL plus backtracking on the whole distilled actor | 742,400 transitions | Every actor proposal was rejected: hard spike crossings remained discontinuous under small recurrent-weight changes | Freeze spiking dynamics; update the continuous readout |
| `smoke-readout` | Readout/log-std PPO only | 3,712 transitions | Exact KL `8.6e-5`–`2.8e-4`; updates accepted; completion retained | Promote to a full bounded run |
| `warm-readout` | Readout/log-std PPO only | 1,484,800 transitions, 79 s | Best validation update 80: 100% completion, return `270.86`, no saturation; test return `269.25` versus distilled `269.44` | Working PPO fine-tune, but no performance-improvement claim |
| `scratch-3m` | Random SNN, whole-network PPO, actor LR `1e-5` | 2,969,600 transitions | Completion `91.4% -> 95.3%`, return `192.18 -> 193.72`; joint/paw tracking worsened | Survival/local optimum; reject as close imitation |
| `scratch-lr1e4` | Random SNN, actor LR `1e-4` under the same KL guard | 2,227,200 transitions | Best return `195.35`, completion 96.9%; root drift improved while limb tracking stayed poor | Stop short-budget scratch iteration |

## Frozen conclusion

The implemented PPO machinery is sequence-correct and can train the controller
through closed-loop physics. At workshop-scale budget, the reliable path is
sequence distillation followed by an optional continuous-readout PPO fine-tune.
Strict scratch PPO learns a torso/survival solution but does not closely imitate
the reference motion. The local TRACK-MJX recurrent recipe uses a 6B-transition
budget, so these probes bound the workshop result rather than proving an
impossibility.

## Controller-generated prior positive control

This follow-up was requested after inspecting the original beta/RSA result, so
it is explicitly exploratory. The intervention changed only the Demo H prior's
training release: complete modern-MJX rollouts from distilled SNN seeds 0 and 2
replace the feedback-projection rows for train/validation. SNN seed 1 remains
test-only and supplies the fixed synthetic-neural recording. The PPO budget,
speed range, beta grid, fixed-input RSA design, and validation-only gait metrics
remain unchanged.

| Iteration | Budget | Outcome | Decision |
|---|---:|---|---|
| Export closed-loop SNN data | 4,466 candidate clip/controller pairs, 71.7 s | 4,406 complete finite clips retained (98.66%); train/validation exclude seed 1 | Accept as a leakage-safe positive-control release |
| Fit matched-data prior | 4,500 gradient steps, 69.4 s | Held-out seed-1 test: 44.6% state skill over persistence, 75.5% command win rate, 78.0% closed-loop action skill | Promote to PPO |
| Matched beta sweep | 6 × 30M transitions, seed 0 | All policies qualify; beta `.025` has best track (`.984`) and speed RMSE (`.215`); high beta degrades | Keep all points; do not select on neural metrics |
| Fixed-input hidden-2 RSA | 6 policies × held-out SNN seed 1 | beta `0` is highest (`.644`); alignment is not monotonic | Reject the simple “matched data makes higher beta more SNN-like” hypothesis |
| Layer/output sensitivity | Same saved arrays | hidden-1 behavior-partial RSA peaks mildly at `.025` (`.383` vs `.355` at zero); output RSA is nearly flat/high | Report as post-hoc architectural diagnosis, not a rescue claim |

The controller release is slower than the PPO task distribution (train median
`0.696`, 90th percentile `1.647` versus task commands `1.5–4.0`) and modern-MJX
data still deploy in legacy Brax. These are now the leading, testable mismatch
hypotheses; increasing beta further is not justified by this run.

## Aligned 1,000-bin follow-up

This later experiment is exploratory and uses a new frozen contract. It does
not retroactively change the accepted short-clip controller or the original
beta analysis.

### Frozen contract

- Use only the accepted `1.75x` projected train/validation/test splits.
- Fit the 4-frame, 16-D whitened-PCA tokenizer on training features only.
- Train on 1,000-bin autoregressive sequences; reset every neuronal variable
  only at the episode boundary and truncate gradients every 50 bins.
- Disclose the repeated 32-frame reference as synthetic periodic data.
- Compare preview horizons and seeds using validation action MSE only.
- Record every SNN seed on the same fixed 1,032-frame Demo H trace.
- Control RSA with each recording's exact raw SNN input and keep neural/gait
  metrics out of training and checkpoint selection.
- Treat input-weight exclusion as a post-hoc sensitivity, never a primary
  neuron-selection rule.

### Experiments

| Run | Change | Budget | Outcome | Decision |
|---|---|---:|---|---|
| `preview1` | 1 token / 80 ms | 64 x 256 x 1,000 bins | Validation MSE `.014159` | Keep as short-horizon ablation |
| `preview4` | 4 tokens / 320 ms | same | Best MSE `.013805` | Small advantage; no large-horizon claim |
| `preview8-seed0` | 8 tokens / 640 ms | same, 22.6 s | MSE `.013916`; stable quarter metrics | Primary aligned recording contract |
| `preview8-seed1/2` | Independent SNN seeds | same | MSE `.014041/.014213` | Keep all seeds for crossed analysis |
| `preview8-128` | Double optimization | 128 x 256 x 1,000 bins, 39.3 s | MSE `.012901` | Report convergence sensitivity; do not replace frozen recordings |
| `balanced-pretrain` | Equal six-speed sampling | 64 x 256 x 1,000 bins | Balanced validation MSE `.021797` | Reject; scarce high-speed clips are repeatedly over-weighted |
| `periodic-control` | Exact and online-PD periodic controls | 1,000 steps | Fail around 85–180 steps | Reject as sustainable physical teacher |
| `motion-graph` | Open-loop nearest transition graph | 1,000 steps | Fail around 74–199 steps | Reject; no continuous-data claim |
| `readout-ppo-100` | Frozen core, 100-step collection | 2.56M transitions, 128 s | Stable test rollout; one speed tracks well | Reject after horizon audit |
| `readout-ppo-1000` | Matched 1,000-step collection/evaluation and balanced speeds | 2.56M transitions, 179 s | All six survive; mean track `.241`, speed MAE `2.21` | Retain as honest negative functional probe |
| `full-input-rsa` | Exact 205-D per-seed input control | 3 SNN x 18 H checkpoints | beta 0 highest: RSA `.869`, partial `.680` | Reject monotonic-beta hypothesis |
| `exclude-input-q4` | Remove top input-weight-norm quartile | same | beta 0 remains highest: `.882/.707` | Result is not an input-proximity artifact |

### Frozen conclusion

Longer intention and exact temporal alignment do not rescue a monotonic beta
effect. They do establish a cleaner result: Demo H and the SNN share
representational geometry beyond the SNN's exact input and beyond a 200 ms
delay control, but stronger prior regularization does not systematically make
Demo H more SNN-like. The 1,000-step functional controller is stable yet not a
successful speed-conditioned imitation policy, so only the original 58-step
controller is presentation-ready.

### Visual-audit correction

The `readout-ppo-1000` environment termination flag stayed false in all six
episodes, but the final side-by-side video shows roughly half of the SNN
rollouts losing locomotion partway. The earlier words “survive” and “stable”
described only coarse nontermination and uninterrupted recurrent activity;
they must not be interpreted as functional locomotion success. Quantitatively,
the realized speeds are `[-0.09, 1.01, 1.72, -0.27, 0.30, 0.28]` for commands
`[1.5, 2.0, 2.5, 3.0, 3.5, 4.0]`, and several cases have zero or near-zero foot
contact switches. The controller is an explicit negative functional result.

## Native-clip correction

Visual inspection exposed a structural error in the aligned experiment: the
32-frame target repeated periodically, and the physical controller often failed
at the artificial boundary. The release contains independent 64-frame clips,
so there is no evidence that the last state of one period can transition to its
first state.

### Frozen replacement contract

- Treat one 64-frame source clip as one 63-action recurrent episode.
- Reset every SNN neuronal variable and the previous action at each clip
  boundary.
- Encode only complete four-frame future blocks that remain inside the clip.
- Zero unavailable tail tokens and include one validity bit per token.
- Train with full 63-step backpropagation through time and sample native clips
  uniformly.
- Evaluate physical imitation only over the native clip duration.
- Record SNN and Demo H activity on all 64 matched state frames for RSA. At the
  terminal state, record the SNN update but discard its unused action readout.
- Keep speed, contact, RSA, and naturalness metrics out of training and model
  selection.
- Do not restore long-horizon PPO without genuinely continuous references or a
  separately validated continuous trajectory generator.

The periodic checkpoint, its long-horizon PPO, and its beta/RSA trend are
rejected rather than treated as baselines for the replacement experiment.

### Replacement experiments

| Run | Budget | Outcome | Decision |
|---|---:|---|---|
| `native-seed0` | 2,000 x 256 x 63 bins, 40.5 s | validation MSE `.008826` | Keep for crossed-seed analysis |
| `native-seed1` | same, 40.5 s | best validation MSE `.008775` | Select for physical test/video |
| `native-seed2` | same, 40.2 s | validation MSE `.008796` | Keep for crossed-seed analysis |
| `native-test` | all 342 held-out clips | 100% finite-episode completion; median joint RMSE `.0722 rad`, speed MAE `.1047` | Accept within the 1.26 s clip boundary |
| `native-rsa-64` | 3 SNN x 18 H checkpoints, 10 conditions | beta-zero mean RSA `.806`, exact-input partial `.325`; both decrease overall with beta | Reject the higher-beta similarity hypothesis |
| `native-exclude-input-q4-64` | retain 192/256 neurons | beta-zero `.847/.381` and remains highest | Not an input-proximity artifact |

The native RSA uses all 64 state frames of 30 healthy fixed Demo H trials,
resets the SNN for each trial, and controls all 209 raw SNN inputs. There are
still only 63 physical transitions; the 64th SNN action readout is explicitly
unused. Its ten estimable speed-by-contact conditions and three seeds per model
support a descriptive ordering, not a precise dose-response or mechanistic
claim.

Adding the terminal state changes crossed-seed beta means by at most `.0083`
for raw RSA and `.0322` for exact-input-partial RSA relative to the provisional
63-bin calculation; the qualitative conclusion is unchanged.

## Continuous-prior beta pilot

This exploratory follow-up changes only the Demo H frozen prior and matched PPO
checkpoints. It uses the sparse continuous-data candidate with JAX SHA-256
`a81a116fe37011c742b370e9c8c79b27c8108b60bbe38e6b32e86ed1321e74f2`.
The fixed 30-trial, 64-state-frame protocol and the three accepted native SNN
recordings are otherwise unchanged.

| Beta | PPO return | RSA | Exact-input partial RSA | 200 ms delay |
|---:|---:|---:|---:|---:|
| 0 | 1088.24 | **.900** | **.684** | .665 |
| .025 | 1021.68 | .783 | .441 | .638 |
| .05 | 1042.75 | .851 | .571 | .610 |
| .075 | 1062.81 | .849 | .540 | .628 |
| .10 | 1052.56 | .848 | .552 | .599 |
| .15 | 604.25 | .808 | .405 | .475 |

This pilot crosses one Demo H training seed with all three SNN seeds; it is not
a multiseed PPO estimate. Beta zero remains highest, so the better offline and
long-rollout scores of the candidate prior do not rescue a monotonic
beta-to-SNN-alignment claim. The synchronized six-beta by six-speed video and
its rollout metrics live under
`demo_h/out/prior_iteration/beta_v2_speed_sweeps/`.
