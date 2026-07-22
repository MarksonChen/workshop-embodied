# Demo J results

This is the compact result ledger after the 2026-07-22 cleanup. Generated
arrays and videos are gitignored; [AUTORESEARCH.md](AUTORESEARCH.md) retains the
iteration history and rejected variants.

## Accepted short-clip imitation

The accepted controller is sequence distillation from the independent bounded
feedback controls paired with the Demo F/H motion release. It is imitation
learning, but not scratch reference-tracking PPO and not a model of biological
learning.

The actor has 128 LIF and 128 adaptive-LIF units and emits one action every
20 ms from four hard-spike 5 ms substeps. Each of three 2,000-update seeds
trained in 31–33 seconds on the H100.

| Seed | Validation completion | Root error | Joint RMSE | Paw RMSE | Return | Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100.00% | .0559 | .0361 rad | .0464 | 264.04 | 29.74 Hz |
| 1 | 99.64% | .0499 | .0328 rad | .0405 | 268.72 | 29.39 Hz |
| 2 | 99.64% | .0555 | .0349 rad | .0453 | 266.03 | 29.39 Hz |

Seed 1 was selected using validation imitation metrics only. On all 342 test
clips it completes 340 (99.42%), with `.0321 rad` median joint RMSE, `.0396`
paw RMSE, return `269.44`, no silent neurons, and no saturated actions. The
canonical video is `out/snn_imitation_speed_sweep.mp4`.

Short-budget scratch PPO learned survival rather than close limb tracking. A
readout-only PPO warm start preserved the distilled policy but did not improve
test imitation. Those trainers and checkpoints were removed from the supported
surface; this negative result is retained only as provenance.

## Native-clip aligned SNN

The replacement aligned workflow uses the same `1.75x` data and 20 ms action
clock as Demo H. A train-only whitened PCA converts four 60-D feature frames
into a 16-D token. One recurrent episode follows one independent 64-frame clip
for 63 actions, then resets.

At each step, up to eight future tokens represent at most 640 ms of remaining
in-clip motion. A token is zeroed and its mask is false whenever its complete
four-frame block would cross the clip tail. Body state, previous action, future
motion, and recurrent state never wrap to the first frame.

Training and held-out native-clip results are pending regeneration after this
contract correction.

## Rejected periodic experiment

The earlier workflow repeated a screened 32-frame segment for 1,000 bins. That
made the target periodic without evidence that its endpoint could transition
back to its start. Visual inspection exposed impossible boundary targets and
four of six showcased physical rollouts failed partway.

Its future-preview sweep produced action MSEs `.014159`, `.013805`, and
`.013916` for 1, 4, and 8 tokens. Its exact-input RSA peaked at beta zero
(`.869` raw, `.680` behavior-partial), and its readout PPO realized speeds
`[-.09, 1.01, 1.72, -.27, .30, .28]` for requests `[1.5, 2, 2.5, 3, 3.5, 4]`.
Those values are retained only as an audit trail: the artificial periodic
contract invalidates them as evidence about intention horizon, long-horizon
imitation, or beta-dependent neural similarity.

The periodic preprocessing, long-horizon environment, and readout-PPO trainer
have been removed from the supported package. A long-horizon claim requires
genuinely continuous references or a separately validated trajectory generator.

## Biological bridge

The historical exact-source-clock DLS analysis joined six Coltrane sessions
without interpolating 20 ms spike counts. All six session increments were
negative; median incremental prediction was `-0.0595 bits/spike`. Therefore
the synthetic SNN benchmark is not biologically validated. Only the compact
final report `out/source_clock_dls_encoding_stable.json` is retained; the
superseded DLS command stack and intermediate arrays were pruned.

## Claim boundary

The defensible workshop result is narrow:

- a small differentiable SNN can closely imitate short physically realized
  Fetch motion sequences;
- the periodic beta/RSA result is invalidated and awaits a native finite-trial
  replacement;
- the current SNN does not add predictive power for real Coltrane DLS activity;
- the retired 1,000-step controller failed functional locomotion;
  uninterrupted neural activity is not behavioral success.

Synthetic spikes are a controlled reference, not biological ground truth, and
representational similarity does not establish shared mechanism or causality.
