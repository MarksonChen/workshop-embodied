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

Three 2,000-update seeds use uniform native-clip minibatches and full 63-step
backpropagation through time. Each run takes about 40.4 seconds on the H100.

| Seed | Best validation action MSE | Time |
|---:|---:|---:|
| 0 | .008826 | 40.5 s |
| 1 | **.008775** | 40.5 s |
| 2 | .008796 | 40.2 s |

Seed 1 was selected only by validation action MSE. A single batched physical
audit then evaluated all 342 test clips. It completes every finite episode,
with median root error `.0757`, joint RMSE `.0722 rad`, foot RMSE `.0694`, and
forward-speed absolute error `.1047` over the 326 positive-speed clips. It has
no silent neurons or saturated actions and fires at `29.47 Hz` on average.

The six-example video is `out/aligned/snn_native_clip_speed_sweep.mp4`. For
reference speeds `[1.51, 2.01, 2.53, 2.86, 3.34, 3.79]`, its realized speeds
are `[1.04, 2.09, 2.35, 2.88, 3.62, 3.92]`. The recording ends after the
source-supported 1.26 seconds; it is not evidence for indefinite locomotion.

### Native finite-trial RSM/RSA

Three independently trained SNN seeds and the 18 frozen Demo H beta-sweep
checkpoints receive the same 30 fixed trials: six speeds, five repeats, and all
64 state frames per trial. The SNN resets at every trial. It updates once at
each state frame; the terminal readout is recorded for activity alignment but
is never applied to physics. Its exact 209-D input is used as the nuisance
geometry:

```text
body state                         60
autoregressive previous action     10
8 future tokens x 16              128
8 token-validity bits               8
command                              3
                                   ---
                                   209
```

After an eight-bin warmup, ten speed-by-contact conditions have at least five
samples in every repeat. Crossed-seed means are:

| beta | RSA | 200 ms delay | Exact-input partial RSA | Partial delay |
|---:|---:|---:|---:|---:|
| 0 | **.806** | .718 | **.325** | .095 |
| .025 | .767 | .702 | .280 | .120 |
| .05 | .734 | .702 | .242 | .057 |
| .075 | .653 | .643 | -.103 | -.129 |
| .10 | .681 | .636 | -.073 | -.132 |
| .15 | .593 | .677 | -.241 | -.028 |

This finite-trial correction preserves the qualitative negative result but
narrows its interpretation. Beta zero has the highest mean alignment, and
stronger prior regularization does not make Demo H more SNN-like. Only the
three lowest-beta conditions have means above their delayed controls after
partialling the exact SNN input. The analysis has just ten estimable conditions
and three seeds per model, so the curves are descriptive rather than a precise
monotonic dose-response estimate.

Excluding the top input-weight-norm quartile raises beta-zero RSA/partial RSA
to `.847/.381` and leaves beta zero highest. The result is therefore not
created by a few input-proximal neurons. Final artifacts are
`out/aligned/beta_rsa_native.{json,npz}` and `out/aligned/rsa_native/`.

Relative to the provisional 63-transition-bin analysis, including the terminal
state changes the beta means by at most `.0083` for raw RSA and `.0322` for
partial RSA. The qualitative ordering is therefore insensitive to this final
state-alignment correction.

### Continuous-prior sensitivity check

A later pilot repeated the same 30 fixed trials and all three native SNN seeds
using one matched Demo H PPO seed trained against the continuous-data candidate
prior. Mean raw/partial RSA at beta zero is `.900/.684`; the corresponding
values at beta `.10` are `.848/.552`. Beta zero remains highest, while
beta `.15` also loses substantial task return. Thus improving the prior's
frozen rollout audit does not reverse the neural-alignment result. This is a
single-PPO-seed sensitivity check, not a replacement for the crossed-seed
result above. Its report and plot are under `out/prior_iteration_v2/`.

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
- on matched finite trials, higher Demo H beta does not improve similarity to
  the native-clip SNN; beta zero has the highest crossed-seed mean;
- the current SNN does not add predictive power for real Coltrane DLS activity;
- the retired 1,000-step controller failed functional locomotion;
  uninterrupted neural activity is not behavioral success.

Synthetic spikes are a controlled reference, not biological ground truth, and
representational similarity does not establish shared mechanism or causality.
