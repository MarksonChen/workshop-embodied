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

## Aligned 1,000-bin SNN

The aligned follow-up uses the same `1.75x` data and 20 ms action clock as Demo
H. A train-only whitened PCA converts four 60-D feature frames into a 16-D
token. SNN state and autoregressive previous action persist for 1,000 bins.

The release has independent 64-frame clips, not natural 20-second recordings.
Training therefore repeats a wrap-screened 32-frame segment and labels the
clock `synthetic-periodic-20ms`. No continuity beyond that segment is claimed.

### Future-preview ablation

| Preview | Horizon | Validation action MSE | Time |
|---:|---:|---:|---:|
| 1 token | 80 ms | .014159 | 22.0 s |
| 4 tokens | 320 ms | **.013805** | 21.9 s |
| 8 tokens | 640 ms | .013916 | 22.6 s |

The differences are small: the original five-frame/100 ms intention was not
catastrophically short, and 640 ms offers no clear gain. At eight tokens,
seeds 0/1/2 score `.013916`, `.014041`, and `.014213`. Doubling optimization
from 64 to 128 episode batches lowers seed-0 MSE to `.012901` in 39.3 seconds,
so the short run is useful but not fully converged.

Equal six-speed sampling repeats only real rows, but the release contains just
2 of 266 eligible training cycles in the 3.75–4.0 stratum. Balanced validation
MSE worsens to `.021797`; this variant is rejected for neural comparisons.

## Exact-input RSM/RSA

Three SNN seeds and 18 Demo H checkpoints are evaluated on the same 30 fixed
episodes: six speeds, five repeats, 1,000 bins. The behavior control is each
SNN recording's exact raw 205-D input:

```text
body state                         60
autoregressive previous action     10
8 future tokens x 16              128
phase + command                   4+3
                                   ---
                                   205
```

Conditions are speed by exact four-foot contact pattern. Sixty-eight
conditions contain at least five samples in every repeat. The score is a
Spearman comparison of diagonally noise-normalized crossvalidated RDMs; the
control delays Demo H by 200 ms without wraparound.

| beta | RSA | Delayed | Exact-input partial RSA | Partial delayed |
|---:|---:|---:|---:|---:|
| 0 | **.869** | .779 | **.680** | .547 |
| .025 | .842 | .753 | .635 | .500 |
| .05 | .755 | .669 | .454 | .366 |
| .075 | .800 | .683 | .539 | .380 |
| .10 | .749 | .667 | .439 | .346 |
| .15 | .771 | .679 | .470 | .360 |

Aligned scores exceed delayed scores at every beta, including after exact-input
partialling. Beta zero is highest and `.025` second, so the aligned experiment
rejects a monotonic “more prior makes Demo H more SNN-like” claim.

Every recurrent SNN neuron receives dense input; there is no anatomical input
layer. Excluding the top quartile by input-weight norm retains 192/256 neurons
per seed, raises beta-zero RSA/partial RSA to `.882/.707`, and preserves the
same beta ordering. The result is not driven by a few input-proximal units.

Final artifacts:

- `out/aligned/beta_rsa_full_input.{json,npz}`
- `out/aligned/rsa_full_input/beta_rsa.{png,svg}`
- `out/aligned/rsa_full_input/rsm_examples.{png,svg}`
- matching `beta_rsa_exclude_input_q4` sensitivity files

## Functional long-horizon probe

An audit found that an early readout-PPO version collected 100-step episodes
while evaluating for 1,000 steps. It was rejected. The corrected trainer
requires identical collection and evaluation horizons and balances six speed
strata. Its 2.56M-transition run takes 179 seconds.

The environment's coarse termination condition remains false for all six
1,000-step episodes, but that is **not** locomotion survival or success. Visual
inspection of the retained comparison video shows roughly half the SNN
rollouts losing locomotion partway. Only the 2.0 and 2.5 cases make substantial
forward progress, both well below their commands; none solves speed control:

```text
requested  1.50  2.00  2.50  3.00  3.50  4.00
realized  -0.09  1.01  1.72 -0.27  0.30  0.28
```

Mean track reward is `.241`. Several cases also have zero or near-zero contact
switches for at least one foot. The retained video
`out/aligned/snn_1000_step_speed_sweep_aligned.mp4` therefore demonstrates a
failed long-horizon functional controller, despite uninterrupted recurrent
activity and the permissive termination metric. Continuous source trajectories,
on-policy imitation such as DAgger, or a TRACK-MJX-scale PPO budget would be
needed before promoting this controller.

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
- Demo H and SNN populations share fixed-input representational geometry beyond
  exact inputs and a temporal-delay control;
- increasing Demo H's prior strength does not reliably or monotonically
  increase that similarity;
- the current SNN does not add predictive power for real Coltrane DLS activity;
- the aligned 1,000-step controller fails functional locomotion and is retained
  only as a negative result; uninterrupted neural activity is not behavioral
  success.

Synthetic spikes are a controlled reference, not biological ground truth, and
representational similarity does not establish shared mechanism or causality.
