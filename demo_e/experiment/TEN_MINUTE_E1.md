# Pipeline-v6 E1 ten-minute result

Date: 2026-07-20. Status: **diagnostic, not workshop-reportable**.

This is the first non-smoke E1 run after the Demo B transition scorer cleared
the frozen source-data and physical-transfer gates. It answers the narrow
question: what does the current task-plus-prior algorithm learn in about ten
minutes of wall-clock training?

## Frozen setup

- Policy: fresh 1024/512/256 PPO high-level policy, seed 0
- Action: 16-D intention consumed by the published frozen TRACK-MJX decoder
- Body/task: native-reset MIMIC-MJX `RodentJoystick`, 100 Hz control
- Prior: validation-selected Coltrane candidate
  `research_contrastive_w10_m01_val_s0`
- Prior export SHA-256:
  `cb5bef8ffb90dd6795b2fa8ff5ac421f96e806c94418227a8e8648903c3e33b5`
- Reward scale: `clip((logp + 1.5) / 0.75, 0, 1)`, `beta=1`
- Prior update rate: 12.5 Hz after the 0.64 s causal warm-up

The run saved 9,830,400 transitions. PPO reported 542.323 s of training work;
the final checkpoint landed after 582.520 s end to end (9.71 min). The launch
ceiling was 13,107,200 transitions only to obtain the same Brax checkpoint
grid; the process was intentionally stopped after the atomic 9.83M checkpoint.

## Learning curve

Evaluation used six commands, three deterministic seeds, 5 s trials, and no
reward-side intervention (`beta=0` during evaluation).

| transitions | survival | functional | vx MAE | raw logp | command top-1 | command margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.278 | 0.0149 | 0.197 | -0.994 | 0.167 | -0.087 |
| 3.28M | 0.436 | 0.0425 | 0.192 | -1.000 | 0.154 | -0.152 |
| 6.55M | 0.363 | 0.0523 | 0.182 | -0.975 | 0.171 | -0.129 |
| 9.83M | 0.450 | 0.0551 | 0.195 | -0.981 | 0.206 | -0.126 |

The PPO KL diagnostic was 25.928, 0.090, and 36.496 at the three trained
checkpoints. The large first/final spikes warrant caution, but no intermediate
checkpoint contains a hidden locomotor solution.

## What the video shows

At a fixed 0.30 m/s straight command, the final policy **stands up into an
almost upright, two-legged posture**. This is real posture learning and should
not be described as merely sitting or flopping. It is nevertheless not the
requested quadrupedal locomotion:

- three-seed mean forward velocity: approximately 0.013 m/s;
- seed-0 displacement before termination: 0.0346 m;
- seed-0 mean speed before termination: 0.0114 m/s;
- seed-0 termination: 3.04 s, at about -59.1 degrees root pitch;
- environment criterion: 60-degree torso-angle `fallen` threshold;
- mean action-rate metric: approximately 3.3e-5.

The renderer holds the terminal state and now labels subsequent frames
`terminated`; older clips called every such frame `fall`, which misleadingly
suggested that the visible body had already flopped flat. The learned strategy
is best described as **upright standing with negligible translation, followed
by orientation-limit termination**.

## Scorer diagnosis

The frozen prior does not confuse this strategy with the confirmed gait. At
the 0.30 m/s command:

- ten-minute E1 upright stance: about -1.055 nats per latent dimension;
- confirmed 52.4M walking controller: about -0.924 nats per latent dimension.

The walking motion therefore receives about 0.174 more normalized score at a
12.5 Hz prior update, equivalent to only about 0.022 reward per 100 Hz control
step. The failed policy's matching-command top-1 rate is 0% at 0.30 m/s, so the
counterfactual diagnostic also recognizes that the realized near-static motion
does not match the requested future displacement.

The evidence supports a limited conclusion: the prior is a valid diagnostic
and modest realism regularizer, but it does not by itself bootstrap this
difficult gait at 9.83M transitions. Failed motion still earns a positive
absolute likelihood bonus, and the incremental conditional advantage of the
correct gait is small until exploration discovers the gait.

## Budget interpretation

The native upstream task-only curve reaches stable standing at 13.1M
transitions but does not show the confirmed 0.30 m/s gait until 52.4M. This E1
run contains fewer transitions than the upstream 13.1M checkpoint because the
scorer lowers throughput. It therefore falsifies a ten-minute locomotion claim;
it does not establish that transition-matched or full-budget E1 cannot work.

The next scientifically useful run is a paired, identical-transition-budget
E0/E1 comparison. Do not select a new `beta`, reward transform, or model using
the report seeds from this diagnostic.

## Local artifacts (gitignored)

- run: `demo_e/out/policy/e1-report-seed0-20260720-111121/`
- evaluation JSON SHA-256:
  `c853c4b1748e2873543e8cdff9d8f997a1346bfad1fd4e7b279477498d78f540`
- evaluation NPZ SHA-256:
  `54e9c2a61843f07a4672c7bbb4fef81b0865bbed64c40b8e8197bf225500d6f5`
- video SHA-256:
  `223e30f1e6133a4eeacd5f0a2ed1bb45a24cad085acbb33d09e625c12d7faa30`
- video: `demo_e/out/videos/e1-p6-10min-vx030-seed0-5s.mp4`

These generated outputs remain excluded by repository policy; the hashes make
the tracked result note auditable against a local or transferred artifact set.
