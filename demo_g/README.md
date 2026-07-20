# Demo G plan — Demo A + Demo F

Demo G is the workshop's combined SSL + RL demonstration on one simple,
fast-training body:

```text
Demo A: task reward teaches Fetch what works
Demo F: retargeted data teaches what rodent-derived Fetch motion looks like
Demo G: the same Fetch PPO policy is rewarded for both
```

The target is the unmodified Brax v1 Fetch body and Demo A locomotion task. This
removes the difficult 38-torque skeletal-rodent control problem that prevented
Demo E from becoming a five-minute workshop result.

## 1. Scientific and teaching claim

Demo F learns a conditional likelihood from recorded motion after a declared,
deterministic cross-morphology transform. Demo G freezes that likelihood and
uses it as a reward term while PPO interacts with Fetch physics:

\[
\max_\pi\;\mathbb E_{\tau\sim\pi}
\sum_t\gamma^t\left[
r_t^{\rm task}
+\beta\,\bar\ell_\phi(w_t\mid h_t,c_t)
\right].
\]

Here `w_t` is motion the physical Fetch policy just produced, `h_t` is its
causal motion history, and `c_t` is the same hindsight displacement convention
used to train Demo F. The frozen normalized likelihood `bar(ell)` receives no
gradient; PPO still learns actions only from return.

The careful wording is **rodent-derived motor realism on Fetch**, not literal
rat biomechanics. Retargeting preserves semantic paw/trunk trajectories and
contact timing while changing morphology and scale.

## 2. Controlled comparison

Implement two arms that differ by one scalar term:

| arm | Demo A task reward | frozen Demo F likelihood |
|---|---:|---:|
| G0 task-only | yes | no (`beta=0`) |
| G1 task + prior | yes | yes |

Both arms must share the Fetch body, reset distribution, command schedule,
policy initialization, PPO hyperparameters, random seed, environment count,
and transition budget. `beta=0` must reproduce Demo A step/reward traces within
numerical tolerance before the comparison begins.

## 3. Implementation sequence

### Gate F0 — accept retargeted data

Inspect Demo F's four speed clips and quantitative contact/IK report. Reject or
revise the transform before model tuning if feet skate, limbs saturate, or gait
phase is visibly wrong.

### Gate F1 — freeze the Fetch motion contract

Construct a compact state shared exactly by offline data and live Brax states:
root-local velocity, height/orientation, ten joint angles and velocities, four
foot positions/velocities, and contacts. Add parity tests against the original
Brax kinematics. Split by source session before forming overlapping crops.

### Gate F2 — train and validate Demo F

Start from Demo B's causal-convolutional tokenizer plus conditional Transformer,
but instantiate it with Demo F-owned dimensions and hyperparameters. Optimize
throughput before tuning quality. Candidate changes must be recorded one at a
time and selected on validation sessions.

Freeze a model only if it passes all of the following:

- held-out reconstruction and next-token likelihood beat persistence;
- real sequences score above time-shuffled and independently permuted legs;
- the matching speed/turn command outranks counterfactual commands;
- autoregressive rollouts remain upright-looking and periodic for at least 5 s;
- generated contact, joint-limit, and foot-skate metrics remain in the
  retargeted-data range;
- likelihood computation has sufficient batched JAX throughput for PPO.

### Gate G0 — integrate without changing Demo A

Add a thin Demo A environment wrapper that maintains per-environment causal
feature history. Evaluate the frozen prior only at its token rate and hold the
last shaping reward between token boundaries. Give every episode an explicit
history warm-up during which prior reward is zero.

Unit tests must establish:

- online features equal the offline Demo F definition;
- command conversion matches Demo F's hindsight horizon;
- the exported JAX scorer matches its PyTorch source;
- reward and transitions are identical when `beta=0`;
- resets cannot leak history between vectorized environments.

### Gate G1 — calibrate, then freeze the reward

Before PPO, score fixed datasets consisting of retargeted motion, early Demo A
tumbling, and late Demo A locomotion. Freeze a robust affine/quantile transform
to a bounded reward range and choose a small beta grid using training seeds
only. Report raw log likelihood separately from its reward transform.

### Gate G2 — paired PPO

Run G0 and G1 for the same transition budget. Demo A previously processed
roughly 100M transitions in about two minutes and 500M in about seven minutes on
this machine, so first target 100--250M transitions and reserve the five-minute
wall-clock claim for a measured run including compilation. Increase the budget
only if both arms are still improving and the workshop can use saved playback.

## 4. Evaluation

Evaluate checkpoints at fixed straight speeds and left/right turns, with held
out random seeds. Keep the two axes separate:

**Functional realism**

- forward/yaw tracking error;
- return and survival;
- target/waypoint success if the reach task is retained;
- control energy and lateral drift.

**Distributional motor realism**

- raw Demo F log likelihood after causal warm-up;
- matching-command rank and margin;
- stride-band versus high-frequency power;
- diagonal/contact phase structure across four feet;
- stance-foot speed, penetration, and joint-limit occupancy.

The preferred workshop result is not merely a higher shaped return. G1 must
retain at least 95% of G0's functional score while improving held-out raw
likelihood and at least one direct gait/contact metric across multiple seeds.

## 5. Ablations and failure interpretations

Run only the ablations needed to explain causality:

1. `beta=0` task-only control;
2. frozen correctly conditioned prior;
3. optional shuffled-prior or wrong-command control if the effect is ambiguous.

A policy can exploit model error, stand still for high likelihood, or trade away
task performance. These are failures even if the combined reward rises. A weak
effect may instead mean the successful Demo A gait is already inside Demo F's
high-likelihood region; that is a valid negative result and should be shown as
such.

## 6. Expected code layout

```text
demo_f/
  dataset.py       session-safe retargeted crops
  features.py      frozen offline/online 60-D feature contract
  models.py        independently tunable conditional motion model
  train.py         tokenizer + predictor training
  evaluate.py      likelihood, intervention, rollout, and contact gates
  export_jax.py    frozen scorer for Demo G

demo_g/
  config.py        command grid, beta, budget, and acceptance gates
  env.py           Demo A Fetch task plus causal frozen-prior wrapper
  prior.py         inference-only Demo F scorer
  train.py         paired G0/G1 PPO entry point
  evaluate.py      functional and distributional metrics
  render.py        synchronized G0/G1 videos
  tests/           feature parity, reset, command, beta-zero, export tests
```

Only the plan is created now. Demo G implementation starts after the retargeting
and Demo F model gates pass, so PPO cannot hide a bad data transform.
