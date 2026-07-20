# Demo E — frozen motion likelihood plus task RL

_Updated 2026-07-20. Canonical implementation: pipeline v6 in `demo_e/`._

Demo E is implemented end to end, including a conditional Demo B scorer that
passes the frozen transfer gates and a measured ten-minute E1 run. It does not
yet have a reportable E0/E1 locomotion comparison. The ten-minute E1 policy
learns to stand almost upright on two legs, not to walk quadrupedally.

## 1. Teaching objective

Demo E makes the workshop's two learning signals visible in one experiment:

| signal | learned from | meaning |
|---|---|---|
| Demo B / SSL | recorded rat motion | distributional realism: does the motion resemble the data? |
| Demo A / RL | physical consequences | functional realism: does the motion accomplish the task? |
| Demo E / both | data and interaction | brain–body–environment behavior must satisfy both constraints |

The board-level objective is

\[
\max_\pi\;\mathbb E_\pi\sum_t\gamma^t\left[
r_t^{\rm task}+\beta\frac1{D}\log p_\phi(w_t\mid h_t,c_t)
\right].
\]

`p_phi` is trained before RL and then frozen. The physical policy supplies the
realized future token `w_t`; only the task reward comes from body–environment
interaction. In code, the mean log likelihood is mapped monotonically into a
bounded reward score so its units do not overwhelm the task reward. Raw log
likelihood remains the scientific evaluation quantity.

The controlled comparison is:

- **E0:** task reward only (`beta=0`);
- **E1:** identical task and policy plus the frozen motion score (`beta=1`).

## 2. Why the controller is hierarchical

The MIMIC skeletal rodent has 38 actuators and long, four-joint limb chains.
Direct-actuator PPO repeatedly learned to sit, rear, or fall at short budgets.
Demo E therefore follows the successful TRACK-MJX RodentJoystick transfer:

```text
joystick command + task sensors
              |
              v
new PPO policy (trainable, 16-D intention)
              |
              v
published TRACK-MJX decoder (frozen, 38 torques)
              |
              v
MIMIC rodent at 100 Hz ----> task reward
              |
              +------------> Demo B motion score (E1 only)
```

The published imitation decoder is shared motor infrastructure, not the SSL
manipulation. Both E0 and E1 train their high-level policy from scratch. This
retains a clean one-term comparison while borrowing the low-level coordination
that made the upstream task solvable.

## 3. Frozen Demo B scorer

Demo B keeps two intentionally separate artifacts:

1. `motor_standalone.pt` is the behaviorally accepted workshop generator. Its
   Coltrane videos walk well and it remains unchanged.
2. `motor_prior_demo_e_jax.npz` is the separately selected, inference-only
   Demo E scorer. It adds a simple shuffled-command contrastive term so the
   likelihood is actually conditional enough to use as a reward.

The promoted scorer is
`research_contrastive_w10_m01_val_s0`: contrastive weight 10, margin 0.1, one
shuffled command negative, seed 0, selected only by held-out validation loss.
Its source checkpoint is format v4, 281-D Coltrane; its committed JAX export is
format v6 and about 15 MB.

### Source-data gate

On held-out Coltrane motion, the promoted seed-0 model:

- beats latent persistence by 25.2%;
- gives real command pairs mean logp -0.885 versus -1.112 after shuffling;
- puts all five test speed-bin means on the likelihood diagonal;
- peaks at zero relative speed offset.

An independent training seed also cleared the same frozen source and physical
criteria. The source checkpoint itself remains a regenerated training artifact;
the JAX export records its SHA-256 and fixed session split.

### Physical-transfer gate

The gate uses development-only native-reset trajectories from the reproduced
52.4M controller. Each realized trajectory is paired with its actual 0.62 s
hindsight displacement rather than the requested joystick label, preventing
controller under-tracking from being mislabeled as a prior failure.

| training seed | moving top-1 | mean rank | matched margin | moving − zero command | moving − standing |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.629 | 1.626 | +0.0223 | +0.2285 | +0.1117 |
| 1 | 0.672 | 1.755 | +0.0241 | +0.2831 | +0.1708 |

Both seeds pass. Report seeds were excluded from candidate selection. The
offline implementation is `demo_e/experiment/audit_prior.py`.

### Gaussian likelihood

The causal convolutional encoder converts a 50 Hz, 281-D feature stream into a
16-D token every 80 ms. A causal Transformer reads eight history tokens plus a
future displacement command and predicts eight future token means. With frozen
isotropic `sigma`, one token receives

\[
\frac1D\log p(w\mid h,c)
=-\frac12\operatorname{mean}_i\left[
\left(\frac{w_i-\mu_i}{\sigma}\right)^2
+2\log\sigma+\log(2\pi)\right].
\]

Because `sigma` is fixed, likelihood ranking is exactly negative prediction-
error ranking. The Gaussian supplies interpretable reward units; it is not a
second learned head.

## 4. Physical representation bridge

The feature contract is unchanged from the accepted 281-D Demo B model:

```text
root-local planar velocity                   2
root height                                  1
root orientation increment                   6
67 joint angles + velocities               134
23 root-local keypoints + velocities       138
                                            ---
                                            281
```

MJX lacks the recorded DANNCE landmarks, so each keypoint is represented by a
body-local site calibrated on the same scaled skeleton using only training
sessions. Reconstruction error is 7.29 mm mean and 10.18 mm maximum. Two
channels (`finger_R` angle and velocity, indices 75 and 142) are constant in
the source data; pipeline v6 masks them after normalization rather than
amplifying passive physics motion through a variance floor.

PyTorch/JAX tests use the exact promoted source pair when the ignored PyTorch
checkpoint is present. Encoder parity is bounded at `4e-4`; transition parity
at `3e-3`.

## 5. Time, reset, and command contracts

| process | rate | relation |
|---|---:|---|
| MuJoCo simulation | 500 Hz | 2 ms physics |
| decoder/task control | 100 Hz | five physics steps |
| Demo B feature | 50 Hz | every two controls |
| token/likelihood | 12.5 Hz | every eight controls |

Demo E preserves RodentJoystick's literal native reset: the task state returned
by `mjx.make_data` is not replaced with a forwarded copy. E1 forwards only a
private copy to initialize its kinematic history. Native autoreset task/decoder
state is also preserved; only prior-side causal buffers reset.

The scorer withholds reward for the first eight realized tokens (0.64 s), then
uses the native eight-token block prediction contract. A prior pipeline-v4
draft forwarded the task reset and is rejected because it changed controller
behavior materially.

Demo B conditions on egocentric displacement over 31 frames, or 0.62 s. For a
constant joystick command `(v, omega)`, pipeline v6 integrates the planar arc:

\[
c=\left[
v\frac{\sin(\omega T)}{\omega},
v\frac{1-\cos(\omega T)}{\omega},
\omega T
\right],\qquad T=0.62\text{ s},
\]

with the straight-line limit `[vT, 0, 0]`. This replaces the incorrect draft
that always set lateral displacement to zero while turning.

## 6. Reward calibration

The confirmed 52.4M native-reset controller was scored at six fixed commands.
Across 300 valid post-warm-up tokens, raw likelihood ranged from -1.416 to
-0.772 nats per latent dimension; the pooled 1st/99th percentiles were -1.354
and -0.780. Pipeline v6 freezes

```text
prior_score = clip((raw_logp - (-1.5)) / (-0.75 - (-1.5)), 0, 1)
beta = 1.0
```

The small padding avoids clipping valid gait variation. The score arrives once
per eight 100 Hz task steps, so it remains a shaping term rather than replacing
the task objective. Evaluation always reports both raw likelihood and this
reward-side transform.

## 7. PPO and measured budgets

Canonical PPO settings match the upstream high-level reproduction:

```text
parallel environments       8192
unroll length               20
batch size                  2048
minibatches                 16
updates per batch           4
learning rate               1e-4
entropy cost                1e-2
discount                    0.99
actor/critic layers         1024, 512, 256
episode length              1000 controls (10 s)
```

The reproduced task-only reference requested 50M transitions and Brax rounded
it to 52,428,800. Training took 2,399 s (40.0 min) and the full evaluation about
425 s. Under the literal native reset at a fixed 0.30 m/s:

| transitions | approximate elapsed | survival | mean vx |
|---:|---:|---:|---:|
| 0 | initial | 0.426 | 0.005 m/s |
| 13.1M | 11 min | 1.000 | 0.000 m/s |
| 26.2M | 22 min | 1.000 | -0.002 m/s |
| 39.3M | 34 min | 1.000 | 0.002 m/s |
| 52.4M | 45 min | 1.000 | 0.240 m/s |

The controller learns stable standing long before gait. The saved evidence
bounds native-reset locomotion onset to `(39.3M, 52.4M]` transitions.

### Pipeline-v6 E1 ten-minute diagnostic

The first promoted-prior E1 run saved 9,830,400 transitions in 582.520 s end to
end (9.71 min; 542.323 s PPO-reported work). Six commands × three seeds gave:

| transitions | survival | functional | vx MAE | raw logp | command top-1 | margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.278 | 0.0149 | 0.197 | -0.994 | 0.167 | -0.087 |
| 3.28M | 0.436 | 0.0425 | 0.192 | -1.000 | 0.154 | -0.152 |
| 6.55M | 0.363 | 0.0523 | 0.182 | -0.975 | 0.171 | -0.129 |
| 9.83M | 0.450 | 0.0551 | 0.195 | -0.981 | 0.206 | -0.126 |

At 0.30 m/s the final policy rises into an almost upright, two-legged stance.
That is genuine posture learning, but it is not quadrupedal locomotion. It
averages about 0.013 m/s, travels 3.46 cm in the seed-0 trial, and terminates at
3.04 s when torso orientation reaches the 60-degree `fallen` criterion. The
video continues to display the held terminal pose and labels it `terminated`;
this is not a claim that the visible body has already flopped flat.

The scorer correctly ranks the confirmed gait above this upright stance at the
same command (-0.924 versus -1.055 nats/dim), but the normalized difference is
only about 0.022 reward per 100 Hz step. It is a modest regularizer, not a gait
bootstrap. PPO KL also spiked to 25.9 and 36.5 in two intervals.

This result falsifies a ten-minute locomotion claim. It does not establish that
transition-matched or full-budget E1 fails, because it contains fewer
transitions than even the upstream 13.1M standing checkpoint. See
`demo_e/experiment/TEN_MINUTE_E1.md` for exact hashes and diagnostics.

## 8. Evaluation and acceptance

The fixed grid is `(vx, yaw)` = `(0,0)`, `(0.1,0)`, `(0.2,0)`, `(0.3,0)`,
`(0.3,-0.75)`, and `(0.3,+0.75)`, with three deterministic seeds. The evaluator
runs with `beta=0`, so scoring cannot change either trajectory.

Report separately:

- survival, forward/yaw tracking, task reward;
- lateral motion, energy, and action rate;
- raw conditional log likelihood and reward-normalized score;
- counterfactual matching-command top-1, rank, and margin;
- every checkpoint, not only a favorable final video.

A workshop claim requires E1 survival ≥0.90, at least 90% of paired E0
functional performance, better raw likelihood than E0, matching-command top-1
≥0.50, positive matching margin, and finite metrics. No current paired result
passes these gates.

## 9. Commands and canonical code

```bash
# Contract tests
uv run --extra dev --extra workshop pytest -q demo_e/tests

# Wiring only; not convergence
uv run python -m demo_e.train --arm e0 --smoke
uv run python -m demo_e.train --arm e1 --smoke

# Full controlled runs (same seed and transition budget)
uv run python -m demo_e.train --arm e0 --seed 0
uv run python -m demo_e.train --arm e1 --seed 0

# Evaluate every saved checkpoint and render command index 3 (0.30 m/s)
uv run python -m demo_e.evaluate --all-checkpoints
uv run python -m demo_e.render demo_e/out/evaluation-<stamp>.npz \
  --command 3 --seed 0
```

```text
config.py       frozen task, reward, budget, and command grid
features.py     exact 281-D MJX feature bridge and arc command
prior.py        inference-only JAX encoder, predictor, and Gaussian score
env.py          native task, frozen decoder, causal score, reset wrapper
train.py        sole E0/E1 PPO entry point
evaluate.py     fixed-command and all-checkpoint evaluator
render.py       single or paired rollout renderer
runtime.py      checkpoint/network restoration
provenance.py   fail-closed hashes and controller metadata
experiment/     offline scorer audit and append-only result notes
tests/          split, parity, reset, reward, and command contracts
```

## 10. Next experiment and claim boundary

The next useful experiment is a paired E0/E1 run with identical seeds and
**transition budgets**, first at 13.1M to test standing acquisition and then at
the full gait-onset budget if resources permit. Do not tune on report seeds.
If optimization remains unstable, diagnose PPO KL and reward composition on
development seeds before changing `beta` or the prior definition.

Do not claim that likelihood is biological realism, that Demo B learns actions,
or that ten-minute E1 walks. The supported statement is narrower: Demo B learns
a conditional motion distribution from recorded futures; Demo E freezes that
distribution as a reward-side constraint while PPO still learns actions from
physical task return.
