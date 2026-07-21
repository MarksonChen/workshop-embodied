# Demo H — generative body–action pretraining plus RL post-training

_Accepted 2026-07-21 as the workshop capstone. Operational commands:
[`demo_h/README.md`](../../demo_h/README.md). Append-only decisions:
[`demo_h/experiment/DECISIONS.md`](../../demo_h/experiment/DECISIONS.md)._

## 1. Executive result

Demo H implements the policy-side version of the workshop's SSL + RL thesis:

```text
rodent-derived Fetch motion
  -> predict a short future body-motion plan
  -> predict the exact-Fetch controls that realize that plan
  -> freeze the pretrained state/action policy
  -> train a zero-initialized residual with PPO task reward and reference KL
```

The accepted run uses:

- an empirical `1.75x` temporal dilation of Coltrane/Fetch motion;
- 2,404 exact-physics state/control clips;
- a 16-D motion plan and 50 Hz Gaussian feedback controller;
- task commands uniformly sampled from 1.5 to 4.0 Fetch units/s;
- `beta=0.10`, 30M PPO transitions, 2,048 environments, and seed 0.

The physical projection takes 84.4 seconds, prior training 70.8 seconds, and
PPO 95.2 seconds on the current H100. The complete data-build, pretraining, and
post-training path is therefore about 250 seconds. With the derived data
prepared ahead of time, the two learned stages take 166 seconds.

The one-sentence workshop version is:

> **Generative pretraining learns familiar short body-motion plans and the
> Fetch controls that realize them; reinforcement learning learns a small
> correction that makes those familiar controls accomplish a task.**

This is a body-centric, inverse-dynamics-style world–action model. Its narrow
“world” is the Fetch body and recurring flat-ground contacts. PPO runs in exact
Brax physics, not inside a learned simulator.

## 2. Keep the learning signals honest

The workshop must display where each target originates:

| quantity | source | teaching description |
|---|---|---|
| future body motion | shift the same retargeted trajectory forward | self-supervised target |
| Fetch control | execute a declared feedback controller in exact physics | physics-derived pseudo-label |
| task reward | execute the adapted policy in the environment | reinforcement-learning signal |

The whole first stage may be called **generative behavior pretraining with
physics-derived control pseudo-labels**. Calling every component pure SSL would
blur Demo B's definition: the controls depend on a simulator, contact model,
actuator strength, and feedback law not contained in the recording.

The action labels are also not animal torques. They are controls for the much
larger Fetch morphology under Fetch physics.

## 3. Relationship to A, F, G, and GPC

| demo | data/model role | RL role |
|---|---|---|
| A | none | learn Fetch control from task reward |
| F | predict retargeted future motion | none |
| G | freeze Demo F likelihood as an extra reward | learn a policy from scratch |
| H | pretrain a motion planner and action distribution | adapt the pretrained policy with PPO + KL |

The direct contrast is:

```text
Demo G: data prior -> reward-side score -> scratch PPO
Demo H: data prior -> pretrained actor -> residual PPO + reference KL
```

The local [GPC paper](../papers/GPC.pdf) motivated a reusable generative
controller followed by task adaptation. Demo H does not claim to reproduce it.
GPC first trains a physics-tracking controller, models discrete FSQ skill
tokens, and applies parameter-efficient adaptation while constraining sampling
to frozen-model support. Demo H deliberately omits FSQ, a large vocabulary,
tracking RL, PD-target actions, and DoRA/FiLM. It uses continuous normalized
Fetch controls, a small residual MLP, and an explicit Gaussian KL.

## 4. Accepted physical state–action data

### 4.1 Why the timing changed

The original geometric retargeting enlarges a 0.09355 m rodent trunk to Fetch
length 2.0, a factor of 21.3789, while retaining the source 50 Hz clock. Direct
timing therefore looked accelerated. The theoretical dynamically similar time
factor is:

\[
s_t=\sqrt{21.3789}=4.6237,
\]

but those clips looked too slow for this workshop body. Exact 50 Hz videos at
factors 1.1, 1.25, 1.4, 1.6, 1.8, and 2.0 were inspected at three motion-speed
quantiles. The user selected `1.75x`.

This is an empirical temporal choice, not Froude similarity. Demo H versions it
as `temporal-dilation-1p75-v1`; canonical Demo F keeps its independently
accepted Froude release. Each retimed parent yields one centered 64-frame crop,
so examples do not overlap or join parent clips.

The parent contains:

| split | clips |
|---|---:|
| train | 1,804 |
| validation | 278 |
| final test | 344 |

### 4.2 Why finite-difference inverse dynamics is insufficient

Demo F clips are contact-aware kinematic IK, not dynamically feasible Fetch
rollouts. Root translation and yaw are imposed, roll/pitch are absent, contacts
are heuristic, and temporal interpolation can create acceleration artifacts.
For a floating-base contact system,

\[
M(q)\ddot q+C(q,\dot q)=S^\top\tau+J_c^\top\lambda,
\]

joint torque is not uniquely recoverable without compatible contact forces and
modes. A generic inverse-dynamics output is therefore not automatically the
ten actuator commands used by Demo A.

The deployed contract is instead explicit:

- Brax v1 0.12.3 Fetch with the Demo A config;
- ten one-axis direct-torque actuators;
- normalized control `u` in `[-1,1]^10`;
- requested actuator-axis torque `-300u` before joint-limit gating;
- 0.02-second control step and four physics substeps;
- gravity `-9.8` and friction `0.7745967`.

### 4.3 Transparent exact-physics projection

Treat retargeted joint angles as a soft reference. At every step, execute:

\[
u_t=\operatorname{clip}\left(
\frac{400(q_t^{\rm ref}-q_t)+10(\dot q_t^{\rm ref}-\dot q_t)}{300},
-1,1\right).
\]

The legacy Fetch coordinate convention means positive normalized control raises
the reported angle even though requested actuator-axis torque is `-300u`.
Store only the states produced by executing these controls in the unchanged
simulator. Keep the kinematic reference in separate audit fields.

The temporal convention is permanent:

\[
u_t\text{ acts during }[t,t+1)\text{ and produces }x_{t+1}.
\]

The accepted physical variant is
`exact-fetch-feedback-projection-retime-1p75-v1`:

| measure | accepted value |
|---|---:|
| train / validation / test clips | 1,784 / 278 / 342 |
| total clips / transitions | 2,404 / 151,452 |
| global projection pass rate | 99.09% |
| median-across-shard joint RMSE | 0.103 rad |
| median-across-shard root RMSE | 0.325 |
| mean-across-shard saturation | 1.36% |
| minimum torso height | 1.133 |
| minimum upright | 0.514 |
| build time | 84.4 s |

Independent paired-control replay recovers stored positions, orientations, and
joint angles to approximately `1e-5`; shuffled controls are materially worse.
Session splits remain disjoint, and every shard and parent manifest is hashed.

The realized command-speed distribution is broad but not uniform. On training
clips, the 10th/50th/90th percentiles are approximately 0.34/1.06/2.27. The RL
range deliberately extends to 4.0, making its upper end an adaptation and
extrapolation stress test rather than dense data support.

## 5. Frozen body–action prior

### 5.1 State and temporal representation

Retain Demo F's readable 60-D physical feature:

```text
root-local planar velocity                   2
root height                                  1
root rotation and angular velocity           9
ten joint angles and velocities             20
four local feet and their velocities        24
four realized contact bits                   4
                                            --
                                            60
```

A causal convolutional tokenizer downsamples four 50 Hz frames into one 16-D
token. The planner reads four tokens, representing 16 raw history frames, and
predicts one next token. One plan spans four 50 Hz control phases.

Let `H_k` be the four-token history and `g_k` the 0.62-second egocentric
hindsight displacement command. The learned factorization is:

\[
p_\theta(z_{k+1}\mid H_k,g_k)
\prod_{j=0}^{3}
p_\theta\!\left(
u_{t+j}\mid x_{t+j},z_{k+1},u_{t+j-1},j,g_k
\right).
\]

Read it left to right:

1. Predict a plausible short future body-motion plan.
2. Observe current physical feedback.
3. Generate the next control conditioned on the plan and previous control.
4. Let exact Fetch physics produce the next state.
5. Refresh the motion plan every four controls.

This is state-first generation plus a feedback inverse-dynamics actor. It is
not the causally action-conditioned learned transition
`p(x_{t+1}|x_t,u_t)` used for planning in a general learned simulator.

### 5.2 Training details

Train from scratch for the accepted timing distribution:

- tokenizer: 1,000 updates;
- state predictor: 1,500 updates;
- action decoder: 2,000 updates;
- batch size: 512;
- hidden width: 192;
- four Transformer layers and four heads;
- predicted-plan input on 75% of decoder examples;
- predicted-previous-control input on 75%;
- plan noise standard deviation 0.05.

The action decoder predicts a correction to the previous pre-tanh control mean.
Its per-joint Gaussian standard deviations are calibrated on residuals. Exact
boundary controls remain stored in the data, but recurrent likelihood input is
clipped to `[-0.98,0.98]` because a tanh Gaussian cannot represent exactly
`+/-1` with finite mean.

### 5.3 Held-out and physical gates

The accepted prior trains in 70.8 seconds. On 342 final-test clips:

| measure | result |
|---|---:|
| state skill over persistence | +49.8% |
| matching-command win rate | 82.4% |
| action MSE | 0.0102 |
| shuffled-plan action MSE | 0.0187 |
| closed-loop action skill over repeated initial control | +86.9% |

Copying the immediately preceding 50 Hz control achieves one-step MSE 0.00952,
6.8% better than the learned one-step decoder. Retain this negative result.
The relevant causal rollout nevertheless improves strongly, the plan matters,
and the model beats zero control by almost an order of magnitude.

When the frozen JAX prior acts without PPO at a 1.5 command:

| reset | survival | mean speed | displacement | minimum upright | saturation |
|---|---:|---:|---:|---:|---:|
| in-support physical state | 100% | 0.988 | 4.94 | 0.982 | 0.08% |
| ordinary standing state | 100% | 1.031 | 5.16 | 0.982 | 0% |

Every foot changes contact in both rollouts. This establishes an executable
pretrained policy before RL begins.

## 6. RL post-training

### 6.1 Policy architecture

Freeze the motion tokenizer, planner, base action mean, action standard
deviation, and a separate reference copy. The trainable actor is a two-layer
128-unit residual MLP initialized with zero output weights. It sees a compact
context containing:

- Demo A's 101-D physical observation;
- latest normalized 60-D body feature;
- 16-D predicted plan;
- four-phase one-hot code;
- normalized three-dimensional command;
- ten-dimensional frozen prior mean.

The actor adjusts both pre-tanh mean and scale. Zero initialization makes the
initial policy exactly the frozen prior while retaining enough residual range
to escape imperfect data.

### 6.2 Exact KL implementation

For H2, the environment adds frozen-reference log probability divided by ten
action dimensions. PPO entropy uses the matching coefficient. In expectation,
their sum is:

\[
-\frac{\beta}{10}
D_{\rm KL}\left[
\pi_\psi(\cdot\mid h_t,g_t)
\Vert p_{\theta_0}(\cdot\mid h_t,g_t)
\right].
\]

The complete objective is:

\[
J(\psi)=\mathbb E\sum_t\gamma^t\left[
r_t^{\rm task}-\frac{\beta}{10}D_{\rm KL}(\pi_\psi\Vert p_{\theta_0})
\right].
\]

The task reward retains Demo A's speed tracking, uprightness, and modest control
cost. The target speed is sampled uniformly from 1.5 to 4.0. Contact
participation, cadence, symmetry, stride bandwidth, and all other naturalness
diagnostics are absent from the objective.

### 6.3 Frozen training budget

| setting | value |
|---|---:|
| accepted beta | 0.10 |
| transitions | 30,000,000 |
| parallel environments | 2,048 |
| PPO evaluations | 3 |
| episode length | 1,000 |
| unroll length | 20 |
| batch / minibatches | 256 / 8 |
| updates per batch | 4 |
| learning rate | 3e-4 |
| discount | 0.97 |
| seed | 0 |
| measured training time | 95.2 s |
| throughput | 315k transitions/s |

The 15M midpoint remained materially below the final return in both β=0.075
and β=0.10 high-speed runs, so retain the complete 30M budget.

## 7. Beta selection and accepted behavior

Train β=0.075 and β=0.10 with the same prior, seed, speed distribution, budget,
and evaluator. Put all 12 five-second trajectories in one labeled video.

| aggregate over six commands | β=.075 | β=.10 |
|---|---:|---:|
| mean absolute speed error | 0.128 | **0.079** |
| minimum survival | 96.8% | **100%** |
| strict stride gates passed | **5/6** | 4/6 |
| mean joint-speed RMS | **3.165** | 4.998 |
| mean realized reference KL/dim | **0.393** | 0.430 |

The training coefficient does not order realized KL monotonically because PPO
converges to different local solutions. The user accepted β=0.10 after direct
video inspection, prioritizing its speed tracking and complete survival for the
workshop presentation.

The accepted command sweep is:

| command | realized mean ± temporal std | survival | upright mean | stride gate |
|---:|---:|---:|---:|---:|
| 1.5 | 1.471 ± 0.188 | 100% | 0.996 | pass |
| 2.0 | 2.010 ± 0.221 | 100% | 0.997 | pass |
| 2.5 | 2.479 ± 0.269 | 100% | 0.997 | **fail** |
| 3.0 | 2.974 ± 0.347 | 100% | 0.994 | pass |
| 3.5 | 3.465 ± 0.417 | 100% | 0.996 | pass |
| 4.0 | 3.647 ± 0.593 | 100% | 0.972 | **fail** |

The strict validator checks all four feet independently and is used only for
reporting. At 2.5 one foot lacks a valid backward-stance/forward-swing reset;
at 4.0 one foot is underused for long intervals. These failures remain in the
accepted video rather than being optimized away with a hand-written reward.

## 8. What “accepted” means

Demo H is accepted for a pedagogical demonstration because:

- the state/control data replay exactly in the deployed simulator;
- the conditional prior passes held-out state, command, plan, and closed-loop
  action gates;
- the frozen actor locomotes from standing before RL;
- PPO remains below two minutes and complete pretraining plus PPO below three;
- the accepted policy tracks a broad command range and survives every shown
  rollout;
- all learning signals remain separable in code and explanation.

It is not an algorithm-level result. The accepted high-speed arm has one policy
training seed, no matched high-speed H0/H1 report, and two failed stride cells.
Do not claim that β=0.10 is statistically superior to β=0.075 or scratch PPO.

## 9. Reproduction

Build the selected Demo F timing derivative:

```bash
uv run --extra workshop python -m demo_f.dataset.retime \
  --time-scale 1.75 --crops-per-parent 1 \
  --variant temporal-dilation-1p75-v1 \
  --output-root demo_f/dataset/release_retime_1p75
```

Project it into exact physics:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.dataset.project --splits train validation test
```

Train and export the prior:

```bash
uv run --extra workshop python -m demo_h.dataset.validate
uv run --extra workshop python -m demo_h.train_prior --from-scratch
uv run --extra workshop python -m demo_h.evaluate_prior
uv run --extra workshop python -m demo_h.export_jax
```

Train H2 with the accepted defaults:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.train --arm h2 --seed 0
```

Use `visualize_speeds.py` and `render_speed_comparison.py` as shown in the
operational README. The evaluator batches all requested commands into one
compilation. The renderer reuses static geometry and defaults to 25 fps while
preserving real playback duration.

## 10. Artifact identities

Generated datasets, weights, reports, and videos remain gitignored. The
accepted local artifacts are identified by SHA-256:

| artifact | SHA-256 |
|---|---|
| Demo F 1.75 manifest | `85fe54ee9730fe3c79871c6739197496e92b726f5072d93c4322bd001df82b3f` |
| Demo H physical manifest | `c02c0cc43775dc28ee33106b4841f7dc7a06696c20e956e7d21aeb36dfd76847` |
| PyTorch prior | `181394fe81eba60aeb67a38d2cac229f2c26e7ea844d8701b68d648ff3d4f903` |
| JAX prior | `fc4f5797844c2b2426d7c5f92ed093cb5d8d6ead8113ee2b1dc46cf649203382` |
| β=.10 PPO checkpoint | `e876bf800b17b48d602f28f067033fd4bb48246cd7e8fd7420cfe7cb5357cb44` |
| accepted speed metrics | `632ba2dd1775518042a0f2af6f09f0db88e26ecc94066dd82a4306002a35d710` |
| β comparison video | `e7568edf8f722cb2f6b2a8277877b1bda6436d92d6d4fa2b554af2055d820001` |

## 11. Package map

```text
demo_h/
  config.py                    frozen physical and training constants
  dataset/
    contract.py                schema, variants, and accepted default roots
    project.py                 exact-physics pseudo-label generator
    validate.py                schema, split, replay, and shuffle gates
    render*.py                 physical data inspection
    DATASET_CARD.md             accepted derivation and limitations
  models.py                    feedback Gaussian action decoder
  windows.py                   explicit state/action temporal alignment
  train_prior.py               tokenizer, planner, and decoder fitting
  evaluate_prior.py            held-out state/action gates
  export_jax.py                PyTorch-to-JAX export and parity
  prior.py                     frozen pure-JAX prior
  env.py                       Demo A task plus causal prior state
  policy.py                    frozen base plus zero residual actor
  wrappers.py                  plan refresh and exact KL reward pair
  train.py                     H0/H1/H2 PPO entry point
  evaluate.py                  shaping-disabled rollout evaluation
  gait_metrics.py              validation-only four-limb diagnostics
  visualize_speeds.py          one-compilation command sweep
  render_speed_comparison.py   cached-scene tiled video renderer
  experiment/DECISIONS.md      append-only empirical record
```

## 12. Claim boundary

- Demo H shows generative body/action pretraining followed by RL adaptation in
  one physical morphology.
- It does not infer biological torque or reproduce GPC.
- It does not establish physical or neural similarity to a rodent.
- Its retargeted data contain visible cadence and morphology artifacts; RL
  adaptation is motivated partly by those imperfections.
- Naturalness diagnostics validate behavior but never enter the loss.
- Multiple policy seeds and matched scratch/warm-start arms are required before
  making an algorithm-level claim.
