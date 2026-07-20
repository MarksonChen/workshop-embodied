# Workshop experiment plan — SSL + RL for embodied neuroscience

_Created 2026-07-18; revised for the three-demo A/B/E workshop on 2026-07-19.
Companion to [PROJECT_STATE.md](PROJECT_STATE.md) and the detailed
[Demo E plan](demo_e.md). Demos [C](demo_c.md) and [D](demo_d.md) are retained
as research references, not presented in the core workshop._

Audience: new computational-neuroscience graduate students. Optimize for clear
definitions, visible learning signals, and one honest controlled comparison.

## 1. Thesis

SSL and RL provide complementary constraints:

| | self-supervised learning | reinforcement learning |
|---|---|---|
| realism | distributional: resemble patterns in recorded data | functional: cause useful outcomes in an environment |
| source | data-driven targets constructed from the recording itself | task-driven scalar reward produced by interaction |
| embodied loop | learned internal model of brain/body motion | body–environment action and consequence |

Their combination is the brain–body–environment story:

> **A data-trained internal motion prior says what behavior is plausible; task
> reward says what behavior works; PPO learns a physical policy constrained by
> both.**

“Brain–body” is a computational interpretation in this workshop. Demo B uses
real motion but no spikes, so the workshop does not claim that its network is a
validated biological brain model.

## 2. The three-demo arc

```text
distributional realism
        ^
        | Demo B: conditional motion model
        |                Demo E: frozen motion likelihood + physical PPO
        |
        |                                      Demo A: task-reward PPO
        +--------------------------------------------------------------> function

        data only            data + interaction             interaction only
```

The points are a teaching picture, not one controlled experiment: Demo A uses a
Brax creature-like quadruped while B/E use the MIMIC skeletal rodent. The valid
causal comparison is inside Demo E, where task-only and task+prior PPO differ by
one frozen reward term.

Use one sentence to bridge every demo:

> **Demo A learns what works; Demo B learns what looks plausible; Demo E asks
> one physical policy to do both.**

## 3. Demo A — define reinforcement learning

Status: **complete in `demo_a/`; training is workshop-scale.**

- Body/task: Brax v1 Fetch, a creature-like quadruped, trained either to reach a
  target or sustain a forward speed.
- Algorithm: standard PPO from random initialization.
- Supervision: a scalar task consequence; there are no target actions.
- Visible lesson: checkpoint deciles show behavior changing as expected return
  rises.
- Useful imperfection: task success does not guarantee a natural gait. The
  difference between “works” and “resembles real motion” motivates Demo B.

Definition on the board:

\[
\max_\pi\;\mathbb E_{\tau\sim\pi}\sum_t\gamma^t r_t^{\rm task}.
\]

Do not use Demo A as the neural or same-body baseline for Demo E; its body is
different. Demo E trains its own beta-zero MIMIC-rodent control.

## 4. Demo B — define self-supervised conditional learning

Status: **complete in `demo_b/`, including the frozen likelihood interface used
by Demo E.**

- Data/body: the behaviorally validated strict-locomotion subset of the first
  eight Coltrane sessions, expressed in the 74-coordinate MIMIC skeletal
  contract with the original 281-D motion representation.
- Tokenizer: a causal convolutional MotionVAE converts 50 Hz motion into 16-D
  tokens at 12.5 Hz.
- Conditional predictor: a standard Transformer reads eight past tokens and an
  egocentric motion command and predicts future tokens.
- Target construction: shift the same continuous recording forward; no behavior
  label, action label, task reward, or environment interaction is required.
- Artifact: reconstruct, generate, and steer a kinematic rat; it can look like a
  locomotor trajectory without solving contact-rich physical control.

Definition on the board:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D}
\log p_\phi(w\mid h,c).
\]

The current MSE transition already has the simplest probabilistic reading:

\[
p_\phi(w\mid h,c)=\mathcal N(w;\mu_\phi(h,c),\sigma^2I).
\]

Demo B exposes `log_prob_next` and freezes a training-residual `sigma`. This
makes the bridge to Demo E one line rather than introducing a discriminator or
a second generative architecture. The exact feature/split changes, measured
eligibility results, and acceptance gates are in [demo_e.md](demo_e.md).

## 5. Demo E — combine the two signals

Status: **pipeline-v6 is implemented in `demo_e/`; its replicated conditional
prior passes the frozen source and physical-transfer gates, but no paired E0/E1
result has passed the behavioral gates.** The first ten-minute E1 diagnostic
learns an upright two-legged stance rather than locomotion. The canonical
specification and measurements are in [demo_e.md](demo_e.md).

### Task

Use MIMIC-MJX `RodentJoystick`: track commanded forward velocity and yaw rate in
physics with the skeletal rodent. A standard PPO policy observes the flattened
56-D task state and emits a 16-D intention. The published TRACK-MJX imitation
decoder is frozen and maps intention plus proprioception to 38 torques at 100
Hz. Both E0 and E1 share this motor infrastructure; PPO initializes only the
high-level policy from scratch.

There is no target action, adversarial discriminator, world-model planning
loop, or trainable second policy.

### Objective

\[
\max _\pi \mathbb E_\pi\left[
\sum_t\gamma^t\left(
r_t^{\rm task}+\beta\frac{1}{\dim(w_t)}
\log p_\phi(w_t\mid h_t,c_t)
\right)\right],
\]

where `p_phi` is the frozen Demo B model. The realized physical motion after the
policy acts supplies `w_t`; the past-only physical token history supplies `h_t`;
and joystick velocity/yaw is converted to Demo B's 0.62 s egocentric
displacement command `c_t`.

PPO and the frozen decoder act at 10 ms. Physical features are sampled at 20
ms and one likelihood token is emitted every 80 ms. PPO reward exposes two
named columns:

```text
task reward:   did the rat track the command and remain physically viable?
prior reward:  was the motion just produced likely under real rat motion?
```

The prior is frozen and receives no gradient. PPO is still RL; SSL trained the
reward model beforehand.

### Same-body control

Train paired arms under identical initialization, native RodentJoystick reset,
command stream, decoder, policy, PPO settings, and physical-step budget:

| arm | `r_task` | frozen `log p_phi` |
|---|---:|---:|
| E0 task-only | yes | no (`beta=0`) |
| E1 task+prior | yes | yes |

This is the result to show, rather than comparing E directly with the
different-bodied Demo A.

### Command grid

The fixed diagnostic grid is 0, 0.10, 0.20, and 0.30 m/s straight, plus 0.30
m/s with ±0.75 rad/s yaw. The 0.30 m/s cell is retained because it is the first
command at which the reproduced 52M reference shows unequivocal locomotion.
Every cell must be labelled with whether its displacement command lies inside
Demo B's empirical support.

### Evaluation

Show both axes before any combined score:

- **Functional `F`:** fixed-command velocity/yaw tracking, 10 s survival,
  lateral drift, energy, and optional waypoint success.
- **Distributional `R`:** raw frozen conditional likelihood, reported only on
  post-warm-up tokens, plus stride/phase/skate/joint diagnostics when available.

The success claim requires E1 to preserve E0 function and survival while
improving raw motion likelihood beyond seed noise. A higher transformed reward
alone is not success, and likelihood is not claimed to be biological realism.

Pipeline v6 repairs the prior bridge: source-constant channels are masked, yaw
commands are integrated into a planar 0.62 s arc, and a validation-selected
single-negative conditional scorer is exported separately from Demo B's
behavioral generator. It clears the frozen transfer gate on two training seeds
(moving top-1 0.629/0.672 and positive matched margins). The reward transform
is frozen to `[-1.5, -0.75]` nats per dimension with `beta=1`.

The first 9.83M-transition E1 run is still not a success result. It stands
almost upright on two legs, averages about 0.013 m/s under the 0.30 m/s command,
and crosses the torso-orientation termination threshold around 3.04 s. The
scorer ranks the confirmed gait above this stance, but its incremental shaping
advantage is too small to bootstrap locomotion at this budget. A paired,
transition-matched E0/E1 result remains required.

## 6. Measured workshop runtime

Demos A and B train in workshop time. The successful upstream joystick
reference requested 50M transitions and reached 52,428,800 after PPO batching.
On the H100 it took about 40 minutes of training plus 7 minutes of evaluation,
or 45–47 minutes end to end. It used 8,192 environments, 100-Hz torque control,
and 1024/512/256 actor and critic layers.

The confirmed native-reset clip at 0.30 m/s is steady. Checkpoint evaluation
shows stable standing through 39.3M and forward locomotion at 52.4M, placing
gait onset roughly 34–45 minutes into the run. No five-minute convergence claim
exists. Pipeline-v6 E1 reached 9.83M transitions in 582.5 s (9.71 min) and
learned upright standing, not gait. Its exhaustive four-checkpoint evaluation
took a separate 495.9 s. Use precomputed checkpoints for workshop playback and
always report compilation, training, and evaluation separately.

## 7. Workshop run of show

1. Show Demo A checkpoint deciles and define RL: reward follows interaction;
   actions are never supplied as labels.
2. Slide a window over real motion in Demo B and define SSL: the shifted future
   is the target.
3. Rewrite Demo B's MSE as Gaussian log likelihood, then visibly freeze the
   model.
4. Add `beta * log_prob` to Demo A-shaped PPO reward in the MIMIC rodent.
5. Once a paired result clears the gates, render E0 and E1 side by side under
   the same straight and turning commands; until then, label the ten-minute
   upright-standing clip as a diagnostic failure, not the workshop result.
6. Show task tracking and raw motion likelihood as separate plots, then state
   the likelihood and biological-realism limitations.

The minimal board diagram is:

```text
recording -- self-generated future target --> Demo B p_phi -- freeze --+
                                                                      |
command --> PPO --> intention --> frozen decoder --> physical rodent --+--> prior reward
                    ^                                |
                    +------------- task reward ------+
```

## 8. What not to claim

- Demo B's network is not a measured biological brain.
- Optimizing a learned likelihood does not guarantee biological realism; raw
  likelihood, task behavior, and failure diagnostics must remain separate.
- Demo E does not show that SSL learns actions. PPO learns actions from return.
- Demo E uses MIMIC's published imitation decoder as frozen shared motor
  infrastructure. It does not train or present that decoder as Demo B.
- The 0.30 m/s training/evaluation cell must be labelled against Demo B's
  empirical command support.
- The physical controller uses raw torque actuators at 100 Hz; PPO emits a
  higher-level 16-D intention, not torque directly.
- Demo A/B/E are not themselves a controlled three-way ablation. E0/E1 is.
- No hidden-activity/neural claim follows from better kinematics. Any later
  aligned-neural analysis is exploratory and cannot tune the model.

## 9. Demos C and D

Demos C and D are no longer in the core presentation and should not consume
workshop time:

- [Demo C](demo_c.md) remains a completed world-model/PPO and exploratory
  aligned-neural study. It is useful for frozen-model provenance, JAX/PyTorch
  boundaries, and careful neural evaluation.
- [Demo D](demo_d.md) remains a one-stage hindsight/imitation experiment whose
  negative command intervention exposed conditional-identifiability and
  termination problems. It motivates independent command sampling, physical
  termination, and a same-state intervention test in E.

Keep their code and reports as optional reading. Do not delete the evidence and
do not present their rejected results as Demo E baselines.

## 10. Reproducibility and iteration

Apply `canvas/misc/autoresearch.md`:

- freeze data IDs, objective, command grid, beta rule, nulls, budget, scalar,
  and gates before comparison;
- validate real > shuffled/noised likelihood and command sensitivity before PPO;
- make beta-zero reproduce task-only transitions and rewards;
- establish multi-seed noise before accepting a prior improvement;
- log task/prior reward, behavior, throughput, memory, and per-command metrics;
- change one major block per experiment and maintain an append-only decision
  log;
- never select an intermediate checkpoint or model on final-test behavior.

The workshop succeeds if a new graduate can say:

> **Demo B constructed its target by shifting recorded motion, so it was
> self-supervised. Demo E froze that learned likelihood and used it as one reward
> term. The joystick task reward still came from physical interaction, and PPO
> learned the actions.**

## 11. Literature anchors

- Aldarondo et al. 2024, *A virtual rodent predicts the structure of neural
  activity across behaviors*, Nature, DOI 10.1038/s41586-024-07633-4.
- Merel et al. 2020, *Deep neuroethology of a virtual rodent*.
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*.
- Peng et al. 2021, *AMP: Adversarial Motion Priors for Stylized Physics-Based
  Character Control*, as related context; Demo E uses an explicit frozen
  conditional likelihood rather than an adversarial discriminator.
