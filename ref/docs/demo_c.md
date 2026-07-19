# Demo C — minimal WAM + RL: learn the next step, then learn what to do

_Revised 2026-07-19. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md),
[demo_a.md](demo_a.md), and the completed code in [`demo_a/`](../../demo_a/) and
[`demo_b/`](../../demo_b/). This supersedes the research-oriented
[demo_c_prev.md](demo_c_prev.md)._

**Status: plan only. First priority: make the definitions of SSL and RL visible.**

## 1. What a student should learn

Demo A teaches:

> **RL:** PPO changes a policy so that its actions produce more cumulative reward.

Demo B teaches:

> **SSL:** a model learns from targets constructed from the data itself; in an
> autoencoder, the input is also the reconstruction target.

Demo C should teach exactly one new idea:

> **WAM + RL:** use self-supervised prediction to learn how actions change the world,
> then let PPO learn what actions to take by interacting with that learned world.

By the end, a student should be able to point to the two different learning signals:

- the world model minimizes **next-state prediction error**; its target comes from the
  next row of a recorded trajectory;
- the policy maximizes **return**; PPO receives rewards, not target actions.

Keeping those objectives visually and programmatically separate is more important than
model novelty or final task performance.

## 2. The whole demo in one diagram

```text
REAL SIMULATOR (data collection)

    state x_t ── action a_t ──> state x_{t+1}
         \___________________________/
              shift the trajectory
                       │
                       ▼
SSL: train W_phi(x_t, a_t) ≈ x_{t+1}


LEARNED SIMULATOR (policy training)

    x_t ──> PPO policy pi_theta ──> a_t ──> frozen W_phi ──> x_hat_{t+1}
                  ▲                                      │
                  └──────── known task reward ───────────┘
```

Workshop line:

> **SSL learns “what happens if I do this?” RL learns “what should I do?”**

## 3. Connection to the assigned paper

Section 3.1 of [WorldModel.pdf](../papers/WorldModel.pdf) describes policy models and
controllable world models as different queries of one joint predictive-control
distribution. Demo C uses the simplest explicit factorization of that idea:

```text
p(x_{t+1}, a_t | x_t)
    = pi_theta(a_t | x_t) p_phi(x_{t+1} | x_t, a_t)
      └── action / policy ┘ └── action-conditioned world model ┘
```

In this plan, **minimal WAM** means this factorized, closed world-and-action loop: the
world factor is the transition model and the action factor is the PPO policy. It does not
mean a unified video-action transformer. The factors remain separate so beginners can see
which one is trained by SSL and which one is trained by RL.

Section 4.1 supplies the execution recipe: freeze the learned transition model, use it as
an interactive simulator, and optimize the policy on imagined transitions. We use PPO
because students have already learned it in Demo A. Section 3's single-backbone,
mixture-of-experts, video, and latent-policy variants are useful context, not requirements
for this demonstration.

## 4. Minimal system

### Environment and task

Reuse Demo A's Brax `FetchRun` quadruped, action space, locomotion task, reward, and PPO
implementation. Demo C changes only the source of the transition:

- Demo A: `next_state = real_brax.step(state, action)`
- Demo C: `next_state = learned_world.step(state, action)`

Initialize the policy from an **early Demo A decile checkpoint** that can stand and move
but has not solved the task. Demo A has already shown PPO from scratch; Demo C should show
the new idea—PPO **post-training inside a learned simulator**—without asking a random
policy to leave the world model's training distribution immediately. Select the earliest
decile that stays upright for the measured dream horizon, and record that choice before
post-training. This also matches Section 4.1's reinforcement-post-training framing. The
fully trained Demo A policy remains an upper reference, not the initialization.

### Teaching state

Use Demo A's 101-D observation plus the three quantities needed to evaluate the existing
task visibly and without a learned reward model:

```text
x_t = [observation_t, forward_speed_t, upright_t, torso_height_t]
```

The real environment can provide this vector during collection and deployment. The
learned model predicts the next vector. The dream environment then computes Demo A's
existing reward and fall condition from the predicted speed, upright value, height, and
the current action. The PPO policy still sees only Demo A's original 101-D observation;
the three appended values are simulator state used for reward and termination. Thus the
policy input remains the same between Demo A and Demo C.

This boundary is deliberate:

- **learn dynamics from data** with SSL;
- **keep the task reward hand-written** and identical to Demo A;
- **use the reward only for PPO**.

Do not train reward or termination heads in the first version. Calling a combined
`(next state, reward, done)` loss “pure SSL” would blur the lesson.

### World model

Use one deterministic residual MLP:

```text
delta_hat = MLP(normalize([x_t, a_t]))
x_hat_{t+1} = x_t + denormalize(delta_hat)
L_SSL = mean((x_hat_{t+1} - x_{t+1})^2)
```

Two hidden layers are enough for the first attempt. Normalize each state and action
feature, clip actions to the real environment's limits, and fit only one-step transitions.
Implement it in JAX so the learned step remains compatible with Demo A's vectorized,
jitted environment interface. There is no VQ bottleneck, diffusion model, transformer,
planner, ensemble, or differentiation through Brax.

Why not copy Demo B's convolutional architecture? Demo B's input is structured motion,
where convolution is sensible; Demo A exposes a compact state vector. The continuity is
in the **self-supervised objective**:

```text
Demo B: x                 -> reconstruct x
Demo C: (x_t, action_t)   -> predict x_{t+1}
```

This is a feature, not a compromise: students learn that “self-supervised” describes how
targets are obtained, not whether the network contains convolutions.

## 5. Four-stage demonstration

### Stage 0 — collect transitions in the real simulator

Build `D = {(x_t, a_t, x_{t+1})}` from existing Demo A checkpoints. Use rollouts from
several training deciles, add modest action noise, and include some random-policy rollouts.
The mix supplies standing, falling, partially learned, and running transitions; a purely
random quadruped falls too quickly to teach useful locomotion dynamics.

Split by complete rollout, not by individual transitions, so adjacent frames cannot leak
between train and validation sets. Rewards may be logged for later comparison, but they
are not targets for `L_SSL`.

Teaching moment: construct the target live by shifting one trajectory by one row. No one
annotates “the correct next state”; the trajectory supplies it.

### Stage 1 — train and inspect the world factor

Train the MLP on one-step prediction. Before any RL, show two held-out diagnostics:

1. one-step error compared with the trivial baseline `x_hat_{t+1} = x_t`;
2. a short open-loop rollout plotting predicted and real forward speed, upright value,
   and one representative joint feature.

The plot should visibly stay close at first and then drift. That is not hidden: it
motivates short imagined rollouts and teaches that a learned simulator is approximate.

### Stage 2 — run PPO in short dreams

Freeze the world model. Wrap it behind the same `reset` / `step` interface used by Demo A
and run the same PPO implementation from the selected early-checkpoint initialization.

- Reset from recorded healthy states, including the standard standing start.
- Roll the world model for a short fixed horizon, initially 20 steps.
- At the horizon, end the imagined episode and reset to another recorded state.
- Select the final horizon from the held-out rollout plot; do not roll farther merely to
  obtain more synthetic data.

The phrase for the audience is **“dream briefly, then wake up.”** It conveys the reason
for the resets without introducing a second model-based RL algorithm. PPO treats the
world model as an environment; it does not backpropagate through it.

### Stage 3 — reality check

Freeze the policy and replace `learned_world.step` with the real Brax `FetchRun.step`. No
fine-tuning is allowed for the first evaluation. Render the quadruped and report:

- return, forward speed, upright time, and fall rate in the learned simulator;
- the same four quantities in the real simulator;
- the unchanged early checkpoint as the before-post-training baseline;
- random policy and fully trained Demo A PPO as lower and upper reference points.

The important comparison is the **dream-to-real gap**, not whether Demo C beats Demo A.
If the policy performs well only in the dream, label that as model exploitation rather
than as successful transfer.

After showing the zero-shot result, an optional short real-simulator PPO fine-tune may be
used as a clearly labeled extension. It demonstrates that imagined practice can provide
an initialization while real feedback corrects model error; it is not part of the minimum
demo.

## 6. Success gates

The minimum demo is complete only when all four gates pass:

1. **SSL is real:** held-out one-step prediction beats the no-change baseline.
2. **The trust horizon is measured:** the chosen imagination horizon lies inside the
   region where held-out rollout error remains acceptably small.
3. **RL is real:** PPO improves return over its early-checkpoint initialization inside the
   frozen world model.
4. **The loop closes:** without real-environment fine-tuning, the post-trained policy
   improves real-simulator locomotion over that unchanged starting checkpoint.

Gate 4 need not approach Demo A's fully trained policy. It does need to improve over the
seed checkpoint and remain above random; otherwise the workshop has demonstrated model
exploitation, not yet WAM + RL transfer. The first remedies are broader transition
coverage and a shorter dream horizon—not adding a research-method stack. Starting PPO
from random can remain an optional stress test after the post-training demo works.

## 7. What to show in the workshop

Keep the live narrative to five beats:

1. Recall Demo B's autoencoder and its data-derived reconstruction target.
2. Shift a rollout by one step and add the action: this creates the world-model training
   pairs.
3. Show predicted versus real short trajectories: the learned world works locally and
   drifts with time.
4. Load an early Demo A checkpoint, change one environment switch from `real` to `dream`,
   and watch the familiar PPO curve rise.
5. Switch back to `real` and render the frozen dream-trained policy.

On the final slide, color only two arrows: blue for prediction error (SSL) and orange for
reward/return (RL). Avoid a taxonomy of contemporary WAM architectures.

## 8. Implementation layout and order

```text
demo_c/
  README.md          commands, diagram, and expected workshop outputs
  collect.py         Demo A rollouts -> train/validation transition files
  world_model.py     residual MLP, normalization, SSL training, drift plot
  dream_env.py       frozen model behind the Demo A-style environment interface
  train_policy.py    the existing PPO configuration with real|dream environment switch
  evaluate.py        dream-vs-real metrics and real-simulator render
```

Build in this order:

1. Implement the teaching-state wrapper and verify its reward/done calculation matches
   `FetchRun` on recorded real transitions.
2. Collect decile-plus-noise data and produce the held-out prediction plot.
3. Add the dream environment and test reset, action clipping, reward, and termination.
4. Reuse Demo A PPO unchanged at the algorithm level.
5. Run the zero-shot reality check and record all success-gate metrics.

## 9. Explicitly out of scope

The following ideas from [demo_c_prev.md](demo_c_prev.md) remain possible research
directions, but are removed from the workshop demo: differentiable MJX, mixed-mode
autodiff, LAPO, an intention-IDM, VQ latents, HILP, MTM, TD-MPC2 losses, learned planning,
real-rat mocap alignment, and neural-recording comparisons.

Likewise, this minimal demo does **not** establish that its behavior matches the real-rat
distribution merely because its world model fits self-collected simulator data. It teaches
the mechanics of combining SSL and RL. A later rodent-data extension can test the stronger
distributional and neuroscience claims after the basic lesson works.
