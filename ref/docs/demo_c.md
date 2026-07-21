# Demo C — virtual-rodent world-action model + reinforcement learning

_Revised and implemented 2026-07-19. Companion to
[WORKSHOP_PLAN.md](WORKSHOP_PLAN.md), the completed `demo_a/` and `demo_b/`, and
[dataset.md](dataset.md)._

> **Presentation status.** Retained as an implemented research reference. The
> accepted core workshop presents Demos A, B, F, and H; see
> [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) and [demo_h.md](demo_h.md). The earlier,
> more research-oriented Demo C proposal is preserved in
> [`archive/demo_c_prev.md`](../../archive/demo_c_prev.md).

## 1. The pedagogical contract

The audience starts with three definitions:

> **Self-supervised learning (SSL)** learns from targets constructed from the data
> itself. No person labels the correct target.

> **Reinforcement learning (RL)** changes a policy so its actions obtain more future
> reward. It receives scalar consequences, not correct target actions.

> **A world-action model loop** uses a learned action-conditioned world transition to
> answer “what happens if I do this?”, then uses a policy to answer “what should I do?”

Demo A already teaches PPO on a Brax quadruped. Demo B teaches a convolutional
autoencoder and predictive rodent motion model. Demo C should therefore introduce no
new large algorithm. It freezes Demo B's world factor and lets the familiar PPO
algorithm interact with it.

The two learning signals remain separate and visible:

```text
L_SSL = MSE(predicted future motion, recorded future motion)
J_RL  = expected sum of food-reaching rewards
```

The world model never sees reward. PPO never sees a target action. PPO does not
backpropagate through the world model.

## 2. What “WAM” means here

Demo C implements the paradigm of **Section 4.1 of
[WorldModel.pdf](../papers/WorldModel.pdf), “World Model for Reinforcement Learning”**: a
policy is optimized *inside* a frozen learned world used as a simulator, improving from
imagined rollouts by maximizing expected return (survey eqs 16–18) rather than regressing
actions from data. It is the survey's *first-level* variant — the world model stays frozen
instead of being co-evolved with the policy — so the closest published point of reference
is a frozen-simulator method such as DiWA, not the second-level world-model/policy
co-training loop (World-VLA-Loop, WoVR).

The factorization only borrows its *vocabulary* from Section 3.1's joint predictive-control
distribution and its marginals. Demo C implements the smallest explicit factorization:

```text
p(z_{t+1:t+8}, a_t | z_{t-7:t}, g_t)
  = πθ(a_t | g_t, body_t, [context_t])    # policy model             (survey eq 5)
    pφ(z_{t+1:t+8} | z_{t-7:t}, a_t)       # controllable world model (survey eq 7)
```

This is deliberately a **policy × controllable-world-model** pair, *not* an
inverse-dynamics policy (the survey's Section 3.2, eq 8): PPO never regresses the action
that connects two observed latents, it maximizes food-reaching reward through imagined
transitions. The one Section 3.2 flavor is the optional `context_t` term — feeding the
frozen predictive latent into the policy input is the VPP/Video2Act style of
predictive-feature conditioning — but it rides on top of the RL loop, and the goal-only
versus WAM ablation in §4.4 exists precisely to test whether that conditioning changes the
result.

This is a **pedagogical state-space analogue**, not a claim to reproduce a unified
video-action transformer, mixture-of-experts backbone, or current WAM state of the art.
Keeping the factors explicit is the feature: a new graduate can point to the blue SSL
arrow and the orange RL arrow.

## 3. Why Demo C uses the Demo B rat

The earlier plan reused Demo A's Fetch quadruped. The neuroscience goal makes that the
wrong body:

- the local recordings contain rat pose and rat neural activity;
- Demo B already supplies a causal rat motion tokenizer and an action-conditioned
  predictor;
- the repository already has a physical MIMIC-MJX rat and a frozen joystick policy.

The actual Demo A network cannot be evaluated fairly on rat recordings: it expects a
different body, observation convention, action space, and task. Demo C therefore uses
a **matched goal-only rodent PPO** as the RL-only/Demo-A-like comparator. It differs
from WAM+RL only by the predictive context in its observation.

This means the talk has two levels of comparison:

1. Demos A/B/C are a teaching progression, not a controlled ablation.
2. Goal-only versus WAM-context PPO inside Demo C is a controlled rodent ablation.

## 4. The implemented loop

```text
REAL CONTINUOUS RODENT RECORDING

64 frames ── frozen causal MotionVAE ──> 16 latent tokens
             first 8 + root command ──> SimpleTrans ──> predicted last 8
                         recorded last 8 ──────────────> SSL MSE target


SHORT LEARNED ENVIRONMENT

history ──> frozen predictive context ───────────────┐
goal/body state ─────────────────────────────────────┼─> PPO policy ─> action
                                                    │                 │
history + action ─> frozen predictor ─> decoder ─> body displacement  │
                                                    │                 │
food progress/reach reward <────────────────────────┴─────────────────┘


REALITY CHECK

frozen navigator ─> high-level velocity command ─> frozen joystick ─> MJX rat
```

### 4.1 World factor: predictive SSL

The frozen MotionVAE is Demo B's convolutional autoencoder. A standard six-layer
Transformer encodes eight 80-ms latent tokens and a small MLP predicts the next eight
tokens from a three-dimensional egocentric displacement/turn command.

```text
history: 8 × 16       command: [forward, lateral, turn]
context: 192          target:  8 × 16 recorded future latents
```

The implementation first tested the bundled Demo B transition. Its small original
training scope did not pass every held-out session gate. Following the frozen-metric
practice in `canvas/misc/autoresearch.md`, the architecture and objective were left
unchanged while data coverage was broadened:

- 12 complete training sessions: two each from art, bud, coltrane, duke, freddie,
  and gerry;
- two complete checkpoint-selection sessions;
- four untouched final sessions: two DLS/coltrane and two MC/freddie.

Every sample is a genuine contiguous 64-frame window. Motion-mapper labels are allowed
to flicker, but windows are never stitched. The frozen checkpoint is the best
validation step, not the last training step; the convergence probe found that step 500
was best and later steps overfit.

The final gate is normalized skill over latent persistence:

```text
skill = 1 - MSE(world prediction) / MSE(repeat last latent)
```

It passed all four untouched sessions, with session-balanced mean skill +39.5%.

### 4.2 Dream transition

This transition is the learned simulator of the survey's Section 4.1: the frozen predictor
supplies the imagined state transition (survey eq 16) and PPO improves purely from these
imagined rollouts. Reward and termination are the known analytic signals of §4.3 rather
than a learned reward head — the survey treats these as optional additions to the simulator.

One transition predicts 0.64 s of motion. The model decodes the old history plus the
predicted future. Local root velocity and orientation deltas are integrated to update
the virtual rat's position, yaw, and measured body velocity.

Crucially, the environment does **not** copy the requested command into the next state.
Doing so would let PPO optimize an identity function rather than the learned world.
Non-finite or physically extreme predictions terminate as invalid failures and are
reported; the accepted runs have zero invalid transitions.

### 4.3 Food-reaching task

Each episode places one food target 0.35–0.75 m away in the rat's forward semicircle.
The policy has eight 0.64-s decisions. Its two bounded actions map to forward
displacement and turn commands already supported by Demo B.

```text
reward = 10 × reduction_in_distance
       + 1 × reached_food
       - 0.01 time
       - 0.01 turn²
       - invalid_penalty
```

The reward is deliberately known and readable. There is no learned reward head,
planner, curiosity bonus, replay buffer, or end-to-end world/policy fine-tuning.

### 4.4 Matched PPO conditions

Both policies use a two-layer, 128-unit tanh actor-critic and the same tanh-squashed
Gaussian action distribution, PPO hyperparameters, rollout data, and seeds.

| input | goal-only PPO | WAM-context PPO |
|---|:---:|:---:|
| egocentric food x/y + distance | yes | yes |
| body forward/lateral/yaw velocity | yes | yes |
| previous action | yes | yes |
| frozen 192-D predictive context | no | yes |

The frozen budget is 786,432 dream steps with 256 environments and seeds 0, 1, and 2.
A longer convergence probe selected the budget before the reportable comparison.
Success is evaluated deterministically on 1,024 fixed held-out episodes. The WAM and
goal-only success means are 0.6445 and 0.6393; their +0.0052 difference is smaller than
the predeclared two-standard-deviation noise floor of 0.0088. They are functionally
matched, which makes the later representation comparison easier to interpret.

### 4.5 Zero-shot physics check

At deployment, no network is updated. The navigator's high-level command is held for
0.64 s by the existing frozen MIMIC joystick, which produces intentions for the MJX
rat. This is an honest test of navigation transfer across the dream/physics boundary,
but the scope must be said aloud:

> Demo C learns the high-level navigator in dreams; it reuses an existing physical
> locomotion primitive.

The independently measured response-curve `calibrated` bridge is reportable: it
inverts the joystick's observed low-speed dead zone to realize requested displacement,
without changing the policy. The uncalibrated `raw`, rejected scalar `inverse_gain`,
and model-decoded bridges remain labelled diagnostics and are not silently substituted.

On the frozen eight-goal suite, the response bridge gives 7/8 success for goal-only
and 4/8 for WAM, with no falls. Raw transfer was 0/8 and 2/8. The fact that WAM is
matched in dreams but worse in physics is retained as a model-reliance/dream-to-real
gap, not tuned away or presented as a functional advantage.

## 5. Neural-population comparison

### Question

For matched real rat movements, do frozen representations explain held-out DLS/MC
population activity, and does the WAM+RL policy resemble the data more than a matched
RL-only policy?

The comparison includes:

- raw 281-D kinematics as a nuisance/reference ceiling;
- Demo B's 16-D autoencoder latent;
- Demo B's 192-D predictive context;
- the final 128-D hidden state of all three goal-only PPO seeds;
- the final 128-D hidden state of all three WAM PPO seeds.

Policy seeds are averaged within a session before sessions are balanced. Units never
cross sessions.

### Alignment and leakage controls

- Pose, representations, and spikes stay on the genuine continuous 80-ms token grid.
- Spike counts sum four 20-ms bins and use only the dataset's `active_units` mask.
- A two-second recorded future bearing acts as pseudo-food direction. Food distance is
  fixed at the midpoint of the trained range, avoiding the rejected near-zero-goal
  out-of-distribution probe.
- Splits use 60-s blocks with 5-s gaps and require the entire eight-token history and
  future-goal window inside one block.
- PCA is fit on training rows only and fixed to at most 16 dimensions for every family.
- The encoding metric is population bits/spike from a Poisson GLM against a train-rate
  null. A 20-s circular shift stays within each split block.
- The RSA metric uses nine speed × turn conditions, crossvalidated/shrinkage distances,
  and a condition-label permutation test.
- `loco` is primary. `all_matched` draws the same number of rows from the full behavior
  distribution so subset size cannot explain the result.

### Result and correct claim strength

On locomotion, the WAM+RL policy has 0.0191 shift-corrected population bits/spike and
RSA ρ = 0.701. The matched RL-only policy has 0.0082 and ρ = 0.575. Demo B's predictor
has 0.0193 and ρ = 0.710.

Therefore the workshop may say:

> The action-conditioned SSL representation survives inside a policy that reaches
> food. It is descriptively more neural-like than the matched RL-only policy while
> preserving, rather than improving on, Demo B's predictive representation.

It may **not** say that Demo C is statistically superior to Demo B or RL-only. There
are only four independent sessions; the smallest possible two-sided exact
sign-permutation p-value is 0.125. The all-behavior n-matched control also shows that
slow temporal structure can dominate raw encoding scores. This is a workshop
demonstration and a hypothesis generator, not a publication-level neural claim.

## 6. Five workshop beats

1. **Recall Demo B:** reconstruction is self-supervised because the input constructs
   its own target.
2. **Shift a rat trajectory:** history plus action predicts the recorded future; show
   the persistence null and the 0.64-s horizon plot.
3. **Freeze the blue network:** put PPO beside it and point separately to reward/return.
4. **Run matched dreams:** show goal-only and WAM curves rising to essentially the same
   food-reaching success.
5. **Return to rat data and physics:** render zero-shot MJX control, then show the
   neural table with its caveat that WAM+RL preserves rather than exceeds the predictor.

The final audience sentence is:

> **SSL learned a predictive rat-motion representation from what happened next; RL
> learned which imagined actions reached food; together they made a goal-directed
> controller that retained that representation.**

## 7. Frozen gates

| gate | criterion | result |
|---|---|---|
| SSL validity | positive skill over persistence on every untouched session | pass |
| numerical validity | finite transitions and zero invalid rate | pass |
| RL validity | both learned policies beat the random controller | pass |
| matched function | success difference no larger than `2 × max(seed SD)` | pass |
| physical loop | frozen policy executes in MJX with no fine-tuning | measured by `deploy_physics.py` |
| neural feasibility | strict split, shift null, RSA, four full DLS/MC sessions | pass |
| stronger neural claim | exact paired evidence that Demo C beats all baselines | not established |

## 8. Implementation and reproducibility

The runnable commands, file map, measured tables, and artifact locations are in
[`demo_c/README.md`](../../demo_c/README.md). The append-only research narrative is in
[`demo_c/experiment/DECISIONS.md`](../../demo_c/experiment/DECISIONS.md).

The implementation follows the useful parts of `canvas/misc/autoresearch.md`: frozen
metrics and budgets, convergence probes, a multi-seed noise floor, one-change
iterations, rich diagnostics, genuine temporal continuity, atomic caches, strict
long-block splits, and explicit keep/reject decisions. Neural data are a final test,
not a hyperparameter-tuning signal.
