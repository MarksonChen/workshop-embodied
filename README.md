# embodied

Neuromechanical rodent control experiments built on
[MIMIC-MJX](https://mimic-mjx.talmolab.org) (stac-mjx + track-mjx).

Demos A–J build a graduate workshop on reinforcement learning, self-supervised
learning, and generative pretraining followed by RL post-training. 

Demos A, F, and H becomes 
[`workshop_4.ipynb`](workshop_4.ipynb)

## Workshop walkthrough: from reward to a generative motor prior

[`workshop_4.ipynb`](workshop_4.ipynb) tells one story in three parts. Every part
controls the *same* ten-actuator Brax Fetch quadruped, but each moves the source of
"realism" somewhere new — first a hand-written reward, then recorded rodent motion,
then both at once. The clips below are speed sweeps of the actual trained models
(each is a grid of six independent rollouts).

### Part 1 — Functional realism from reward (Demo A)

Part 1 asks the isolated reinforcement-learning question — *which actions make future
reward larger?* — with no recorded motion at all. The environment is Brax Fetch, a
dog-like quadruped with ten actuated hinge joints; every 20 ms the policy reads a
101-number proprioceptive observation and emits ten bounded motor commands, and
physics returns the next state and a scalar reward. That reward is engineered by hand:
track a target forward speed, stay upright, and don't waste control effort, with a
fall ending the episode. Nothing in it describes a gait — cyclic locomotion has to
*emerge* because stepping is simply what earns return. A ~7,000-parameter 4×32 MLP
policy trained with PPO learns exactly that. Because one lucky rollout proves nothing,
the clip shows six independent environment resets of the same trained policy.

<video src="attachments/part1_demo_a_ppo_from_scratch.mp4" controls muted loop playsinline width="100%"></video>

*Six resets of the Demo A PPO policy on `FetchRun`. If the video does not play inline, open [attachments/part1_demo_a_ppo_from_scratch.mp4](attachments/part1_demo_a_ppo_from_scratch.mp4).*

### Part 2 — Distributional realism from retargeted rodent motion (Demo F)

Part 2 changes the source of realism from reward to data. We take Coltrane — a freely
behaving Long–Evans rat from Aldarondo et al. (2024), recorded at 50 Hz — retarget its
locomotion onto the *same* ten-joint Fetch body, and learn the **distribution** of that
motion with no reward and no physics. The model is self-supervised: given a window of
recent retargeted motion and a hindsight displacement command read from the clip's own
future, predict the motion that comes next. A causal tokenizer compresses motion into
16-D tokens and a small Transformer predicts the next token conditionally, so
maximizing future-motion likelihood reproduces rodent-derived, command-following Fetch
motion. It is still only moving a kinematic skeleton, though — nothing yet proves that
actuators could produce this motion while gravity and contacts push back. The clip is a
six-speed sweep (source-equivalent 0.05–0.30 m/s).

<video src="attachments/part2_demo_f_motion_prior.mp4" controls muted loop playsinline width="100%"></video>

*Demo F conditional motion across six commanded speeds. If the video does not play inline, open [attachments/part2_demo_f_motion_prior.mp4](attachments/part2_demo_f_motion_prior.mp4).*

### Part 3 — A generative motor prior plus RL (Demo H)

Part 3 joins the two halves into a brain–body–environment loop. First the
self-supervised prior is extended from predicting future *states* to predicting future
states **and** the Fetch controls that realize them (the controls are physics-derived
pseudo-labels from a transparent feedback controller replayed in the exact simulator —
not measured animal torques). That state-plus-action prior is frozen, and PPO trains a
small zero-initialized residual policy on top of it, maximizing task reward minus a
β-weighted KL penalty for leaving the prior — so naturalness pressure comes from the
learned distribution rather than a hand-tuned gait checklist. The clip is a six-speed
sweep of the β=0.10 fine-tuned capstone across its 1.5–4.0 Fetch-unit/s training range:
it tracks commands cleanly up to about 3.5 u/s (survival 1.0, four-limb stride gate
passing) and visibly degrades once the 4.0 command exceeds what the body can reliably
sustain. RL repairs task failures, but it cannot turn imperfect demonstrations into a
perfect biological model.

<video src="attachments/part3_demo_h_motor_prior_rl.mp4" controls muted loop playsinline width="100%"></video>

*Demo H (last fine-tuned β=0.10 checkpoint) across six commanded speeds. If the video does not play inline, open [attachments/part3_demo_h_motor_prior_rl.mp4](attachments/part3_demo_h_motor_prior_rl.mp4).*

### Recap

| part | information source | model | objective | realism |
|---|---|---|---|---|
| 1 · PPO locomotion | environment interaction | 4×32 policy MLP | maximize task return | functional |
| 2 · conditional motion | retargeted Coltrane locomotion | causal tokenizer + conditional Transformer | maximize future-motion likelihood | distributional |
| 3 · motor prior + PPO | states, controls, and environment | state predictor + action decoder + residual policy | task return minus prior KL | both |

Three ideas to carry out of the room: **RL is task-driven** (reward says what must work
in the environment), **SSL is data-driven** (the future supplies its own target and
hindsight command), and **pretraining and post-training play different roles** (the
prior proposes data-like motor behavior; RL changes only what the task and physics
require).

## The demos

**Demo A — Fetch PPO from scratch.** The workshop's reinforcement-learning baseline and
the engine behind Part 1. Standard PPO trains the unmodified ten-actuator Brax v1 Fetch
body from random initialization, with no motion clips, action labels, or learned prior;
the `FetchRun` task rewards sustained 3.0-unit/s forward motion, upright posture, and
low action magnitude, and terminates on a fall (a target-reaching task is kept as a
reward-design contrast under `--env fetch`). Cyclic gait is never specified — it emerges
because it maximizes return — and learning-decile snapshots plus stride and
high-frequency foot analyses make that emergence measurable. See
[`demo_a/`](demo_a/) and [`ref/docs/demo_a.md`](ref/docs/demo_a.md).

**Demo B — conditional self-supervised Coltrane motion.** Demo B learns only from
recorded rat motion on the MIMIC skeletal rodent: a causal convolutional VAE turns 50 Hz
skeletal motion into 16-D tokens at 12.5 Hz, and a small Transformer predicts the next
eight tokens from eight past tokens plus a hindsight displacement command drawn from the
same recording. There are no actions, rewards, or physics — future frames supply both
the command and the target. A fixed-σ Gaussian around the predicted mean makes
"maximize likelihood" exactly equivalent to "minimize the original MSE," which is the
probabilistic bridge Demo F later repeats on the Fetch body. The workshop reuses the
original, behaviorally validated Coltrane checkpoint unchanged. See [`demo_b/`](demo_b/).

**Demo C — a small world–action model plus PPO (research reference).** No longer part of
the live arc, Demo C combines the first two demos without inflating the lesson into a
model survey: a frozen Demo B predictor is a "world" factor answering *what happens if I
do this?*, while a small PPO actor–critic answers *what should I do?* in two matched
conditions — one seeing ordinary navigation state, the other also seeing the world
model's predictive context. PPO receives only rewards and the frozen world model gets no
PPO gradients. It is a deliberately factorized, state-space teaching analogue of the WAM
framing, not a unified video/action transformer. See [`demo_c/`](demo_c/).

**Demo D — one-stage hindsight-command imitation RL (research reference).** Demo D trains
a physical virtual rodent from random initialization without loading MIMIC's published
decoder: it turns the difference between two poses of an unlabeled trajectory into an
egocentric command, and a single PPO policy must map that command plus proprioception
directly into 38 joint torques, supervised by a hidden imitation reward and a measured
command-velocity reward. At deployment the command is overwritten by the user and the
reference frame is dropped. It is preserved for its measured **negative** command-control
result. See [`demo_d/`](demo_d/).

**Demo E — task RL with a frozen motion prior (research reference).** Demo E was the
workshop's first literal composition of Demos A and B on the MIMIC skeletal rodent: a
controlled comparison between task reward alone (E0) and task reward plus β × a frozen
Demo B motion score (E1), both arms sharing the same 100 Hz torque physics, native
`RodentJoystick` reset, PPO architecture, and frozen 16-D imitation decoder. Only E1
receives the self-supervised likelihood as reward. It is kept as a research reference;
its long scratch-PPO diagnostic learned to stand rather than to walk — a negative result
the faster same-body Demo G was designed to avoid. See [`demo_e/`](demo_e/).

**Demo F — rodent-derived conditional motion on Fetch.** The engine behind Part 2. Demo F
repeats Demo B's self-supervised construction after retargeting real Coltrane locomotion
onto the ten-joint Fetch body used by Demo A: given a past motion window and a hindsight
displacement command extracted from the same clip, it predicts the shifted future motion,
with no action labels, rewards, or environment rollouts in the objective. Its accepted
checkpoint is the frozen conditional model consumed by Demo G, and its retargeting,
feature, command, and metric primitives are reused by Demo H. Canonical Demo F uses
Froude-similarity timing; Demo H owns a separately versioned 1.75× retimed derivative.
See [`demo_f/`](demo_f/).

**Demo G — Demo A task PPO plus Demo F motion prior.** The workshop's controlled
SSL-plus-RL comparison on a single physical body: G0 optimizes Demo A's task reward, and
G1 adds β × the frozen Demo F motion score, with matched initialization, environment
count, PPO settings, transition budget, and paired evaluation seeds. The best accepted
model (dynamic seed 0) preserves the task while substantially reducing the held-out
gait-distance score; across three seeds the learned likelihood improves every time but
the full gait composite improves in only two. The honest reading is that a data prior can
*shape* physical RL, not that it solves locomotion realism. See [`demo_g/`](demo_g/).

**Demo H — generative body–action pretraining plus RL.** The accepted workshop capstone
and the engine behind Part 3. Demo H extends Demo F from future-motion prediction to a
body-centric world–action prior — predict a short future body-motion plan, then predict
the bounded Fetch controls that realize it — freezes that state/action prior, and trains
a small zero-initialized residual PPO policy on top with task reward minus a β-weighted
KL to the prior. The accepted configuration retargets Coltrane strict-locomotion clips to
Fetch under 1.75× temporal dilation, projects them through exact Brax v1 Fetch physics to
obtain pseudo-label controls, and fine-tunes at β=0.10 over 30M transitions with speeds
sampled from 1.5 to 4.0 Fetch units/s. Four-limb contact, stride, and speed statistics are
validation-only, so gait quality is pressure from the learned distribution rather than a
hand-written checklist. See [`demo_h/`](demo_h/) and
[`ref/docs/demo_h.md`](ref/docs/demo_h.md).

**Demo J — spiking imitation and neural similarity.** Demo J trains a recurrent spiking
controller — 128 LIF plus 128 adaptive-LIF neurons with surrogate-gradient learning — on
the same 1.75× retargeted Fetch-motion release used by Demo H, then treats its 20 ms
spike counts as a synthetic neural recording. Each 64-frame clip is an independent episode
whose recurrent state resets at the clip boundary, and future-motion tokens are masked
wherever a block runs past the clip. The current analysis compares population geometry
(RSM/RSA) against the matched Demo H β-sweep under identical locomotion inputs, using no
Demo H policy, activation, or spike to train or select the network. Naturalness and neural
similarity are measurements, never rewards. See [`demo_j/`](demo_j/).

## Setup

```bash
uv sync --extra cuda12 --extra workshop --extra dev
# Do not use cuda13 on WSL2; see rl/README.md.
```

Demo H's and Demo F's renderers use the legacy `brax.v1` software rasterizer, which the
repo keeps in an isolated environment (`uv run --no-project --isolated --with
'brax==0.12.3' --with 'jax==0.4.30' ...`); see each demo's README for exact commands.

## Repository layout

- `rl/` — published-decoder joystick experiment. Start with `rl/README.md`.
- `attachments/` — the committed workshop showcase videos embedded above (an explicit
  exception to the repo's global "no video files" policy).
- `ref/repos/` — upstream repos as submodules (track-mjx, stac-mjx, DART,
  MotionStreamer).
- `ref/papers/` — local reference papers used by the design documents.
- `ref/docs/` — per-demo design notes, measurements, and the workshop plan.
