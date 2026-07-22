# From reinforcement learning to a generative motor prior

This workshop introduces reinforcement learning and self-supervised learning
through one continuous example: teaching the same simulated Fetch quadruped to
move. Each part adds one idea while keeping the body, task, and visual language
consistent.

By the end, you should be able to explain three complementary views:

- Self-supervised learning supplies **distributional realism**; reinforcement
  learning supplies **functional realism**.
- Self-supervised learning is **data-driven**; reinforcement learning is
  **task-driven**.
- Motion data describes a brain–body relationship, RL describes a
  body–environment relationship, and their combination models a
  brain–body–environment loop.

The code is organized as:

```text
workshop/
  part1/       PPO locomotion from scratch
  part2/       conditional motion pretraining
  part3/       body–action pretraining plus PPO
  data/        generated course datasets
  prepare_data.py
```

Research ablations, rejected models, publishing scripts, neural comparisons,
and historical demo implementations are deliberately outside this package.

## Part 1: learn locomotion from reward

Start with the basic reinforcement-learning loop.

- Call the simulated world the **environment**.
- Call the information given to the controller the **observation**.
- Call the ten motor commands selected by the controller the **action**.
- Call the scalar feedback after an action the **reward**.
- Call the action-producing network the **policy**.
- Call one complete interaction sequence an **episode**.
- Call the discounted sum of future rewards the **return**.

At time step $t$, the policy receives $o_t$, samples $a_t$, and the
environment returns $o_{t+1}$, $r_t$, and a flag saying whether the episode
ended. Reinforcement learning adjusts the policy so that actions leading to
larger future return become more likely.

Part 1 uses a deliberately small reward:

\[
r_t=
\exp\!\left[-\frac{(v_{x,t}-v^*)^2}{2\sigma^2}\right]
+0.1\,\text{upright}_t
-10^{-3}\lVert a_t\rVert^2.
\]

The first term rewards the requested forward speed, the second rewards staying
upright, and the third discourages needlessly large actions. Falling ends the
episode. No motion recording or pretrained model is available.

PPO repeatedly performs four operations:

1. Roll out the current policy in many environments.
2. Estimate whether each action produced more return than expected.
3. Increase the probability of helpful actions and decrease the probability of
   unhelpful actions.
4. Clip large policy changes so one noisy batch cannot destroy the controller.

Train the policy in the pinned legacy-Brax runtime:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  python -m workshop.part1.train
```

Use `--smoke` for a short wiring check. A full 30M-transition run is the
workshop setting. Render the resulting checkpoint with:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'imageio>=2.37' \
  --with 'imageio-ffmpeg>=0.6' \
  python -m workshop.part1.visualize \
  --checkpoint workshop/part1/out/POLICY.pkl
```

Ask students to identify the observation, action, reward, policy, and return in
the code before discussing PPO internals. Then inspect whether a high return
actually corresponds to stable cyclic locomotion.

## Part 2: learn the motion distribution from data

Part 2 introduces self-supervised learning. A learning problem is
self-supervised when its targets are constructed from the data itself rather
than supplied by a human annotator.

The course data contains Coltrane rodent locomotion retargeted to the same Fetch
body. For every motion clip, split the sequence into:

- a history $h_t$, containing motion already observed;
- a future $w_t$, containing the next motion;
- a hindsight command $c_t$, computed from the future displacement.

The model learns

\[
\max_\phi\;\log p_\phi(w_t\mid h_t,c_t).
\]

No reward or environment interaction is used. The future is simultaneously
the training target and the source of its command, which makes the objective
self-supervised.

Each frame has 60 body features: root motion, root orientation, ten joint
angles and velocities, four paw positions and velocities, and four contact
bits. A causal convolutional autoencoder compresses every four frames into a
16-dimensional token. A small conditional Transformer predicts the next token
from four history tokens and the command. During training, the predictor is
also unrolled through four of its own outputs to expose short-horizon drift.

The Gaussian likelihood is a normalized prediction score. With fixed variance,
higher likelihood means smaller latent prediction error. It is useful as a
compact motion-distribution score, but it is not a complete definition of
physical or biological realism.

Train, evaluate, and generate four commanded speeds:

```bash
uv run --extra workshop python -m workshop.part2.train
uv run --extra workshop python -m workshop.part2.evaluate
uv run --extra workshop python -m workshop.part2.generate
```

Render the generated trajectories with the exact Fetch body:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'imageio>=2.37' \
  --with 'imageio-ffmpeg>=0.6' \
  --with 'pillow>=11' \
  python -m workshop.part2.visualize
```

The important checks are that matching commands receive better likelihood than
mismatched commands, generated speed changes with requested speed, and joint
angles remain within the body limits. Part 2 predicts motion; it does not yet
control a simulated body.

## Part 3: combine a generative motor prior with RL

Part 3 adds actions to the self-supervised sequence. A bounded feedback
controller replays each retargeted motion in the exact Fetch simulator. The
stored state is what physics actually realizes, and the stored action is the
normalized actuator control used during that transition.

These controls are physics-derived pseudo-labels. They are not measured animal
torques and should not be presented as biological inverse dynamics.

The pretrained model learns the body-centric world–action distribution

\[
p_\theta(s_{t+1:t+H},a_{t:t+H-1}\mid s_{\leq t},a_{<t},g).
\]

It first predicts a future motion token, then predicts an action conditioned on
the current body state, predicted token, previous action, action phase, and
goal. This factorization makes the lesson concrete: predict what comes next,
then predict the motor command that could produce it.

Train and evaluate the prior, then export it from PyTorch to the pure-JAX form
used inside PPO:

```bash
uv run --extra workshop python -m workshop.part3.pretrain
uv run --extra workshop python -m workshop.part3.evaluate_prior
uv run --extra workshop python -m workshop.part3.export
```

PPO trains a small residual policy around the frozen action prior. The visible
control knob is \(\beta\):

\[
J(\psi)=\mathbb E\sum_t\gamma^t\left[
r_t^{\mathrm{task}}
-\frac{\beta}{10}
D_{\mathrm{KL}}\!\left(\pi_\psi(\cdot\mid h_t,g_t)
\Vert p_{\theta_0}(\cdot\mid h_t,g_t)\right)
\right].
\]

- Set `--beta 0` to optimize only the locomotion task with the pretrained
  architecture.
- Set `--beta 0.10` to use the accepted workshop regularization strength.
- Increase β to stay closer to the prior, at the cost of less freedom to solve
  the task.

Train the regularized policy:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'scipy>=1.15' \
  python -m workshop.part3.train --beta 0.10
```

Generate rollouts from 1.5 to 4.0 Fetch units/s and render one comparison:

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'scipy>=1.15' \
  python -m workshop.part3.visualize \
  --checkpoint workshop/part3/out/POLICY.pkl

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'scipy>=1.15' \
  --with 'imageio>=2.37' \
  --with 'imageio-ffmpeg>=0.6' \
  --with 'pillow>=11' \
  python -m workshop.part3.render \
  workshop/part3/out/speed_sweep/metrics.json \
  --columns 6 \
  --output workshop/part3/out/speed_sweep/comparison.mp4
```

Four-limb contact, stride, speed, uprightness, and action statistics are used
only for validation. They are never added to the training reward. This is an
important scientific distinction: the learned data prior is responsible for
the naturalness pressure rather than a manually designed gait objective.

## Optional data reproduction

Prepared datasets are course assets, so data construction is not part of the
live workshop. To reproduce them on a new machine, first download the public
retargeted release and create the Part 2 and Part 3 timing variants:

```bash
uv run --extra workshop python -m workshop.prepare_data --download

uv run --extra workshop python -m workshop.prepare_data \
  --time-scale 1.75 \
  --crops-per-parent 1 \
  --variant temporal-dilation-1p75-v1 \
  --output-root workshop/data/part3_reference
```

Then generate the Part 3 state–action data in the same pinned physics used by
RL:

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' \
  --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' \
  --with 'scipy>=1.15' \
  python -m workshop.part3.data.project
```

## What the final comparison establishes

Part 1 shows that task reward can discover functional locomotion. Part 2 shows
that future motion can supervise a conditional model without labels. Part 3
shows how a frozen generative body–action distribution can initialize and
regularize task-driven RL.

The demonstration does not establish biological torque recovery, literal
rodent biomechanics, or an algorithm-level advantage over PPO across many
seeds. Retargeted clips contain artifacts, and the current prior is not robust
on every long rollout. Present the result as a compact illustration of SSL–RL
synergy, not as a state-of-the-art locomotion claim.
