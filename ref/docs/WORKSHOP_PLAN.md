# Workshop plan — from RL and SSL to generative pretraining plus RL

_Finalized 2026-07-21 for the accepted Demo A/B/F/H workshop. Detailed pages:
[Demo A](demo_a.md), [Demo B](demo_b.md), [Demo F](demo_f.md), and
[Demo H](demo_h.md). [Demo G](demo_g.md) remains the measured reward-side
comparison. Demos [C](demo_c.md), [D](demo_d.md), and [E](demo_e.md) are
research history rather than live material._

Audience: new computational-neuroscience graduate students. Optimize for
definitions they can repeat accurately, short causal code paths, visible model
behavior, and honest boundaries between data targets, simulator-derived
pseudo-labels, and reward.

## 1. Teaching thesis

Self-supervised learning and reinforcement learning impose complementary
constraints:

| | self-supervised learning | reinforcement learning |
|---|---|---|
| realism | distributional: resemble patterns in recorded data | functional: produce useful consequences |
| supervision | construct targets from the data itself | receive scalar reward from interaction |
| emphasis | data-driven brain/body regularities | task-driven body/environment behavior |

The combined story is:

> **Generative pretraining learns familiar body motion and controls from data;
> RL adapts that prior so the body succeeds in an environment.**

Use the shorthand only after defining it:

```text
SSL       = distributional realism = data-driven brain–body structure
RL        = functional realism     = task-driven body–environment interaction
SSL + RL  = both constraints       = brain–body–environment interaction
```

“Brain–body” is a computational interpretation. The workshop uses behavioral
motion, not neural recordings, and makes no neural-similarity claim.

## 2. Accepted four-demo arc

```text
Demo A: task reward -> PPO from scratch -------------------------------+
                                                                      |
recorded rodent motion                                                 |
  -> Demo B: predict conditional future rodent motion (teach SSL)      |
  -> retarget motion to Fetch                                          |
  -> Demo F: predict conditional future Fetch motion                   |
  -> exact-physics projection: motion + executable controls            |
  -> Demo H pretraining: predict future motion, then control            |
  -> freeze reference + residual PPO <---------------------------------+
```

Say:

> **Demo A learns what works. Demo B learns what motion is predictable.
> Demo F puts that data onto Demo A's body. Demo H pretrains a policy from
> body–action sequences, then uses PPO to make a small task-driven correction.**

Demo F is an engineering bridge, not a third learning definition. The physics
projection in Demo H creates pseudo-labels; it must not be called pure SSL.

## 3. Begin with the absolute basics of RL

Introduce five objects before naming PPO:

1. Explain that the **environment** contains the body and world.
2. Explain that the **observation** is the information available now.
3. Explain that the **action** is the policy's motor output.
4. Explain that the **reward** is one number measuring a consequence.
5. Explain that the **policy** maps observations to an action distribution.

Draw the loop:

```text
observation --> policy --> action --> environment
      ^                                  |
      +-------- next observation --------+
                       + reward
```

Only then define return:

\[
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots,
\qquad \max_\pi\mathbb E_{\tau\sim\pi}[G_0].
\]

Make three points explicit:

- RL is not given the correct action as a label.
- Delayed reward can change how likely an earlier action becomes.
- High reward means behavior satisfies the written objective; it does not
  automatically mean motion looks natural.

Introduce PPO operationally: collect rollouts, estimate which sampled actions
did better than expected, make a bounded policy update, and repeat. Keep the
clipped-surrogate derivation optional.

## 4. Demo A — reinforcement learning on Fetch

Status: **implemented in `demo_a/`.**

Use the ten-actuator Brax v1 Fetch body. `FetchRun` rewards forward speed,
upright posture, and modest control effort:

\[
r_t^{\rm task}
=\exp\left(-\frac{(v_x-v^*)^2}{2\sigma^2}\right)
+0.1u_z-10^{-3}\lVert a_t\rVert^2.
\]

Terminate when the torso is too low or upside down. Keep the standard 101-D
observation and train PPO from random initialization. Use `v*=3` for the
standalone locomotion demonstration.

Show the reward code before the optimizer. Ask students to predict a loophole,
then emphasize that PPO discovers whatever satisfies the written reward. The
30M-transition task-only reference trains in roughly one minute on the current
H100.

## 5. Demo B — self-supervised conditional rodent motion

Status: **implemented in `demo_b/`; the Coltrane generator is behaviorally
accepted.**

Slide a window over one continuous recording. Use past motion `h` as input,
shift the same recording to obtain future `w`, and compute a hindsight
displacement command `c` from that future:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D}
\log p_\phi(w\mid h,c).
\]

There are no action labels, rewards, or environment interactions. This is the
workshop's clean definition of self-supervised learning.

Show the causal convolutional tokenizer and conditional Transformer. Explain
the fixed-variance Gaussian view:

\[
\log p_\phi(w\mid h,c)
=-\frac12\operatorname{mean}_i\left[
\left(\frac{w_i-\mu_{\phi,i}}{\sigma}\right)^2
+2\log\sigma+\log(2\pi)\right].
\]

With fixed `sigma`, maximizing likelihood is minimizing normalized MSE. Render
several command-conditioned trajectories. State that this is kinematic motion
generation, not physical control.

## 6. Demo F — retarget and repeat conditional prediction

Status: **implemented and accepted in `demo_f/`.**

Retarget strict Coltrane locomotion to the Fetch skeleton:

```text
rodent keypoints
  -> trunk frame and semantic paws
  -> contact-aware stance pinning
  -> bounded sequence inverse kinematics
  -> Fetch root, ten joints, feet, contacts
```

Do not copy rotations between anatomically unlike bones. Match semantic paw
endpoints and regularize the complete sequence.

Canonical Demo F applies the theoretical Froude factor
`sqrt(21.3789)=4.6237`, producing 7,483/1,166/1,425 session-split clips. Its
small 60-D conditional model trains in 51.4 seconds, achieves 0.0080/0.0129
units/s validation/test speed MAE, beats shuffled futures, selects the matching
command in every evaluated speed bin, and produces zero joint-limit saturation.

Demo H owns a different, explicitly versioned data derivative. Direct timing
looked accelerated, while 4.6237x looked too slow for the workshop body. A
visual factor sweep selected `1.75x`, with one centered crop per parent. Call
this an empirical temporal dilation—not Froude similarity—and do not replace
canonical Demo F with it.

## 7. Demo H — body-centric world–action prior plus PPO

Status: **implemented and accepted with `beta=0.10`.**

### 7.1 Build executable pseudo-labels

Track every 1.75x kinematic reference with a transparent `kp=400`, `kd=10`
controller in the unchanged Demo A Fetch physics. Save the controls and only
the simulator-realized states:

\[
u_t\text{ acts over }[t,t+1)\text{ and produces }x_{t+1}.
\]

The accepted release contains 1,784/278/342 train/validation/test clips and
151,452 transitions. It passes 99.09% of candidate clips and builds in 84.4
seconds. These ten-dimensional controls are Fetch pseudo-labels with requested
axis torque `-300u`; they are not biological torque.

### 7.2 Pretrain motion first, then control

Reuse Demo F's causal 16-D motion token. Given four history tokens and a goal,
predict one short future-motion plan. At each of the next four 50 Hz phases,
decode a Gaussian control distribution from current physical feedback, the
plan, previous control, phase, and goal:

\[
p_\theta(z_{k+1}\mid z_{k-3:k},g_k)
\prod_{j=0}^{3}
p_\theta(u_{t+j}\mid x_{t+j},z_{k+1},u_{t+j-1},j,g_k).
\]

Exact Fetch physics supplies each next state during deployment. The narrow
“world” in this body-centric world–action model is the body and its recurring
flat-ground contacts; it is not a learned general simulator.

The prior trains from scratch in 70.8 seconds. On final-test sessions, it beats
state persistence by 49.8%, chooses matching over shuffled commands in 82.4%
of windows, and improves 20-step closed-loop control MSE by 86.9% over
repeating the initial control. The frozen prior itself locomotes from an
ordinary standing reset for five seconds without falling or saturating.

### 7.3 Freeze the reference and apply RL post-training

Freeze the planner and base action distribution. Initialize a small residual
actor at exactly zero and train it with PPO:

\[
J(\psi)=\mathbb E\sum_t\gamma^t\left[
r_t^{\rm task}-\frac{\beta}{10}
D_{\rm KL}\!\left(
\pi_\psi(\cdot\mid h_t,g_t)
\Vert p_{\theta_0}(\cdot\mid h_t,g_t)
\right)\right].
\]

The implementation adds reference log probability to environment reward and
uses PPO entropy with the matching coefficient; together they are exactly the
mean per-action-dimension KL term. No hand-written gait or naturalness metric
is optimized.

Freeze:

- `beta=0.10`;
- uniform task commands from 1.5 to 4.0 Fetch units/s;
- 30M transitions, 2,048 environments, three evaluations;
- seed 0 for the accepted workshop checkpoint.

The accepted PPO stage takes 95.2 seconds. Pretraining plus PPO takes 166
seconds; including the one-time physical projection takes about 250 seconds.

## 8. Accepted evidence and caveats

The user selected β=0.10 after one video placed it beside a matched β=0.075
run at six speeds. Preserve the complete record:

| command | realized β=.10 | survival | strict four-limb stride gate |
|---:|---:|---:|---:|
| 1.5 | 1.471 | 100% | pass |
| 2.0 | 2.010 | 100% | pass |
| 2.5 | 2.479 | 100% | **fail** |
| 3.0 | 2.974 | 100% | pass |
| 3.5 | 3.465 | 100% | pass |
| 4.0 | 3.647 | 100% | **fail** |

β=0.10 has mean absolute speed error 0.079 over the six commands and survives
all five-second rollouts. However, β=0.075 passes the stride gate at 5/6 speeds
versus 4/6 and has lower mean joint-speed RMS. A larger training coefficient
does not guarantee a lower realized KL because PPO can converge to a different
solution.

Acceptance therefore means **workshop-ready qualitative demonstration**, not
that every predeclared research gate passed. Always show the 2.5 and 4.0
failures. Require multiple policy-training seeds and matched H0/H1 baselines
before claiming algorithm-level superiority.

## 9. Live run of show

1. Explain environment, observation, action, reward, policy, and return.
2. Show Demo A's reward code and a successful PPO locomotion video.
3. Construct Demo B targets by shifting one recorded sequence.
4. Rewrite normalized MSE as fixed-variance Gaussian likelihood.
5. Show Demo B command-conditioned rodent generations.
6. Retarget one clip to Fetch and explain morphology and timing limitations.
7. Show Demo F's conditional prediction and shuffled-command check.
8. Execute a stored Demo H control sequence in exact physics.
9. Show the state-first motion-plan and feedback-control factorization.
10. Roll out the frozen prior from standing before any RL.
11. Add the zero-initialized residual actor and the single KL term to PPO.
12. Train or replay the accepted 95-second β=0.10 run.
13. Show all six speeds in one video with validation labels visible.
14. Ask students to identify every data target, pseudo-label, and reward.

The final sentence students should be able to say is:

> **Demo H predicted future body motion from shifted data, learned executable
> Fetch controls from physics-derived pseudo-labels, then used PPO reward to
> adapt that frozen generative prior to a locomotion task.**

## 10. Demo G — optional reward-side contrast

Demo G remains useful when time permits:

```text
Demo G: Demo F likelihood -> extra reward -> scratch PPO
Demo H: state/action prior -> pretrained policy -> residual PPO + KL
```

Demo G is evaluated over three policy seeds. It improves learned likelihood in
3/3 seeds while retaining task tracking, but its full nine-measure gait
composite improves in only 2/3. Use it to contrast reward-side and policy-side
priors, not as an additional required live training block.

## 11. Claim boundaries

- “Rodent-derived motion statistics on Fetch” is accurate; “rat biomechanics on
  Fetch” is not.
- Retargeting changes morphology, length, time, contacts, mass, and actuation.
- Physics-derived controls are Fetch pseudo-labels, not measured animal torque.
- A Gaussian likelihood or KL is a learned data-distribution proxy, not a
  complete naturalness or biological-realism metric.
- Better kinematics do not imply better neural similarity.
- Demo B and canonical Demo F do not learn actions.
- Demo H does not train inside a learned simulator; it acts in exact Brax
  physics while using a learned body/action reference.
- The accepted Demo H result uses one PPO training seed and supports a
  pedagogical demonstration, not an algorithm-level claim.

## 12. Research-history demos

- [Demo C](demo_c.md): PPO inside a frozen learned rodent world model.
- [Demo D](demo_d.md): one-stage hindsight-command imitation and its command
  identifiability failure.
- [Demo E](demo_e.md): Demo B likelihood on the full MIMIC skeletal rodent; its
  long scratch-PPO diagnostic learned standing rather than locomotion.
- [Demo G](demo_g.md): accepted limited evidence for a reward-side motion prior.

Keep these as engineering and scientific context rather than crowding the live
beginner arc.

## 13. Reproducibility discipline

Continue to follow `canvas/misc/autoresearch.md`:

- freeze data IDs, splits, objectives, commands, beta, budgets, and gates before
  a controlled run;
- validate paired versus shuffled controls and command sensitivity before PPO;
- keep naturalness diagnostics out of the training loss;
- report raw task reward, reference KL, direct behavior, and runtime separately;
- change one major block per experiment and keep append-only decisions;
- require multiple policy seeds before making an algorithm-level claim;
- retain negative results and failed validation cells after qualitative
  acceptance.

## 14. Literature anchors

- Aldarondo et al. 2024, *A virtual rodent predicts the structure of neural
  activity across behaviours*.
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*.
- GPC, the local [`GPC.pdf`](../papers/GPC.pdf), as architectural motivation for
  reusable generative-control pretraining and RL post-training. Demo H is not a
  reproduction of its FSQ, tracking-RL, or adapter stack.
- Peng et al. 2021, *AMP: Adversarial Motion Priors for Stylized Physics-Based
  Character Control*, as related motion-prior context.
