# Workshop plan — from RL and SSL to SSL-guided RL

_Updated 2026-07-20 for the current Demo A/B/F/G workshop. Detailed pages:
[Demo A](demo_a.md), [Demo B](demo_b.md), [Demo F](demo_f.md), and
[Demo G](demo_g.md). Demos [C](demo_c.md), [D](demo_d.md), and
[E](demo_e.md) remain research history rather than live material._

Audience: new computational-neuroscience graduate students. Optimize for
definitions they can repeat accurately, short causal code paths, visible model
behavior, and one honest controlled comparison.

## 1. Teaching thesis

Self-supervised learning and reinforcement learning impose complementary
constraints:

| | self-supervised learning | reinforcement learning |
|---|---|---|
| realism | distributional: resemble patterns in recorded data | functional: produce useful consequences |
| supervision | construct targets from the data itself | receive a scalar reward from interaction |
| emphasis | data-driven brain/body regularities | task-driven body/environment behavior |

The combined story is:

> **SSL learns what motion is plausible from recorded behavior. RL learns what
> motion works in an environment. SSL-guided RL asks one physical policy to
> satisfy both constraints.**

Use the shorthand only after defining it:

```text
SSL       = distributional realism = data-driven brain–body structure
RL        = functional realism     = task-driven body–environment interaction
SSL + RL  = both constraints       = brain–body–environment interaction
```

“Brain–body” is a computational interpretation. The workshop uses behavioral
motion, not neural recordings, and makes no neural-similarity claim.

## 2. Four-demo arc

```text
recorded rodent motion
        |
        +--> Demo B: learn conditional rodent motion (teach SSL)
        |
        +--> retarget and dynamically rescale motion to Fetch
                |
                +--> Demo F: repeat conditional SSL on Fetch -- freeze --+
                                                                          |
Demo A: train Fetch from task reward (teach RL) ---------------------------+--> Demo G
                                                                               task + prior PPO
```

Say:

> **Demo A learns what works. Demo B learns what looks plausible. Demo F puts
> that data knowledge onto Demo A's body and time scale. Demo G trains one
> policy with both signals.**

Demo F is an engineering bridge, not a third learning definition.

## 3. Start from the absolute basics of RL

Introduce five objects before naming PPO:

1. The **environment** contains the simulated body and world.
2. The **observation** is the information given to the learner now.
3. The **action** is the learner's motor output.
4. The **reward** is one number measuring the action's consequence.
5. The **policy** maps observations to a distribution over actions.

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
- A delayed reward can change how likely an earlier action becomes.
- High reward means behavior satisfies the written objective; it does not
  automatically mean motion looks natural.

Introduce PPO operationally: collect rollouts, estimate which sampled actions
produced better-than-expected return, update without one excessively large
policy step, and repeat. Keep the clipped-surrogate derivation optional.

## 4. Demo A — reinforcement learning on Fetch

Status: **implemented in `demo_a/`.**

Use the unmodified ten-actuator Brax v1 Fetch body. `FetchRun` rewards forward
speed, upright posture, and modest control effort:

\[
r_t^{\rm task}
=\exp\left(-\frac{(v_x-v^*)^2}{2\sigma^2}\right)
+0.1u_z-10^{-3}\lVert a_t\rVert^2.
\]

Terminate when the torso is too low or upside down. Keep the standard 101-D
observation and train PPO from random initialization. Use `v*=3` for the
standalone Demo A locomotion demonstration. Demo G reuses exactly this task
class and optimizer but chooses a slower, data-aligned target.

The transition-matched 30M task-only arms used in Demo G train in 58–60 seconds
inside `ppo.train`. This gives a live RL example comfortably inside five
minutes.

## 5. Demo B — self-supervised conditional rodent motion

Status: **implemented in `demo_b/`; the Coltrane generator is behaviorally
accepted.**

Show a continuous recording. Choose past motion `h`, shift that recording to
obtain future `w`, and compute a hindsight displacement command `c` from the
future. The recording supplies input and target:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D}
\log p_\phi(w\mid h,c).
\]

There are no action labels, rewards, or environment interactions. This is the
workshop's definition of self-supervised learning.

The causal convolutional tokenizer and conditional Transformer predict future
tokens from past tokens plus command. Explain its fixed-variance Gaussian view:

\[
\log p_\phi(w\mid h,c)
=-\frac12\operatorname{mean}_i\left[
\left(\frac{w_i-\mu_{\phi,i}}{\sigma}\right)^2
+2\log\sigma+\log(2\pi)\right].
\]

With fixed `sigma`, maximizing this likelihood is minimizing normalized MSE.
Render kinematic rodent motion at multiple commands. State that generation is a
learned motion distribution, not physical control.

## 6. Demo F — retarget, rescale, and repeat SSL

Status: **implemented and accepted in `demo_f/`.**

Retarget Coltrane strict locomotion to the Fetch skeleton:

```text
rodent keypoints
  -> trunk frame and semantic paws
  -> contact-aware stance pinning
  -> bounded sequence inverse kinematics
  -> Fetch root, ten joints, feet, contacts
```

Do not copy relative rotations between anatomically unlike bones. Match semantic
paw endpoints and regularize the whole sequence.

Then explain the physical-scale fix. The spatial transform enlarges trunk
length by 21.3789x. Retaining the source clock made the old Fetch target about
3 units/s and produced moon-like dynamics. Froude similarity dilates time by
`sqrt(21.3789)=4.6237`, mapping 0.20 m/s source locomotion to 0.924747 Fetch
units/s and command `[0.573343, 0, 0]` over 0.62 seconds.

Build four disjoint target-time crops within each parent clip; never join clips.
A 1% joint-saturation gate leaves 7,483/1,166/1,425 session-split clips.

Repeat Demo B's model pattern in a 60-D Fetch representation. The accepted
small model reads four 16-D history tokens and predicts one next token. During
training it recursively consumes four of its own predictions and receives a
joint-limit penalty. This fixes rollout drift without replacing the workshop
architecture.

Measured evidence:

- train in 51.4 seconds;
- achieve 0.0080/0.0129 m/s validation/test speed MAE;
- beat shuffled futures by +5.81/+5.56 mean log likelihood;
- select the matching command in 5/5 speed bins on both splits;
- peak locally at the exact matched speed on both splits;
- produce zero joint-limit saturation in all evaluated rollouts.

Freeze the model after these data-only gates. Do not tune it on Demo G PPO.

## 7. Demo G — combine task and data constraints

Status: **implemented, evaluated over three policy seeds, and accepted only for
a limited claim.**

Reuse Demo A's environment and PPO code with the Froude-aligned target
`v*=0.924747` and `sigma=v*/3`. Compare:

| arm | Demo A task reward | frozen Demo F score |
|---|---:|---:|
| G0 | yes | no (`beta=0`) |
| G1 | yes | yes (`beta=0.1`) |

Train:

\[
\max_\pi\;\mathbb E\sum_t\gamma^t
\left[r_t^{\rm task}+\beta\,
\operatorname{sigmoid}\left(\frac{\ell_\phi(w_t\mid h_t,c)+20}{5}\right)
\right].
\]

The pure-JAX Demo F network is frozen. It sees the same 60-D feature contract,
with four unsupported yaw-only roll/pitch channels masked. Collect 32 causal
frames, score all 2,048 environments in one batch every four frames, and keep
task reward frame-by-frame.

Use 30M transitions and three PPO evaluations. Measured G0 runs take 57.8–59.8
seconds and G1 runs 68.0–69.5 seconds. A sequential pair is about 2.1 minutes.

Evaluate with shaping disabled using five paired rollout seeds per policy seed:

| training seed | raw log-p improvement | direct composite | tracking retained |
|---:|---:|---:|---:|
| 0 | +18.11 (5/5 wins) | +5.78 (5/5) | 100.15% |
| 1 | +32.76 (5/5 wins) | +3.66 (5/5) | 100.58% |
| 2 | +17.28 (5/5 wins) | -0.30 (0/5) | 100.15% |

Across training seeds, raw likelihood improves by `22.72 ± 7.11`; tracking and
survival are retained in every seed. Airborne fraction, stance-foot speed,
approximate world-foot slip, and joint-speed RMS move toward the held-out
reference in 3/3 seeds. The complete nine-measure gait distance improves in only
2/3 seeds, and cyclicity improves in 0/3.

Select seed 0 as the best presentation model because it passes every
single-seed gate and gives the largest direct improvement. It reduces contact
switching 11.48→3.11 Hz, world-foot slip 1.92→0.75, and vertical acceleration
1.78→0.91 g while preserving near-target speed. Retain its failures: maximum
flight duration worsens, cyclicity worsens, and the rendered posture is
crouched.

## 8. Acceptance gates

| gate | result |
|---|---|
| complete one matched pair in under five minutes | pass: about 2.1 min inside `ppo.train` |
| validate dynamic data and frozen prior before PPO | pass on validation and final test |
| clear causal state at vectorized resets | pass: permanent CPU regression |
| match G0/G1 budget and PPO settings | pass: 30M, 2,048 envs, three evals |
| evaluate with shaping disabled | pass: five paired rollouts per training seed |
| retain at least 95% tracking and survival | pass in 3/3 seeds |
| improve held-out raw likelihood | pass in 3/3 seeds and 15/15 rollouts |
| improve at least one direct measure in every seed | pass for four measures |
| improve the full gait composite in every seed | **fail: only 2/3** |

Present the limited claim that passed. Do not redefine the failed gate or use
learned likelihood as a synonym for physical realism.

## 9. Live run of show

1. Explain environment, observation, action, reward, policy, and return.
2. Show the few lines defining Demo A's reward.
3. Train or replay Demo A and plot forward speed plus task return.
4. Slide a window over recorded rodent motion to construct Demo B's future.
5. Show Demo B generations and rewrite MSE as Gaussian likelihood.
6. Show spatially retargeted Fetch clips and explain the time-scale correction.
7. Train or replay Demo F and show its held-out likelihood matrix.
8. Freeze Demo F visually and add one `beta * prior_score` term to PPO.
9. Train one matched G0/G1 pair or replay all three measured pairs.
10. Plot task tracking and raw likelihood separately before showing videos.
11. Show the four robust direct measures, then the seed-2 composite failure.
12. Ask students which signal came from data and which came from interaction.

The final sentence students should be able to say is:

> **Demo F constructed targets by shifting retargeted recordings, so it was
> self-supervised. Demo G froze that learned motion score as one reward term,
> while task reward still came from physical interaction and PPO still learned
> the actions.**

## 10. Claim boundaries

- “Rodent-derived motion statistics on Fetch” is accurate; “rat biomechanics on
  Fetch” is not.
- Retargeting changes morphology, scale, contacts, and dynamics.
- A Gaussian score is a learned data-likelihood proxy, not biological realism.
- Better kinematics do not imply better neural similarity.
- Demo B and Demo F do not learn actions.
- G0 versus G1—not Demo A versus Demo F—is the controlled causal comparison.
- Demo G is aligned with Demo A's task form and code, not its standalone
  3-unit/s target.
- Report seed 2 and the failed direct-composite gate in every presentation.

## 11. Research-history demos

- [Demo C](demo_c.md): world/action model plus PPO and exploratory neural
  analysis.
- [Demo D](demo_d.md): one-stage hindsight-command imitation; its
  identifiability failures motivate Demo F's explicit likelihood audit.
- [Demo E](demo_e.md): the same SSL+RL idea on the full MIMIC skeletal rodent.
  The published decoder locomotes well, but the ten-minute scratch-PPO run
  learned an upright stance rather than workshop-ready locomotion.

Keep these as valuable negative and engineering results, not live baselines.

## 12. Reproducibility discipline

Continue to follow `canvas/misc/autoresearch.md`:

- freeze data IDs, splits, objectives, commands, beta, budgets, and gates before
  a controlled run;
- validate real-versus-shuffled likelihood and command sensitivity before PPO;
- keep beta-zero parity and reset isolation as regression tests;
- report raw task reward, raw likelihood, transformed reward, behavior, and
  runtime separately;
- change one major block per experiment and keep append-only decision logs;
- require multiple policy seeds before making an algorithm-level claim;
- stop at the declared experiment boundary instead of tuning on final reports.

## 13. Literature anchors

- Aldarondo et al. 2024, *A virtual rodent predicts the structure of neural
  activity across behaviors*.
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms*.
- Peng et al. 2021, *AMP: Adversarial Motion Priors for Stylized Physics-Based
  Character Control*, as related context. Demo G uses an explicit frozen
  conditional Gaussian score rather than an adversarial discriminator.
