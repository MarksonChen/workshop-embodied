# Demo C matched-policy decision log

This is an append-only research narrative following `canvas/misc/autoresearch.md`.
Generated checkpoints, curves, metrics, and `results.tsv` live under `demo_c/out/`
and are intentionally git-ignored.

## Frozen problem (do not edit during the comparison)

- **World:** Demo B's frozen `motor_standalone.pt`; one transition is 0.64 s. The
  planar state update is decoded from predicted velocity/orientation features, never
  copied from the command.
- **Task:** reach one randomly placed food target 0.35--0.75 m away in at most 8
  dream steps. Reward is `10 * progress + arrival - time/turn costs`.
- **Matched inputs:** both policies receive egocentric goal, distance, measured body
  velocity, and previous action. `wam` additionally receives the frozen 192-D
  predictive context. Dynamics, resets, reward, policy head, PPO, and seeds match.
- **Primary scalar:** deterministic held-out dream success over 1,024 episodes.
  Diagnostics: return, final distance, invalid transition rate, throughput, memory.
- **Hard gates:** non-finite metrics, invalid-transition rate > 0, or performance no
  better than the frozen random controller.
- **Budget:** 786,432 environment steps, 256 environments, 8-step rollouts. Three
  seeds (0, 1, 2); fixed evaluation seed 10,000.
- **References:** random controller is the null; the transparent go-to-goal heuristic
  is a task ceiling/reference, not a learned baseline.

The original Demo A Fetch policy is not used as the neuroscience baseline because it
has a different body, observation convention, and task. `goal_only` is the valid
matched rodent analogue of RL without a predictive representation.

## 0. Smoke and convergence calibration — complete

- Both variants passed shape/finite gates on CPU and H100.
- At 262,144 steps, seed 0: `goal_only` success 0.472; `wam` 0.432. The WAM train
  curve was visibly rising at cutoff, so the provisional budget was rejected.
- 6x probes (1,572,864 steps): `goal_only` success 0.469; `wam` 0.509. Curves wobble
  around their plateau from roughly 600--800k onward. Therefore the reportable fixed
  budget is 786,432 steps. Probe artifacts have a `convergence6x` suffix and never
  replace reportable checkpoints.

## 1. Matched three-seed baseline

Complete at the frozen 786,432-step budget:

| policy | success (mean ± SD) | return | final distance |
|---|---:|---:|---:|
| goal-only PPO | 0.4759 ± 0.0091 | 2.8248 ± 0.0417 | 0.3056 ± 0.0033 m |
| WAM-context PPO | 0.4837 ± 0.0158 | 3.0259 ± 0.0659 | 0.2863 ± 0.0052 m |

Both have zero invalid transitions. The primary success delta is +0.0078 while the
conservative noise threshold is `eta = 2 * max(SD) = 0.0316`. **Verdict: within
noise; the two policies are functionally matched on navigation.** The lower WAM final
distance is retained as a diagnostic, not promoted to a win after seeing the result.
This is a useful controlled starting point for the independently frozen neural test:
any representational difference is not confounded by a large task-performance gap.

## 2. World-data coverage block — bundled transition rejected, broader fit kept

The bundled Demo B transition was first evaluated without changing its architecture.
Its exact held-out future-latent skill was positive on only two of four untouched
neural sessions, with high between-session variance. It therefore failed the frozen
"positive on every session" gate. This was a coverage failure, not evidence for a
larger model.

One thing changed: training data. The tokenizer, six-layer standard Transformer,
plain MSE, command definition, and persistence null stayed fixed. The new data block
uses 6,972 genuine contiguous locomotion crops from 12 complete sessions (two each
from six rats), with 1,227 crops from two separate validation sessions. The four final
DLS/MC sessions remain untouched.

An 8,000-step convergence probe peaked at step 500:

| step | validation skill over persistence |
|---:|---:|
| 1 | -12.5% |
| **500** | **+42.0%** |
| 1,000 | +38.5% |
| 2,000 | +33.0% |
| 8,000 | +32.8% |

The best validation checkpoint, not the last one, was saved. On the four untouched
sessions it achieved +42.0%, +24.7%, +45.1%, and +46.0% skill over persistence over
53, 101, 149, and 23 held-out windows. Session-balanced mean was +39.5%; every output
was finite. **Decision: keep the broader checkpoint and freeze it.**

## 3. Task-support iteration — full-circle navigation closed, forward food kept

The first matched block sampled food over the full circle. It produced the results in
Section 1, but transfer diagnostics showed that the Demo B locomotion support does not
include reliable in-place U-turns. With the broader world checkpoint, the fixed raw
physics bridge reached only 1/8 goals for each policy. This was a task/world support
mismatch: many episodes began with food behind the rat.

One task field changed before the final block: food bearing was restricted to the
forward semicircle. Radius, horizon, reward, model, PPO architecture, budget, seeds,
and evaluation seed stayed fixed. Both policies were retrained from scratch.

Final dream results:

| policy | success (mean ± seed SD) | return | final distance |
|---|---:|---:|---:|
| goal-only PPO | 0.6393 ± 0.0044 | 4.4251 | 0.1643 m |
| WAM-context PPO | 0.6445 ± 0.0029 | 4.4733 | 0.1603 m |

Random success was 0.0859 and the transparent heuristic reached 0.5518. Both learned
conditions had zero invalid transitions. The WAM success delta, +0.0052, is inside
`eta = 0.0088`. **Decision: keep; final policies are functionally matched.**

## 4. Frozen-joystick bridge block — nonlinear response bridge kept

No policy was changed in this block. The frozen joystick's response was measured on a
command grid before selecting a bridge. It has a pronounced low-speed dead zone: a
single scalar fit gives forward gain 0.346 and turn gain 1.072, but hides the
nonlinearity.

All bridges were evaluated on the same eight forward goals:

| bridge | goal-only success | WAM success | goal-only/WAM falls | verdict |
|---|---:|---:|---:|---|
| raw command | 0/8 | 2/8 | 1/0 | reject: cannot cover far goals reliably |
| inverse scalar gain | 4/8 | 1/8 | 0/0 | reject: over-drives WAM |
| **measured response curve** | **7/8** | **4/8** | **0/0** | **keep** |

The accepted bridge monotonizes and inverts the measured zero-turn displacement curve,
then uses the independently measured turn gain. It changes interface units, not a
network. WAM's 4/8 physical result remains below goal-only's 7/8 despite matched dream
success. **Interpretation: the physical loop closes, but WAM exhibits a larger
dream-to-real/context-reliance gap; do not claim a functional transfer advantage.**

The raw and inverse-gain metrics remain in separately named files under
`demo_c/out/physics/`.

## 5. Neural block — feasible and promising, superiority not established

The world checkpoint, task, PPO checkpoints, representation families, four sessions,
16-PC bottleneck, Poisson settings, RSA conditions, temporal split, shuffle null, and
paired-session statistics were frozen before the reportable neural pass. Policy seeds
are averaged within session before sessions are balanced.

One preprocessing probe was rejected on distributional grounds. It fed the literal
2-s recorded displacement as the food goal. Median distances were only 0.002--0.006 m
across sessions and about 0.1% of rows lay in the trained 0.35--0.75 m goal range. This
unfairly collapsed the goal-only baseline. The accepted cache retains the recorded
future **bearing**, clips it to the frozen forward field, and fixes radius at the task
midpoint. This rule was accepted because it restores the predeclared input support,
not because of a neural score.

Primary locomotion-only session-balanced means:

| representation | raw population bits/spike | shift-corrected bits/spike | RSA rho |
|---|---:|---:|---:|
| RL-only policy | 0.0141 | 0.0082 | 0.575 |
| Demo B autoencoder | 0.0660 | 0.0141 | 0.625 |
| Demo B predictor | 0.0671 | 0.0193 | 0.710 |
| **Demo C WAM+RL policy** | **0.0655** | **0.0191** | **0.701** |
| kinematics | 0.0751 | 0.0148 | 0.643 |

The WAM-minus-RL descriptive deltas are +0.0514 raw bits/spike, +0.0109 corrected
bits/spike, and +0.125 RSA. Their exact sign-permutation p-values are 0.25, 0.375, and
0.25. WAM is essentially tied with the Demo B predictor (deltas -0.0015, -0.0002,
-0.0093).

The deterministic n-matched full-behavior control gives WAM/RL RSA 0.754/0.688 and raw
bits/spike 0.0667/0.0193, but shift-corrected bits/spike 0.0019/0.0057. Slow temporal
structure therefore explains much of the unrestricted raw encoding advantage; show
the corrected metric and RSA together.

With four sessions, the smallest possible two-sided exact sign-permutation p-value is
0.125. **Decision: keep the analysis and its null result. The defensible workshop
claim is that WAM+RL preserves Demo B's predictive neural structure while adding
goal-directed function; statistical superiority over Demo B or RL-only is not
established.** Neural data will not be used for another architecture-tuning loop.

## 6. Final accepted system

- Broadened but architecturally unchanged action-conditioned Demo B transition.
- Forward-semicircle, eight-step food task.
- Matched goal-only and WAM-context PPO at 786,432 steps and three seeds.
- Nonlinear measured-response bridge to the frozen MIMIC joystick.
- Four-session strict neural evaluation, with locomotion primary and n-matched control.

This block meets the workshop objective: prediction targets visibly train the world,
reward visibly trains the policy, the same frozen policy can be tested in MJX, and the
neural comparison produces an honest empirical outcome rather than a preassigned win.
