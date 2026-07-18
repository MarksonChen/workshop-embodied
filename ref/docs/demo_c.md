# Demo C — WAM + RL synthesis: the "intention-IDM" (plan)

_Created 2026-07-18. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) — this **supersedes
the WAM recipe in §5** there. Assets: [PROJECT_STATE.md](PROJECT_STATE.md)._

**Status: not started. Stage 1 needs the mocap (have it now); Stage 2 reuses the joystick
RL stack. Prototype the Stage-1 gradient path first (§7).**

---

## 1. Role in the thesis

Demo C is the **both** corner (WORKSHOP_PLAN §2): functional **and** distributional — the
synthesis that fills the empty top-right of the 2×2.

## 2. The core idea — we already own what LAPO must *learn*

LAPO ([ref/repos/LAPO](../repos/LAPO)) discovers latent actions from **action-free**
observation: an inverse-dynamics model `(s_t, s_{t+1}) → z` trained jointly with a
**learned** forward world-model that must reconstruct `s_{t+1}` from `s_t` + `z` (a
bottleneck forces `z` to encode the true action). We verified this scaffold in code
(`lapo/stage1_idm.py`: `vq_loss = idm.label(batch); wm_loss = wm.label(batch)`).

**We don't need the learned world model** — we have the two pieces it approximates:
- the **frozen NPMP decoder** = a differentiable inverse-dynamics map `(intention, proprio)
  → torque`;
- **MJX** = exact, differentiable forward dynamics `(torque, state) → next state`.

Replace LAPO's learned FDM with `MJX ∘ frozen_decoder` and train an **intention-IDM**:

> `g: (s_t, s_{t+1}) → z ∈ ℝ¹⁶`  such that  `MJX_step( decoder(z, proprio_t) ) ≈ s_{t+1}`
> on real mocap transitions.

This inverts our own controller + physics to recover the 16-D intention that would have
produced the observed rat motion — turning 65 GB of **action-free** mocap into exact
`(state → intention)` supervision, with **no learned world model and no imagination**. It
is an inverse model paired with a forward model = the **Wolpert–Kawato internal model** =
the SSL / brain–body half of the thesis.

## 3. Stage 1 — SSL: learn intentions from mocap
- **Scaffold:** LAPO's Stage-1 (~1.7k LOC), **ported to JAX** so the loss can backprop
  through **differentiable MJX**. Swap LAPO's **VQ** (discrete) bottleneck for a **16-D
  continuous** latent matching the decoder's intention space.
- **Loss:** single-step state reconstruction
  `‖ MJX_step(decoder(g(s_t, s_{t+1}), proprio_t)) − s_{t+1} ‖` (+ smoothness / consistency).
- **#1 technical risk + fallback:** differentiating through contact-rich MJX is stiff. Use
  **single-step** supervision (tractable); if gradients misbehave, fall back to a
  **learned-FDM proxy** (literal LAPO) or a **distilled decoder**, then fine-tune. Prototype
  this before the full build (§7).
- **Optional shaping (stack-native):** **HILP** ([ref/repos/HILP](../repos/HILP), JAX/Flax)
  to give the intention a temporal-distance / directional structure — a smoother,
  goal-directed manifold.
- **Lower-risk alternative (Role A):** **MTM** ([ref/repos/mtm](../repos/mtm)) state-only
  masked-trajectory pretraining to warm-start just the PPO *encoder*, if explicit
  intention-labeling proves brittle.
- **Output:** a pretrained state encoder + BC targets `(state → intention)`.

## 4. Stage 2 — RL: brax PPO fine-tune in MJX (reuse the joystick stack)
- Warm-start the high-level policy from Stage 1 (BC on intentions and/or the encoder); PPO:
  policy → 16-D intention → frozen decoder → torques → MJX. = our joystick pipeline **+ SSL
  warm-start.**
- **Borrow from TD-MPC2** ([ref/repos/tdmpc2](../repos/tdmpc2); read in code,
  `tdmpc2/tdmpc2.py:262–276`): add its self-predictive latent-consistency as a pure
  **auxiliary loss** (predict next latent → stop-grad target encoding). No planner, no
  learned dynamics-for-control.
- **The synergy contrast (isolates SSL's value):** Demo A (PPO from scratch) vs the existing
  joystick walker (decoder + PPO, *no* WAM) vs Demo C (WAM + decoder + PPO). Frame the gap
  **distributionally** (WORKSHOP_PLAN §1), not as asymptotic reward.

## 5. Where it sits in the world-model survey
[ref/papers/WorldModel.pdf](../papers/WorldModel.pdf) (*World Model for Robot Learning: A
Comprehensive Survey*, Hou et al. 2026):
- Occupies **§3.2 "Inverse-Dynamics Policies with World Models"** + **§3.6 "Policies with
  Latent-Space World Modeling."**
- **Vacates §4.1 "World Model for Reinforcement Learning"** (RL-in-imagination — Dreamer /
  TD-MPC planning) and **all of §5 "World Model for Robotic Video Generation."** Because MJX
  is the exact world model, we borrow *only* TD-MPC2's representation objective, never its
  MPPI planner.
- Slide line: **"learned representation (SSL) + exact simulator (MJX) + model-free PPO."**

## 6. Repos (cloned to `ref/repos`)
| Repo | Framework | Role here |
|---|---|---|
| [`LAPO`](../repos/LAPO) | PyTorch | Stage-1 scaffold (action-free latent-action = intention discovery) — **port to JAX** |
| [`HILP`](../repos/HILP) | JAX/Flax | optional intention shaping (temporal-distance directional latent) |
| [`mtm`](../repos/mtm) | PyTorch | fallback Role-A representation warm-start |
| [`tdmpc2`](../repos/tdmpc2) | PyTorch | contrast exhibit + borrowable self-predictive aux loss |

Blueprints without public code (design references only): Radosavovic *humanoid locomotion as
next-token prediction* (action-masking of mocap is exactly our regime) and RPT.

## 7. Build order
0. **Prototype** the JAX intention-IDM on a few merged mocap bouts — test the
   differentiable-MJX gradient path (the make-or-break risk) before anything else.
1. Full Stage-1 intention-IDM on the merged locomotion bouts. Data gotchas
   ([dataset.md](dataset.md)): **merge bouts** (label flicker shreds raw runs), **finite-diff
   `qvel`**, and pick one **Aldarondo ↔ MIMIC-MJX `qpos` convention**.
2. Stage-2 PPO warm-start + TD-MPC2 aux loss; render + score.
3. (Phase 2) neural axis — MIMIC GLM/RSA vs DLS/MC.

## 8. Evaluation (shared axes, WORKSHOP_PLAN §4)
Functional (velocity tracking, MJX) — strong; **kinematic-distributional** (gait stats vs
real mocap) — should be **best of the three demos**; neural (MIMIC GLM/RSA vs DLS/MC) —
Phase 2.

## 9. The neuroscience tie-in (why the ML recipe *is* the thesis)
- **intention-IDM** `g:(s,s') → z` = the **inverse model**.
- **`decoder ∘ MJX`** `z → s'` = the **forward model**.
- Jointly = the **Wolpert–Kawato paired forward/inverse internal model** = the predictive /
  cerebellar **brain–body** half (SSL).
- **Stage-2 RL** = the reward-driven **body–environment** half (basal-ganglia / DLS).
- So the ML term ("WAM") and the neuroscience thesis point at the same object — and both
  halves can be scored against the DLS (RL) + MC (motor) recordings.
