# Demo C — World Model + RL: learn a simulator, then dream inside it (plan)

_Created 2026-07-19. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) (thesis),
[demo_a.md](demo_a.md) (RL demo), [PROJECT_STATE.md](PROJECT_STATE.md) (assets)._
_Supersedes [demo_c_prev.md](demo_c_prev.md), which proposed a research-grade recipe
(differentiable MJX + LAPO port + intention-IDM). See §9 for what changed and why._

**Status: not started. Buildable now — no dataset dependency (uses Demo A's body + self-collected data).**

> **North star: pedagogical simplicity.** The workshop teaches new grads the *definitions*
> of SSL and RL from scratch. Demo A taught RL (PPO on a quadruped). Demo B taught SSL (a
> convolutional autoencoder over motion). Demo C must show — in the simplest honest way — how
> the two **combine**. It is not a research contribution; it is the "aha" that closes the arc.

---

## 1. Role in the thesis

Demo C is the **both** corner of the 2×2 (WORKSHOP_PLAN §2): **functional and
distributional** realism at once. The one-line idea:

> **Learn a world model with SSL, then train a policy with RL entirely *inside* it —
> and it still works when deployed in the real simulator.**

This is the textbook "World Models" recipe (Ha & Schmidhuber 2018) and is *exactly*
[WorldModel.pdf](../papers/WorldModel.pdf) **§4.1 "World Model for Reinforcement Learning"**:
the world model becomes a *learned simulator* in which the policy rolls out, receives
rewards, and improves through **imagined** interaction.

It fills the empty corner because the two objectives constrain different things:
- the **world model is learned from data** (SSL → *distributional*: it matches the
  distribution it was trained on), and
- the **policy is shaped by reward** (RL → *functional*: it achieves the goal),
- so the policy can only act *through a model of the world learned from data* — reward and
  data manifold constrain it jointly.

## 2. The core idea — the simplest possible SSL + RL synthesis

One joint distribution ties the whole workshop together
([WorldModel.pdf](../papers/WorldModel.pdf) **§3.1**, the load-bearing slide):

> `p(future states, future actions | current state)`
> - the **world model** is one query of it: `p(sₜ₊₁ | sₜ, aₜ)` — *learned by SSL* (predict the data);
> - the **policy** is another: `p(aₜ | sₜ)` — *shaped by RL* (maximize reward).
>
> SSL learns **how the world responds**; RL learns **what to do about it.** Demo C trains
> one, then the other, and closes the loop. Same object, two queries — that *is* the synthesis.

Why this is the pedagogically right choice, concretely:

- **It reuses Demo A verbatim.** Stage 2 is the *same brax PPO on the same fetch body* — the
  only thing that changes is that the environment is now a *learned* model instead of the
  physics engine. "Identical RL, new simulator" is the entire lesson, and it's one diff.
- **It reuses Demo B's idea.** Demo B learned to reconstruct motion clips with no labels
  (an autoencoder). Demo C's world model predicts the *next state given the action* with no
  labels — the same self-supervision, now applied to **dynamics**. A world model is just
  "Demo B's SSL, made action-conditioned."
- **It is the paper the user assigned.** §3.1 gives the "why" (one distribution, two queries);
  §4.1 gives the "what we build" (RL in a learned simulator). Nothing exotic in between.
- **It is cheap and reliable.** The world model is a small MLP; rolling it out is *faster*
  than MJX (no contact solver), so PPO-in-imagination fits the live-workshop budget with room
  to spare — and there is no 65 GB dataset, no differentiable physics, no cross-framework bridge.

## 3. The build (one body, four short stages)

**Body/env: Demo A's brax `fetch` quadruped, task `FetchRun`** (constant-speed run, fall
termination) — reused verbatim from [demo_a/fetch_run.py](../../demo_a/fetch_run.py). Same
obs (101-d), same reward (`track + upright − ctrl`), same fall test. Keeping the body
identical to Demo A makes the A-vs-C comparison a **clean ablation of SSL's contribution**
(§7). Everything stays in one stack (brax/JAX) — no torch bridge.

### Stage 0 — collect data ("motor babbling")
Roll out a **random policy** (optionally mixed with the trained Demo A policy for coverage)
in the *real* brax env; save transitions `D = {(sₜ, aₜ, sₜ₊₁, rₜ, doneₜ)}`. Teaching line:
*"we learn physics by watching a body flail" — this is motor babbling* (§4).

### Stage 1 — SSL: learn the world model
A small MLP `f_θ: (sₜ, aₜ) → (Δsₜ, rₜ, doneₜ)`, trained by MSE/BCE on `D`. **Pure
self-supervision** — the only "label" is the next state the world already showed us.
- *Connect to Demo B out loud:* same objective family (predict data, no labels), now a
  **dynamics** model = a world model.
- *Show its quality (mirror of Demo B's "honest to ~2.5 s open-loop" note):* roll `f_θ`
  open-loop from a real start state and plot predicted vs. true trajectory — **the dream
  stays coherent for ~N steps, then drifts.** That drift plot motivates Stage 2's short
  imagination horizon.

### Stage 2 — RL: train PPO inside the world model
Wrap `f_θ` as a brax `Env` (`reset` → sample a real start state from `D`; `step(a)` → `f_θ`).
Run **the same PPO as Demo A** on this learned env. The policy trains **entirely in
imagination** — it never touches real physics.
- *Reliability technique (teachable, standard):* **short imagination horizon** with resets to
  real start states (MBPO / "when to trust your model") so the policy can't escape into
  fantasy. If pure-imagination transfer proves flaky in dry runs, fall back to **Dyna**
  (Sutton 1991): use imagined rollouts to *augment* real experience rather than replace it.

### Stage 3 — deploy in reality (the money shot)
Drop the imagination-trained policy into the **real** brax simulator; measure forward
speed / `FetchRun` reward. Two outcomes, both teachable:
- **it transfers** → *"RL in a dream produced a real controller — the learned model captured
  enough physics"*; or
- **it exploits model error** → *"the policy found free reward that exists only in the dream"* —
  the direct analog of Demo B's physics-falsification money shot, and the concrete reason
  model fidelity matters (§4.1's "second-level" co-evolution problem, shown, not lectured).

### Files (same shape/size as demo_a, demo_b)
```
demo_c/
  README.md         what it is, how to run, the teaching point + the money shot
  collect.py        Stage 0 — roll out random(+Demo-A) policy in brax fetch -> transitions D
  world_model.py    Stage 1 — MLP f(s,a)->(Δs,r,done); MSE train loop; open-loop drift plot
  dream_env.py      Stage 2 — wrap f_θ as a brax Env (reset from real states; step = f_θ)
  train_dream.py    Stage 2 — PPO (reuse demo_a plumbing) inside dream_env
  evaluate.py       Stage 3 — deploy dream-trained policy in REAL brax; speed/reward; A-vs-C
  # reuses demo_a/fetch_run.py for the real env + metrics
```

## 4. The neuroscience tie-in (for the comp-neuro audience)

The ML recipe maps 1:1 onto a real motor-learning story — one a comp-neuro grad already half
knows:

- **Motor babbling** (Stage 0) — random exploration to sample your own dynamics; how infants
  and animals bootstrap control.
- **Forward model** (Stage 1) — the learned `f_θ` predicting next sensory state from an action
  is a **cerebellar-style internal forward model** (Wolpert–Kawato). *SSL = the brain–body half.*
- **Mental rehearsal / motor imagery** (Stage 2) — practicing a skill *inside* your internal
  model, no movement required; a documented way animals and humans improve motor skill
  (Jeannerod). *This is literally "RL in imagination."*
- **Execution** (Stage 3) — the basal-ganglia / **DLS** reward-driven controller acting in the
  world. *RL = the body–environment half.*

So: **babble → build a forward model → rehearse in imagination → execute.** The ML term
("world model + RL") and the neuroscience ("cerebellar forward model + striatal RL") point at
the same object — the workshop's thesis, in one sentence a new grad can repeat.

## 5. Where it sits in the world-model survey
[WorldModel.pdf](../papers/WorldModel.pdf) (Hou et al. 2026):
- **Occupies §4.1 "World Model for Reinforcement Learning"** — policy improved by imagined
  rollouts in a *learned* simulator (Eqs. 16–18).
- Uses **§3.1's joint-distribution framing** as the conceptual backbone (world model and
  policy as two queries of one distribution).
- **Does not touch §5** (robotic video generation) — our states are low-dimensional
  proprioception, not pixels; the world model is a tiny MLP, not a video diffusion model.
- Slide line: **"learn the simulator with SSL, do the RL inside it — the oldest world-model
  idea, and still the clearest."**

## 6. Literature anchors (all already in WORKSHOP_PLAN §7)
- **Ha & Schmidhuber, World Models, 2018** (arXiv:1803.10122) — the canonical "train a
  controller inside a learned world model, then deploy" result. *Primary anchor.*
- **Dreamer / DreamerV3** (arXiv:2301.04104) — learns behaviors purely in imagination from a
  learned world model (modern, scaled version of the same idea).
- **Sutton, Dyna, 1991** — the classic "learned model generates simulated experience for RL"
  (the reliable fallback framing, §3 Stage 2).
- **Janner et al., MBPO, 2019** (arXiv:1906.08253) — short-horizon model rollouts branched
  from real states (the reliability technique, §3 Stage 2).
- **WorldModel.pdf §4.1** — the modern VLA instantiation of exactly this loop.

## 7. Evaluation — the one figure, three real points

Shared axes (WORKSHOP_PLAN §4), one point per demo:
- **Demo A** (RL in the *real* sim, reward only): functional, **not** distributional — the
  scramble gait (see [demo_a.md](demo_a.md)).
- **Demo B** (SSL on real motion, no physics/reward): distributional, **not** functional —
  foot-skate, falls through the floor.
- **Demo C** (SSL world model + RL inside it): **both** — runs in real physics *and* is
  grounded in a learned data-driven model.

Because **Demo A and Demo C share the fetch body, the `FetchRun` task, and the gait/speed
metrics**, the A-vs-C comparison is a genuinely controlled ablation of what the learned
world model buys (unlike the previous plan, where the bodies differed). Demo B sits on a
different body (the rodent) illustrating the SSL-only corner; the three are still legible on
one figure because the **evaluation axes** (functional + distributional) are shared. Be
honest about the body mismatch rather than paper over it.

## 8. The central design decision (stated, not hidden)

**Which body carries Demo C — the fetch quadruped (chosen) or the rodent?**
- **Chosen: fetch** (Demo A's body). It reuses Demo A's RL machinery *directly*, keeps
  everything in one framework, needs no dataset, and makes A-vs-C a clean ablation. The
  "distributional" content is honest but *abstract*: "the world model matches the data
  distribution it was trained on."
- **Rejected for the core demo: rodent + real mocap.** It would make the distributional claim
  *literal* ("matches the real-rat manifold"), but it drags back the whole rodent-MJX stack,
  the frozen decoder, and the action-free-mocap labeling problem — i.e. the complexity the
  user is explicitly moving away from.
- **Optional Phase-2 upgrade (note, don't build):** train the *same* world model on
  real-rat-derived transitions instead of self-collected fetch data. The recipe is unchanged;
  only the data source is. That is where the literal "real-rat distributional realism" claim —
  and the neural-axis scoring (MIMIC GLM/RSA vs DLS/MC) — would re-enter, once the dataset is local.

## 9. What we drop from the previous plan, and why
[demo_c_prev.md](demo_c_prev.md) was a research proposal, not a lesson. It is dropped wholesale:

| Dropped | Why it was too much for a *definitions* workshop |
|---|---|
| Differentiable MJX + forward-mode autodiff (`jacfwd`) through the constraint solver | fragile, research-grade; nothing to do with the SSL/RL *definitions* |
| LAPO ported to JAX (VQ→continuous, ~1.7k LOC) | a large port to explain a bottleneck the demo doesn't need |
| intention-IDM (invert `decoder ∘ MJX` to label action-free mocap) | clever, but the hardest thing in the plan |
| HILP / MTM / TD-MPC2 aux losses | a zoo of SOTA repos; each needs its own lecture |
| 65 GB rat mocap dependency | blocks Phase 0; the demo works with self-collected data |

**The crux of the simplification:** the previous plan *vacated* §4.1 — it used exact MJX as
the world model, so there was no *learned* model and no RL-in-imagination. The user asking us
to read §4.1 is the tell. This plan **embraces §4.1**: the world model is **learned** (that's
the SSL lesson), and RL happens **inside** it (that's the synthesis). Replacing "exact MJX as
world model" with "a small MLP the students watch us train" is the single move that turns a
SOTA proposal into a demonstration of the basics.

## 10. Build order
0. **Stage 0 + 1 first** — collect from a random policy, fit `f_θ`, produce the open-loop
   drift plot. This alone is a complete, self-contained SSL demo (a learned simulator) and
   de-risks everything downstream.
1. **Stage 2** — wrap `f_θ` as an env, run Demo A's PPO inside it with a short imagination
   horizon. Watch reward climb *in the dream*.
2. **Stage 3** — deploy in real brax; measure. Whichever outcome (transfer vs. exploitation),
   it is the money shot; if exploitation dominates, iterate Dyna-style (§3) for a clean live result.
3. Write `demo_c/README.md` in the demo_a/demo_b house style; add the third point to the 2×2 figure.
4. **(Phase 2, optional)** swap Stage-0 data for real-rat transitions (§8) → literal
   distributional realism + the neural axis.
