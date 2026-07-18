# Workshop Experiment Plan — SSL + RL synergy for embodied neuroscience

_Created 2026-07-18. Companion to [PROJECT_STATE.md](PROJECT_STATE.md)._

Audience: **new computational-neuroscience grad students** (a workshop talk, **not**
a publication). Optimize for engaging, teachable, *demonstrable* — with our own
virtual rodent — over novelty.

---

## 1. Thesis

**Embodied neuroscience needs a synergy of self-supervised learning (SSL) and
reinforcement learning (RL).** The two objectives constrain *different* things:

| | SSL | RL |
| --- | --- | --- |
| realism it buys | **distributional** (matches the data manifold) | **functional** (achieves the goal) |
| signal | **data-driven** (predict the data) | **task-driven** (maximize reward) |
| interaction | **brain–body** (internal / forward+inverse model) | **body–environment** (goal in the world) |

**SSL + RL = brain–body–environment.**

### The sharpened claim (defend this version)
The ML literature shows predictive (SSL) pretraining mainly buys *sample-efficiency
and better representations* — from-scratch RL can eventually match final **task**
performance given enough simulator rollouts. So on the **functional** axis, SSL is a
convenience, not a necessity. **Point the synergy claim at the distributional axis
instead:**

> RL reaches functional realism at any compute — but it **never** reaches
> distributional (neural / kinematic) realism, because nothing constrains it to the
> data manifold. SSL is what pulls the representation onto the real-rat distribution.
> **RL buys function; SSL buys the brain; embodied neuroscience needs both.**

This survives the obvious pushback ("but RL eventually walks fine") — yes, and its
gait and its "neurons" are still wrong.

---

## 2. The one figure the talk is built around

A scatter over two axes, one point per demo:

```
 distributional
 realism  ▲
 (neural/ │   SSL-only ●                 ● WAM+RL   ← the target corner
  kinematic)         (MotionStreamer)      (both)
          │
          │                              ● RL-only
          │                                (from scratch)
          └───────────────────────────────────────▶ functional realism
                                                     (physics / goal)
```

Each demo fills a different corner; the empty-then-filled top-right **is** the
thesis. Three archetypes, one shared pair of axes (Section 4).

---

## 3. The three demos

> These are three **regimes/archetypes**, not a controlled ablation (they differ in
> more than SSL-vs-RL: physics present or not, what is predicted, the data). Frame
> them that way, and hold the **body + evaluation fixed** across all three so the
> comparison is legible.

### Demo A — RL-only (functional, not distributional)
- **What:** train the rodent body to locomote with **pure RL reward** (velocity +
  alive), **from scratch, on raw torque control — WITHOUT the pretrained decoder.**
- **Note:** our existing joystick walker is **not** this — it rides an
  imitation-pretrained (NPMP-style) decoder, so it already carries a data prior and
  sits near the WAM+RL corner. Demo A must be the true no-prior baseline.
- **Teaching point:** a **competent-but-unnatural** walker — moves forward fine, but
  gait and internal representation don't match the real rat. *Not* a flailing
  failure (a failure conflates "RL can't walk on a 38-DoF body" — a compute artifact
  — with "RL isn't distributionally real," the actual point). If from-scratch RL
  won't produce a competent gait cheaply, use a curriculum or a simpler reward; keep
  it functional-but-ugly.
- **Feasibility:** buildable **now** (rodent model + MJX + brax PPO; no dataset).

### Demo B — SSL-only (distributional, not functional)
- **What:** a **generative motion model over rodent kinematics** (no physics).
  Options, cheapest first:
  1. a light autoregressive / VAE model on the mocap joint-angle sequences;
  2. **DART** (`ref/repos/DART`) — real-time motion primitives, already vendored;
  3. **MotionStreamer** (`ref/repos/MotionStreamer`) — strongest, but it is
     **human text-to-motion** and would need **retraining on rodent kinematics** (a
     real lift). Recommend 1 or 2 for the demo unless retraining MotionStreamer is
     independently wanted.
- **Teaching point:** produces realistic-*looking* walking, then a **physics check
  falsifies it** — foot-skating, no ground-reaction forces, limbs through the floor.
  Distributional realism with **zero** functional realism. Exposing the physics
  violations is the money shot.
- **Feasibility:** needs the **mocap clips** (Section 6). Medium.

### Demo C — WAM + RL (both) — the synthesis
Two stages (concrete recipe in Section 5):
- **Stage 1 (the "WAM"):** SSL **masked-trajectory prediction of future
  proprioceptive state + intention** from mocap clips — a joint forward+inverse
  model.
- **Stage 2 (RL):** **brax PPO** fine-tunes the pretrained representation for a
  locomotion task; policy outputs **intention** → frozen decoder → torques → **MJX**.
- **Teaching point:** functional **and** distributional. Compare against Demo A
  (same task, no pretraining) to show the synergy.
- **Feasibility:** needs mocap clips (stage 1) + physics (stage 2, have it). Hardest.

---

## 4. Shared evaluation axes (compute for ALL three)

- **Functional realism:** commanded-velocity tracking / goal-reaching success in
  MJX physics. *Demo B scores ~0 here — that is the point.*
- **Distributional realism:**
  - **Kinematic:** distance between the demo's gait/joint-angle **statistics** and
    the **real-rat mocap distribution** (e.g., distributions of stride frequency,
    duty factor, joint-angle ranges, inter-joint correlations). Buildable once clips
    are local; needs no neural data.
  - **Neural (when the recordings arrive):** run the MIMIC comparison — Poisson
    encoding GLM + cross-validated RSA (crossnobis / whitened-cosine) — of each
    controller's activations against **DLS/MC**. Does the controller out-predict raw
    kinematics? (This is Aldarondo 2024's test, re-run per demo.)

---

## 5. The WAM + RL recipe (concrete)

Grounded in the locomotion-proven predictive-pretraining lineage, **not** the
video-world-model/manipulation lineage of the cited survey (see Section 7).

**Stage 1 — masked-trajectory model (MTM-style), over `[state, intention]` tokens.**
- Predict **future proprioceptive state + intention** — **not raw torques.** The
  frozen decoder already maps (intention, proprioception) → torques, so the useful,
  low-dimensional, controllable variable to model is the **16-D intention** (exactly
  what the downstream PPO policy will output).
- **Objective:** masked-trajectory reconstruction with a random-autoregressive mask
  (MTM, arXiv:2305.02968), preferred over pure next-step because (a) it natively
  handles **action-free mocap** — our clips have states but no intention labels, so
  mask the missing modality; (b) one net then serves as forward-model, inverse-model,
  and encoder; (c) MTM's own results show the learned reps **accelerate downstream RL
  on Walker locomotion**. Simpler intuitive alternative: modality-aligned next-token
  prediction (Radosavovic, arXiv:2402.19469), also locomotion-proven.
- Where intention labels are missing in raw mocap: either infer them via the frozen
  decoder's encoder, or treat intention as a masked/latent token to be predicted.

**Stage 2 — PPO fine-tune in MJX (do NOT plan in a learned model).**
- Attach a brax PPO policy/value head on the (frozen-then-unfrozen) WAM encoder;
  policy outputs **intention** → frozen decoder → torques → **MJX steps real
  physics**.
- **Avoid** Dreamer/TD-MPC "RL-in-imagination": we already have exact physics in MJX,
  so a learned dynamics model + planner is redundant machinery. Use the WAM only as a
  **representation / pretraining signal**; let MJX be the simulator. This keeps
  everything inside the existing JAX/MJX + brax PPO stack.
- **Baseline for the synergy claim:** PPO-from-scratch (= Demo A's controller) vs.
  WAM-pretrained PPO — expect faster learning + better representation, framed as
  distributional (Section 1), not asymptotic task score.

**The tidy tie-in:** a WAM = joint model of **future state + action** = **forward
model + inverse model** = the Wolpert–Kawato internal model = the cerebellar /
**brain–body** half. Stage-2 RL = **body–environment**. The ML term and the
neuroscience thesis point at the same object.

---

## 6. Data dependencies — what we're waiting for, what's buildable now

**BLOCKED on the dataset the user is getting:**
- **Rodent mocap clips** (HF `talmolab/MIMIC-MJX`, ~66 GB) → needed for Demo B, WAM
  Stage 1, and the kinematic-realism metric.
- **DLS/MC recordings** (Harvard Dataverse `10.7910/DVN/FB0MZT`; Ölveczky-lab
  lineage) → needed for the neural-realism axis of Section 4.
- The MIMIC RSA / encoding-GLM pipeline (`github.com/diegoaldarondo/virtual_rodent`,
  on request) → needed to score the neural axis.

**Buildable NOW, no dataset (do while waiting, only if asked):**
- Demo A (RL-only from-scratch walker) — rodent model + MJX + brax PPO.
- Metric/eval scaffolding (functional tracking metric; kinematic-stats harness ready
  to point at clips once local).
- WAM Stage-1 code scaffold (masked-trajectory model; unit-test on synthetic
  trajectories).

**Phasing:**
- **Phase 0 (no data):** Demo A + eval scaffolding + Stage-1 scaffold.
- **Phase 1 (mocap clips):** Demo B; WAM Stage 1 pretraining; kinematic-realism axis;
  Demo C Stage 2.
- **Phase 2 (neural data):** neural-realism scoring for all three → the full 2×2.
- **Phase 3 (optional):** swap the decoder for a **BrainPy SNN** (ties to the SNN arm
  in PROJECT_STATE §6) and re-score.

> **STATUS: waiting on the dataset before executing Phase 1+. Phase 0 is buildable
> now but not started (per the user's "wait" instruction).**

---

## 7. Literature anchors (verify 2026 IDs before a slide)

**Predictive pretraining → control (our real recipe lineage):**
- MTM — Masked Trajectory Models, ICML 2023, arXiv:2305.02968 *(locomotion-proven; primary anchor)*
- Humanoid locomotion as next-token prediction, NeurIPS 2024, arXiv:2402.19469
- SMART, ICLR 2023, arXiv:2301.09816 (DMC locomotion)
- RPT — Robot Learning w/ Sensorimotor Pre-training, CoRL 2023, arXiv:2306.10007 (manipulation)
- LAPO — Learning to Act without Actions, ICLR 2024, arXiv:2312.10812

**World-model RL (the "don't-do-this-here" contrast — MJX is our sim):**
- World Models, Ha & Schmidhuber 2018, arXiv:1803.10122
- DreamerV3, arXiv:2301.04104 · TD-MPC2, ICLR 2024, arXiv:2310.16828

**"WAM" as a named term (2026, manipulation/VLA — cite provenance honestly):**
- World Action Models: The Next Frontier in Embodied AI, arXiv:2605.12090
- World Model for Robot Learning: A Comprehensive Survey, arXiv:2605.00080
  *(the originally-cited survey; its §3/§4.1 are video-world-models for manipulation
  + RL-in-imagination — NOT our mocap→PPO locomotion recipe)*

**Neuroscience grounding:**
- Aldarondo et al. 2024, *Nature* (virtual rodent ≈ DLS/MC; inverse dynamics) —
  DOI 10.1038/s41586-024-07633-4 (+ 2025 Author Correction)
- Merel et al. 2020, Deep neuroethology of a virtual rodent, arXiv:1911.09451
- Wolpert–Kawato internal models · Todorov & Jordan 2002 (OFC) · cerebellum-as-
  forward-model (Boven/Costa 2023) · DLS = striatum = RL substrate

---

## 8. Framing corrections to bake into the slides
Carried from the review (see also PROJECT_STATE §4/§5):
1. **`16 / 277 / 38` are our MIMIC-MJX build's numbers**, not the Nature paper's
   (which reports a 60-D latent; only the 38 actuators are confirmed there).
2. **Our "frozen decoder + high-level policy" is the Merel NPMP design**; MIMIC (2024)
   trained end-to-end. Say "NPMP-style frozen primitive decoder, evaluated à la
   Aldarondo 2024."
3. **No clean MC-layer vs DLS-layer split** — both are best predicted by the decoder's
   first layer (same inverse-dynamics computation).
4. **Synergy = distributional, not sample-efficiency** (Section 1) — the load-bearing
   framing choice.
5. Our recordings are **DLS + MC** — DLS *is* the RL/basal-ganglia substrate, so we're
   data-rich on the RL pole and inferring the SSL/cerebellar pole from MC + behavior +
   theory. Frame that as scoping, not weakness.

---

## 9. Success criterion
The talk works if a new grad leaves able to say: **"RL made it move; SSL made it look
like a real brain; you need both, and here's the one scatter plot that shows why."**
The deliverable is that scatter (Section 2) with three real points from our rodent.
