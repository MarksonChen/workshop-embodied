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

The canonical recipe is now [demo_c.md](demo_c.md):
- **World factor (SSL):** learn the action-conditioned next-state map
  `W(x_t, a_t) → x_{t+1}` from Demo A rollouts. The target is created by shifting a
  trajectory by one step.
- **Action factor (RL):** use the frozen world factor as a short-horizon learned
  simulator and PPO-post-train an early Demo A checkpoint inside it.
- **Reality check:** deploy the frozen post-trained policy back in the real Brax
  environment and expose the dream-to-real gap.
- **Teaching point:** SSL learns *what happens if I do this?*; RL learns *what should I
  do?* The core demo teaches how the objectives combine. A later real-rat-data extension
  is still required for the stronger distributional-realism claim.
- **Feasibility:** buildable now from the completed Demo A assets; no mocap dependency.

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

[demo_c.md](demo_c.md) supersedes the former MTM/intention-IDM recipe. Its minimal
factorization follows WorldModel.pdf §3.1:

```text
p(x_{t+1}, a_t | x_t)
    = pi(a_t | x_t) p(x_{t+1} | x_t, a_t)
      └─ action factor ┘ └────── world factor ──────┘
```

1. **Collect:** roll out Demo A checkpoints with modest action noise and save
   `(x_t, a_t, x_{t+1})`.
2. **Learn the world factor with SSL:** fit a small residual MLP to predict the next
   state. Keep Demo A's reward function explicit; do not hide it in the SSL loss.
3. **Learn the action factor with RL:** freeze the world model, initialize from an early
   Demo A checkpoint, and run the same PPO in short imagined episodes with frequent
   resets to recorded states. This is the learned-simulator role in WorldModel.pdf §4.1.
4. **Return to reality:** evaluate the unchanged policy in real Brax and report both
   imagined and real return, speed, upright time, and falls.

The architecture stays factorized so a beginner can identify the two learning signals:
next-state error trains the world; reward and return train the policy. The research-grade
intention-IDM, differentiable MJX, LAPO port, VQ bottleneck, and auxiliary-method stack
remain archived in [demo_c_prev.md](demo_c_prev.md), not in the workshop build.

---

## 6. Data dependencies — what we're waiting for, what's buildable now

**BLOCKED on the dataset the user is getting:**
- **Rodent mocap clips** (HF `talmolab/MIMIC-MJX`, ~66 GB) → needed for the
  kinematic-realism metric and the optional real-rat extension of Demo C, not its core
  SSL + RL lesson.
- **DLS/MC recordings** (Harvard Dataverse `10.7910/DVN/FB0MZT`; Ölveczky-lab
  lineage) → needed for the neural-realism axis of Section 4.
- The MIMIC RSA / encoding-GLM pipeline (`github.com/diegoaldarondo/virtual_rodent`,
  on request) → needed to score the neural axis.

**Buildable NOW, no external dataset:**
- Demo C's complete minimal pipeline — its transition data comes from Demo A rollouts.
- Metric/eval scaffolding (functional tracking metric; kinematic-stats harness ready
  to point at clips once local).

**Phasing:**
- **Core workshop:** Demos A and B are built; build the minimal Demo C from Demo A data.
- **Real-rat extension:** add mocap-based kinematic-realism scoring and, if wanted, train
  a rodent-data world factor.
- **Neural extension:** neural-realism scoring for all three → the full 2×2.
- **Optional:** swap the decoder for a **BrainPy SNN** (ties to the SNN arm
  in PROJECT_STATE §6) and re-score.

> **STATUS: Demo C core is buildable now and not started. External data blocks only the
> stronger real-rat distributional/neural claims.**

---

## 7. Literature anchors (verify 2026 IDs before a slide)

**Predictive-pretraining research extensions (context, not the core demo):**
- MTM — Masked Trajectory Models, ICML 2023, arXiv:2305.02968 *(locomotion-proven; primary anchor)*
- Humanoid locomotion as next-token prediction, NeurIPS 2024, arXiv:2402.19469
- SMART, ICLR 2023, arXiv:2301.09816 (DMC locomotion)
- RPT — Robot Learning w/ Sensorimotor Pre-training, CoRL 2023, arXiv:2306.10007 (manipulation)
- LAPO — Learning to Act without Actions, ICLR 2024, arXiv:2312.10812

**World-model RL (the current Demo C lineage):**
- World Models, Ha & Schmidhuber 2018, arXiv:1803.10122
- Sutton, Dyna, 1991 · MBPO, NeurIPS 2019, arXiv:1906.08253
- DreamerV3, arXiv:2301.04104 *(modern context, not an implementation dependency)*

**"WAM" as a named term (2026, manipulation/VLA — cite provenance honestly):**
- World Action Models: The Next Frontier in Embodied AI, arXiv:2605.12090
- World Model for Robot Learning: A Comprehensive Survey, arXiv:2605.00080
  *(§3 supplies the factorized predictive-control view; §4.1 supplies RL in a learned
  simulator. Demo C is their state-space teaching analogue, not a video/VLA reproduction.)*

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
4. **Synergy = distributional, not sample-efficiency** (Section 1) remains the
   load-bearing research claim. The minimal Demo C teaches the SSL + RL mechanism; do
   not count self-collected simulator data as evidence for real-rat distributional
   realism. That claim waits for the mocap/neural extension.
5. Our recordings are **DLS + MC** — DLS *is* the RL/basal-ganglia substrate, so we're
   data-rich on the RL pole and inferring the SSL/cerebellar pole from MC + behavior +
   theory. Frame that as scoping, not weakness.

---

## 9. Success criterion

The core workshop works if a new grad can say: **"SSL learned what happens next from
the trajectory itself; RL used reward to choose actions; Demo C let PPO practice inside
the learned world."**

The stronger research story is complete only when the mocap/neural extension supports
the original claim: **"RL made it move; SSL made it look like a real brain."** Its
deliverable remains the Section 2 scatter with three empirically measured points, not
points assigned from method labels alone.
