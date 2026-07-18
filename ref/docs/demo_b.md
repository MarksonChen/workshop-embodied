# Demo B — SSL-only generative motion (as-built + review)

_Created 2026-07-18. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) (thesis). Code +
gritty details: [`demo_b/README.md`](../../demo_b/README.md)._

**Status: built ✅ (Phases 0 / 1 / 2b + heuristic waypoint). Two additions recommended to
place it on the 2×2 — see §4.**

---

## 1. Role in the thesis

Demo B is the **SSL-only** corner (WORKSHOP_PLAN §2): **distributional realism, ~zero
functional realism.** A generative motion model over rodent kinematics (no physics, no
reward); it walks convincingly, then a physics check falsifies it.

## 2. What was built (as-built)

A DART-style **latent-diffusion motion model** — a frozen motion tokenizer + a diffusion
transition (`MotionTrans`) — over the rodent mocap, reusing the **CANVAS** package as a
library (`canvas/out/motor_ckpt.pt`); everything is driven by `qpos(74)` in the vnl rodent
convention. PyTorch — a **separate stack** from the JAX/MJX `rl/`.

Files: `foot_metrics.py` (the instrument), `rollout.py`, `reproduce.py`, `train_phase1.py`,
`train_phase2.py`, `drive.py`, `waypoint.py`, `rerender.py`.

### Findings (from `demo_b/README.md` + code)
- **Foot-quality instrument validated** against real motion: skate ~0.003, penetration
  ~1 mm, jerk ~570 — a real-motion reference, so the target is *"as good as real"*, not
  *"exactly zero."*
- **Phase 1 (anti-jitter) ✅:** DART-style decode-space losses cut jerk **15–21 % toward
  real**; the rendered-channel jerk penalty (`sm`) and the aux velocity-consistency (`vc`)
  perform comparably; `vc` is canonical. Residual jerk is partly **frozen-tokenizer-bound**.
- **Phase 2 (Two-Forward) — negative result, kept:** scheduled sampling did **not** fix the
  orientation "flying" drift. Correctly root-caused: the drift is **open-loop orientation
  integration** in `reconstruct_qpos` (`R = R @ d6`), *not* teacher-forcing exposure bias —
  so no latent-space trick can anchor it.
- **Phase 2b — rollout-time re-anchoring ✅:** `anchor_orientation` (a leaky pull-to-level)
  cuts pitch drift **+27° → +2°** while keeping gait bob; the fix lives at the reconstruction
  layer, the correct one.
- **Waypoint reaching (heuristic, no RL) ✅:** a ~50-line closed-loop controller reaches
  **12/13** waypoints across three shapes → *the command is already an egocentric go-to-goal
  signal, so navigation is essentially steering; RL over the command is unnecessary.*

## 3. Review

Strong, honest, well-instrumented — the negative result + root-cause diagnosis are
exemplary. Two things for how it serves the **thesis**:

1. **The falsification is milder than the plan wants — lead with the "flying" drift.**
   Because the generator is trained on real mocap, its skate is *already real-like* and
   penetration is cosmetically fixable (`fix_floor`); it does **not** dramatically put
   "limbs through the floor." The vivid, honest falsification is the **open-loop orientation
   drift** — the trunk rears nose-up (+27° over 16 s) because nothing anchors pitch. Make
   *that* the money shot; foot-skate is the quantitative supporting number.
2. **Add a *functional* score — the one thing to build.** Demo B measures kinematic
   violations but never drops the generated `qpos` into **MJX physics**. For the 2×2's
   functional axis, set the generated motion as an MJX reference and show it is
   **dynamically infeasible** (can't be tracked / falls without forces) → the clean
   "functional realism ≈ 0" point. Small addition — `foot_metrics` already uses `mujoco` FK.

**Insight that reshapes Demo C.** The "RL-over-command is unnecessary for steering" finding
means RL earns its keep only at the **latent level or for extra objectives** (contact,
obstacles, timing). [demo_c.md](demo_c.md) is framed around that, not navigation.

## 4. For the 2×2 scatter
| Axis | Demo B score | Source |
|---|---|---|
| Kinematic-distributional | **high** (real-like gait) | `foot_metrics` + gait stats — measured |
| Functional | **~0** | needs the MJX-infeasibility score (addition #2 above) |
