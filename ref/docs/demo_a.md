# Demo A — RL-only from-scratch walker (plan)

_Created 2026-07-18. Companion to [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md) (thesis) and
[PROJECT_STATE.md](PROJECT_STATE.md) (assets)._

**⚠️ PIVOT (2026-07-19): `RodentMaintainVelocity` reward-hacks — moving to a retargeted
quadruped.** The spartan rodent env (forward-velocity reward only, raw torque,
`noslip_iterations=0`) doesn't learn to *walk* — it learns to **twitch-slide** (vibrate in
impossible postures and slide on foot friction). The fix is to copy a **proven** MuJoCo
quadruped locomotion env and retarget its body to a rodent:
- **Phase 1 DONE:** MuJoCo Playground **`Go1JoystickFlatTerrain`** (12-DoF, position/PD
  actuators, velocity-tracking + foot air-time/slip shaping) trains a **real walker in
  6.7 min** on one H100 (reward 28.6, full-episode survival). Trainer: `demo_a/train_go1.py`
  (with a `mjx.make_data` nconmax→naconmax shim). ~358k sps (18× the rodent env).
- **Phase 2 DONE:** a **12-DoF rodent** = Go1 physics verbatim + rodent primitive visuals
  (`demo_a/models/rodent_go1.xml`, self-contained, injected via `train_go1.py --model`).
  Trains a real walker (**0.79 m/s, upright, full-episode**) in **3.9 min** on one H100 at
  1e8 steps — the **hard <10-min live-workshop budget met with margin** (rat visuals add
  ~28% FK cost, so cap at 1e8, not the stock 2e8). This is Demo A's walker: reduced DoF,
  reliable, a real gait (not the RodentMaintainVelocity twitch-slide), rodent-shaped.
  Video: `demo_a/out/go1_go1_<step>.mp4`. Render with
  `render_go1.py <ckpt> 500 demo_a/models/rodent_go1.xml`.

The `RodentMaintainVelocity` probe below is kept for the record (it *is* a clean example of
reward-hacking for the talk), but is **not** the Demo A walker.

---

**Status: convergence probe DONE (2026-07-19).** A competent-but-**veering** walker emerges
from scratch — **~0.4 m/s forward by ~35 M steps (~35 min, one H100)**, then survival keeps
improving (fallen 100%→50% by 53 M). Not flailing (§2 risk retired), not natural (it curves —
on-thesis). Full results + the metric lesson (reward conflates speed×survival → track speed +
fallen% separately) in [demo_a/README.md](../../demo_a/README.md). Remaining: reward-shaping
variants + the distributional (gait-vs-real) axis.

---

## 1. Role in the thesis

Demo A is the **RL-only** corner of the 2×2 (WORKSHOP_PLAN §2): **functional realism,
~zero distributional realism.** It must be a *competent-but-unnatural* walker — moves
forward fine, but its gait and internal representation don't match the real rat.

It is the **true no-prior baseline** — explicitly **not** our existing joystick walker,
which rides the imitation-pretrained NPMP decoder, already carries a data prior, and
therefore sits near the *both* corner (see §5).

## 2. The one risk that governs the whole design

From-scratch RL on a 38-DoF torque-actuated body can **flail** instead of walking. A
flailing Demo A conflates "RL can't control 38 DoF" (a compute artifact) with "RL isn't
distributionally real" (the actual point). So the plan is built around making it
**competent-but-ugly, cheaply**, and *not* rewarding naturalness — the ugliness must be
emergent, not imposed.

## 3. Stack

- Same MJX rodent body + brax PPO as `rl/`, but with the **decoder removed**: the policy
  outputs **raw torques (38-dim action)** from proprioception. **Env:
  `RodentMaintainVelocity`** (registered in vnl_playground; `torque_actuators=True`, a
  forward-velocity reward, and *no upright reward* — so it must learn to stay up) — trained
  directly via track-mjx `scripts/train_task.py` (**not** `train_highlvl.py`, which wraps the
  decoder). No new env to build.
- Reuse `rl/` tooling: the `train_joystick`-style launcher, `watch_health.py`,
  `render_joystick.py`, and the `LD_LIBRARY_PATH` + wandb fixes (PROJECT_STATE §5).

## 4. Staged build

### Stage 0 — de-risk (before spending real GPU)
Short smoke run with a **well-shaped locomotion reward**:
- forward-velocity tracking + alive + upright/orientation + energy/torque penalty +
  **action-rate smoothness** (the term that most reliably turns flailing into a gait).
- Confirm a *stable* gait emerges cheaply. If it flails: add a **curriculum** (ramp
  commanded speed, forgiving termination) or adopt a proven MuJoCo-Playground
  quadruped/locomotion reward before committing.

### Stage 1 — train
Train to a competent-but-ugly walker. Budget **~100–300 M steps** (raw-torque-from-scratch
is harder than the decoder-based joystick, which needed ~50 M) → real H100 time.
Checkpoint + monitor as in PROJECT_STATE.

### Stage 2 — score on the shared axes (WORKSHOP_PLAN §4)
- **Functional:** commanded-velocity tracking in MJX — should score **well**.
- **Kinematic-distributional:** gait statistics (stride frequency, duty factor,
  joint-angle ranges, inter-joint correlations) vs the real-rat mocap distribution —
  should score **poorly** (the point). Reuse `demo_b/foot_metrics.py` + a small gait-stats
  module over the Aldarondo `qpos`. Mind the data gotchas ([dataset.md](dataset.md)):
  **merge locomotion bouts** (raw runs are shredded by label flicker) and **finite-diff
  `qvel`**.
- **Neural (Phase 2):** MIMIC Poisson-GLM + RSA of its activations vs DLS/MC — should
  out-predict raw kinematics **least** of the three demos.

## 5. The minimum-viable thesis demo (do this first)

**Demo A + our existing joystick walker** is *already* a clean two-point demonstration of
the thesis, buildable now with no dataset dependency:
- raw-torque RL (no prior) → moves, but gait/representation wrong;
- decoder prior + PPO → moves **and** looks real.

Treat this pair as the MVP; the SSL-WAM ([demo_c.md](demo_c.md)) is the principled upgrade
of the "data prior."

## 6. Open questions
- ~~Torque-control env?~~ **Resolved: `RodentMaintainVelocity`** (raw torque,
  forward-velocity reward, no upright reward) via `scripts/train_task.py`.
- How unnatural is the emergent gait, quantitatively, vs real? *That number is the demo.*
- The neural axis is blocked until the DLS/MC comparison pipeline exists (shared with
  [demo_c.md](demo_c.md) and the story-map "Arc 01").
