# demo_b/ — DART-style locomotion RL (side project)

**Scope.** A self-contained exploration, **not part of CANVAS**. It reuses the CANVAS package only as a *library*
(the trained motor generative model + the MuJoCo rodent + constants) and touches nothing under `canvas/` or the CANVAS
docs. **No neural activity** — this is purely about kinematic locomotion quality.

**Goal.** Explore the DART recipe for locomotion (`ref/docs/lit/motor_generation.md`): a compact motion model whose
rollout *walks*, with the classic kinematic artifacts — **foot slide + floor penetration + jitter** — measured and then
suppressed. DART's design: keep the generator trained on reconstruction + consistency losses, and shape foot contact
either as a training loss on the decoded output, as test-time latent-noise optimization, or (eventually) as an RL policy
over the model's latent action space with skate/floor-contact rewards.

## Substrate

The DART-style "primitive decoder" here is CANVAS's **motor** model (motion tokenizer + transition), loaded from a
checkpoint (`canvas/out/motor_ckpt.pt` / `ms_ckpt.pt`). The MuJoCo rodent walker
(`ref/repos/vnl-playground/.../rodent/xmls/`, via `canvas.utils.build_model()`) provides forward kinematics for the
foot metrics and rendering. Everything is driven by `qpos(74)` in that rodent convention.

## Layout

```
demo_b/
  README.md          this file
  foot_metrics.py    DART-style foot-quality metrics (skate / penetration / jerk) via MuJoCo FK; fix_floor + fix_upright + ground
  rollout.py         load the motor model (CANONICAL = Phase-1), roll out under constant/hindsight command, measure + render
  reproduce.py       faithful reproduction of the autoresearch motor rollout (stage3.motor_rollout) + save + score
  train_phase1.py    Phase 1: + DART decode-space losses (velocity-consistency / rendered-jerk) to cut jitter
  drive.py           drive the model with a FIXED command (go straight / walk in cycles) + top-down path plot
  waypoint.py        closed-loop HEURISTIC waypoint controller (no RL) -- steer the command to custom goals
  rerender.py        re-render the comparison clips as raw | grounded (no retrain)
  out/               models (.pt) + videos (.mp4) + logs  (all git-ignored)
```

**Kinematic re-grounding for rendering.** `reconstruct_qpos` integrates root height and orientation open-loop, so
over a long rollout both drift: paws sink through the floor and the trunk pitches nose-up ("flies"). `fix_floor`
snaps the lowest paw to the floor; `fix_upright` flattens the accumulated pitch/roll to yaw-only; `ground()` does
both. These are **render-side cosmetic corrections** (like DART's `fix_floor`) — the model-side fix for the drift is
scheduled sampling (Phase 2, not yet built).

## Run (this box: RTX 3080, WSL2 → osmesa render)

```bash
# measure the metric instrument on real motion (validation + the reference baseline the rollout must approach)
uv run python -m demo_b.foot_metrics                       # or: uv run python demo_b/foot_metrics.py
```

The scripts set `MUJOCO_GL=osmesa` themselves. Metrics are reported in the rodent's world frame; the arena floor plane
is at **z = 0**. Real STAC-fit motion sits near the floor (paw soles dip to ~−20 mm from marker-fit slack), so
`foot_metrics` prints a **real-motion reference** to contextualize generated numbers — the target is *"as good as real"*,
not *"exactly zero"*.

## Status

**Phase 0 — see + measure ✅.** `foot_metrics.py` is a validated instrument (real motion scores skate ~0.003,
penetration ~1 mm, jerk ~570). `reproduce.py` faithfully reproduces the autoresearch motor rollout — it *walks*
(modest slide + mild penetration + mild jitter); the earlier "vigorous jitter + falling" was the wrong artifact
(live_link `motor_ckpt.pt`) + an off-distribution seed, both eliminated. `fix_floor` removes penetration for free.

**Phase 1 — anti-jitter loss ✅.** Added DART-style decode-space losses to the transition training, step-matched to
the baseline. Key finding: DART's delta-consistency does not transfer verbatim — CANVAS's `reconstruct_qpos` reads
joint angles (`jq`) directly, so the `qd`/`kpd` velocity channels are *auxiliary* (never rendered). Built both the
literal port (`vc`, velocity-consistency on aux channels) and the correct-for-CANVAS lever (`sm`, a jerk penalty on
the rendered `jq`+root channels — DART's `calc_jerk`). They perform comparably.

| jerk (9 s rollout) | baseline | vc | sm | real |
|---|---|---|---|---|
| seed 07_29 | 754 | **636** | 657 | 570 |
| seed 07_30 | 1095 | **888** | 865 | 570 |

Both cut jerk ~15–21% toward real with skate staying real-like; `vc` reaches it at lower fidelity cost, so it is the
**canonical** model (`out/motor_phase1_vc.pt`, loaded by default). Best recipe = a jerk loss **+ `fix_floor`**:
penetration → 0, skate ≈ real, jerk ~640–890. Residual jerk (vs real 570) is partly **frozen-tokenizer-bound**
(`vc(tokenizer recon)=0.99`) — closing it needs a Stage-1 tokenizer re-mint (deferred; would break `CLAIMS#c1`).

**Phase 2 — Two-Forward ❌ (negative result, kept).** Ported MotionStreamer's Two-Forward (= scheduled sampling; the
exposure-bias fix CANVAS's full Stage-C Transition already uses at `train.py:637-647`, which the motor
reimplementation had dropped) into the motor trainer (`train_phase2.py`). **It did not fix the orientation drift**
("flying"): pitch @16 s stayed +25→+39° across baseline / Phase-1 / Phase-2 alike; foot quality unchanged.
*Why:* the drift is **not** teacher-forcing exposure bias. It is open-loop **orientation integration** — `reconstruct_qpos`
integrates the 6D delta `d6` (`R = R @ d6`) *after* decode, and the latent-space training loss never sees it; the
transition predicts *relative* deltas and never observes absolute orientation, so no latent-space trick (Two-Forward
or deeper) can anchor it. Confirmed the frozen tokenizer adds no `d6` bias (mean `d6` real ≈ recon). ⇒ the fix must
live in the **reconstruction/rollout** path, not the transition.

**Phase 2b — rollout-time re-anchoring ✅.** `anchor_orientation` re-integrates the trunk orientation from the original
`d6` deltas with a gentle gravity pull each frame (a leaky `fix_upright`). On the 16 s straight roll, pitch drift
**+27° → +2°** while **1.3° of gait bob survives** (the hard yaw-only `fix_upright` kills it to 0°). Since XY integration
uses only yaw, re-leveling never moves the path. This confirms the drift is a reconstruction-path problem, fixable
there — the transition (relative deltas, blind to absolute orientation) is the wrong layer. `ground(gq, alpha=0.03)`
= leaky anchor + floor is the render default for `drive.py`.

**Waypoint reaching — heuristic first (no RL) ✅.** Before building an RL policy, the honest baseline: a ~50-line
closed-loop controller (`waypoint.py`) that re-points the egocentric command at the current goal each step (turn
clamped to the feasible range, always some forward so it arcs; position feedback by decoding the stream). It reaches
**12/13 waypoints** across three shapes (square 4/4, star 5/5, zigzag 3/4) with **no learning**. The one miss is a
control-precision orbit at the tail of the longest aggressive-turn path, not a steering failure. **Conclusion: the
command is already an egocentric go-to-goal signal, so navigation is essentially steering — RL over the *command* is
unnecessary.** RL earns its keep only for latent-level control or extra objectives (foot contact, obstacles, timing);
that is the open question if the project continues.

**Not yet built:** test-time DNO latent guidance, RL over the *latent* action space (only if a task needs more than
steering), tokenizer re-mint (Phase 3).
