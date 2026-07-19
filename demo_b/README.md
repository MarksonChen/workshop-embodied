# demo_b — rodent locomotion motor model, standalone

A minimal, **self-contained** version of the `rl/` side project: load a trained rat-locomotion motor model, roll it
out under egocentric commands, measure foot-contact quality, and drive it to custom waypoints. **The runtime imports
nothing outside this directory** — only `torch`, `numpy`, `mujoco`, `matplotlib`, `mediapy` (deps, not project code).
Lifted from the CANVAS repo; see `../rl/` for the research history that produced the model and these techniques.

## What the model is

A frozen **motion tokenizer** (causal conv VAE, 64-frame crop → 16×16 latent) + a **transition** that predicts the
next 8 motion latents from the last 8, conditioned on a per-step egocentric command `γ = [dx_ego, dy_ego, dψ]`.
Rolled out autoregressively and decoded to `qpos(74)` for the MuJoCo rodent.

The transition (`models.SimpleTrans`) is deliberately a **standard `nn.TransformerEncoder` + an MLP head, trained
with plain MSE** — no rotary/QK-norm attention, no diffusion sampler, no per-session table. That is the *graduated*
design from the ablation in `exploration/` (`DECISIONS.md`): a sequential autoresearch loop showed the shipped
model's RoPE attention, 10-step diffusion head, and session embedding each change the rollout by *less than the
measured noise floor*, so they were removed. The simplified model walks the same (slightly smoother) and trains ~2×
faster.

## One-time setup (bootstrap the weights)

Runtime code is standalone, but the 20 MB weights+seed bundle is generated once from the parent CANVAS checkpoints
(the repo does not commit `*.pt`). From the CANVAS repo root:

```bash
uv run python demo_b/make_assets.py        # trains SimpleTrans on parent loco data + bundles it (imports canvas)
```

This trains the simplified transition (~3 min) on the parent repo's locomotion data with the frozen tokenizer, and
writes `assets/motor_standalone.pt`. It is the ONLY file here that reaches outside `demo_b`.
`assets/arena.xml` + `assets/rodent.xml` (the MuJoCo model, primitives only — no meshes) are already bundled.
Dependencies: `torch numpy mujoco matplotlib mediapy imageio-ffmpeg pillow` (already in the CANVAS env; `uv run …`).

## Run

```bash
uv run python demo_b/drive.py --seconds 16 --render     # go straight / walk in cycles + path plot
uv run python demo_b/waypoint.py --shape square --render  # closed-loop reach 4 waypoints (also: star, zigzag)
```

Outputs (mp4/png) land in `out/` (git-ignored). This box renders headless via `MUJOCO_GL=osmesa` (set automatically).

## Layout

```
demo_b/
  constants.py     frozen grid + feature layout (FPS, CLIP, NLAT, DM, H, K, FM, slices)
  geometry.py      quat/6D rotation helpers + reconstruct_qpos (root-local vel + orient delta -> qpos)
  models.py        MotionVAE (frozen tokenizer) + SimpleTrans (std Transformer + regression head) + load_motor()
  mujoco_rodent.py build the arena+rodent MjModel from assets/*.xml
  foot_metrics.py  skate / penetration / jerk (MuJoCo FK) + fix_floor / fix_upright / anchor_orientation / ground
  rollout.py       autoregressive roll (hindsight / constant / fixed command) + MuJoCo render
  drive.py         fixed-command demo (straight / circle) + top-down path plot
  waypoint.py      closed-loop heuristic waypoint controller (steer the command; no RL)
  make_assets.py   BOOTSTRAP ONLY — train SimpleTrans + bundle assets/motor_standalone.pt (imports canvas)
  exploration/     the autoresearch ablation that simplified the transition (objective/lib/run + DECISIONS.md)
  assets/          arena.xml, rodent.xml (tracked) + motor_standalone.pt (generated, git-ignored)
```

## Two things to know about the rollout

- **Kinematic re-grounding.** `reconstruct_qpos` integrates root height + orientation open-loop, so over long rolls
  the paws sink and the trunk pitches nose-up ("flies"). `ground(gq)` = `anchor_orientation` (a leaky gravity pull that
  removes the drift while keeping gait bob) + `fix_floor` (snap paws to the floor). Render-side only; the model is
  honest to ~2.5 s open-loop.
- **Waypoint reaching needs no RL.** The command is already an egocentric go-to-goal signal, so a closed-loop
  controller that re-points it at the goal each step reaches waypoints on its own (square 4/4, star 5/5, zigzag 3/4).
  RL would only be worth it for latent-level control or extra objectives (foot contact, obstacles, timing).
