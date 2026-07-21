# Embodied — Project State & Next Steps

_Last updated: 2026-07-21_

A working log of the neuromechanical rodent-control project: what it is, what's
built, what's proven, what's uncertain, and what to do next. Written as a handoff
document — someone (or some agent) picking this up should be able to continue
from here without re-deriving anything.

> **Workshop overlay (updated 2026-07-21).** Sections 1–8 below preserve the
> original `rl/` joystick-project history. The current live presentation is now
> Demo A (Fetch PPO), Demo B (conditional rodent motion), Demo F (the same SSL
> construction after contact-aware retargeting to Fetch), and Demo H
> (generative body/action pretraining followed by residual PPO with a frozen
> reference KL). See
> [WORKSHOP_PLAN.md](WORKSHOP_PLAN.md), [demo_a.md](demo_a.md),
> [demo_b.md](demo_b.md), [demo_f.md](demo_f.md), and [demo_h.md](demo_h.md).
> Demo H uses an explicitly separate `1.75x` empirical retiming of Demo F data,
> exact-physics action projection, a learned state/action prior, and a 30M-step
> PPO post-training run. The accepted single-seed checkpoint uses `beta=0.10`,
> survives every 5-second evaluation at commands 1.5–4.0, and has mean speed
> MAE 0.0790; strict stride validation passes 4/6 speeds. The complete local
> physical-build + prior + PPO path takes about 250 seconds. This is a
> pedagogical result, not an algorithm-level claim: the nearby `beta=0.075`
> run passes 5/6 stride gates, and naturalness metrics remain validation-only.
> Demo G is retained as an optional, three-seed reward-side contrast. Demos C,
> D, and E remain research references. In
> particular, Demo E's ten-minute full-skeletal-rodent run learned an upright
> stance rather than locomotion and must not be presented as the workshop
> result.

---

## 1. Goal

Drive the MIMIC-MJX virtual rodent from point A to point B, and (a separate,
longer-horizon aim) compare an SNN controller's activity against the motor-cortex
(MC) / dorsolateral-striatum (DLS) recordings from the MIMIC paper.

Two distinct "policies" are involved — don't conflate them:

- **Imitation policy** (published, frozen): reproduces recorded motion. Its
  encoder consumes a window of *future reference joint angles*, so it has **no
  goal input** and cannot be commanded to a target.
- **High-level joystick policy** (we train): the paper's Section 2.5 transfer.
  The imitation **decoder** is frozen and reused; a fresh policy is trained to
  emit the 16-dim "intention" latent the decoder consumes, conditioned on a
  `[vx, vyaw]` velocity command. A→B = pin/steer that command toward a target.

```
task_obs (56, incl. command) --> [high-level policy, TRAINED] --> intention (16) --.
                                                                                    |--> frozen decoder --> 38 torques --> physics
                                              proprioception (277) -----------------'
```

Only the decoder is reused (frozen); the high-level policy is trained from
scratch. The original imitation encoder is discarded. See Section 5 for why
"finetune the policy from pretrained weights" is not applicable here.

---

## 2. Repository layout

```
embodied/
├── pyproject.toml        uv project (py3.12); track-mjx editable + mirrored git pins
├── uv.lock
├── demo_a/               PPO quadruped workshop demo (complete)
├── demo_b/               self-supervised kinematic rodent demo (complete)
├── demo_c/               archived workshop prototype: world-action + PPO + neural eval
├── demo_d/               archived workshop prototype: hindsight-command torque PPO
├── demo_e/               frozen-motion-likelihood + paired RodentJoystick PPO
├── demo_f/               retargeted Coltrane-to-Fetch dataset and SSL prior
├── demo_g/               same-body Fetch task-only/task+prior PPO (complete, limited claim)
├── demo_h/               accepted body/action pretraining + residual PPO capstone
├── rl/                   OUR code (see Section 3)
│   ├── check_compat.py       proprioception-compatibility check (277 == 277)
│   ├── run_imitation.py      roll out + render the published imitation policy
│   ├── train_joystick.py     high-level transfer training launcher
│   ├── watch_health.py       compact per-eval learning-curve / collapse monitor
│   ├── render_joystick.py    roll out + render a trained joystick checkpoint
│   ├── README.md             per-script notes + all compatibility shims
│   ├── remote/               turnkey H100 remote-training setup
│   └── runs/                 training outputs (gitignored)
├── ref/
│   ├── repos/                upstream as submodules: track-mjx, stac-mjx, DART, MotionStreamer
│   ├── papers/               PDFs (gitignored)
│   └── docs/                 this document
├── model_checkpoints/    pretrained decoder from HF (gitignored, auto-downloaded)
└── videos/               rendered rollouts (gitignored)
```

Upstream pins: `vnl-playground` and `mujoco_playground` are git deps pinned in
both `ref/repos/track-mjx/pyproject.toml` and (mirrored, because uv does not
inherit a dependency's `tool.uv.sources`) the root `pyproject.toml`. Keep them in
sync.

---

## 3. What's built (all committed, all validated end-to-end)

| Script | Purpose | Status |
| --- | --- | --- |
| `check_compat.py` | Verify RodentJoystick proprioception (277) matches the decoder | PASS |
| `run_imitation.py` | Reference-driven imitation rollout + ghost render | Works; produces `videos/imitation.mp4` (reward ~4.71) |
| `train_joystick.py` | Frozen-decoder transfer training (RodentJoystick) | Works; converges (balance), see Section 4 |
| `watch_health.py` | Parse training log → one health line per eval + collapse flags | Works |
| `render_joystick.py` | Rebuild policy from checkpoint, roll out with pinned command, render + measure displacement | Works |
| `rl/remote/` | One-shot H100 setup + train script and workflow doc | Tested on a real H100 SXM (2026-07-17); two fixes folded in (see Section 5, #7 + wandb no-tty) |

---

## 4. Current status — RESOLVED: stand-still local optimum, root-caused and fixed

_Updated 2026-07-18 after the remote H100 run._

The plateau was a **stand-still local optimum**, not a benign curriculum — and a
one-line reward fix (`tracking_sigma` 0.25 → 0.05) breaks it and produces a
walker.

**Diagnosis (remote H100, default 3e8 config, stopped at ~100 M).** The baseline
reproduced the plateau exactly: balance saturated by ~30 M (episode_len ~1000,
fallen ~0 %) while forward speed stayed flat — `vx` 0.024 → 0.040 across 10
evals from 10 M to 94 M, ~5.5× short of the ~0.22 commanded. Ten datapoints on a
flat line settle the "curriculum vs optimum" question: it is an optimum.

**Root cause (the reward formula).** `RodentJoystick._tracking_lin_vel_reward`
(`vnl_playground/tasks/rodent/joystick.py:294`) is
`weight * exp(-(v_cmd - v_actual)² / sigma)` — note the denominator is `sigma`,
**not** `2·sigma²`. With `sigma=0.25` and mean command ~0.23, a **standing** rat
collects `exp(-0.23²/0.25) ≈ 0.81` of the maximum tracking reward. Moving
perfectly adds only the last 19 %, set against the −100 termination risk of
falling — so standing is a strong attractor. The `alive` bonus (0.1 → ~100/ep of
pure-survival reward) reinforces it.

**Fix + result (`tracking_sigma=0.05`, 50 M run).** Sharpening the peak drops the
standing payoff to `exp(-0.23²/0.05) ≈ 0.35`. Locomotion emerges:

| metric | baseline (sigma 0.25) | fixed (sigma 0.05) |
| --- | --- | --- |
| eval `vx` (random cmds) | 0.024 → 0.040 (flat, to 94 M) | 0.001 → 0.048 → 0.085 → **0.097** (climbing, by 52 M) |
| pinned `vx=0.3` render | 0.02–0.03 m / ~0.01 m/s | **0.373 m / 0.149 m/s over 2.5 s** (steady gait) |

That's a large improvement in a straight-line walk at 52 M steps. The corrected
renderer pins all three command copies (reward, flattened actor observation,
and nested decoder observation); the confirmed 2.5 s clip averages 0.149 m/s.
Older 0.204 m/s notes predate that renderer audit and should not be cited as the
current fixed-command measurement.

**Caveat / open question:** the `vx` climb was still flattening at 52 M
(slope +.047, +.037, +.012 over successive 13 M intervals), so sigma-alone may
converge somewhere around 0.10–0.15 eval-avg rather than the full command. A
longer run and/or the complementary levers (raise `tracking_lin_vel.weight`, cut
`alive.weight`) are the way to close the rest of the gap — see Section 6.

**Artifacts (gitignored).** Baseline run `RodentJoystick-highlvl-20260717-224040`
(ckpts 0…94 M); fixed run `RodentJoystick-highlvl-20260718-012844` (ckpts
13/26/39/52 M) + `joystick_52428800.mp4`. wandb project `embodied-joystick`
(`markson-team`): baseline `...224040`, fixed `...012844`, render `7q8gcqjz`.

Note: `tracking_sigma` tightening was **counterintuitive at first** — the 13 M
eval dipped to `vx≈0.001` (sharper peak → less gradient for a novice policy → it
nails balance first), then climbed hard from 26 M on. Don't kill a sigma run at
the first eval.

---

## 5. Key findings & gotchas (do not re-learn these)

### Compatibility shims (upstream skew: published checkpoint/data vs pinned repos)
All are applied in our code and documented in `rl/README.md`:

1. **WSL2 GPU: use `--extra cuda12`, not `cuda13`.** cuda13 pulls a CUDA 13.2+
   runtime that segfaults in `cuInit()` against WSL's stub libcuda
   (NVIDIA/cuda-samples#433). Not a driver-version issue. Does **not** apply on a
   real Linux datacenter GPU, but cuda12 is safe there too.
2. **Typed-key crash:** track-mjx branches on `key.ndim == 1` (raw-key era) but
   `jax.random.key(0)` is now a typed key (ndim 0). Shim:
   `jax.random.key = jax.random.PRNGKey`.
3. **`jax.device_put_replicated` removed** (brax 0.14 vs JAX 0.10). Alias to the
   repo's `replicate_for_pmap`.
4. **Eval callback hangs:** stock `create_policy_params_fn` runs a 1000-step
   Python-loop rollout + video render + ffmpeg fork every eval; the first
   post-training eval stalled at 99% GPU. Replaced with a checkpoint-only
   callback. Videos are rendered separately from checkpoints.
5. **Legacy ReferenceClips joint-name bug** (imitation only): loader returns all
   74 `names_qpos` instead of `names_qpos[7:]` (67); we strip the 7 root DoF.
6. **API/version drift:** `rollout.create_environment` takes `cfg.env_config`
   (not full `cfg`); no `render` module in the pinned commit; `TrackMjxObsWrapper.unwrapped`
   returns `self` (reach the inner env via `.env`); ghost cameras are suffixed
   (`close_profile-rodent` / `-ghost`).
7. **Remote box `LD_LIBRARY_PATH` shadows the venv CUDA libs.** Some rental GPU
   images preset `LD_LIBRARY_PATH=/usr/local/cuda/lib64`, whose system cuSPARSE
   (e.g. 12.5.8.93) is older than what jaxlib 0.10.2 expects and shadows the pip
   `nvidia-cusparse-cu12` (12.5.10.65) wheel in the venv → JAX raises "Unable to
   load cuSPARSE" and **silently falls back to CPU** (training "runs" but ~100×
   slower). Fix: `unset LD_LIBRARY_PATH` so JAX auto-locates the venv wheels; now
   done automatically at the top of `setup_and_train.sh`. Verify with
   `jax.devices()` → must print `CudaDevice`, not `CpuDevice`.

### Operational
- **Local compile is RAM-bound.** The 3080 box (15 GB RAM, ~4 GB free) took
  ~18 min to compile the 4096-env graph. A datacenter GPU with 100+ GB RAM
  compiles in ~2–3 min — this was the single biggest local time sink.
- **stdout is block-buffered when redirected;** eval metrics lag behind the
  flushed checkpoint prints. `train_joystick.py` now sets `PYTHONUNBUFFERED=1`
  (applies to new runs).
- **wandb online in a no-tty shell needs the key in the *env*, not just
  `~/.netrc`.** `wandb.init` raised `UsageError: api_key not configured (no-tty)`
  even with a valid `machine api.wandb.ai` netrc entry present. `WANDB_API_KEY`
  in the environment is always honored. `setup_and_train.sh` now hoists the key
  from `~/.netrc` into `WANDB_API_KEY` when it isn't already exported.
- **Memory is not the constraint:** 4096 envs used ~5 GB of 10 GB on the 3080.

### brax logs only eval metrics
No entropy / policy-loss / KL is surfaced by the progress callback, so collapse
is detected **behaviorally** (`watch_health.py`): episode_length rising, fallen%
falling, reward not crashing from peak, no NaN termination.

---

## 6. Next steps

### Done: diagnosis (see Section 4). The immediate question is answered.
The remote H100 run confirmed the stand-still optimum and `tracking_sigma=0.05`
produced a walker (0.204 m/s pinned). **Timing reality check:** at ~20k sps on
one H100 SXM, 10 M steps ≈ 10 min wall-clock, so **3e8 ≈ 5 hours ≈ ~$15**, not
the ~15 min this doc originally estimated. Budget accordingly; 5e7 (~50 min,
~$2.6) is enough to see a reward change take effect. Launch a tuned run directly:

```
export WANDB_API_KEY=...            # or rely on ~/.netrc; see remote/ notes
env -u LD_LIBRARY_PATH WANDB_MODE=online uv run python rl/train_joystick.py \
  --num_envs 8192 --batch_size 2048 --num_timesteps 5e7 --eval_every 10000000 \
  --env "reward_config.tracking_sigma=0.05" 2>&1 | tee rl/runs/train_sigma05.log
```
(`env -u LD_LIBRARY_PATH` + `WANDB_MODE=online` are the two shims from Section 5
when NOT going through `setup_and_train.sh`.)

### Next: close the remaining speed gap (eval-avg 0.097 → commanded ~0.22)
The sigma-0.05 `vx` climb was still flattening at 52 M, so it may not reach full
command on its own. In rough priority, and now that we know standing is the
attractor:
- **Train longer** at `sigma=0.05` (e.g. `--num_timesteps 2e8`, ~3 h, ~$10) —
  simplest; see if it keeps climbing past ~0.1. No resume is wired, so this
  restarts from scratch.
- **Cut `reward_terms.alive.weight`** (0.1 → 0.0): removes ~100/ep of
  pure-survival reward that rewards standing regardless of tracking shape.
- **Raise `reward_terms.tracking_lin_vel.weight`** (1.0 → 2–3): amplify the
  moving-vs-standing gap in absolute terms.
- A combined `"reward_config.tracking_sigma=0.05 reward_terms.alive.weight=0.0
  reward_terms.tracking_lin_vel.weight=3.0"` is the strongest single shot at a
  full-speed walker (at the cost of clean attribution).
Change one axis at a time if attribution matters; `--env` takes several
space-separated dotted overrides at once if it doesn't.

### A→B closed-loop control (now unblocked — a straight-line walker exists)
The joystick policy tracks `[vx, vyaw]`. For A→B, at inference replace the random
command sampler with a controller that points `vyaw` at the target bearing and
sets `vx`, stopping on arrival. Extend `render_joystick.py` (which already pins a
constant command) with a target-seeking command function. **Caveat:** the
sigma-0.05 checkpoint is validated only for straight-line `vx` (0.204 m/s pinned);
**turning quality (`vyaw` tracking) is untested** — the eval average mixes in
turns and was lower, so verify a `vyaw`-pinned render before trusting closed-loop
steering, or fold `vyaw` into the next training run's success criteria.

### SNN arm (longer horizon)
For SNN-vs-MC/DLS comparison, the SNN must replace the **decoder** (the layer
MIMIC found most MC/DLS-like), trained on **imitation** against behavior-matched
clips — *not* the joystick high-level policy (that's the encoder-level layer,
least MC/DLS-like, and joystick has no matching neural recordings). Cheapest path:
distill the pretrained ANN decoder into an SNN (supervised on rolled-out
(intention, proprioception)→action pairs), then optionally RL-finetune. Frame
"SNN spikes match real spikes better" as a hypothesis to test, not a given —
MIMIC's Poisson GLMs already bridge rate→spike.

---

## 7. Data & assets inventory (what git does NOT carry)

No secrets anywhere in the repo (public HuggingFace, offline wandb, no
credentials). A `git push`/`bundle` carries all code + history (~120 KB bundle).
What it omits, and whether it matters:

| Item | Size | Needed for remote training? | How to obtain |
| --- | --- | --- | --- |
| `.venv/` | 7 GB | no | regenerated by `uv sync` |
| `model_checkpoints/` (decoder) | 159 MB | **yes** | auto-downloaded from HF by setup script |
| `~/data/mimic-mjx/` (reference clips + `art/`) | 66 GB | **no** (imitation only, not joystick) | HF dataset `talmolab/MIMIC-MJX` if ever needed |
| `ref/papers/` (PDFs) | 34 MB | no | reference reading only |
| `rl/runs/` (checkpoints, logs, videos) | — | no | training *output* |
| `wandb/` | — | no | offline logs |

**Bottom line:** for the remote training path, git + the setup script's HF
download = everything. No large-file transfer, no secrets, no manual data copy.

### Local checkpoints on hand
`rl/runs/RodentJoystick-highlvl-20260717-143949-684649/` — steps `0`, `10485760`,
`20971520` (balance-only, pre-locomotion). The local run has been stopped.

---

## 8. Environment quick reference

- Python 3.12, `uv`. Install: `uv sync --extra cuda12` (WSL2) — see Section 5.
- Local GPU: RTX 3080 10 GB, WSL2, driver 591.86.
- Key env vars: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MUJOCO_GL=egl`
  (render only), `WANDB_MODE=offline`, `PYTHONUNBUFFERED=1`.
- Pretrained decoder run id: `feedforward_260210_013247_285744` (also the default
  `--mimic_checkpoint` in `scripts/train_highlvl.py`).
