# Embodied — Project State & Next Steps

_Last updated: 2026-07-17_

A working log of the neuromechanical rodent-control project: what it is, what's
built, what's proven, what's uncertain, and what to do next. Written as a handoff
document — someone (or some agent) picking this up should be able to continue
from here without re-deriving anything.

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
| `rl/remote/` | One-shot H100 setup + train script and workflow doc | Ready (untested on a real instance) |

---

## 4. Current status — the key result

The transfer **works, and does not collapse**, but has **not yet produced a
walking policy**. Evidence from the local run (RTX 3080), pinned forward command
`vx=0.3 m/s`:

| step | episode_len | fallen | reward | fwd speed (measured) |
| ---: | ---: | ---: | ---: | ---: |
| 0 (random) | 223/1000 | 100% | 98 | 0.000 m/s |
| 10.5 M | 905/1000 | 20% | 1389 | 0.009 m/s |
| 21.0 M | (upright, stable) | — | — | 0.011 m/s |

**Interpretation:** the policy mastered **balance** (survival 223→905 steps, no
collapse, reward 14×) but forward **locomotion has not emerged** — ~0.01 m/s vs a
commanded 0.3. Confirmed two ways: the env's own eval metrics (`vx 0.005` at 10 M
under real sampled commands) and pinned-command renders (displacement 0.022 m /
0.029 m over 2.5 s at 10 M / 21 M).

This is **20 M steps = ~7 % of the planned 3e8 run.** Two readings, unresolved:

1. **Balance-first curriculum (benign):** locomotion often emerges only after
   balance is solid; the paper's rat imitation needed ~0.65 B steps.
2. **Stand-still local optimum (needs a reward fix):** the reward is asymmetric —
   `termination -100` dwarfs `alive +0.1` / `stand_still -0.5`, so "stand still,
   don't risk falling" may be a strong attractor.

Two datapoints can't distinguish these. The fast remote run (Section 6) is the
cheapest way to find out.

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

### Operational
- **Local compile is RAM-bound.** The 3080 box (15 GB RAM, ~4 GB free) took
  ~18 min to compile the 4096-env graph. A datacenter GPU with 100+ GB RAM
  compiles in ~2–3 min — this was the single biggest local time sink.
- **stdout is block-buffered when redirected;** eval metrics lag behind the
  flushed checkpoint prints. `train_joystick.py` now sets `PYTHONUNBUFFERED=1`
  (applies to new runs).
- **Memory is not the constraint:** 4096 envs used ~5 GB of 10 GB on the 3080.

### brax logs only eval metrics
No entropy / policy-loss / KL is surfaced by the progress callback, so collapse
is detected **behaviorally** (`watch_health.py`): episode_length rising, fallen%
falling, reward not crashing from peak, no NaN termination.

---

## 6. Next steps

### Immediate: remote H100 run to answer "curriculum or local optimum?"
Recommended: **H100 SXM 80 GB** (~$2.99/hr; a ~15-min run ≈ $0.75). Setup is
turnkey in `rl/remote/` (see its README). Flow:

1. `git bundle create /tmp/embodied.bundle --all` → scp to instance → clone +
   `git submodule update --init ref/repos/track-mjx`.
2. `bash rl/remote/setup_and_train.sh` (uv sync cuda12, download 158 MB decoder,
   train `num_envs 8192 batch 2048 3e8`, checkpoint every 10 M).
3. `uv run python rl/watch_health.py rl/runs/train.log` in another shell.
4. Bring the final checkpoint back; render locally with `render_joystick.py`.

**Read the `vx` trend across evals:**
- Forward speed climbs toward the command → curriculum; let it finish, we have a
  walker.
- Still ~0.01 m/s at 3e8 → stand-still optimum; tune rewards (below) and rerun
  (~15 min each).

### If locomotion doesn't emerge: reward tuning (one-line `--env` overrides)
Candidates, in rough priority:
- Raise `reward_terms.tracking_lin_vel.weight` (default 1.0).
- Cut `reward_terms.alive.weight` (default 0.1) — less pure-survival incentive.
- Soften `reward_terms.termination.weight` (default -100) — reduce fall-fear.
- Possibly lower `command_config.zero_prob` (default 0.1) so the policy is asked
  to move more often.
Change one axis at a time; the H100 makes each test ~15 min.

### A→B closed-loop control (once a walker exists)
The joystick policy tracks `[vx, vyaw]`. For A→B, at inference replace the random
command sampler with a controller that points `vyaw` at the target bearing and
sets `vx`, stopping on arrival. Extend `render_joystick.py` (which already pins a
constant command) with a target-seeking command function.

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
