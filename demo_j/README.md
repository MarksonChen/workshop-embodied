# Demo J — spiking imitation and neural similarity

Demo J trains a recurrent spiking controller on the same `1.75x` retargeted
Fetch-motion release used by Demo H, then treats its 20 ms spike counts as a
synthetic neural recording. The current comparison asks how Demo H activity
changes across prior strengths `beta` under identical locomotion inputs.

The package intentionally supports only three workflows:

1. short-clip SNN imitation, which is the accepted behavioral result;
2. aligned 1,000-bin pretraining plus an exploratory readout-PPO probe;
3. fixed-input RSM/RSA against the matched Demo H beta sweep.

Historical ANN probes, short-horizon PPO variants, controller-generated-prior
experiments, motor-distance overlays, and the superseded Poisson/DLS command
stack were removed. Their conclusions remain summarized in [RESULTS.md](RESULTS.md)
and the append-only [AUTORESEARCH.md](AUTORESEARCH.md).

## Scientific contract

- Data: accepted Coltrane locomotion clips retargeted to Fetch and slowed
  `1.75x`, with session-disjoint train/validation/test splits.
- Physics: paired controls are deterministically replayed in modern MJX; this
  is not an ML reconstruction or animal inverse dynamics.
- Actor: 128 LIF plus 128 adaptive-LIF neurons, four 5 ms substeps per 20 ms
  action, hard forward spikes, and BrainPy surrogate gradients.
- Independence: no Demo H policy, hidden activation, beta value, or biological
  spike is used to train or select the SNN.
- Analysis: compare population geometry rather than aligning units one-to-one;
  use exact SNN input and a 200 ms delayed Demo H control.
- Naturalness and neural similarity are validation measurements, never rewards.

## Supported workflows

Install and verify the runtime:

```bash
uv sync --extra demo-j --extra dev
uv run python -m demo_j.cli smoke --output demo_j/out/runtime_smoke.json
uv run pytest -q demo_j/tests
```

Build the deterministic modern-MJX cache and train the accepted short-clip
controller:

```bash
uv run python -m demo_j.cli build-cache
uv run python -m demo_j.cli train-imitation --seed 1
uv run python -m demo_j.cli evaluate-imitation \
  --checkpoint demo_j/out/snn_distilled_seed1_<stamp>.pkl \
  --split test --output demo_j/out/snn_canonical_test.npz
MUJOCO_GL=egl uv run python -m demo_j.cli render-imitation \
  --recording demo_j/out/snn_canonical_test.npz \
  --output demo_j/out/snn_imitation_speed_sweep.mp4
```

Run the aligned long-horizon experiment:

```bash
uv run python -m demo_j.cli fit-tokenizer
uv run python -m demo_j.cli train-aligned \
  --preview-tokens 8 --seed 0 --episode-batches 64 \
  --episode-steps 1000 --chunk-steps 50
uv run python -m demo_j.cli train-ppo \
  --init-checkpoint demo_j/out/aligned/snn_aligned_preview8_seed0_<stamp>.pkl \
  --num-updates 20 --num-envs 128 --minibatch-envs 16 \
  --episode-steps 1000 --unroll-steps 1000 --eval-every 2 \
  --run-id aligned-long-balanced
uv run python -m demo_j.cli evaluate-aligned \
  --checkpoint demo_j/out/aligned/snn_aligned-long-balanced_seed0_<stamp>.pkl \
  --steps 1000 --speeds 1.5 2.0 2.5 3.0 3.5 4.0 \
  --output demo_j/out/aligned/rollout_1000_long_balanced_seed0.npz
MUJOCO_GL=egl uv run python -m demo_j.cli render-aligned \
  --recording demo_j/out/aligned/rollout_1000_long_balanced_seed0.npz \
  --output demo_j/out/aligned/snn_1000_step_speed_sweep_aligned.mp4
```

The `analysis` bridge is invoked through `export-h-trace` and
`export-h-activations` in Demo H's isolated legacy environment. It exports one
fixed trajectory bank plus activations. In the modern environment, record each
SNN seed on that bank, compare, and plot:

```bash
uv run python -m demo_j.cli record-aligned \
  --checkpoint demo_j/out/aligned/snn_aligned_preview8_seed0_<stamp>.pkl \
  --trace demo_j/out/aligned/h_fixed_trace_1032.npz \
  --output demo_j/out/aligned/snn_fixed_1000_seed0.npz
uv run python -m demo_j.cli compare-rsa \
  --recording demo_j/out/aligned/snn_fixed_1000_seed[012].npz \
  --trace demo_j/out/aligned/h_fixed_trace_1032.npz \
  --activation demo_j/out/aligned/*_activations.npz \
  --output demo_j/out/aligned/beta_rsa_full_input.json --permutations 1000
uv run python -m demo_j.cli plot-rsa \
  --report demo_j/out/aligned/beta_rsa_full_input.json \
  --output-dir demo_j/out/aligned/rsa_full_input
```

For a notebook, import the supported functions from `demo_j.api`; internal
modules are not intended as a stable presentation interface.

## Package map

```text
api.py                       stable notebook-facing functions
artifacts.py                 paths, hashes, checkpoint compatibility, JSON I/O
cli.py                       single lazy command dispatcher
data/
  dataset.py                 accepted data and source provenance
  physics.py                 Fetch MJCF access and joint/site mappings
  projection.py              deterministic modern-MJX replay cache
control/
  config.py, snn.py          SNN timing contract and LSNN dynamics
  policy.py, imitation.py    short imitation policy and sequences
  tracking.py                short reference-tracking environment
  aligned.py                 PCA tokens and disclosed periodic sequences
  aligned_tracking.py        long functional locomotion environment
  ppo.py                     minimal PPO math
experiments/
  train_imitation.py         accepted short sequence distillation
  evaluate_imitation.py      held-out short evaluation
  aligned.py                 tokenizer fitting and aligned pretraining
  aligned_rollout.py         long rollout and fixed-input spike recording
  train_ppo.py               full-horizon readout PPO
  render.py                  short comparison video
  render_aligned.py          long comparison video and speed audit
analysis/
  bridge.py, contracts.py    legacy Demo H exports and provenance checks
  rsa.py, compare.py         RDM primitives and crossed-seed comparison
  plot.py                    final RSA and RSM figures
```

Only `api.py`, `artifacts.py`, and `cli.py` form the public top-level surface.
Use `python -m demo_j.cli --help` for workflow discovery; nested modules are
implementation details rather than additional workshop entry points.

## Current boundary

The accepted 58-step SNN closely imitates held-out references. The aligned SNN
can emit uninterrupted 1,000-bin activity for representation analysis, but its
functional controller is a failed locomotion result: the comparison video shows
roughly half the showcased rollouts losing locomotion partway, and none tracks
its requested speed closely. The environment's coarse termination flag did not
detect those failures, so it must not be reported as survival or success. The
source release has independent 64-frame clips, so its repeated 32-frame aligned
training sequence is explicitly marked synthetic-periodic rather than claimed
as a natural continuous 20-second trajectory.

Generated files under `demo_j/out/` are disposable. The retained local set is
limited to the reference cache, canonical checkpoints/videos, final aligned
rollout, and final RSA reports/figures. Raw activation banks and rejected runs
should be regenerated when needed, not accumulated indefinitely.
