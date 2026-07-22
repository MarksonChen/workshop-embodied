# Demo J — spiking imitation and neural similarity

Demo J trains a recurrent spiking controller on the same `1.75x` retargeted
Fetch-motion release used by Demo H, then treats its 20 ms spike counts as a
synthetic neural recording. The current comparison asks how Demo H activity
changes across prior strengths `beta` under identical locomotion inputs.

The package intentionally supports only three workflows:

1. short-clip SNN imitation, which is the accepted behavioral result;
2. native-clip token-conditioned SNN imitation;
3. finite-trial RSM/RSA against the matched Demo H beta sweep.

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
- Episode: one independent 64-frame clip supplies 63 actions. Recurrent state
  resets at the clip boundary; states, actions, and intentions never wrap.
- Intention: each four-frame future-motion token is valid only when its whole
  block lies inside the clip. Invalid tail tokens are zero and explicitly
  masked.
- Independence: no Demo H policy, hidden activation, beta value, or biological
  spike is used to train or select the SNN.
- Analysis: compare population geometry rather than aligning units one-to-one;
  use all 64 state-aligned frames, exact SNN input, and a 200 ms delayed Demo H
  control. The terminal SNN readout is measured but never applied as an action.
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

Run native-clip token-conditioned imitation:

```bash
uv run python -m demo_j.cli fit-tokenizer
uv run python -m demo_j.cli train-aligned \
  --preview-tokens 8 --seed 0 --updates 2000 --batch-size 256
uv run python -m demo_j.cli evaluate-aligned \
  --checkpoint demo_j/out/aligned/snn_native_clip_seed0_<stamp>.pkl \
  --speeds 1.5 2.0 2.5 3.0 3.5 4.0 \
  --output demo_j/out/aligned/native_clip_rollout_seed0.npz
MUJOCO_GL=egl uv run python -m demo_j.cli render-aligned \
  --recording demo_j/out/aligned/native_clip_rollout_seed0.npz \
  --output demo_j/out/aligned/snn_native_clip_speed_sweep.mp4
```

This workflow audits every held-out test clip in one batched rollout and stores
six speed-matched examples for the video. It evaluates only the duration
supported by each source clip and does not manufacture a long reference by
repeating a short segment. A genuine long-horizon experiment requires genuinely
continuous references or a separate trajectory generator and is outside the
current data contract.

The `analysis` bridge is invoked through `export-h-trace` and
`export-h-activations` in Demo H's isolated legacy environment. It exports one
fixed trajectory bank plus activations. In the modern environment, record each
SNN seed on that bank, compare, and plot:

```bash
uv run python -m demo_j.cli record-aligned \
  --checkpoint demo_j/out/aligned/snn_native_clip_seed0_<stamp>.pkl \
  --trace demo_j/out/aligned/h_native_trace_64.npz \
  --output demo_j/out/aligned/snn_native_fixed_seed0.npz
uv run python -m demo_j.cli compare-rsa \
  --recording demo_j/out/aligned/snn_native_fixed_seed[012].npz \
  --trace demo_j/out/aligned/h_native_trace_64.npz \
  --activation demo_j/out/aligned/demo-j-beta-v1_*_native_activations.npz \
  --output demo_j/out/aligned/beta_rsa_native.json --permutations 1000
uv run python -m demo_j.cli plot-rsa \
  --report demo_j/out/aligned/beta_rsa_native.json \
  --output-dir demo_j/out/aligned/rsa_native
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
  aligned.py                 PCA tokens and finite native-clip sequences
experiments/
  train_imitation.py         accepted short sequence distillation
  evaluate_imitation.py      held-out short evaluation
  aligned.py                 tokenizer fitting and native-clip training
  aligned_rollout.py         finite rollout and fixed-trial spike recording
  render.py                  short comparison video
  render_aligned.py          native-clip comparison video and speed audit
analysis/
  bridge.py, contracts.py    legacy Demo H exports and provenance checks
  rsa.py, compare.py         RDM primitives and crossed-seed comparison
  plot.py                    final RSA and RSM figures
```

Only `api.py`, `artifacts.py`, and `cli.py` form the public top-level surface.
Use `python -m demo_j.cli --help` for workflow discovery; nested modules are
implementation details rather than additional workshop entry points.

## Current boundary

Both finite-clip SNN workflows now closely imitate held-out references. The
native token-conditioned seed selected by validation completes all 342 test
clips over their supported 1.26-second duration. Its matched finite-trial RSA
does not support the proposed “higher beta is more SNN-like” ordering: beta
zero has the highest crossed-seed mean, with substantial uncertainty across
the three Demo H seeds. This is a short-episode imitation and descriptive
representation result, not evidence for indefinite locomotion or a biological
mechanism.

The previous 1,000-bin workflow and its readout PPO remain rejected. They
repeated a short segment, created discontinuities at every wrap, and roughly
half of the showcased physical rollouts failed partway.

Generated files under `demo_j/out/` are disposable. The retained local set is
limited to the reference cache, canonical checkpoints/videos, final native
rollout, and final RSA reports/figures. Raw activation banks and rejected runs
should be regenerated when needed, not accumulated indefinitely.
