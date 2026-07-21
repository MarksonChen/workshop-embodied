# Demo H — generative body–action pretraining plus RL

Demo H is the accepted workshop capstone. It extends Demo F from future-motion
prediction to a body-centric world–action prior, then adapts that prior with the
same PPO idea introduced in Demo A.

```text
retargeted motion
  -> predict a short future body-motion plan
  -> predict the Fetch controls that realize that plan
  -> freeze this state/action prior
  -> train a small residual policy with task reward - beta * KL
```

The future-state targets are self-supervised because they are shifted from the
same motion sequence. The controls are physics-derived pseudo-labels produced
by a transparent feedback controller in exact Fetch physics; they are neither
animal torques nor pure SSL targets. PPO still learns from scalar task reward.

## Accepted configuration

- Coltrane strict-locomotion clips retargeted to the ten-joint Fetch body;
- one centered 64-frame crop after `1.75x` temporal dilation at 50 Hz;
- exact Brax v1 Fetch projection with `kp=400`, `kd=10`;
- 2,404 accepted physical clips: 1,784 train, 278 validation, 342 test;
- 16-D causal motion token, four-token history, and one-token motion plan;
- 50 Hz Gaussian feedback action decoder;
- zero-initialized residual PPO actor with a frozen reference copy;
- task speed sampled uniformly from 1.5 to 4.0 Fetch units/s;
- `beta=0.10`, 30M transitions, 2,048 environments, seed 0.

The 1.75 factor is an empirically selected temporal dilation, not the 4.6237
Froude-similarity factor used by canonical Demo F. Keeping these as separately
versioned datasets prevents either interpretation from being silently changed.

## Build and validate the physical dataset

The derived datasets are generated artifacts and remain gitignored:

```bash
uv run --extra workshop python -m demo_f.dataset.retime \
  --time-scale 1.75 --crops-per-parent 1 \
  --variant temporal-dilation-1p75-v1 \
  --output-root demo_f/dataset/release_retime_1p75

env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.dataset.project --splits train validation test

uv run --extra workshop python -m demo_h.dataset.validate
```

For an independent exact-physics replay check, run the validator inside the
isolated Brax environment with `--replay-clips 16`. Stored control `u[t]` acts
during `[t,t+1)` and produces stored state `x[t+1]`.

The accepted build takes 84.4 seconds and passes 99.09% of candidate clips.
Across shards, median joint tracking RMSE is 0.103 rad, mean actuator
saturation is 1.36%, minimum torso height is 1.133, and minimum uprightness is
0.514. Replaying paired controls recovers saved trajectories to approximately
`1e-5`; shuffled controls are materially worse.

## Train and validate the frozen prior

```bash
uv run --extra workshop python -m demo_h.train_prior --from-scratch
uv run --extra workshop python -m demo_h.evaluate_prior
uv run --extra workshop python -m demo_h.export_jax

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.evaluate_physics --target-speed 1.5
```

The accepted prior trains in 70.8 seconds. On the held-out test split it:

- improves next-state prediction 49.8% over persistence;
- chooses the matching command over a shuffled command in 82.4% of windows;
- makes shuffled motion plans 83% worse for action prediction;
- improves 20-step closed-loop action MSE 86.9% over repeating the initial
  control.

Copying the previous 50 Hz control is 6.8% better for exactly one step, which
is expected for very smooth controls and is not used as the rollout gate. From
an ordinary standing reset, the frozen prior survives five seconds, travels
5.16 units, remains upright, switches all four foot contacts, and never
saturates an actuator.

## Train the accepted RL policy

```bash
env -u LD_LIBRARY_PATH uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax[cuda12]==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.train --arm h2 --seed 0
```

The H2 objective is task reward plus reference cross-entropy and PPO entropy,
which together implement the mean per-action-dimension KL regularizer:

\[
J(\psi)=\mathbb E\sum_t\gamma^t\left[
r_t^{\rm task}-\frac{\beta}{10}
D_{\rm KL}\!\left(\pi_\psi(\cdot\mid h_t,g_t)
\Vert p_{\theta_0}(\cdot\mid h_t,g_t)\right)\right].
\]

No contact, cadence, symmetry, stride, or other naturalness metric appears in
the reward. Those measurements are validation-only diagnostics.

The accepted 30M-transition run takes 95.2 seconds on the current H100. The
frozen-prior and PPO stages together take 166 seconds; including the one-time
physics projection gives approximately 250 seconds, still under five minutes.

## Inspect the accepted policy

Set the checkpoint emitted by training, then create all six rollouts in one
batched compilation and render one comparison video:

```bash
DEMO_H_CHECKPOINT=demo_h/out/h2_seed0_20260721-020035.pkl

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  python -m demo_h.visualize_speeds --arm h2 \
  --checkpoint "$DEMO_H_CHECKPOINT" \
  --speeds 1.5 2.0 2.5 3.0 3.5 4.0 \
  --label 'beta=0.10' \
  --output-dir demo_h/out/accepted_speed_sweep

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu \
  uv run --no-project --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' \
  --with 'jaxlib==0.4.30' --with 'scipy>=1.15' \
  --with 'imageio[ffmpeg]' --with pillow \
  python -m demo_h.render_speed_comparison \
  demo_h/out/accepted_speed_sweep/metrics.json \
  --output demo_h/out/accepted_speed_sweep/comparison.mp4
```

The renderer reuses static scenes, renders every second 50 Hz physics frame at
25 fps, and preserves the five-second playback duration. Six-speed rollout
evaluation takes about 20 seconds including startup instead of about 45
seconds with six separate compilations.

| command | realized mean | survival | four-limb stride gate |
|---:|---:|---:|---:|
| 1.5 | 1.471 | 100% | pass |
| 2.0 | 2.010 | 100% | pass |
| 2.5 | 2.479 | 100% | fail |
| 3.0 | 2.974 | 100% | pass |
| 3.5 | 3.465 | 100% | pass |
| 4.0 | 3.647 | 100% | fail |

The user accepted β=0.10 after a direct video comparison with β=0.075. This is
a workshop-level qualitative selection, not a multiseed algorithm claim. The
4.0-unit/s command is a visible stress case, and the two failed validation
cells above must remain visible when presenting the result.

## Artifact identities

Generated artifacts are not committed. The accepted local run is identified
by:

- Demo F 1.75 manifest: `85fe54ee...2b3f`;
- Demo H physical manifest: `c02c0cc4...76847`;
- PyTorch prior: `181394fe...f903`;
- JAX prior: `fc4f5797...3382`;
- β=0.10 PPO checkpoint: `e876bf80...cb44`.

Full hashes and the append-only experiment record are in
[`experiment/DECISIONS.md`](experiment/DECISIONS.md).

## Claim boundary

Demo H demonstrates that a compact generatively pretrained body/action policy
can initialize and regularize task-driven RL in the same physical body. It does
not establish rat biomechanics, biological torque recovery, neural similarity,
or an algorithm-level advantage over scratch PPO. Those claims require better
retargeted data, paired baselines, and multiple policy-training seeds.

See [`ref/docs/demo_h.md`](../ref/docs/demo_h.md) for the workshop-facing design,
evidence, and teaching sequence.
