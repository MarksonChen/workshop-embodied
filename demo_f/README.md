# Demo F — rodent-derived conditional motion on Fetch

Demo F is **Demo B retrained in Fetch space after semantic motion retargeting**.
It is intentionally a separate demo rather than a modification of Demo B: the
retargeting representation, tokenizer capacity, transition model, likelihood
calibration, and data filters can all be tuned without destabilizing the
accepted rodent generator used in the earlier workshop section.

The teaching definition is unchanged. Given a continuous retargeted sequence,
the past is the input, a displacement extracted from its later frames is the
command, and the shifted future is the target:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D_{\rm Fetch}}
\log p_\phi(w\mid h,c).
\]

There are still no actions, rewards, or environment interactions in Demo F.

## Current stage: validated standalone release and conditional prior

The speed-binned renders passed the visual inspection gate. The same transform
has now been applied to the full Coltrane strict-locomotion subset and packaged
as a separate, versioned dataset:

<https://huggingface.co/datasets/MarksonChen/aldarondo2024-retargeted>

Only `dataset/build.py` reads raw Aldarondo HDF5. `dataset/loader.py` is the
canonical Demo F training input and fails closed unless it sees the complete
public schema.

The data transform is:

```text
Coltrane keypoints
  -> smoothed trunk frame + four semantic paws
  -> body-size normalization
  -> foot-contact detection and stance pinning
  -> bounded, sequence-level inverse kinematics
  -> 10 Fetch joint angles + root trajectory
```

Relative rodent bone rotations are not copied. The loss matches the four paw
endpoints and regularizes the target pose, velocity, and acceleration. This is
the minimum contact-aware cross-morphology recipe supported by animal-to-robot
retargeting practice.

Generate the four inspection clips:

```bash
uv run --extra workshop python -m demo_f.retarget

env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' --with 'jaxlib==0.4.30' \
  --with 'imageio[ffmpeg]' python -m demo_f.render
```

The accepted regression clips cover 0.100, 0.150, 0.200, and 0.217 m/s real
Coltrane motion over 128 frames (2.54 s). The last value is deliberate: no
128-frame Coltrane clip passing Demo B's strict gait screen reaches 0.25 m/s.
Every video names its exact source session and frame offset.

The accepted videos were inspected for:

- recognizable four-leg gait phase rather than synchronized paddling;
- stable stance feet without obvious skating;
- no persistent foot penetration or floating;
- smooth trunk and joint trajectories;
- no repeated saturation at the +/-60 degree Fetch joint limits;
- visibly increasing stride speed across the four examples.

## Standalone dataset

Build and validate the Hugging Face-ready local release:

```bash
uv run --extra workshop python -m demo_f.dataset.build
uv run --extra workshop python -m demo_f.dataset.validate
```

The release contains one compressed shard per source session, an ODC-By dataset
card, immutable session splits, source and shard SHA-256 values, code/config
provenance, and per-clip target-feasibility metrics. Public upload is a separate
validated operation:

```bash
HF_HOME=/root/.cache/huggingface uv run --extra workshop \
  python -m demo_f.dataset.publish --confirm-public-upload
```

## Frozen Fetch representation

The conditional model uses only quantities available both in the retargeted
dataset and a live Demo A Fetch state:

```text
root-local planar velocity                   2
root height                                  1
root orientation / angular velocity          9
10 joint angles + velocities                20
4 root-local feet + velocities              24
4 foot-contact bits                          4
                                            --
                                            60 dimensions
```

Schema v1 freezes this 60-D contract. `config.PriorConfig` owns the architecture
and training hyperparameters independently of Demo B. The canonical release has
2,903 clips (2,156 train / 335 validation / 412 test), with all splits separated
by recording session. Train only from the standalone release:

```bash
uv run --extra workshop python -m demo_f.train
```

On the canonical seed-0 run, training takes about 14 seconds on the current GPU.
The validation-selected predictor reaches latent MSE 0.745 versus 1.733 for a
last-token persistence baseline, and the matching command scores better than a
reversed validation command for 80.0% of clips. Generation/rollout stability and
online Demo G integration remain separate acceptance gates.

Generate deterministic conditional-mean rollouts at four source-equivalent
speeds, then render them with the original Brax v1 Fetch body. The predictor
proposes its trained eight-token horizon, advances one token, and replans; the
continuous latent stream is decoded once to avoid visible 32-frame pauses.

```bash
uv run --extra workshop python -m demo_f.generate

env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' --with 'jaxlib==0.4.30' \
  --with 'imageio[ffmpeg]' python -m demo_f.render \
  --mode generated --input-dir demo_f/out/generated
```

The requested speeds are rat-scale labels. `generate.py` robustly maps them to
Fetch-space displacement commands using only straight clips in the training
split, and every video reports both requested and realized equivalent speed.
It also writes 50 Hz instantaneous-speed and joint-activity traces, a speed plot,
and the fraction of post-seed time spent below 25% of the requested speed.

## Layout

```text
config.py       source clips and independently tunable retarget/model settings
kinematics.py   differentiable kinematics of the original 10-DoF Brax Fetch
retarget.py     semantic preprocessing, contact pinning, and sequence IK
render.py       exact Brax v1 Fetch inspection videos and synchronized grid
dataset/             the complete standalone-data surface
  contract.py        public schema, frozen session splits, repository identity
  build.py           the sole raw-HDF5 -> release path
  validate.py        checksums, shapes, splits, limits, and FK parity
  publish.py         confirmation-gated Hugging Face upload
  loader.py          standalone-release-only Demo F training loader
  DATASET_CARD.md    Hugging Face dataset-card template
  release/           generated/downloaded canonical release (gitignored)
features.py          offline/online 60-D Fetch feature contract
models.py            tunable causal tokenizer and conditional Transformer
train.py             dataset-only conditional Gaussian-prior training
generate.py          command calibration and autoregressive latent rollout
out/retarget/   generated trajectories, metrics, HTML, and MP4 artifacts
```
