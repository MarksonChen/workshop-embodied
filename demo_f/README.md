# Demo F — rodent-derived conditional motion on Fetch

Demo F repeats Demo B's self-supervised construction after retargeting real
Coltrane locomotion to the ten-joint Fetch body used by Demo A. Its output is a
frozen conditional motion model for Demo G; it is not a controller and never
interacts with physics during training.

Given a past motion window `h`, Demo F extracts a future displacement command
`c` from the same recording and predicts the shifted future `w`:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D_{\rm Fetch}}
\log p_\phi(w\mid h,c).
\]

There are no action labels, rewards, or environment rollouts in this objective.

## Data path

The spatial retargeting is:

```text
Coltrane strict-locomotion keypoints
  -> smoothed trunk frame and four semantic paws
  -> body-size normalization
  -> contact detection and stance pinning
  -> bounded sequence inverse kinematics
  -> Fetch root, 10 joint angles, feet, and contacts
```

Relative rodent joint rotations are not copied between unlike skeletons. The
optimizer matches semantic paw endpoints and regularizes pose, velocity, and
acceleration. The public, session-split parent release is hosted at
<https://huggingface.co/datasets/MarksonChen/aldarondo2024-retargeted> and has
2,156/335/412 train/validation/test clips.

The parent release is useful for inspecting spatial retargeting, but it enlarges
a 0.09355 m rodent trunk to Fetch's 2.0-unit trunk while retaining the original
50 Hz clock. That 21.3789x length change is not dynamically similar under
gravity. The accepted training release therefore applies the Froude time scale

\[
s_t=\sqrt{21.3789}=4.6237.
\]

Each parent clip is independently interpolated, never joined to another clip,
and yields four disjoint 64-frame target-time crops. A stricter 1% joint-limit
saturation gate leaves 7,483/1,166/1,425 clips. The declared mapping is:

- source speed: 0.20 m/s;
- Fetch target speed: 0.924747 units/s;
- command horizon: 0.62 s;
- Fetch displacement command: `[0.573343, 0, 0]`.

Build and validate this local derived release:

```bash
uv run --extra workshop python -m demo_f.dataset.retime
uv run --extra workshop python -m demo_f.dataset.validate \
  --root demo_f/dataset/release_dynamic
```

## Frozen representation and model

Each physical frame has 60 quantities available both in the dataset and in a
live Fetch state:

```text
root-local planar velocity                   2
root height                                  1
root orientation / angular velocity          9
10 joint angles + velocities                20
4 root-local feet + velocities              24
4 foot-contact bits                          4
                                            --
                                            60
```

The accepted model deliberately remains small: a causal convolutional tokenizer
with 16-D tokens, four history tokens (0.32 s), and a four-layer conditional
Transformer that predicts one next token (0.08 s). During fitting, that same
one-token predictor is unrolled through four of its own predictions (0.32 s),
and decoded joint-limit excursions receive weight 10. This closes the mismatch
between safe one-step prediction and drifting autoregressive generation without
making the workshop architecture larger.

The fixed-variance Gaussian score is the average latent log likelihood. With
fixed `sigma`, ranking motion by this score is equivalent to ranking its
normalized prediction error, while retaining a calibrated scalar that Demo G
can use as a frozen reward term.

## Accepted evidence

Seed 0 trains in 51.4 seconds on the current GPU. It uses 37,415 training and
5,830 validation predictor windows; the selected predictor is step 1,600 of
2,000.

| held-out measure | validation | final test |
|---|---:|---:|
| rollout objective | 0.0536 | 0.0862 |
| speed tracking MAE | 0.0080 m/s | 0.0129 m/s |
| skill over last-token persistence | 21.5% | 21.5% |
| matching command beats reversed | 82.7% | 82.7% |
| real minus shuffled-future log likelihood | +5.81 | +5.56 |
| actual-speed bins selecting matching command | 5/5 | 5/5 |
| local likelihood peak at exact match | yes | yes |
| maximum generated joint saturation | 0% | 0% |

Every frozen finite-output, prediction, command-use, likelihood, root-height,
and joint-limit gate passes on both splits. This demonstrates conditional
sensitivity at the tested speed scale; it does not make the score a complete
measure of physical or biological realism.

## Reproduce the accepted prior

The accepted settings and dynamic dataset are now the command-line defaults:

```bash
uv run --extra workshop python -m demo_f.train
uv run --extra workshop python -m demo_f.evaluate \
  --output demo_f/out/dynamic/evaluation_validation.json
uv run --extra workshop python -m demo_f.evaluate --split test \
  --output demo_f/out/dynamic/evaluation_test.json
uv run --extra workshop python -m demo_f.generate \
  --output-dir demo_f/out/dynamic/generated
uv run --extra workshop python -m demo_f.export_jax
```

Render generated trajectories with the original Brax v1 Fetch body:

```bash
env -u LD_LIBRARY_PATH uv run --isolated \
  --with 'brax==0.12.3' --with 'jax==0.4.30' --with 'jaxlib==0.4.30' \
  --with 'imageio[ffmpeg]' python -m demo_f.render \
  --mode generated --input-dir demo_f/out/dynamic/generated
```

The canonical checkpoint SHA-256 is
`2a83780327f11f66fb8d3a196633e0bacad96e5d258f353c35e55778a207fbd3`.
The pure-JAX archive SHA-256 is
`7a3b8b1641512d61a014ec97535bcae7f44f4fb871c9a8ff1f35c810ece796a7`;
its source hash is embedded in the archive. PyTorch/JAX prediction and
likelihood parity pass at `5e-4` tolerance.

All datasets, checkpoints, reports, and videos above are generated artifacts
and remain gitignored. The original accepted kinematic checkpoint is preserved
locally as `demo_f/out/dynamic/prior_kinematic_legacy.pt`.

## Physical-transfer boundary

Retargeted root orientation is yaw-only. Four roll/pitch-related channels have
numerical-zero source variance but naturally vary in Fetch physics. Demo G
masks precisely these unsupported channels before scoring. Describe this as a
planar, rodent-derived motion prior—not literal rat biomechanics or a model of
full 3-D Fetch dynamics.

## Layout

```text
config.py              retarget and accepted prior settings
retarget.py            semantic preprocessing and sequence IK
features.py            shared 60-D offline/online feature contract
dataset/build.py       raw Aldarondo data -> public kinematic release
dataset/retime.py      parent release -> Froude-scaled dynamic release
dataset/validate.py    fail-closed schema, checksum, and geometry audit
train.py               conditional Gaussian-prior training
evaluate.py            held-out rollout and likelihood gates
generate.py            receding-horizon conditional generation and traces
export_jax.py           provenance-carrying frozen scorer for Demo G
experiment/            append-only experiment decisions
```

See [`ref/docs/demo_f.md`](../ref/docs/demo_f.md) for the workshop-facing role
and [`demo_g/README.md`](../demo_g/README.md) for the physical PPO comparison.
