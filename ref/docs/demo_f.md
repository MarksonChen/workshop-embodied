# Demo F — dynamically retargeted conditional Fetch motion

_Updated 2026-07-20. Operational details:
[`demo_f/README.md`](../../demo_f/README.md)._

## Workshop role

Demo F solves the body and time-scale mismatch between Demo B's rodent data and
Demo A's Fetch physics. It retargets Coltrane locomotion to Fetch, applies the
gravity-relative Froude time scale, and repeats Demo B's self-supervised future
prediction objective. The resulting model is frozen for Demo G.

Demo F is not RL: it has no actions, rewards, or environment interactions.
Demo H reuses the retargeting pipeline but owns a separately versioned `1.75x`
timing derivative. That empirical variant does not replace Demo F's accepted
Froude-scaled dataset or checkpoint.

## Data construction

```text
rodent keypoints
  -> semantic paws and trunk frame
  -> contact-aware sequence inverse kinematics
  -> Fetch root, ten joints, feet, contacts
  -> 4.6237x time dilation for dynamic similarity
```

The 21.3789x spatial enlargement maps 0.20 m/s rodent motion to 0.924747 Fetch
units/s. The 0.62-second command is `[0.573343, 0, 0]`. Session splits remain
disjoint, parent clips are never joined, and a 1% saturation gate leaves
7,483/1,166/1,425 train/validation/test clips.

## Model and evidence

Use the same 60-D feature contract offline and in physical Fetch. A small causal
tokenizer produces 16-D tokens. The Transformer reads four tokens, predicts one
next token, and is trained through four of its own recursive predictions with a
joint-limit loss.

Keep one explicit command convention. For predictor anchor `a`, Demo F measures
the egocentric 0.62-second command from frame `4a` to `4a+31`. Demo H starts at
`4a-1` because control `u[4a-1]` produces the first predicted state `x[4a]`.
Thus the first anchor is 16→47 in F and 15→46 in H; the one-frame shift is the
causal action boundary, not an inconsistency.

Training takes 51.4 seconds. The accepted seed-0 checkpoint passes every frozen
gate on validation and final-test sessions:

| measure | validation | test |
|---|---:|---:|
| rollout objective | 0.0536 | 0.0862 |
| source-equivalent speed MAE | 0.0080 m/s | 0.0129 m/s |
| skill over last-token persistence | 21.5% | 24.0% |
| matching command beats reversed | 82.7% | 83.7% |
| real minus shuffled log likelihood | +5.81 | +5.56 |
| matching likelihood bins | 5/5 | 5/5 |
| local peak at matched command | yes | yes |
| generated joint saturation | 0% | 0% |

The Gaussian likelihood is a normalized future-prediction score, not a complete
physical-realism metric. Demo G therefore evaluates direct contacts, flight,
foot slip, acceleration, joint speed, and cyclicity in addition to likelihood.

For notebook code, import `load_split`, `load_prior`, `evaluate_checkpoint`,
and `generate_rollouts` from `demo_f.api`. The shared implementation lives in
`commands`, `features`/`jax_features`, `models`/`jax_models`, `metrics`,
`losses`, `prior`, and `artifacts`; Demo H reuses these modules directly.

## Live commands

```bash
uv run --extra workshop python -m demo_f.dataset.retime
uv run --extra workshop python -m demo_f.dataset.validate \
  --root demo_f/dataset/release_dynamic
uv run --extra workshop python -m demo_f.train
uv run --extra workshop python -m demo_f.evaluate
uv run --extra workshop python -m demo_f.generate \
  --output-dir demo_f/out/dynamic/generated
uv run --extra workshop python -m demo_f.export_jax
```

Retargeted orientation is yaw-only. Demo G masks four unsupported roll/pitch
channels. Present the result as a planar, rodent-derived Fetch motion prior, not
literal rat biomechanics or full 3-D likelihood.
