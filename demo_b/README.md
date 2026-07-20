# Demo B — conditional self-supervised Coltrane motion

Demo B learns only from recorded rat motion. A causal convolutional VAE turns
50 Hz skeletal motion into 16-D tokens at 12.5 Hz; a small Transformer predicts
the next eight tokens from eight past tokens and a hindsight displacement
command. Future frames from the same recording supply both command and target:
there are no actions, task rewards, or physics interactions.

The workshop checkpoint is the original, behaviorally validated Coltrane model
from `rl_standalone/`. Its weights are unchanged. Demo B adds a fixed Gaussian
interpretation around the predicted mean:

```text
p(next token | history, command) = Normal(predicted mean, sigma^2 I)
```

Because `sigma` is fixed, maximizing this likelihood is exactly equivalent to
minimizing the model's original MSE. The likelihood API therefore does not
change generation. It teaches the probabilistic bridge that Demo F repeats on
the Fetch body before Demo G uses the resulting frozen score.

## Frozen representation and locomotion subset

Each frame uses the original 281-D representation:

```text
root-local planar velocity                  2
root height                                 1
root orientation increment                  6
67 fitted joint angles + velocities       134
23 root-local keypoints + velocities      138
                                           ---
                                           281
```

The source is the first eight Coltrane sessions. Locomotion selection is the
strict geometric rule used by the known-good model: non-overlapping 64-frame
blocks must exceed 0.10 m/s planar speed, exhibit coordinated gait, turn less
than 90 degrees, and keep front-spine vertical drift below 10 mm. Adjacent
accepted blocks are merged before stride-16 training crops are made. The
bundled real locomotion seed contains 320 frames from
`coltrane/2021_07_29_1`.

The Freddie experiments are retained under `demo_b/out/` as rejected
comparisons. Even with the restored 281-D representation and a Freddie-matched
tokenizer, slow rollouts twitched badly. They are not workshop assets.

## Restore and validate

From the repository root:

```bash
uv run --extra workshop python -m demo_b.promote_coltrane
uv run --extra workshop python -m demo_b.evaluate
```

`promote_coltrane` copies the proven predictor weights unchanged and calibrates
one scalar `sigma`. The current calibration set has 1,344 standalone training
windows. It yields:

- full future-token MSE: 0.00264;
- matching conditioned speed wins all five population bins;
- per-window speed-bin top-1 accuracy: 91.9% versus 20% chance;
- the local likelihood curve peaks at zero speed mismatch.

This establishes the teaching bridge and the behaviorally good generator. The
speed audit is in-sample, so it demonstrates that the frozen conditional model
uses its command; it is not an independent biological-realism result.

The archived Demo E research path can still build its framework-neutral export
with:

```bash
uv run --extra workshop python -m demo_b.export_jax
```

It contains the unchanged encoder/predictor, likelihood calibration, and a
fixed 23-marker forward-kinematic bridge fitted only on the eight source
sessions. The bridge's current mean marker RMSE is 7.29 mm (10.18 mm maximum).
It intentionally contains no physical reset bank: aligned Demo E preserves the
native RodentJoystick reset and withholds likelihood reward for 0.64 s while
the policy fills a real causal history.

## Kinematic demonstrations

```bash
uv run --extra workshop python -m demo_b.speed_sweep --render
uv run --extra workshop python demo_b/drive.py --seconds 16 --render
uv run --extra workshop python demo_b/waypoint.py --shape square --render
```

The accepted inspection videos are:

- `out/restored_coltrane_v010_straight.mp4`
- `out/restored_coltrane_v015_straight.mp4`
- `out/restored_coltrane_v020_straight.mp4`
- `out/restored_coltrane_v025_straight.mp4`

These are kinematic generations, not physical control. The current workshop
next retargets the same locomotion concept to Fetch in Demo F; Demo G then adds
that same-body prior to physical PPO.

## Layout

```text
models.py              causal MotionVAE, Transformer, and Gaussian score
promote_coltrane.py    unchanged-weight rollback + likelihood calibration
speed_sweep.py         fixed-speed rollout and video generation
strict_locomotion.py   exact geometric locomotion rule
marker_bridge.py       calibrated skeleton sites for all 23 keypoints
export_jax.py          compact scorer retained for archived Demo E research
train_full_prior.py    controlled Coltrane/Freddie research comparisons
geometry.py            decoded features -> skeletal qpos
rollout.py             autoregressive generation and rendering
```

See [`demo_f/README.md`](../demo_f/README.md) for the current retargeted-data
bridge and [`ref/docs/demo_b.md`](../ref/docs/demo_b.md) for the workshop notes.
