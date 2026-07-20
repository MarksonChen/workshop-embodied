# Demo C — a small world-action model plus PPO, on the virtual rodent

> **Research reference (2026-07-19).** Demo C remains implemented and evaluated,
> but it is no longer part of the core live workshop. The A/B/E presentation and
> replacement synthesis are specified in
> [`ref/docs/WORKSHOP_PLAN.md`](../ref/docs/WORKSHOP_PLAN.md) and
> [`ref/docs/demo_e.md`](../ref/docs/demo_e.md).

Demo C combines the ideas from the first two workshop demos without turning the
lesson into a state-of-the-art model survey:

> **Self-supervised learning learns “what happens if I do this?”  Reinforcement
> learning learns “what should I do?”**

The world factor is Demo B's action-conditioned rodent motion predictor. It learns
future motion targets made by shifting a real recording forward in time. The action
factor is the same small PPO actor-critic in two matched conditions: one sees ordinary
navigation state; the other also sees the world's predictive context. PPO receives
rewards, never target actions, and the frozen world model receives no PPO gradients.

This is a deliberately factorized, state-space teaching analogue of the WAM framing
in `ref/papers/WorldModel.pdf` Sections 3 and 4.1. It is not a unified video/action
transformer.

## The complete loop

```text
REAL RAT RECORDINGS                         SHORT DREAMS

motion history + command                   food goal
          │                                    │
          ▼                                    ▼
 frozen motion tokenizer              PPO policy πθ(s, [context])
          │                                    │ action
          ├── future latent target             ▼
          ▼                            frozen predictor Wφ
 action-conditioned predictor Wφ               │ predicted motion
          │                                     ▼
          └── MSE: SSL                    known progress reward
                                                │
                                                └── return: RL

                                      ZERO-SHOT REALITY CHECK

                                 frozen πθ → frozen joystick → MJX rat
```

One dream step is 0.64 s: eight 80-ms history tokens predict the next eight tokens.
Decoded local velocity and orientation—not the requested command—move the dream rat.
The task reward remains explicit: make progress toward food, arrive, and avoid wasting
time or turning excessively.

## Why the comparison is matched inside Demo C

Demo A's completed PPO agent controls a different Brax quadruped. Feeding rat motion
into that network would not be a meaningful neural baseline. Demo C therefore trains
an **RL-only rodent analogue** with the same task, learned dynamics, policy head, PPO
budget, and seeds as the WAM condition.

| condition | ordinary 8-D navigation state | 192-D predictive context |
|---|:---:|:---:|
| goal-only PPO (RL-only analogue) | yes | no |
| WAM-context PPO | yes | yes |

The frozen task uses food 0.35–0.75 m away in the forward semicircle, an eight-step
horizon, 256 parallel dreams, 786,432 environment steps, and seeds 0/1/2. The forward
field avoids asking the Demo B locomotion model for out-of-distribution in-place
U-turns. `config.py` is the single frozen problem definition.

## Setup

From the repository root:

```bash
uv sync --extra cuda12 --extra workshop --extra dev
export ALDARONDO_ROOT=/workspace/data/Aldarondo2024
```

Prerequisites:

- `demo_b/assets/motor_standalone.pt`, generated as described in Demo B's README;
- the Aldarondo HDF5 tree described in `ref/docs/dataset.md` for world-model and
  neural evaluation;
- the frozen joystick checkpoint under `rl/runs/` for the MJX reality check. Pass a
  different path with `--joystick` when needed.

All generated checkpoints, caches, figures, videos, and metrics go under
`demo_c/out/`, which is git-ignored.

## Run the pipeline

### 1. Fit and validate the world factor with SSL

```bash
uv run --extra cuda12 --extra workshop python -m demo_c.train_world
uv run --extra cuda12 --extra workshop python -m demo_c.validate_world
```

The tokenizer remains frozen. The transition has the same simple Transformer + MSE
architecture as Demo B, but is fit on real contiguous crops from 12 sessions balanced
across six rats. Two complete sessions select the checkpoint; the four DLS/MC neural
sessions remain untouched. Validation compares it with the honest latent-persistence
null on held-out time blocks.

### 2. Train the two matched PPO conditions in dreams

```bash
for variant in goal_only wam; do
  for seed in 0 1 2; do
    uv run --extra cuda12 --extra workshop \
      python -m demo_c.train --variant "$variant" --seed "$seed"
  done
done
uv run --extra cuda12 --extra workshop python -m demo_c.compare
```

For a wiring check, add `--smoke`; smoke artifacts never replace the frozen anchors.
Training curves, resolved configs, provenance, throughput, memory, and the append-only
`results.tsv` are recorded automatically.

### 3. Compare hidden activity with real neural populations

```bash
uv run --extra cuda12 --extra workshop python -m demo_c.prepare_neural
uv run --extra cuda12 --extra workshop python -m demo_c.neural_eval --scope loco
uv run --extra cuda12 --extra workshop python -m demo_c.neural_eval --scope all_matched
```

The primary locomotion analysis uses two held-out DLS sessions and two held-out motor
cortex sessions. A recorded two-second future bearing supplies the pseudo-food
direction; its distance is fixed at the middle of the trained task range so neither
policy is evaluated on an out-of-distribution near-zero goal.

Each representation is reduced to 16 train-only PCs. A Poisson GLM predicts active-unit
spike counts, and crossvalidated RSA compares speed × turn population geometry. The
split uses 60-s temporal blocks, 5-s gaps, and whole-window containment. A 20-s
within-block circular shift is the encoding noise floor. `all_matched` repeats the
analysis on the same number of samples drawn from the full behavioral range.

### 4. Put the frozen navigator back in MJX physics

```bash
uv run --extra cuda12 --extra workshop python -m demo_c.calibrate_bridge
uv run --extra cuda12 --extra workshop python -m demo_c.deploy_physics --episodes 8
uv run --extra cuda12 --extra workshop python -m demo_c.deploy_physics \
  --goal-index 7 --render
```

The policy is unchanged. Its high-level displacement is converted to a velocity
command for the existing frozen MIMIC joystick, which drives the physical rodent.
This tests zero-shot navigation transfer at the high-level control boundary; it does
not claim the locomotion primitive itself was learned in Demo C. The independently
measured response-curve bridge is reportable because it maps requested displacement
through the joystick's observed low-speed dead zone without changing the policy.
`--bridge raw` and `--bridge inverse_gain` retain the two rejected bridge probes;
`--bridge decoded` is an explicitly labelled diagnostic.

### 5. Run tests

```bash
uv run --extra cuda12 --extra workshop --extra dev pytest -q demo_c/tests
```

## Frozen results from the completed run

### SSL gate

The broadened transition beat latent persistence in every untouched neural session:

| held-out session | region | windows | skill over persistence |
|---|---:|---:|---:|
| coltrane 2021-08-05 | DLS | 53 | +42.0% |
| coltrane 2021-08-06 | DLS | 101 | +24.7% |
| freddie 2022-05-16 | MC | 149 | +45.1% |
| freddie 2022-05-17 | MC | 23 | +46.0% |

Session-balanced mean skill was **+39.5%**. These are real contiguous 64-frame
windows; no locomotion fragments are stitched.

### Matched dream task

| policy | success, mean ± seed SD | return | final distance |
|---|---:|---:|---:|
| goal-only PPO | 0.6393 ± 0.0044 | 4.4251 | 0.1643 m |
| WAM-context PPO | 0.6445 ± 0.0029 | 4.4733 | 0.1603 m |

The success difference is +0.0052, inside the predeclared noise threshold
`η = 2 × max(seed SD) = 0.0088`; both policies have zero invalid transitions. This is
the desired controlled result: task performance is functionally matched before asking
whether the representations differ.

### Zero-shot MJX reality check

| policy | success | fall rate | final distance |
|---|---:|---:|---:|
| goal-only PPO | 7/8 (0.875) | 0 | 0.084 m |
| WAM-context PPO | 4/8 (0.500) | 0 | 0.156 m |

The independently measured nonlinear response bridge was selected before this final
pass. The raw bridge gave 0/8 and 2/8; a rejected scalar inverse-gain bridge gave 4/8
and 1/8. The response curve improves both the paired average and the primary WAM
transfer result without changing a network. WAM's lower real success despite matched
dream success is an important dream-to-real gap, consistent with greater reliance on
learned context rather than evidence that WAM improves transfer.

### Neural comparison

Primary locomotion-only, session-balanced means:

| representation | population bits/spike | shift-corrected bits/spike | RSA ρ |
|---|---:|---:|---:|
| RL-only rodent policy | 0.0141 | 0.0082 | 0.575 |
| Demo B autoencoder latent | 0.0660 | 0.0141 | 0.625 |
| Demo B predictive context | 0.0671 | 0.0193 | 0.710 |
| **Demo C WAM+RL policy** | **0.0655** | **0.0191** | **0.701** |
| raw kinematics reference | 0.0751 | 0.0148 | 0.643 |

The honest interpretation is:

- Demo C is clearly promising relative to the matched RL-only policy in descriptive
  magnitude, especially for population geometry;
- it is approximately tied with Demo B's predictor, not better than it—the policy
  preserves the predictive representation while adding goal-directed control;
- four sessions give a minimum two-sided exact sign-permutation p-value of 0.125, and
  no pairwise “Demo C wins” claim is statistically established;
- the n-matched full-behavior control preserves the RSA ordering but has a near-zero
  shift-corrected encoding advantage, so both views must be shown.

The actual Demo A quadruped is intentionally absent from this neural table because its
body and observation space cannot be aligned to these rat recordings. The matched
goal-only rodent policy is the defensible Demo A-like baseline.

## File map

```text
demo_c/
  config.py             frozen task, PPO budget, policy dimensions, seeds
  motor.py              frozen Demo B tokenizer/predictor adapter
  dream_env.py          short learned rodent environment and explicit reward
  policy.py             shared two-layer actor-critic
  train_world.py        broader real-session SSL fit
  validate_world.py     held-out persistence gate and horizon plot
  train.py              PPO, evaluation, curves, provenance, append-only table
  compare.py            matched three-seed noise-floor verdict
  deploy_physics.py     zero-shot high-level transfer to MIMIC-MJX
  calibrate_bridge.py   optional command-response diagnostic
  neural_data.py        causal representation/spike alignment and atomic caches
  prepare_neural.py     cache builder
  neural_eval.py        strict Poisson/RSA evaluation and paired contrasts
  experiment/DECISIONS.md
  tests/test_core.py
```

The workshop design and its caveats are expanded in `ref/docs/demo_c.md`. The earlier
research-oriented proposal remains archived unchanged as `ref/docs/demo_c_prev.md`.
