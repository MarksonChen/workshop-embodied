# Experimental SNN-controller rollout release

This local, generated release is a positive-control arm for the question:
does Demo H behave more coherently when its prior is trained on closed-loop
state/action trajectories emitted by a successful imitation controller rather
than directly on feedback-projection trajectories?

It does **not** replace Demo H's accepted workshop release. The generator runs
the Demo J recurrent SNN controller for all 63 transitions in modern MJX and
stores each executed normalized action with the state it actually produces.
The projected Demo F motion is only the controller's future-reference input.

## Leakage and split contract

- SNN seeds 0 and 2 generate train and validation trajectories.
- SNN seed 1 generates test trajectories only.
- Seed 1 remains the held-out synthetic-neural benchmark for RSA.
- Original session-level train/validation/test splits remain unchanged.
- Failed or non-finite 63-transition rollouts are rejected as whole clips.

The complete build contains 3,521 train, 546 validation, and 339 test clips.
It retains 98.66% of 4,466 candidate controller/clip pairs and takes 71.7
seconds on the current H100. The manifest SHA-256 is
`b6401326bfc85f1ad1ac508fb77828817a4a24b87082acb2b901c6e492dee811`.

The held-out test controller is independent at the network-seed level but not
at the supervision-source level: all three SNNs were distilled from the same
independent feedback-controller labels. This experiment tests dynamic
feasibility and controller-seed generalization, not a new source of biological
actions.

## Important domain limits

The rollout engine is modern MJX, whereas Demo H post-training remains pinned
to legacy Brax v1. Thus the release removes open-loop state/action mismatch in
the data-generating engine but does not eliminate the deployment-engine shift.

The generated locomotion is also slower than the original release. Over the
31-frame command horizon, train-set speed has median `0.696`, 90th percentile
`1.647`, and maximum `3.702` Fetch units/s. Demo H's task still samples
`1.5–4.0`, so stronger prior regularization increasingly asks the policy to
extrapolate beyond its data.

`requested_actuator_torque` retains Demo H's legacy actuator-axis convention
`-300 * normalized_control` for schema compatibility. Modern MuJoCo exposes a
positive gear in its own coordinate convention; use `normalized_control` as
the cross-runtime action identity and do not interpret this compatibility
field as measured animal torque.

## Reproduction

```bash
uv run python -m demo_h.dataset.controller_rollouts \
  --training-checkpoints \
    demo_j/out/snn_distilled_seed0_<stamp>.pkl \
    demo_j/out/snn_distilled_seed2_<stamp>.pkl \
  --test-checkpoint demo_j/out/snn_distilled_seed1_<stamp>.pkl

uv run python -m demo_h.dataset.validate \
  --dataset-root demo_h/dataset/release_snn_controller \
  --dataset-variant demo-j-snn-controller-rollouts-v1
```

The release, checkpoints, and arrays are gitignored generated artifacts. The
manifest records every controller hash, shard hash, runtime version, accepted
clip, and rejected parent clip.
