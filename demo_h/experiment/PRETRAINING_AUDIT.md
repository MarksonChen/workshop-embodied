# Demo H pretraining audit

This audit checks whether the frozen prior used by the PPO reference term is
actually the causal, deployment-compatible controller described by Demo H. It
also separates action-distribution fidelity from locomotion quality: a correct
KL implementation can only pull PPO toward the distribution that the frozen
prior learned.

## Verdict

The state/action timing, train-only normalization, held-out splits, bounded
Gaussian likelihood, PyTorch-to-JAX export, and PPO reference-KL algebra are
correct. The saved beta sweep also confirms that the regularizer is active:
mean analytic action KL per dimension falls from `6.803` at `beta=0` to `0.623`
at `beta=0.025` and `0.455` at `beta=0.05` over the six-speed rollouts.

The stronger claim that the prior is a robust natural-locomotion controller is
not established. From an ordinary standing reset, its deterministic mean
policy survives for five seconds at all six commands, but requested speeds
`1.5, 2.0, 2.5, 3.0, 3.5, 4.0` produce only `0.872, 0.988, 1.584, 1.309, 1.339,
0.194` Fetch units/s. All six rollouts fail the strict four-limb stride gate.
Consequently, increasing beta does pull toward the learned action distribution,
but that is not equivalent to directly pulling the physical motion toward the
retargeted state distribution.

## Checks that pass

- The physical release is generated in the unchanged Brax v1 Fetch simulator.
  `normalized_control[t]` is executed over `[t,t+1)` and produces state
  `t+1`; shuffled controls fail the independent replay check.
- Train, validation, and test contain 26, 6, and 6 disjoint sessions. Feature,
  token, and command normalizers use the training split only. Validation
  selects checkpoints and calibrates variance; the test split is evaluation
  only.
- For an anchor whose history ends at state `t`, the planner sees only states
  through `t`, predicts the token containing states `t+1:t+4`, and the action
  decoder targets controls `u[t:t+3]`. The command begins at state `t`.
- The convolutional encoder is causal. Changing frames after state `t` does not
  alter any history token available at `t`.
- The exported JAX prior matches PyTorch with maximum plan error `5.49e-4` and
  action-mean error `4.98e-5` on the end-to-end parity example.
- The action head models a diagonal Gaussian in pre-tanh coordinates and the
  offline likelihood includes the exact tanh Jacobian. Brax receives the same
  mean and standard deviation through its softplus scale parameterization.
- `beta * log p0(a|h,g) / 10` in the reward plus PPO entropy coefficient
  `beta / 10` is, in expectation, `-beta * KL(pi || p0) / 10` for the shared
  tanh transform.

## Limitations and mismatches

- The prior is behavior cloning with a predicted motion plan, not a
  closed-loop imitation-RL controller. Its held-out one-step action MSE is
  `6.8%` worse than simply copying the previous 50 Hz control, although its
  20-step action rollout is `86.9%` better than repeating one control forever.
  Covariate shift beyond those short dataset rollouts remains possible.
- The training command distribution is much slower than the PPO task range.
  Train-set forward-command speed has median `0.962`, 95th percentile `2.831`,
  and 99th percentile `3.787` Fetch units/s. Thus the upper part of the
  `1.5–4.0` PPO range is sparse-tail conditioning, and `4.0` is slightly beyond
  the 99th percentile.
- Pretraining encodes each full 64-frame clip once, whereas deployment
  re-encodes a rolling 16-frame buffer. The causal encoder's receptive field
  makes these histories identical at the first anchor but not later anchors.
  On the test split, rolling-buffer evaluation raises mean next-token MSE from
  `0.2490` to `0.2513` (about `0.9%`), so this is real but too small to explain
  the frozen controller's high-speed failure by itself. A future prior version
  should train on exactly the same rolling-buffer encoding used online.
- Offline projection defines contact at any raw contact velocity above `1e-7`;
  the live Fetch observation uses squared magnitude above `1e-5`. A same-GPU
  replay of 128 held-out clips disagrees on `0.152%` of foot/frame values. This
  should be unified in a new feature-contract version, but its measured size is
  also too small to be the main failure.
- The saved `beta=0` checkpoint used PPO entropy coefficient `0.01`; the
  positive-beta checkpoints used `beta / 10`. The requested video is a valid
  comparison of those saved policies, but beta zero is not an otherwise exact
  KL-dose ablation. A future controlled sweep must freeze the non-reference
  entropy treatment across all arms or use zero at beta zero.

## Claim boundary

The current evidence supports: generative body/action pretraining supplies a
working initialization and a mathematically effective action-space reference
for residual PPO. It does not support: the frozen prior alone tracks the full
task-speed range, every action close to the prior produces data-like motion, or
larger beta monotonically implies more natural physical locomotion.
