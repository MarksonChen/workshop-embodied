# Demo B — conditional self-supervised rodent motion

_Updated 2026-07-20. Operational details are in
[`demo_b/README.md`](../../demo_b/README.md)._

## Purpose

Demo B defines self-supervised learning from scratch. It learns from recorded
motion only; it never acts in an environment and never receives a task reward.

Given a continuous recording:

1. take a past motion window `h`;
2. shift the same recording forward to obtain future motion `w`;
3. compute an egocentric displacement command `c` from that future;
4. predict `w` from `(h, c)`.

The future target is constructed from the data itself:

\[
\max_\phi\;\mathbb E_{(h,c,w)\sim\mathcal D}
\log p_\phi(w\mid h,c).
\]

This is self-supervised even though `c` is used as a condition: `c` is a
hindsight quantity computed from the same recording, not a human behavior
label.

## Accepted workshop model

- animal: Coltrane;
- data: strict-locomotion blocks from the first eight sessions;
- frame rate: 50 Hz;
- feature dimension: 281;
- tokenizer: causal convolutional MotionVAE, 16-D tokens at 12.5 Hz;
- predictor: standard conditional Transformer;
- objective: future-token MSE;
- probabilistic interpretation: fixed-isotropic Gaussian around the predicted
  mean.

The accepted checkpoint is restored from `rl_standalone/` without changing its
weights. The Gaussian wrapper only calibrates one scalar residual `sigma`.

## Why MSE is a likelihood here

For fixed `sigma`:

\[
\frac1D\log p(w\mid h,c)
=-\frac12\operatorname{mean}_i\left[
\left(\frac{w_i-\mu_i(h,c)}{\sigma}\right)^2
+2\log\sigma+\log(2\pi)\right].
\]

The last two terms are constant, so maximizing likelihood and minimizing MSE
rank models and examples identically. No discriminator or second learned head
is needed.

## Evidence and demonstration

The current calibration uses 1,344 standalone windows. Matching speed wins all
five population likelihood bins, per-window top-1 is 91.9% versus 20% chance,
and the local likelihood curve peaks at zero speed mismatch. This audit is
in-sample and establishes command use, not biological generalization.

Show the accepted kinematic videos at 0.10, 0.15, 0.20, and 0.25 m/s. Ask the
audience what is absent: contacts cannot make the generated body fall, and no
action was learned.

```bash
uv run --extra workshop python -m demo_b.promote_coltrane
uv run --extra workshop python -m demo_b.evaluate
uv run --extra workshop python -m demo_b.speed_sweep --render
```

## Bridge to Demo F

Demo B and Demo A use different bodies, so comparing their scores directly
would not be controlled. Demo F applies an explicit contact-aware retargeting
to Coltrane locomotion and repeats this conditional-learning construction in
Fetch coordinates. This makes Demo G’s G0/G1 comparison same-body.

## Claim boundary

Demo B learns a motion distribution, not torques or actions. Its kinematic
generation is not evidence of physical task competence, and its hidden units
are not claimed to model recorded neural populations.
