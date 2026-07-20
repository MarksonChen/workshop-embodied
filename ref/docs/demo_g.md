# Demo G — dynamically aligned SSL-guided RL on Fetch

_Updated 2026-07-20. Reproduction details:
[`demo_g/README.md`](../../demo_g/README.md)._

## Teaching role

Demo G makes one controlled change to Demo A:

```text
G0 reward = physical task consequence
G1 reward = physical task consequence + frozen Demo F motion score
```

PPO never receives a target action. Demo F is trained first from recordings and
then frozen; PPO still learns motor actions from return.

## Aligned objective

The Froude-scaled data maps a 0.20 m/s rodent command to 0.924747 Fetch units/s
and a 0.62-second displacement `[0.573343, 0, 0]`. Demo G uses this same target
and preserves Demo A's `sigma / target = 1/3` tracking width.

\[
\max_\pi\;\mathbb E\sum_t\gamma^t
\left[r_t^{\rm task}+\beta\,
\operatorname{sigmoid}\left(\frac{\ell_\phi(w_t\mid h_t,c)+20}{5}\right)
\right],
\]

with `beta=0` for G0 and `beta=0.1` for G1. The score is batched over 2,048
environments every four control frames; task reward remains frame-by-frame.

## Runtime and result

Each arm trains for 30M transitions with three PPO evaluations. G0 takes
57.8–59.8 seconds and G1 takes 68.0–69.5 seconds inside `ppo.train`; a sequential
pair is about 2.1 minutes.

Five shaping-disabled rollout seeds evaluate each of three matched policy seeds:

| seed | raw log-p gain | direct composite | tracking retained |
|---:|---:|---:|---:|
| 0 | +18.11 (5/5 wins) | +5.78 (5/5) | 100.15% |
| 1 | +32.76 (5/5 wins) | +3.66 (5/5) | 100.58% |
| 2 | +17.28 (5/5 wins) | -0.30 (0/5) | 100.15% |

Raw likelihood improves by `22.72 ± 7.11`, and survival is 100% in every seed.
Airborne fraction, stance-foot speed, approximate world-foot slip, and
joint-speed RMS move toward held-out motion in 3/3 seeds. The full nine-measure
gait distance improves in only 2/3 seeds, and cyclicity improves in 0/3.

Seed 0 is the best presentation checkpoint because it passes all single-seed
gates and gives the largest direct improvement. It reduces contact switching
from 11.48 to 3.11 Hz, approximate stance-world foot speed from 1.92 to 0.75,
and vertical acceleration from 1.78 g to 0.91 g while retaining tracking.
Nevertheless, it remains crouched, its maximum flight duration worsens, and its
cyclicity moves away from the data.

## Teaching statement

> **The frozen data-driven prior shifts physical PPO toward recorded motion
> statistics without sacrificing the task, but likelihood is an incomplete
> realism proxy and the full behavioral improvement is not seed-robust.**

Show task tracking, raw likelihood, direct metrics, and the seed-0 paired video.
End with the seed-2 composite failure. This distinction is the central lesson:
optimizing a learned score is not identical to optimizing every property we
intended that score to represent.
