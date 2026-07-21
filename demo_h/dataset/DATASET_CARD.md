# Demo H exact-Fetch body–action data

This derived dataset pairs the empirically selected 1.75x-temporally-dilated
Coltrane/Fetch motion references with bounded controls executed in the unchanged
Brax v1 Fetch physics. It is not animal torque data and it is not ordinary
analytical inverse dynamics.

The parent is Demo F variant `temporal-dilation-1p75-v1`: every original
retargeted clip is interpolated at 50 Hz, and one centered 64-frame crop is
kept. The 1.75 factor was selected through visual tempo inspection. It is not
the 4.6237 Froude factor used by canonical Demo F, and the two releases remain
separately versioned.

For every clip, a fixed `kp=400`, `kd=10` feedback controller tracks the ten
reference joint angles. The release stores the simulator-realized body states
and the controls that produced them. Original Demo F poses remain in separate
`reference_*` fields for auditing only.

The temporal contract is exact: `normalized_control[t]` acts during
`[t, t + 1)` and produces `realized_*[t + 1]`. Requested actuator-axis torque
is `-300 * normalized_control`, before Brax joint-limit gating. It must not be
interpreted as biological torque.

## Accepted release

- variant: `exact-fetch-feedback-projection-retime-1p75-v1`;
- 1,784 train, 278 validation, and 342 final-test clips;
- 2,404 clips and 151,452 physical transitions total;
- 99.09% projection pass rate;
- median-across-shard joint tracking RMSE: 0.103 rad;
- mean-across-shard actuator saturation: 1.36%;
- minimum torso height: 1.133;
- minimum uprightness: 0.514;
- build time: 84.4 seconds on the current H100;
- parent manifest SHA-256:
  `85fe54ee9730fe3c79871c6739197496e92b726f5072d93c4322bd001df82b3f`;
- release manifest SHA-256:
  `c02c0cc43775dc28ee33106b4841f7dc7a06696c20e956e7d21aeb36dfd76847`.

Session-level train/validation/test assignments are inherited unchanged from
Demo F. Independent paired-control replay on the same H100/CUDA backend
reproduces the stored state to about `1e-5`, while shuffled controls are
materially worse. CPU replay is not an exact-integrity check for Brax's legacy
contact-rich PBD trajectory and is rejected explicitly.

See `manifest.json` in a built release for shard hashes, package versions, the
exact Fetch-config hash, gains, rejection gates, actuator ordering, and
per-session provenance. Generated releases remain gitignored; reproduce them
with the commands in [`../README.md`](../README.md).
