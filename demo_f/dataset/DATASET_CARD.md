---
license: odc-by
pretty_name: Aldarondo 2024 Coltrane locomotion retargeted to Brax Fetch
tags:
  - motion-capture
  - reinforcement-learning
  - robotics
  - neuroscience
  - motion-retargeting
---

# Aldarondo 2024 locomotion retargeted to Fetch

This dataset contains the strict locomotion subset of all 38 sessions from the
`coltrane` rat in Aldarondo et al. (2024), deterministically retargeted onto the
unmodified 10-DoF Brax v1 Fetch body. It is the data-only input to Demo F in the
Embodied SSL + RL workshop.

Each row is a non-overlapping 64-frame trajectory sampled at 50 Hz. Rows contain
only the derived Fetch trajectory and source provenance; raw rat keypoints,
neural recordings, behavior labels, and video are not redistributed.

## Data fields

Each `data/<split>/<session>.npz` shard contains:

| field | shape | dtype | meaning |
|---|---:|---|---|
| `joint_angles` | `(N,64,10)` | float32 | Fetch joint angles, radians |
| `root_position` | `(N,64,3)` | float32 | Fetch root trajectory, Fetch model units |
| `root_quaternion` | `(N,64,4)` | float32 | scalar-first root quaternion |
| `feet_local` | `(N,64,4,3)` | float32 | realized Fetch foot endpoints in root frame |
| `contacts` | `(N,64,4)` | uint8 | source-derived stance mask |
| `command` | `(N,3)` | float32 | hindsight forward/lateral displacement and yaw |
| `source_start` | `(N,)` | int32 | first frame in the named source session |
| `source_speed_mps` | `(N,)` | float32 | net rat displacement speed; may be small on curved paths |
| `source_path_speed_mps` | `(N,)` | float32 | mean rat path speed |
| `ik_foot_rmse` | `(N,)` | float32 | semantic endpoint fitting error |
| `contact_speed_mean` | `(N,)` | float32 | realized Fetch stance-foot speed |
| `minimum_foot_height` | `(N,)` | float32 | minimum realized foot height |
| `joint_limit_fraction` | `(N,)` | float32 | fraction within 1% of a joint limit |

Foot ordering is front-right, front-left, back-right, back-left. Joint ordering
and every transformation parameter are recorded in `manifest.json`.

## Construction

The builder applies Demo B's frozen geometric locomotion screen to
non-overlapping 64-frame blocks: mean path speed above 0.10 m/s, coordinated joint
motion, less than 90 degrees turning, and limited neck-height drift. Retargeting
then uses a smoothed trunk frame and four semantic paws, body-size normalization,
stance detection/pinning, and bounded sequence-level inverse kinematics with
pose/velocity/acceleration regularization. Rodent bone rotations are not copied.

Train, validation, and test splits contain 26, 6, and 6 entire recording
sessions. No session appears in more than one split.

## Intended use and limitations

The dataset supports conditional motion modelling, motion-prior diagnostics,
and simple control experiments on Fetch. It represents **rodent-derived Fetch
motion**, not rat biomechanics: morphology, length, mass, actuation, and physical
feasibility differ. The trajectories are kinematic IK results and are not torque
supervision or guaranteed dynamically feasible demonstrations.

## Source, attribution, and license

The source recordings are:

> Aldarondo, D. et al. *A virtual rodent predicts the structure of neural
> activity across behaviours.* Nature 632, 594–602 (2024).
> https://doi.org/10.1038/s41586-024-07633-4

Original dataset: https://doi.org/10.7910/DVN/FB0MZT

The Harvard Dataverse source declares the Open Data Commons Attribution License
(ODC-By) 1.0. This derived dataset is redistributed under ODC-By 1.0 and retains
the required source attribution. Users should also cite the original paper and
dataset. The Fetch implementation originates in Brax, licensed under Apache 2.0.

## Reproduction

The exact schema, source hashes, code hashes, configuration, per-session counts,
quality gates, and shard SHA-256 values are in `manifest.json`. Builder and
validator source live in the companion workshop repository under
`demo_f/dataset/`.
