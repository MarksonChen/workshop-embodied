# Aldarondo 2024 dataset

> **Scope.** Local layout and on-disk schema of the *Aldarondo et al. 2024* dataset accompanying the MIMIC paper ([`ref/papers/MIMIC.pdf`](../papers/MIMIC.pdf)), as downloaded under `/workspace/data/Aldarondo2024/`.

---

## 1. What this dataset is

Freely-behaving rats with simultaneous (i) markerless 3D pose tracking from a six-camera array (DANNCE → 23 anatomical-landmark skeleton; MIMIC Methods *Behavioural apparatus*, *Videography*, *Pose tracking*) and (ii) extracellular electrophysiology from custom **128-channel tetrode drives** (32 tetrodes × 4 channels) implanted unilaterally in either dorsolateral striatum (DLS) or motor cortex (MC) — **never both** in the same animal (MIMIC Methods *Electrophysiology*). Each session is a continuous recording of either **2 or 3 hours** in a **cylindrical 1 m-diameter arena with 60 cm walls** (MIMIC Methods *Behavioural apparatus*, *Recording protocol*).

The dataset spans **7 female Long-Evans rats**: `art`, `bud`, `coltrane`, `duke`, `espie`, `freddie`, `gerry` (209 sessions total in this download). The MIMIC paper headline totals are **607 h** of simultaneous video + ephys (353.5 h DLS + 253.5 h MC). One implanted animal yielded no spike-sorted neurons and is included for behavioral data only — in our download this is `espie` (no `/ephys` group).

The mocap stream has additionally been (i) registered to a MuJoCo rat body model via STAC inverse kinematics, producing per-frame `qpos` (MIMIC Methods *Skeletal registration*; STAC = Wu et al. 2013, ref. 44) and (ii) labelled per-frame by **motion-mapper** (Berman et al. 2014) — a wavelet-then-tSNE behavioral clusterer (MIMIC Methods *Behavioural classification*) whose output is mapped to ~16 high-level behavior names.

---

## 2. Local layout

```
~/data/Aldarondo2024/
├── art/        62 sessions  (e.g., 2020_12_22_1.h5, ..., 2021_02_23_1.h5)
├── bud/        26 sessions
├── coltrane/   38 sessions
├── duke/       30 sessions
├── espie/       6 sessions
├── freddie/    25 sessions
└── gerry/      22 sessions
```

File naming: `<rat>/<YYYY>_<MM>_<DD>_<X>.h5` where `X` is the recording index for that day. Per-session file sizes range from ~62 MB (partial session) to ~370 MB. Each file is self-contained — no shared metadata files.

### Obtaining it (e.g. on a fresh / remote box)

The full set (209 `.h5`, ~65 GB, all 7 dirs incl. `espie` which is motion-only) is mirrored to the **private** HF
dataset repo **`MarksonChen/CANVAS`** (`repo_type="dataset"`), preserving the `animal/<file>.h5` layout `prepare.py`'s
glob expects. Pull it into the default `CANVAS_DATA_ROOT` so paths just work:

```python
from huggingface_hub import snapshot_download   # not a project dep -- run via: uv run --with 'huggingface_hub[hf_xet]'
snapshot_download("MarksonChen/CANVAS", repo_type="dataset", local_dir="~/data/Aldarondo2024")
```

Needs a **read**-scoped `HF_TOKEN` in the environment. The token is a machine/secret detail, not project knowledge —
keep it out of the repo (env var or a git-ignored local note), never here. On the original box the global `hf` CLI is
broken (typer/click clash on py3.10); the `--with` form above sidesteps it. The mirror was seeded from the OpenNeuro
source, so a fresh box can equally re-download from OpenNeuro/DANDI at datacenter bandwidth instead of via HF.

---

## 3. Per-session HDF5 schema

| Path | Shape | Dtype | Units | Description |
|---|---|---|---|---|
| `/behavior/motion_mapper` | `(T,)` | `uint8` | — | Per-frame behavior label ID, **1-indexed**: IDs run `1–100` and map to the length-100 `names` attribute as **`names[id - 1]`**. ID `0` never occurs. There is no "unclassified" sentinel. See §4. |
| `/ephys/spike_counts` | `(T, U)` | `uint8` | spikes/frame | Per-frame spike counts on `U` putative single units. **`U` varies per session** (range 2–708 across the dataset). The `/ephys` group is **absent** for all `espie` sessions. See §5. |
| `/pose/keypoints` | `(T, 3, 23)` | `float64` | mm | 3D Cartesian positions of 23 named keypoints. The `names` attribute lists keypoint names in channel order. |
| `/pose/qpos` | `(T, 74)` | `float64` | m, rad | MJX joint coordinates after STAC fitting. `names` attribute uses `walker/<joint>` form; the first 7 entries are root translation+quaternion (no per-joint name). |

Root-group attributes: `animal` (e.g., `b'art'`), `session` (e.g., `b'2020_12_22_1'`).

**Frame count `T` varies per session** — see §7. The four datasets above are jointly indexed by frame, so all share the same `T` within a file.

### Keypoint names (`/pose/keypoints@names`, in order)

```
Snout, EarL, EarR, SpineF, SpineM, SpineL, TailBase,
ShoulderL, ElbowL, WristL, HandL, ShoulderR, ElbowR, WristR, HandR,
HipL, KneeL, AnkleL, FootL, HipR, KneeR, AnkleR, FootR
```

### Joint names (`/pose/qpos@names`, in order)

74 entries. First 7 are the free root joint (3 translation + 4 quaternion components), each carrying the placeholder name `'walker/'`. Remaining 67 are named joints of the rodent MJX model (vertebrae C1–C29, hip/knee/ankle/toe L/R, scapula/shoulder/elbow/wrist/finger L/R, cervical/atlas/mandible). The MIMIC-MJX-side names are identical modulo the `walker/` prefix, as verified against the local MIMIC-MJX assets.

---

## 4. Behavior labels

> ⚠️ **Corrected 2026-07-19 — the label→name mapping in this section was off by one, and every number derived from it
> was wrong.** *(Scope: §4 only — the ID convention, the composition figures, and the locomotion/bout tables. §5 Neural
> data, §6 Pose, and §7 frame counts never used the mapping and are untouched.)*
>
> **What was wrong.** This section read IDs as `0–99` into `names`, with `100` an unmapped "unclassified" sentinel. The
> labels are in fact **1-indexed** (`names[id - 1]`), so every frame was attributed to its *neighbouring* cluster's
> name, and the 0.48 % of frames at ID `100` — a real `Walk` cluster — were discarded as junk.
>
> **How it was caught, and why the correction is trusted.** Three independent checks, two of them physical:
> *(i)* over all 209 files, ID `0` **never occurs** while `1–100` do — under the old reading cluster 0 is
> inexplicably unused *and* 100 needs an ad-hoc sentinel; under 1-indexing the 100 IDs and 100 names are a clean
> bijection. *(ii)* **Root speed** is monotone across the semantic ladder only under 1-indexing
> (`ProneStill` 0.003 → `Amble` 0.030 → `Walk` 0.084 → `WalkFast` 0.266 m/s); under the old reading clusters named
> `Walk` sit at 0.005 m/s, i.e. not moving. *(iii)* **Root height** separates `Rear*` from `Prone*` by **+7.3 cm**
> under 1-indexing and by **−0.4 cm** under the old reading, which makes `RearHigh` the *lowest* posture in the
> dataset. Speed and height are orthogonal signals and agree.
>
> **What survives.** The schema, the 16-name vocabulary, the ID-multiplicity structure, and the qualitative shape of
> the corpus (a mostly-stationary rat, locomotion a ~10 % minority) all stand. **What changed most:** the dominant
> category is `ProneStill` at 50 %, not `RearMid` at 14 %; locomotion is **11.3 %**, not 13.5 %; and locomotion bouts
> are **substantially longer and less flickery** than the old numbers implied (mean 0.37 s vs 0.17 s) — the flicker
> caveat below is correspondingly weaker. **No ledger row consumed these numbers** (checked by grep over `ref/`), so
> nothing outside this file needed revising.

`/behavior/motion_mapper` is **dense** — every frame of every session has an integer label. **The labels are 1-indexed:** IDs run `1–100` and map into the length-100 `names` attribute as **`names[id - 1]`**. ID `0` never occurs in any of the 209 files, and there is **no** unclassified sentinel — every ID that occurs is a real motion-mapper cluster (99 of the 100 appear somewhere in the corpus; one is simply never used). (The 1-indexing is unsurprising given motion-mapper's MATLAB origin; Berman et al. 2014.) Several IDs collapse to the same name: the ID is the raw cluster from the unsupervised motion-mapper, while the name is a manually assigned coarse category.

**Vocabulary (16 distinct names across the dataset):**

```
Amble, AmbleGroom, FaceGroom, Hunch, ProneSlow, ProneSniff, ProneStill,
RearDown, RearHigh, RearLow, RearMid, RearSniff, Walk, WalkFast,
(check), TrackingError
```

`(check)` and `TrackingError` flag frames where the clusterer / tracker had low confidence — exclude these from any pairing or downstream analysis.

**ID multiplicity.** A session uses **84–97** of the 100 IDs (median 95); 99 of the 100 appear somewhere in the corpus. Multiple IDs map to the same name — `ProneSlow` is the widest, spanning 30 IDs (6, 8, 15, 18, 20, 21, 25, 35, 42, 48, 51, 52, …), while `Walk` spans 5 (16, 55, 59, 90, 100). Use the name for coarse grouping and the ID for fine-grained sub-mode identity.

**Composition (all 209 files, `names[id - 1]`).** The corpus is dominated by stationary behavior — half of all frames are a single category:

| Name | % of frames | Hours | median root speed | p90 |
|---|---|---|---|---|
| `ProneStill` | 50.13 % | 298.3 | 0.003 m/s | 0.011 |
| `ProneSlow` | 21.61 % | 128.6 | 0.006 | 0.030 |
| `ProneSniff` | 8.96 % | 53.3 | 0.005 | 0.027 |
| `Amble` | 7.45 % | 44.4 | 0.030 | 0.119 |
| `Walk` | 3.37 % | 20.1 | 0.084 | 0.237 |
| `RearHigh` | 2.40 % | 14.3 | 0.027 | 0.133 |
| `FaceGroom` | 2.36 % | 14.1 | 0.017 | 0.084 |
| `RearMid` | 0.80 % | 4.8 | 0.027 | 0.168 |
| `AmbleGroom` | 0.74 % | 4.4 | 0.047 | 0.188 |
| `RearDown` | 0.63 % | 3.7 | 0.094 | 0.422 |
| `RearLow` | 0.48 % | 2.9 | 0.006 | 0.034 |
| `WalkFast` | 0.43 % | 2.6 | 0.266 | 0.473 |
| `RearSniff` | 0.39 % | 2.3 | 0.008 | 0.034 |
| `TrackingError` | 0.18 % | 1.0 | 0.237 | 1.884 |
| `Hunch` | 0.03 % | 0.2 | 0.038 | 0.211 |
| `(check)` | 0.02 % | 0.1 | 0.007 | 0.024 |

The top five names cover **91.5 %** of frames. Note `TrackingError`'s speed — a median 50× the corpus median and a p90 of 1.9 m/s, physically impossible in a 1 m arena — which is both a sanity check on the mapping and a reminder to drop it (see §6 for the artifact tail).

### Locomotion totals and bout durations

Computed over all **209 files** of this download (locomotion = `Walk` + `WalkFast` + `Amble`; 50 Hz ⇒ 1 frame = 20 ms). Provenance for every number in §4 and §6's speed subsection, including the mapping audit: `exploration/behavior_label_audit.py`.

| Category | Hours | % of all frames |
|---|---|---|
| **Locomotion (total)** | **67.0 h** | **11.26 %** |
| `Amble` | 44.4 h | 7.45 % |
| `Walk` | 20.1 h | 3.37 % |
| `WalkFast` | 2.6 h | 0.43 % |

Total corpus = 107 100 000 frames = 595.0 h (consistent with the §7 frame-count table). Per-session locomotion share ranges **1.5–35.4 %** (median 10.3 %) — the spread is largely *per-animal*, not per-session noise: `freddie` and `gerry` locomote 2–3× as much as `art`, `bud`, `duke`, and `espie`.

**Bout durations.** Defining a *segment* as a maximal contiguous run of locomotion frames (a single frame of any non-locomotion label ends it), the 209 sessions hold **643 362 segments**, right-skewed — mean 0.37 s, median 0.14 s (7 frames), max 17.72 s; percentiles p75 = 0.34 s, p90 = 0.90 s, p95 = 1.66 s, p99 = 3.72 s.

| Segment duration | Segments | % of segments | % of locomotion time |
|---|---|---|---|
| < 0.1 s | 230 146 | 35.8 % | 4.2 % |
| 0.1 – 0.2 s | 151 269 | 23.5 % | 8.5 % |
| 0.2 – 0.5 s | 150 780 | 23.4 % | 18.7 % |
| 0.5 – 1 s | 52 912 | 8.2 % | 15.1 % |
| 1 – 2 s | 33 476 | 5.2 % | 19.5 % |
| 2 – 5 s | 22 140 | 3.4 % | 26.9 % |
| 5 – 10 s | 2 535 | 0.4 % | 6.6 % |
| ≥ 10 s | 104 | 0.02 % | 0.5 % |

**Caveat — frame-level flicker.** The motion-mapper labels still switch in and out of locomotion frame-by-frame, so segments are *numerically* dominated by short fragments: **82.7 % of segments are < 0.5 s**. But those fragments hold only **31.4 %** of locomotion time, while the **9.1 %** of segments lasting ≥ 1 s hold **53.5 %** of it. So most locomotion *time* is in respectable multi-second bouts even though most *segments* are flicker. Merging across small gaps and/or dropping ≤2-frame fragments is still the right move for bout-level analysis — just note the flicker is a smaller problem than this section previously claimed (under the off-by-one it read as 94 % of segments / 70 % of time).

---

## 5. Neural data

`/ephys/spike_counts` is `(T, U) uint8`. The `U` columns are putative single units recovered from spike sorting on the 32-tetrode (128-channel) drive; **`U` varies per session** rather than being fixed per-rat. **Crucially, `U` counts *all* spike-count columns — the per-session *usable* subset (`active_units == True`, see below) is much smaller, a median of ~15 % of `U`.** Always mask to `active_units` before analysis. Observed ranges in this download (region per the user-confirmed dataset-site mapping):

| Rat | Region | raw `U` (columns) | active units / session (min – med – max) | Sessions |
|---|---|---|---|---|
| `art` | DLS | 6 – 169 | 0 – 13 – 59 | 62 |
| `bud` | DLS | 2 – 258 | 0 – 4 – 19 | 26 |
| `coltrane` | DLS | 206 – 708 | 8 – 25 – 38 | 38 |
| `duke` | MC | 36 – 197 | 10 – 20 – 36 | 30 |
| `espie` | MC* | — (no `/ephys` group) | — | 6 |
| `freddie` | MC | 61 – 162 | 18 – 26 – 32 | 25 |
| `gerry` | MC | 40 – 255 | 7 – 19 – 31 | 22 |

\*`espie` is the MC implant that yielded no neurons; excluded (no `/ephys` group). **6 of the 203 ephys sessions have zero active units** (all `art`/`bud`), leaving 197 sessions with usable neurons.

The MIMIC paper reports the 7 animals as **3 DLS implants (353.5 h, 2 654 neurons total) + 4 MC implants (253.5 h, 1 177 neurons total across the 3 productive MC implants)** — the same paragraph that reports the per-region totals in Results §1 specifies "DLS: three animals … MC: three animals", and MIMIC Methods *Electrophysiology* notes "One animal implanted in MC yielded no neurons and was thus excluded from electrophysiological analyses", giving the 4th MC implant. The per-animal region assignment is **not** stored as a per-file attribute; the user-confirmed mapping (from the dataset site) is **DLS: `art`, `bud`, `coltrane`; MC: `duke`, `freddie`, `gerry`** (plus `espie`, the excluded MC implant). Note that **raw column count is not a reliable region proxy**: `coltrane` (DLS) has by far the most raw columns, but its *active*-unit yield (median 25) is comparable to the MC animals `duke`/`freddie`/`gerry` (median 19–26), and the two other DLS animals (`art`, `bud`) yield fewer — see the active-unit analysis below. `espie` is the excluded MC implant — its sessions have no `/ephys` group at all in this download.

Per-session attributes on `/ephys/spike_counts`:

- **`active_units`** (boolean attr, length `U`): per-session quality mask — units judged usable in this session. Inactive columns are still present but should be masked out for analysis.
- **`unique_labels`** (int attr, length `U`): per-column spike-sorter cluster ID.
- **`original_labels_map`** (JSON string attr): per-column provenance. Each entry has `channel_group` (tetrode index, 0–31) and `label` (cluster ID inside that tetrode). Downstream code typically only needs `active_units`.

Spike-count columns from different sessions of the same rat are **not** identity-aligned (`U` and `unique_labels` differ session to session). Pairing units across sessions requires a separate cross-session matching step that this dataset does not provide.

Bin width equals one mocap frame (1 / 50 Hz = 20 ms; see §7), aligned to the camera trigger via the acquisition FPGA (MIMIC Methods *Electrophysiology*).

### Active units, firing rates, and dynamic range

Computed over all **203 ephys sessions** of this download (active units only — `active_units == True`; region per the mapping above; 1 bin = 20 ms ⇒ rate (Hz) = mean spike count × 50).

**Active-unit yield by region (session-summed).**

| Region | Animals | Σ active unit-sessions |
|---|---|---|
| DLS | `art` 1 011 + `bud` 180 + `coltrane` 938 | **2 129** |
| MC | `duke` 610 + `freddie` 639 + `gerry` 442 | **1 691** |
| **Total** | | **3 820** |

⚠️ These are **session-summed unit counts, not unique neurons**. They do **not** equal the paper's per-region totals (2 654 DLS / 1 177 MC): the grand totals happen to be close (3 820 vs 3 831), but the DLS/MC split differs markedly because the paper counts neurons by a different method and **spike-count columns are not identity-aligned across sessions** (see below). Do not equate summed `active_units` with neuron counts.

**Firing rates (mean spikes/s per active unit, pooled across sessions).** Strongly right-skewed; **MC units fire faster than DLS**:

| Population | n | median | mean | p10 | p90 | max |
|---|---|---|---|---|---|---|
| All | 3 820 | 6.35 Hz | 11.5 | 0.54 | 31.0 | 118.6 |
| DLS | 2 129 | 6.08 Hz | 9.3 | 0.48 | 20.7 | 118.6 |
| MC | 1 691 | 6.61 Hz | 14.3 | 0.70 | 40.2 | 109.9 |

```
firing-rate distribution (Hz), active units pooled
  <0.5     355   9.3%  ####
  0.5–1    294   7.7%  ###
  1–2      394  10.3%  #####
  2–5      654  17.1%  ########
  5–10     744  19.5%  #########
  10–20    741  19.4%  #########
  20–50    509  13.3%  ######
  >=50     129   3.4%  #
```

**Sparsity & dynamic range.**

- Per-session nonzero-bin fraction over active columns: median **15.0 %** (range 0.9–31.6 %) — i.e. a typical active unit is silent in ~85 % of 20 ms bins.
- **The largest spike count in any 20 ms bin across the entire dataset is `15`** — far below the `uint8` cap of 255. There is no clipping/saturation; counts comfortably fit even `int8`, and 0 active units come anywhere near the cap.
- No active unit is silent for a whole session (0 of 3 820), confirming `active_units` is a meaningful usability mask.

---

## 6. Pose

Two parallel pose representations are stored:

- **`/pose/keypoints`** is the raw 3D-tracked marker stream (output of the upstream DANNCE pipeline), in **mm**, in the lab/world frame.
- **`/pose/qpos`** is STAC's fit of the MuJoCo rat body model to those keypoints, in **m and radians**, with a free root joint expressed as 3D translation + unit quaternion (XYZW convention follows MJX).

The two are redundant by design: `qpos` provides a physics-ready state, while `keypoints` lets you re-run STAC or train a different fitting pipeline. **Note:** MIMIC-MJX provides its own STAC fit of the same source keypoints — the resulting `qpos` numerics differ between Aldarondo and MIMIC-MJX because of different fitting and smoothing choices and possibly a different coordinate-frame origin.

`/pose/qpos` does **not** include `qvel`, `xpos`, or `xquat` — those have to be either finite-differenced from `qpos` or recomputed via forward kinematics, OR sourced from the mimic-mjx-side reference file when one exists.

### Root speed distribution

Speed here = horizontal root translation, `‖diff(qpos[:, :2])‖ × 50` (m/s). Computed over all 209 files (595 h); per-frame quantiles from a 10 % systematic subsample (10.7 M frames), window quantiles exact over all 1 673 345 windows. Provenance: `exploration/behavior_label_audit.py`.

**The distribution is unimodal and heavily right-skewed — it is not bimodal, and there is no natural rest/locomote split on speed alone.** In log space it is one near-stationary mode at ~4 mm/s with a long tail; any locomotion threshold is a cut through that tail, not a valley between two modes. This is worth knowing before treating a speed cutoff as principled — it is a choice of operating point.

| | p1 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|
| per-frame (20 ms) | 0.0002 | 0.0011 | 0.0022 | **0.0047** | 0.0138 | 0.0469 | 0.0961 | 0.288 | 0.0225 |
| 1.26 s window mean | 0.0019 | 0.0030 | 0.0044 | **0.0079** | 0.0200 | 0.0555 | 0.0923 | 0.193 | 0.0222 |

The median frame is at 4.7 mm/s — that is tracking jitter on a stationary animal, not movement. Fraction of the corpus above a given speed:

| threshold (m/s) | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|---|---|
| per-frame | 19.3 % | 9.5 % | **4.8 %** | 2.9 % | 1.9 % | 0.9 % | 0.3 % |
| 1.26 s window | 25.0 % | 11.2 % | **4.3 %** | 1.9 % | 0.9 % | 0.3 % | 0.1 % |

The bolded column is `prepare.py`'s `SPEED_MIN = 0.10` — i.e. the speed gate alone admits **~4–5 %** of the corpus. (That gate is one of four; see the note below.)

⚠️ **The far tail is tracking artifact, not behavior.** Per-frame speed reaches **25.7 m/s**, which is impossible for a rat in a 1 m arena — these are pose-fit glitches, and they are concentrated in the `TrackingError` label (§4: median 0.237 m/s, p90 1.88 m/s). Time-averaging suppresses but does not remove them (the 1.26 s window mean still reaches 2.44 m/s). **Any speed statistic computed without excluding `TrackingError` or clipping the tail will be contaminated** — the mean is ~4.8× the median for exactly this reason. Prefer medians/quantiles, and prefer window means to per-frame values.

**Relation to the frozen loco subset.** `canvas/prepare.py`'s `loco_clips()` gates on speed **and** three other conditions (`GAIT_MIN`, `TURN_MAX`, `NECK_MAX`). Measured on 8 `coltrane` sessions (24 h): the speed gate passes 4.1 % of windows, the gait-coordination gate 46.8 %, turn 99.0 %, neck 94.1 % — but the **conjunction passes only 1.67 %**. It is **gait coordination, not speed, that co-binds**: the two gates are near-independent, so the conjunction lands well below the product of the marginals. Within the retained subset, speed is tightly concentrated: median 0.132 m/s, p95 0.238 m/s.

**How much loco data exists in total.** Running the frozen `loco_clips()` over all 209 sessions yields **18 930 clips = 6.73 h**, i.e. **1.13 %** of the 595 h corpus. Per animal:

| animal | sessions | clips | yield | loco wall-clock |
|---|---|---|---|---|
| `gerry` | 22 | 4 182 | 2.25 % | 89.2 min |
| `freddie` | 25 | 4 182 | 2.02 % | 89.2 min |
| `coltrane` | 38 | 3 394 | 1.11 % | 72.4 min |
| `duke` | 30 | 2 013 | 0.80 % | 42.9 min |
| `art` | 62 | 3 466 | 0.75 % | 73.9 min |
| `bud` | 26 | 1 445 | 0.69 % | 30.8 min |
| `espie` | 6 | 248 | 0.49 % | 5.3 min |

(`freddie` and `gerry` landing on the same 4 182 total is coincidence — different session counts and different per-session distributions.) 18 682 of the 18 930 clips come from sessions that have `/ephys`; only `espie`'s 248 do not, so the loco subset is barely reduced by requiring neural data.

⚠️ **Yield is not the only restriction — session scope is the bigger one.** `build()` only opens the sessions named by `CANVAS_ANIMALS` (default `coltrane`) up to `CANVAS_MAX_SESS` (default 8), so a default run sees the loco clips of **8 of the 209 sessions — 1 125 clips = 24 min**, about **6 %** of the 6.73 h that exists. The loco gate is FROZEN (`prepare.py`); the session scope is **not** — it is an env var, and widening it is the cheapest way to get more motion data.

---

## 7. Frame counts, timing, and partial sessions

Camera shutter trigger rate is **50 Hz** (MIMIC Methods *Videography*: "controlled synchronously by a 50 Hz Arduino hardware trigger"); the dataset abstract states the corpus totals 607 h sampled at 50 Hz. Frame count `T` therefore directly encodes session duration: `duration_sec = T / 50`. The MIMIC paper *Recording protocol* specifies sessions of "2 or 3 h", which matches the observed distribution exactly.

Distribution of `T` across the 209 files:

| `T` | Duration | Count | Notes |
|---|---|---|---|
| 540 000 | 3 h | 181 | Standard later-protocol session. |
| 360 000 | 2 h | 24 | Earlier-protocol art and a few bud / coltrane sessions. |
| 270 000 | 1.5 h | 1 | `freddie/2022_06_06_1.h5` |
| 180 000 | 1 h | 2 | `coltrane/2021_08_21_1.h5`, `coltrane/2021_08_23_1.h5` |
| 90 000 | 0.5 h | 1 | `bud/2021_07_12_1.h5` |

Partial-length files indicate truncated recordings; downstream code that assumes a uniform `T` must check per-file shape.

---

## 8. Pairing with mimic-mjx

The MIMIC-MJX dataset is **a re-processing of the same DANNCE keypoints** that yields per-frame `qvel/xpos/xquat`. It does not include behavior labels or neural data, so any task that needs neural activity must be anchored on the Aldarondo side.

Coverage of mimic-mjx vs Aldarondo:

| Aldarondo rat | Sessions | Sessions with a mimic-mjx counterpart |
|---|---|---|
| `art` | 62 | **62** but clip-concatenated, not time-aligned (see below) |
| `coltrane` | 38 | **1 frame-aligned** (`2021_07_29_1`, the smoothed file) + **1 partial** (`2021_07_28_1`, contributes 842 clips to the curated pool) |
| `bud`, `duke`, `espie`, `freddie`, `gerry` | 109 | **0** |

**Frame-by-frame pairing is only possible for one session.** Empirical inspection shows that the per-session `art/art_*_reference.h5` files are concatenations of independently-fit 250-frame STAC clips — frame-to-frame discontinuities at every multiple of 250 are ~30× larger than within-clip transitions. This means **MIMIC-MJX art frames do not correspond to raw-session frames at any simple index mapping**. The only MIMIC-MJX file that is genuinely continuous is `rodent_reference_coltrane_2021_07_29_1_smoothed.h5`, which matches `coltrane/2021_07_29_1.h5` 1:1 (verified Snout↔TailBase correlation = 1.0000 over 50 000 frames).

For the curated 842-clip pool (`rodent_reference_clips.h5`) and the per-session art files, recovering a frame-by-frame mapping to the original Aldarondo recording requires an upstream snip-extraction artifact (`fit_snips.h5` / `transform_snips.h5` per the MIMIC-MJX STAC config), which is **not shipped in this download**.

---

## 9. Reference

- Aldarondo, D. et al. *A virtual rodent predicts the structure of neural activity across behaviors.* Nature (2024). Paper local copy: [`ref/papers/MIMIC.pdf`](../papers/MIMIC.pdf).
