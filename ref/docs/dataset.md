# ref/docs/data/aldarondo.md

> **Scope.** Local layout and on-disk schema of the *Aldarondo et al. 2024* dataset accompanying the MIMIC paper ([`ref/papers/MIMIC.pdf`](../../papers/MIMIC.pdf)), as downloaded under `/workspace/data/Aldarondo2024/`.

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

---

## 3. Per-session HDF5 schema

| Path | Shape | Dtype | Units | Description |
|---|---|---|---|---|
| `/behavior/motion_mapper` | `(T,)` | `uint8` | — | Per-frame behavior label ID (0–100). IDs 0–99 map to names via the `names` attribute (length-100 array); ID `100` is an unmapped "unclassified" sentinel present in **every** file (~0.4% of frames). See §4. |
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

74 entries. First 7 are the free root joint (3 translation + 4 quaternion components), each carrying the placeholder name `'walker/'`. Remaining 67 are named joints of the rodent MJX model (vertebrae C1–C29, hip/knee/ankle/toe L/R, scapula/shoulder/elbow/wrist/finger L/R, cervical/atlas/mandible). The MIMIC-MJX-side names are identical modulo the `walker/` prefix — see [ref/docs/data/mimic_mjx.md §3](mimic_mjx.md).

---

## 4. Behavior labels

`/behavior/motion_mapper` is **dense** — every frame of every session has an integer label in `[0, 100]`. The `names` attribute (a length-100 array of Python strings) maps IDs `0–99` to a higher-level behavior name. Several IDs collapse to the same name: the ID is the raw cluster from the unsupervised motion-mapper, while the name is a manually assigned coarse category. **Label `100` falls outside the `names` array** — it is an unmapped "unclassified" sentinel found in every file (~0.4% of all frames; exclude it like `(check)` / `TrackingError`).

**Vocabulary (16 distinct names across the dataset):**

```
Amble, AmbleGroom, FaceGroom, Hunch, ProneSlow, ProneSniff, ProneStill,
RearDown, RearHigh, RearLow, RearMid, RearSniff, Walk, WalkFast,
(check), TrackingError
```

`(check)` and `TrackingError` flag frames where the clusterer / tracker had low confidence — exclude these from any pairing or downstream analysis.

**ID multiplicity.** A typical art session uses 90–100 of the 100 IDs; a typical coltrane session uses 92–96. Multiple IDs map to the same name (e.g., `ProneSlow` appears under IDs 7, 14, 41, 47, 50, 66, 88, …). Use the name for coarse grouping and the ID for fine-grained sub-mode identity.

**Approximate composition.** Across recordings, the most common high-level categories are `RearMid`, `ProneSlow`, `ProneStill`, `RearHigh`, and `Amble` — together covering ~50–70% of frames. Locomotion (`Walk` + `WalkFast` + `Amble`) typically accounts for 5–15% of a session.

### Locomotion totals and bout durations

Computed over all **209 files** of this download (locomotion = `Walk` + `WalkFast` + `Amble`; 50 Hz ⇒ 1 frame = 20 ms):

| Category | Frames | Hours | % of all frames |
|---|---|---|---|
| **Locomotion (total)** | **14 447 853** | **80.3 h** | **13.5 %** |
| `Amble` | 10 047 855 | 55.8 h | 9.4 % |
| `Walk` | 3 830 711 | 21.3 h | 3.6 % |
| `WalkFast` | 569 287 | 3.2 h | 0.5 % |

Total corpus = 107 100 000 frames = 595.0 h (consistent with the §7 frame-count table). Per-session locomotion share ranges **3.0–26.9 %** (median 13.2 %), i.e. mid-range of the 5–15 % rule of thumb above.

**Bout durations.** Defining a *segment* as a maximal contiguous run of locomotion frames (a single frame of any non-locomotion label ends it), the 209 sessions hold **1 702 035 segments** with a strongly right-skewed duration distribution — mean 0.17 s, median 0.10 s (5 frames), max 28.08 s; percentiles p75 = 0.20 s, p90 = 0.36 s, p95 = 0.54 s, p99 = 1.04 s.

| Segment duration | Segments | % of segments |
|---|---|---|
| < 0.1 s | 777 435 | 45.7 % |
| 0.1 – 0.2 s | 473 558 | 27.8 % |
| 0.2 – 0.5 s | 348 215 | 20.5 % |
| 0.5 – 1 s | 83 074 | 4.9 % |
| 1 – 2 s | 16 530 | 1.0 % |
| 2 – 5 s | 2 653 | 0.2 % |
| 5 – 10 s | 464 | 0.03 % |
| ≥ 10 s | 106 | 0.01 % |

**Caveat — frame-level flicker.** The motion-mapper labels switch in and out of locomotion frame-by-frame, so raw segments are dominated by very short fragments: **94 % of segments are < 0.5 s** (yet hold only ~70 % of locomotion time), and just **1.2 % are ≥ 1 s** (holding 11.5 % of the time). For genuine *bout*-level analysis, merge runs across small gaps and/or drop ≤2-frame fragments rather than using raw contiguous runs.

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

The two are redundant by design: `qpos` provides a physics-ready state, while `keypoints` lets you re-run STAC or train a different fitting pipeline. **Note:** mimic-mjx provides its own STAC fit of the same source keypoints — the resulting `qpos` numerics differ between Aldarondo and mimic-mjx (different optimizer settings, smoother regularization, possibly different coordinate-frame origin). See [ref/docs/data/mimic_mjx.md §6](mimic_mjx.md).

`/pose/qpos` does **not** include `qvel`, `xpos`, or `xquat` — those have to be either finite-differenced from `qpos` or recomputed via forward kinematics, OR sourced from the mimic-mjx-side reference file when one exists.

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

The mimic-mjx dataset ([ref/docs/data/mimic_mjx.md](mimic_mjx.md)) is **a re-processing of the same DANNCE keypoints** that yields per-frame `qvel/xpos/xquat`. It does not include behavior labels or neural data, so any task that needs neural activity must be anchored on the Aldarondo side.

Coverage of mimic-mjx vs Aldarondo:

| Aldarondo rat | Sessions | Sessions with a mimic-mjx counterpart |
|---|---|---|
| `art` | 62 | **62** but clip-concatenated, not time-aligned (see below) |
| `coltrane` | 38 | **1 frame-aligned** (`2021_07_29_1`, the smoothed file) + **1 partial** (`2021_07_28_1`, contributes 842 clips to the curated pool) |
| `bud`, `duke`, `espie`, `freddie`, `gerry` | 109 | **0** |

**Frame-by-frame pairing is only possible for one session.** Empirical inspection (see [ref/docs/data/mimic_mjx.md §3.5](mimic_mjx.md)) shows that the per-session `art/art_*_reference.h5` files are concatenations of independently-fit 250-frame STAC clips — frame-to-frame discontinuities at every multiple of 250 are ~30× larger than within-clip transitions. This means **mimic-mjx art frames do not correspond to raw-session frames at any simple index mapping**. The only mimic-mjx file that is genuinely continuous is `rodent_reference_coltrane_2021_07_29_1_smoothed.h5`, which matches `coltrane/2021_07_29_1.h5` 1:1 (verified Snout↔TailBase correlation = 1.0000 over 50 000 frames).

For the curated 842-clip pool ([`rodent_reference_clips.h5`](mimic_mjx.md)) and the per-session art files, recovering a frame-by-frame mapping to the original Aldarondo recording requires an upstream snip-extraction artifact (`fit_snips.h5` / `transform_snips.h5` per the mimic-mjx STAC config), which is **not shipped in this download**.

---

## 9. Reference

- Aldarondo, D. et al. *A virtual rodent predicts the structure of neural activity across behaviors.* Nature (2024). Paper local copy: [`ref/papers/MIMIC.pdf`](../../papers/MIMIC.pdf).
