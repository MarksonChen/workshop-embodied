# Demo J — spiking motion imitation and a neural benchmark

**Status (2026-07-21):** implementation checkpoint. The BrainPy LSNN,
verified source alignment, modern Fetch port, reference-tracking environment,
state-only PPO probe, sequence-distillation fallback, and closed-loop spike
recorder are implemented. The beta sweep and neural encoding analysis remain
future work.

## Current implementation checkpoint

- The 1,784/278/342 train/validation/test clips resolve through the retimed
  release to verified unretimed Coltrane starts; no rounded timestamp is trusted
  without checking both parent releases.
- The BrainPy hard-spike/surrogate-gradient smoke test runs 1,000 control bins
  on GPU with finite non-zero gradients.
- A conventional MLP failed to overfit one feasible clip after 10.2M state-only
  PPO transitions (147 s). Direct PPO is therefore rejected at this checkpoint,
  rather than attributing the failure to spiking neurons.
- The declared fallback distills contiguous 58-step sequences from the
  independent feedback controller. A 2,000-update LSNN run took 31 s and
  reached held-out action MSE `0.00309`, with finite activity and no silent
  neurons.
- On the first 64 legacy-target validation clips, that SNN completed 84.4%,
  with median per-frame joint RMSE `0.0646 rad` and no saturated actions. This
  is useful evidence but does **not** pass the predeclared 95% gate.
- A modern-MJX replay cache is implemented to remove legacy/modern global-root
  drift. Its first cross-batch replay check differed by up to `0.054` in qpos,
  so it remains experimental until projection and evaluation use a proven
  batch-invariant replay contract. Do not call it an accepted release yet.

Generated checkpoints, cache arrays, and recordings remain under ignored
`demo_j/out/`. The next behavioral step is to close the replay discrepancy,
retrain against the accepted modern target, and apply sequence aggregation if
validation completion remains below 95%. Only then should the beta sweep and
neural comparison be run.

Demo J will train an independent spiking controller to imitate the physically
realizable Fetch trajectories derived from Demo F. Its hidden spikes become a
synthetic reference recording paired with Fetch motion. The main experiment
will then ask whether otherwise matched Demo H policies become more similar to
that recording as the strength of their frozen generative prior, `beta`, is
changed.

This supersedes the earlier headline comparison `Demo A < Demo F < Demo H`.
Demo F is a kinematic generative model rather than a controller, and the three
demos have different inputs and architectures. A within-Demo-H beta sweep is a
cleaner controlled experiment:

- hold the task, body, observations, prior, residual-policy architecture,
  training budget, evaluation trajectories, and random seeds fixed;
- vary only the coefficient multiplying the KL to the frozen prior;
- compare only the trainable Demo H residual-controller activity across beta;
- retain Demo A/F/H comparisons, if useful, as a clearly labelled appendix.

The primary question is therefore:

> Among Demo H policies that can perform the locomotion task, how does beta
> affect the ability of their controller activity to predict an independent
> imitation controller's 20 ms spike counts beyond what is already predictable
> from body motion and the reference command?

Do not assume that the answer is monotonic. A stronger prior may improve the
match, plateau, or eventually impair task performance and the match. The
accepted `beta=0.10` policy is one pre-existing point on this curve, not a
result selected using Demo J.

## Workshop story

```text
Demo F retargeted locomotion
        |
        | reference-tracking imitation RL
        v
Demo J recurrent spiking actor ----> 20 ms synthetic spike recording

same held-out Fetch trajectory and command
        |
        +----> Demo H beta=0.000 residual activity --+
        +----> Demo H beta=0.025 residual activity   |
        +----> Demo H beta=0.050 residual activity   +--> neural encoding score
        +----> Demo H beta=0.075 residual activity   |
        +----> Demo H beta=0.100 residual activity   |
        +----> Demo H beta=0.150 residual activity --+

source-aligned Coltrane clips
        |
        +----> Demo J spikes versus real Coltrane DLS spike counts
```

This supports three beginner-friendly lessons:

1. Imitation can be posed as reinforcement learning when demonstrations give
   desired motion but not the actions that produced it.
2. A spiking policy can be trained by differentiating through simulated neuron
   dynamics while the environment remains non-differentiable.
3. Neural similarity is a held-out measurement, not a reward. It must be
   interpreted after controlling for shared behavior.

Call Demo J's spikes a **synthetic reference recording** or an **operational
ground truth for the controlled benchmark**. They are not biological ground
truth. A good match to Demo J means that a Demo H representation resembles this
particular independently trained inverse controller; it does not by itself
show that either network is brain-like.

## Design choices from the literature and local references

| Evidence | Decision for Demo J |
|---|---|
| [MIMIC](https://www.nature.com/articles/s41586-024-07633-4) found that inverse-controller activity predicted sensorimotor neural activity beyond kinematics and used neural encoding models and representational geometry. The full paper is also available at [`ref/papers/MIMIC.pdf`](../ref/papers/MIMIC.pdf). | Analyze the spiking **controller**, use a behavior-only baseline, make cross-validated Poisson prediction the primary score, and use representational similarity only as a secondary analysis. |
| The local [`track-mjx` intention networks](../ref/repos/track-mjx/track_mjx/agent/ff_ppo/intention_network.py) separate a short future-reference encoder from a proprioceptive action decoder, while its [recurrent PPO implementation](../ref/repos/track-mjx/track_mjx/agent/recurrent_ppo/networks.py) carries and resets controller state. | Encode five future reference frames into a compact intention, then replace the recurrent decoder—not the critic—with an SNN. Reset every neuronal state on episode reset. |
| [CoMic](https://proceedings.mlr.press/v119/hasenclever20a.html), [DeepMimic](https://arxiv.org/abs/1804.02717), and TRACK-MJX learn motion imitation with physics-in-the-loop reference tracking. | Train the primary model with recurrent PPO from state-only demonstrations; do not supervise it on Demo H's pseudo-controls. |
| [BrainPy](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f11394cdd377aab9ff5e2a4e9f27367f-Abstract-Conference.html) provides differentiable neuron dynamics on JAX/XLA, and the local [training quickstart](../ref/repos/BrainPy/docs/quickstart/training.ipynb) demonstrates gradient-based LIF training. | Implement the neuron dynamics in BrainPy and use surrogate gradients through spikes. Keep the model small enough to inspect and render. |
| Adaptive spiking neurons give recurrent SNNs longer temporal memory in [LSNNs](https://papers.nips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html). | Use a mixed LIF/adaptive-LIF recurrent population instead of a large feed-forward stack. |
| [SuperSpike](https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/) gives a practical voltage-based surrogate derivative for otherwise non-differentiable spikes. | Start with a fast-sigmoid surrogate and verify non-zero, finite gradients before environment training. |
| [PopSAN](https://proceedings.mlr.press/v155/tang21a.html) demonstrates a spiking actor paired with an ordinary deep critic for continuous control. | Keep the actor spiking and the PPO critic non-spiking. Never include critic activity in the neural comparison. |
| [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) compares representations of different widths, while [crossvalidated distances and whitened unbiased RDM similarity](https://nbdt.scholasticahq.com/article/27664-comparing-representational-geometries-using-whitened-unbiased-distance-matrix-similarity) reduce bias in RSA. | Use behavior-residualized linear CKA as a secondary check and crossnobis/WUC RSA as an exploratory population-level analysis. |

There is no single universally standard ANN-to-spike comparison. For this
dataset, reproducing MIMIC's encoding-model logic is more defensible than
matching units one-to-one or selecting a convenient generic similarity score.

The biological qualifier applies to the model dynamics, not the entire learning
algorithm. LIF/ALIF neurons, synaptic filtering, and emitted spikes are more
biophysically grounded than an MLP. Dense connectivity, PPO, backpropagation
through time, and surrogate gradients are not thereby biologically plausible.
Do not advertise Demo J as a model of how the brain learns. An e-prop or
Dale-constrained variant can be a later ablation, not part of the first
workshop implementation.

## Experimental independence and leakage rules

Demo J may use the same **training distribution** as Demo H because that is the
scientific question, but it must remain an independently trained controller.

- Train Demo J only from reference states, imitation reward, and task resets.
- Do not initialize it from Demo H, query any Demo H policy, use Demo H actions,
  optimize neural similarity, or inspect biological spikes during training.
- Do not use Demo J or biological scores to choose a Demo H checkpoint, beta,
  locomotion reward, or naturalness loss.
- Keep Demo F's existing session-level train/validation/test split. Perform all
  headline neural comparisons on untouched test sessions.
- Select SNN hyperparameters and the canonical Demo J seed using imitation and
  non-degenerate-spiking metrics on train/validation data only.
- Report two additional Demo J seeds as a sensitivity analysis; never select a
  seed because it gives the desired beta ordering.

The gait, symmetry, stance, contact, and spike-similarity metrics remain
validation-only. Reference-tracking errors are legitimate imitation rewards;
hand-engineered naturalness metrics are not.

## Data and time contracts

### Control-training release

Build Demo J's references from the `1.75x` temporal-dilation view of Demo F and
the accepted exact-physics projection described by [Demo H](../demo_h/README.md).
This keeps the visually accepted timing and gives every target sequence a known
physically feasible realization. Discard the projection controls when training
the primary SNN; PPO must discover controls that track the state sequence.

Each example must contain:

- the complete 64-frame, 50 Hz Fetch state sequence;
- the Demo F 60-D feature at each frame;
- Fetch generalized position/velocity and reference root/paw transforms needed
  for tracking reward and diagnostics;
- `session`, parent clip ID, exact original `source_start`, split, retiming
  variant, and all dataset/manifold hashes;
- an exact source-frame map. Store integer indices for the unretimed parent and
  floating-point source coordinates for retimed views.

Do not infer alignment from the current rounded `source_start` alone. Add a
versioned sidecar before any neural claim. Validate monotonic source indices,
clip bounds, session identity, split disjointness, and ten manually sampled
motion/ephys alignments.

### Two deliberately separate neural clocks

The main synthetic comparison uses the control clock: four 5 ms SNN steps are
summed into each 20 ms Fetch control bin. This exactly matches the bin width of
the stored recordings but does not make the retimed motion simultaneous with
the original rat spikes.

The real-neural bridge uses the unretimed parent clip at its original 50 Hz
source clock. Teacher-force the SNN with the corresponding retargeted state and
future reference, record its responses, and align those frames directly to
`/ephys/spike_counts`. This probe measures representation under source-timed
input; it is not a physics rollout and must be labelled as a clock/domain-shift
analysis.

Do not interpolate integer spike counts onto the `1.75x` clock and then fit a
Poisson model as though they were observed counts. A time-warped, phase-based
RSA may be shown separately, with the time warp stated explicitly.

Only Coltrane has exact provenance for this release, so the biological analysis
is a DLS analysis. Follow [`ref/docs/dataset.md`](../ref/docs/dataset.md): use
the per-session `active_units` mask, treat one frame as one 20 ms bin, and never
assume that unit columns have identity across sessions.

## Imitation task

Use reference-tracking PPO rather than behavior cloning as the primary method.
Demo F provides desired motion, not animal torques. This mirrors MIMIC and
TRACK-MJX: the policy learns actions through physical interaction while reward
measures agreement with a motion demonstration.

At reset:

1. sample a training session, clip, and valid starting frame;
2. initialize Fetch near the reference state, with small train-only pose and
   velocity noise after the basic task works;
3. reset membrane voltage, synaptic current, adaptation, filtered-spike traces,
   previous action, and PPO recurrent state;
4. expose the next five reference frames, root-relative to the current body.

At every 20 ms control step, reward reference agreement in:

- root position, height, orientation, and linear/angular velocity;
- ten joint angles and joint velocities;
- four root-relative paw positions;
- bounded action magnitude and action change.

Normalize every squared tracking error by a robust scale estimated from the
training split before applying exponential kernels. Start with the relative
term weights in the closest TRACK-MJX Fetch adaptation, inspect per-term
magnitudes, then freeze them before full validation. Terminate on large root
drift, unrecoverable torso orientation/height, non-finite state, or clip end.
Do not reward a particular contact sequence, cadence, limb symmetry, or gait.

Use this curriculum:

1. overfit one short feasible clip and require near-exact tracking;
2. train on many starts from a small clip set;
3. train on all training sessions;
4. introduce reset noise and modest perturbations only after nominal imitation
   passes;
5. evaluate once on validation, freeze the design, then evaluate test sessions.

## SNN actor

Mirror TRACK-MJX's intention/decoder split while keeping the lesson compact.

### Inputs

- `proprioception`: the current normalized Demo F 60-D Fetch feature;
- `previous_action`: ten normalized actuator commands;
- `reference`: five future 60-D features expressed as errors or root-relative
  targets from the current state.

A small deterministic reference MLP maps the flattened five-frame reference to
a 32-D intention. Avoid a VAE in the first implementation: stochastic intention
latents add a second lesson without helping the SNN question. A learned affine
current encoder maps `[proprioception, previous_action, intention]` into the
spiking population.

### Recurrent decoder

Use one recurrent population of 256 neurons:

- 128 current-based LIF neurons;
- 128 adaptive-threshold LIF neurons;
- exponential input and recurrent synaptic currents;
- 5 ms integration and four internal steps per 20 ms action;
- initial time constants `tau_m=20 ms`, `tau_syn=10 ms`, and
  `tau_adapt=500 ms`;
- a fast-sigmoid surrogate derivative at the hard spike threshold;
- a 20 ms exponential filtered-spike trace feeding the action readout.

In schematic form,

\[
\tau_m \dot v = -(v-v_{rest}) + I_{in} + W_{rec}z,\qquad
z=\mathbb 1[v>\vartheta+a],
\]

with a slow adaptation state `a` for the ALIF half. A linear readout maps the
filtered population spikes to the mean of a ten-dimensional tanh-Gaussian
policy. Learn one state-independent log standard deviation initially. Use a
two-layer 256-unit ordinary MLP critic with privileged reference information.

Record the hard spikes used in the forward pass, not surrogate values or
membrane voltages. Save substep spikes with shape `[T, 4, 256]` and define the
canonical neural observation as their integer sum `[T, 256]` in 20 ms bins.
Membrane voltage, adaptation, and filtered traces may be retained for debugging
but are not the headline activity.

Apply only weak, train-only activity stabilization sufficient to avoid a silent
or saturated network. Choose its coefficient from imitation performance,
finite-gradient checks, and broad non-degeneracy bounds—not by matching the DLS
firing-rate histogram. Report the coefficient and the full firing-rate
distribution.

### Optimization

Adapt recurrent PPO from [`track_mjx/agent/recurrent_ppo`](../ref/repos/track-mjx/track_mjx/agent/recurrent_ppo):

- unroll complete contiguous sequences and never shuffle individual time steps;
- truncate gradient history initially at 32 control steps;
- propagate the BrainPy neuronal state between chunks but stop gradients at
  explicit truncation boundaries;
- reset every state with the environment `done` mask;
- clip global gradients and log membrane, adaptation, firing-rate, entropy, and
  action-saturation summaries;
- compute the critic and generalized advantage estimates conventionally;
- differentiate the PPO actor loss through the BrainPy simulation using the
  surrogate spike derivative, not through the physics simulator.

Before PPO, require a tiny supervised gradient test in which the spike readout
learns a known action sequence. This separates SNN/autodiff failures from
imitation-environment failures.

## Runtime architecture and fail-fast fork

The current local stacks cannot simply be imported together:

- Demo F/H's live Fetch environment uses legacy `brax.v1` with JAX `0.4.30`;
- the current local BrainPy uses `brainevent>=0.0.7`, whose resolver requires a
  newer JAX generation;
- the current local TRACK-MJX stack already uses modern JAX `0.10.2`, Brax
  `0.14.0`, and MuJoCo/MJX.

Do not loosen package pins until imports happen to work. Use a dedicated Demo J
runtime and exchange versioned NumPy artifacts with legacy Demo H.

### Preferred path: direct BrainPy actor in modern MJX

1. Port the same Fetch morphology, joint limits, masses, actuators, ground,
   timestep, and observation mapping into an MJCF model supported by current
   MJX.
2. Compare neutral kinematics, actuator signs/ranges, one-step dynamics, and
   fixed open-loop control sequences against legacy Fetch.
3. Accept the port only when differences are documented and small enough that
   the 60-D feature and reference contracts remain meaningful.
4. Train the BrainPy actor directly with recurrent PPO in the modern runtime.

Physics need not be byte-identical across engines, but model scaling and action
semantics must be explicit. Keep cross-Demo comparisons teacher-forced on the
same feature arrays whenever engine differences could confound them.

### Fallback: independent teacher plus sequence distillation

If the MJX port fails the parity or close-imitation gate, do not build a custom
cross-version on-policy transport. Instead:

1. train an ordinary recurrent reference-tracking teacher in the legacy Fetch
   environment, independently of every Demo H beta policy;
2. collect teacher state/reference/action sequences;
3. train the BrainPy SNN with surrogate-gradient sequence behavior cloning;
4. export the SNN equations and weights to a minimal pure-JAX legacy inference
   function and verify step-by-step spike/action parity;
5. aggregate corrective labels from states visited by the SNN, following the
   dataset-aggregation logic of [DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), until closed-loop drift passes.

This remains imitation learning and may be the more reliable workshop artifact,
but it must be named **teacher distillation**, not direct imitation PPO. Record
which path produced the accepted checkpoint in its manifest.

## Behavioral acceptance gates

Establish an ordinary recurrent ANN tracking baseline with the same observation
and reward. Promote a Demo J checkpoint only if all of the following hold on
validation before touching the test set:

- at least 95% of held-out 64-frame references complete without early failure;
- normalized tracking return reaches at least 90% of the matched ANN baseline;
- median joint-angle RMSE is at most 0.15 rad;
- median root-planar and paw-position RMSE are each at most 0.15 Fetch units;
- fewer than 2% of actions lie within 0.01 of saturation;
- no non-finite state, gradient, voltage, or spike count occurs;
- the recurrent state resets exactly and batching does not change a single
  sequence's deterministic output beyond numerical tolerance;
- a tiled target-versus-policy video shows at least six held-out clips spanning
  the retained speed distribution.

If the physically projected reference itself exceeds one of the absolute error
thresholds, also report error relative to that reference's generating replay.
Do not relax a gate after looking at neural similarity.

## Synthetic recording protocol

After accepting Demo J, freeze its checkpoint and generate one immutable test
recording. For every held-out clip save:

- canonical 60-D current state and five-frame reference;
- commanded and measured speed, phase, action, reward terms, and done mask;
- 5 ms hard spikes, 20 ms integer spike counts, and optional neuronal states;
- reference and realized Fetch generalized state;
- session/source mapping, model/data/config hashes, seed, runtime versions, and
  whether the recording is teacher-forced or closed-loop.

Use deterministic policy means for the canonical recording. Stochastic action
repeats can be a variability analysis, but must not replace it. Generated
recordings, checkpoints, and videos stay under `demo_j/out/` and remain
gitignored; commit compact manifests and aggregate metrics only.

The primary fixed-trajectory recording teacher-forces every network on the same
held-out physically feasible state/reference arrays. It isolates representation
from behavioral divergence. Closed-loop rollouts are a required secondary
analysis because they reveal what each policy actually visits.

## Controlled Demo H beta sweep

Train a new matched sweep at:

```text
beta = [0.000, 0.025, 0.050, 0.075, 0.100, 0.150]
seed = [0, 1, 2]
```

Keep 30M transitions, 2,048 environments, the frozen prior hash, initialization
scheme, observation contract, speed distribution, and all PPO hyperparameters
fixed. Retrain even when an old checkpoint has the same nominal beta; old runs
are useful pilots but are not a controlled sweep. Add beta and sweep ID to the
checkpoint envelope itself rather than relying only on adjacent JSON.

For each fixed input sequence, extract:

- the first 128-D residual-actor hidden layer after SiLU;
- the second 128-D residual-actor hidden layer after SiLU;
- the ten-dimensional residual mean correction and final policy mean.

Do **not** use the frozen prior's activations as the beta-dependent signal. They
are identical across beta when evaluated on identical inputs. The critic is
also excluded. Select a single residual layer using validation data, lock it,
and report every layer in a supplement so layer selection is auditable.

Show task and gait validation next to neural similarity. Report every beta,
including policies that fail. Restrict the headline trend to a predeclared
behavior-qualified subset and repeat it with tracking score as a covariate. This
prevents a stationary or failed policy from appearing neurally similar merely
because its inputs have little variation.

Fit both a linear beta trend and a quadratic trend. The preregistered qualitative
possibilities are monotonic improvement, saturation, an intermediate optimum,
or no reliable relationship. Do not rewrite the hypothesis around whichever
curve appears.

## Primary neural comparison

### Why an encoding model

Demo J has 256 spike channels and Demo H has continuous hidden units with no
one-to-one correspondence. Directly correlating sorted channels is arbitrary.
Instead ask whether a regularized mapping from a candidate representation can
predict held-out target spike counts.

For target channel `u`, fit a Poisson generalized linear model with log link:

\[
y_{t,u}\sim\operatorname{Poisson}(\lambda_{t,u}),\qquad
\log\lambda_{t,u}=b_u+B_{t-4:t}w_u+X^{(\beta)}_{t-4:t}v_u.
\]

Here `B` is the behavior/reference baseline and `X` is one Demo H residual
layer. Use a causal five-bin window. The strong baseline contains the current
60-D state, previous action, five-frame future reference, command, and phase;
it contains no learned model activity. Candidate activity is a deterministic
nonlinear function of these inputs, so incremental prediction means it supplies
a useful learned feature basis—not new causal information.

Follow the MIMIC analysis as closely as the locomotion-only setting allows:

1. standardize from training folds only;
2. reduce each candidate representation by training-fold PCA retaining 90% of
   variance, with an equal maximum dimensionality across beta;
3. begin with elastic-net mixing `0.5` and regularization `0.01`, changing them
   only by nested validation;
4. use five blocked folds made of contiguous four-second chunks;
5. score held-out cross-validated log-likelihood ratio in bits per spike and
   Poisson deviance pseudo-`R^2`;
6. compute the primary increment

\[
\Delta\mathrm{CVLL}_{J}(\beta)
=\mathrm{CVLL}(B+X^{(\beta)}\rightarrow J_{spikes})
-\mathrm{CVLL}(B\rightarrow J_{spikes}).
\]

Aggregate channels within clip/session, then perform a hierarchical paired
bootstrap over H seed, Demo J replication seed, and source session. Neurons are
not independent experimental replicates. Publish the full per-session table,
confidence intervals, and the paired beta contrasts.

Required negative and positive controls are:

- time-shift candidate activity by offsets larger than the temporal window;
- permute clip identity within coarse speed/contact strata;
- compare an untrained SNN of the same architecture;
- compare the behavior-only model;
- verify that Demo J's own filtered/readout activity predicts its spikes better
  than shuffled activity without using this self-score as the headline result.

Do not use Victor–Purpura or van Rossum spike-train distance as the primary
metric. There is no justified unit correspondence between Demo H hidden units,
Demo J neurons, or biological neurons, and the biological data are already
binned at 20 ms.

### Secondary representation measures

- Regress the behavior baseline from Demo J counts and Demo H activations using
  training folds, then compute held-out linear CKA on the residuals.
- Construct speed-bin × gait-phase × turn-bin conditions with enough repeated
  samples, estimate crossvalidated Mahalanobis distances, and compare RDMs with
  whitened unbiased cosine similarity.
- Treat RSA as exploratory because Demo F contains a narrow locomotion subset,
  unlike MIMIC's broad behavioral repertoire.

The primary result is the encoding-model beta curve. CKA and RSA may support or
qualify it; they do not replace it if they disagree.

## Bridge to real neural recordings

Use only the untouched unretimed test-session parent clips for the main DLS
bridge. For each Coltrane session:

1. load `/ephys/spike_counts[:, active_units]`;
2. index the exact original frames recorded in the clip;
3. teacher-force Demo J at the original 20 ms clock and extract its integer
   spike counts;
4. fit the same blocked Poisson models, separately per session because unit
   identity and count differ across sessions;
5. compare behavior-only prediction with behavior plus Demo J spikes;
6. report session-level `Delta CVLL_DLS` and pseudo-`R^2`, then bootstrap
   sessions rather than pooling all unit columns.

As an exploratory extension, extract every Demo H beta representation under the
same source-clock teacher forcing and ask whether its DLS predictivity shows the
same beta curve as the synthetic benchmark. Label this clearly: Demo H and Demo
J were trained on a different physical clock, and the analysis is DLS-only.

Biological spikes are never used to train Demo J, tune beta, choose a layer,
choose a time constant, select a seed, or decide whether a result is shown. A
positive biological result supports the synthetic benchmark's relevance. A
null result does not invalidate the controlled synthetic result, but it prevents
claims that the beta curve is biologically validated.

## Planned implementation map

Create code only after the runtime spike succeeds. Keep the eventual package
small enough to combine with Demo F/H in one workshop notebook:

```text
demo_j/
  README.md            this plan and final accepted evidence
  config.py            frozen data, neuron, policy, and analysis contracts
  dataset.py           reference release and exact source/ephys sidecar
  fetch_mjx.py         modern Fetch port and parity probes
  env.py               reference-tracking task and reward decomposition
  snn.py               BrainPy intention encoder, LSNN actor, state reset
  train.py             recurrent PPO entry point
  record.py            fixed-trajectory and closed-loop spike recordings
  compare.py           beta encoding models, CKA, RSA, and controls
  visualize.py         target/policy video, raster, beta and DLS figures
  artifacts.py         fail-closed manifests and hash validation
  tests/               contracts, gradients, reset, alignment, leakage, parity
  out/                 generated checkpoints, recordings, figures, videos
```

Do not copy TRACK-MJX wholesale. Reuse the ideas and the minimum tested
components needed for future references, recurrent PPO sequence handling, and
checkpointing. Put no dataset-building scripts outside `demo_j/`.

## Implementation sequence and stop/go gates

### Phase 0 — runtime and one-neuron spike

- Create the isolated modern BrainPy/MJX runtime.
- Differentiate a loss through 1,000 recurrent neuron steps.
- Verify finite non-zero surrogate gradients, deterministic reset, batching,
  JIT, GPU execution, and 20 ms binning.
- Stop if a version pin or BrainPy API is being bypassed with monkey patches.

### Phase 1 — data provenance and Fetch parity

- Build the exact source-frame/ephys sidecar and session-leakage tests.
- Port Fetch to modern MJX and validate kinematics, action signs, joint limits,
  gravity, contacts, and open-loop dynamics.
- Choose the direct or distillation fork based on the documented parity result.

### Phase 2 — ordinary imitation baseline

- Implement reference sampling, reward terms, resets, and termination.
- Train an ANN recurrent policy and pass the behavioral gates.
- Stop and fix the task if the ANN cannot closely imitate; do not compensate by
  enlarging the SNN.

### Phase 3 — SNN imitation

- Pass the toy gradient test and single-clip overfit.
- Train the full SNN with three seeds and promote a canonical seed from
  validation imitation only.
- Generate one six-clip target-versus-policy comparison video and inspect it.

### Phase 4 — frozen recording and beta sweep

- Freeze and hash Demo J.
- Add activation hooks and beta metadata to Demo H without changing its policy
  computation.
- Retrain the six-beta, three-seed controlled sweep.
- Export fixed-trajectory and closed-loop arrays across the runtime boundary.

### Phase 5 — neural analysis

- Lock preprocessing, layer choice, temporal window, and GLM regularization on
  validation data.
- Run the synthetic test analysis and all nulls once.
- Run the source-clock Coltrane DLS bridge once.
- Produce machine-readable metrics before presentation figures.

### Phase 6 — workshop packaging

- Load accepted artifacts in the live notebook rather than running the full
  SNN and 18-policy sweep during class.
- Offer a short single-clip SNN training cell to expose surrogate gradients.
- Show one synchronized panel containing target Fetch, imitating Fetch, and a
  20 ms raster; then show beta similarity beside task performance.
- State the synthetic-ground-truth and source-clock limitations on the slide,
  not only in speaker notes.

Do not promise a five-minute full Demo J run before Phase 0 profiling. The six
Demo H beta values already require 18 matched runs for three seeds, even though
one current Demo H run is about 95 seconds on the available H100. Precompute the
scientific sweep; keep only the explanatory micro-example live.

## Definition of done

Demo J is shippable when:

- the runtime path and physics differences are explicit and reproducible;
- the controller passes every held-out imitation gate and its video visibly
  follows the reference rather than merely staying upright;
- hard spikes are non-degenerate, exactly binned at 20 ms, and reproducible from
  the artifact manifest;
- source frames align exactly to masked Coltrane DLS counts with no neural data
  leakage;
- every Demo H beta point is trained under one matched contract and all points
  are reported;
- the primary behavior-controlled Poisson comparison, shuffle nulls, CKA/RSA
  sensitivity checks, and task-performance context are saved;
- the conclusion describes the observed beta curve without claiming that an
  SNN is biological ground truth or that representational predictivity proves a
  shared mechanism.
