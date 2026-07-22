# Workshop 4: From reward to a generative motor prior

A quadruped trained only to move forward may discover a motion that works but looks strange.
A model trained only on recorded motion may generate something that looks plausible but cannot
survive contact with the ground. In this workshop, we make the two kinds of learning cooperate.

We build one idea at a time:

1. **Reinforcement learning** learns *what works* from task reward.
2. **Self-supervised learning** learns *what motion looks like* from data.
3. A **generative motor prior + RL** learns motion that is both data-like and useful in physics.

In short: **SSL gives distributional realism; RL gives functional realism.**


# Section 0: Setup

Imports, causal-padding details, data loading, training loops, runtime isolation, checkpoints,
metrics, and video rendering live here. **You do not need to read this section.**

Set `FULL_TRAINING = False` for a quick wiring check. The later architecture cells are kept
inline and match the models instantiated by the tested trainers under `workshop/`.



```python
import json
import math
import os
import pathlib
import subprocess
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from IPython.display import Video, display

ROOT = (
    pathlib.Path.cwd()
    if (pathlib.Path.cwd() / "workshop").is_dir()
    else pathlib.Path.cwd().parent
)
PART1 = ROOT / "workshop" / "part1"
PART2 = ROOT / "workshop" / "part2"
PART3 = ROOT / "workshop" / "part3"
FULL_TRAINING = True
DEV = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


class CausalConv1d(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride=1):
        super().__init__()
        self.padding = kernel - stride
        self.convolution = nn.Conv1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        return self.convolution(F.pad(values, (self.padding, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, inputs, outputs, kernel, stride):
        super().__init__()
        self.trim = kernel - stride
        self.convolution = nn.ConvTranspose1d(inputs, outputs, kernel, stride=stride)

    def forward(self, values):
        output = self.convolution(values)
        return output[..., : -self.trim] if self.trim else output


def sinusoidal_positions(length, width, device):
    position = torch.arange(length, device=device)[:, None].float()
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device).float() * (-math.log(10_000.0) / width)
    )
    output = torch.zeros(length, width, device=device)
    output[:, 0::2] = torch.sin(position * frequency)
    output[:, 1::2] = torch.cos(position * frequency)
    return output


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def show_video(path, width=900):
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Video(str(path), embed=True, width=width))


def show_videos(paths, width=440):
    for path in paths:
        print(pathlib.Path(path).stem)
        show_video(path, width=width)


def show_generation_speeds(path):
    values = json.loads(pathlib.Path(path).read_text())
    rows = [
        {
            "rodent command (m/s)": round(row["requested_source_speed_mps"], 2),
            "realized Fetch speed (units/s)": round(
                row["realized_fetch_forward_speed"], 2
            ),
            "rodent-equivalent speed (m/s)": round(
                row["realized_source_equivalent_speed_mps"], 2
            ),
        }
        for row in values["videos"]
    ]
    display(rows)
    return rows


def show_report(path, keys=None):
    values = json.loads(pathlib.Path(path).read_text())
    display(values if keys is None else {key: values.get(key) for key in keys})
    return values


def _run(command, *, cpu=False):
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    if cpu:
        environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _modern(module, *arguments):
    _run(
        [
            "uv",
            "run",
            "--extra",
            "workshop",
            "python",
            "-m",
            module,
            *map(str, arguments),
        ]
    )


def _legacy(module, *arguments, media=False, cpu=False):
    jax_package = (
        "jax[cuda12]==0.4.30"
        if torch.cuda.is_available() and not cpu
        else "jax==0.4.30"
    )
    packages = ["brax==0.12.3", jax_package, "jaxlib==0.4.30", "scipy>=1.15"]
    if media:
        packages += ["imageio>=2.37", "imageio-ffmpeg>=0.6", "pillow>=11"]
    command = ["uv", "run", "--no-project", "--isolated"]
    for package in packages:
        command += ["--with", package]
    command += ["python", "-m", module, *map(str, arguments)]
    _run(command, cpu=cpu)


def _latest(directory, pattern):
    paths = list(pathlib.Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime)


print(f"ready; PyTorch on {DEV}; full training = {FULL_TRAINING}")
```

    ready; PyTorch on cuda; full training = True



```python
def describe_motion_data():
    manifest = json.loads(
        (ROOT / "workshop" / "data" / "part3_reference" / "manifest.json").read_text()
    )
    clips = manifest["counts"]["train"]
    scaling = manifest["dynamic_scaling"]
    values = {
        "source": "Coltrane rat locomotion retargeted to Fetch",
        "training clips": clips,
        "body features": (clips, 64, 60),
        "hindsight commands": (clips, 3),
        "sampling": "50 Hz; 64 frames = 1.28 s",
        "timing": f"{scaling['time_scale']}x temporal dilation",
        "0.20 m/s rodent command": f"{scaling['recommended_fetch_speed']:.2f} Fetch units/s",
    }
    display(values)
    return values


def describe_action_data():
    manifest = json.loads(
        (ROOT / "workshop" / "data" / "part3" / "manifest.json").read_text()
    )
    clips = manifest["counts"]["train"]
    values = {
        "clips": clips,
        "realized body states": (clips, 64, 60),
        "bounded controls": (clips, 63, 10),
        "one action": "10 actuator controls",
    }
    display(values)
    return values


def run_part1(full=True, seeds=range(6)):
    arguments = [] if full else ["--smoke"]
    _legacy("workshop.part1.train", *arguments)
    checkpoint = _latest(PART1 / "out", "policy_*.pkl")
    output_dir = PART1 / "out" / "notebook_rollouts"
    videos, reports = [], []
    for seed in seeds:
        video = output_dir / f"seed_{seed}.mp4"
        _legacy(
            "workshop.part1.visualize",
            "--checkpoint",
            checkpoint,
            "--output",
            video,
            "--seed",
            seed,
            media=True,
            cpu=True,
        )
        videos.append(video)
        reports.append(video.with_suffix(".json"))
    return {
        "checkpoint": checkpoint,
        "videos": videos,
        "reports": reports,
    }


def run_part2(full=True):
    dataset = ROOT / "workshop" / "data" / "part3_reference"
    checkpoint = PART2 / "out" / "notebook_prior.pt"
    generated = PART2 / "out" / "notebook_generated"
    video = PART2 / "out" / "notebook_motion_sweep.mp4"
    arguments = ["--dataset-root", dataset, "--output", checkpoint]
    if not full:
        arguments.append("--smoke")
    _modern("workshop.part2.train", *arguments)
    _modern(
        "workshop.part2.evaluate",
        "--checkpoint",
        checkpoint,
        "--dataset-root",
        dataset,
        "--output",
        PART2 / "out" / "notebook_evaluation.json",
    )
    _modern(
        "workshop.part2.generate",
        "--checkpoint",
        checkpoint,
        "--dataset-root",
        dataset,
        "--output-dir",
        generated,
    )
    _legacy(
        "workshop.part2.visualize",
        "--input-dir",
        generated,
        "--output",
        video,
        media=True,
        cpu=True,
    )
    return {
        "checkpoint": checkpoint,
        "video": video,
        "evaluation": PART2 / "out" / "notebook_evaluation.json",
        "generation": generated / "metrics.json",
    }


def pretrain_part3(full=True):
    checkpoint = PART3 / "out" / "prior.pt"
    exported = PART3 / "out" / "prior_jax.npz"
    arguments = ["--output", checkpoint]
    if not full:
        arguments.append("--smoke")
    _modern("workshop.part3.pretrain", *arguments)
    _modern(
        "workshop.part3.evaluate_prior",
        "--checkpoint",
        checkpoint,
        "--output",
        PART3 / "out" / "prior_evaluation.json",
    )
    _modern(
        "workshop.part3.export",
        "--checkpoint",
        checkpoint,
        "--output",
        exported,
    )
    return {
        "checkpoint": checkpoint,
        "exported": exported,
        "evaluation": PART3 / "out" / "prior_evaluation.json",
    }


def run_part3(beta=0.10, full=True):
    prior = PART3 / "out" / "prior_jax.npz"
    if not prior.is_file():
        pretrain_part3(full=full)
    arguments = ["--prior", prior, "--beta", beta]
    if not full:
        arguments.append("--smoke")
    _legacy("workshop.part3.train", *arguments)
    checkpoint = _latest(PART3 / "out", "policy_*.pkl")
    sweep = PART3 / "out" / f"speed_sweep_beta_{beta:g}"
    video = sweep / "comparison.mp4"
    _legacy(
        "workshop.part3.visualize",
        "--checkpoint",
        checkpoint,
        "--prior",
        prior,
        "--output-dir",
        sweep,
        "--label",
        f"beta={beta:g}",
        cpu=True,
    )
    _legacy(
        "workshop.part3.render",
        sweep / "metrics.json",
        "--output",
        video,
        "--columns",
        6,
        media=True,
        cpu=True,
    )
    return {
        "checkpoint": checkpoint,
        "video": video,
        "metrics": sweep / "metrics.json",
    }
```

# Part 1: Functional realism from reward

Part 1 deliberately uses **no recorded motion**. The controller only receives a task and a
simulated body. This isolates the central question of reinforcement learning:

> **Which actions make future reward larger?**


# Section 1: Reinforcement learning from zero

Our environment is **Brax Fetch**, a simulated dog-like quadruped. It has a segmented torso,
four two-link legs, and **ten actuated hinge joints**. Every 20 ms (50 Hz), one interaction happens:

1. Fetch reports a **101-number observation** describing torso orientation, body positions and
   velocities in the torso frame, and contacts.
2. The policy chooses **ten actuator commands**, each bounded to $[-1,1]$.
3. The physics simulator applies those commands, advances gravity, joints, and contacts by one step,
   and returns the next observation and reward.

An episode lasts at most 1,000 steps (20 s), and ends early if Fetch falls. This gives concrete
meanings to six RL words:

- The **environment** is Fetch's body, ground, and physics.
- The **observation** $o_t$ is the 101-number description available now.
- The **policy** $\pi(a_t\mid o_t)$ turns an observation into an action distribution.
- The **action** $a_t$ contains the ten bounded motor commands.
- The **reward** $r_t$ is one number judging the latest transition.
- The **return** $G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots$ values the future, not just now.

One interaction step is

```text
observation ──► policy ──► action ──► Fetch + ground physics
     ▲                                      │
     └──────── next observation + reward ◄──┘
```

Learning changes the policy so episodes with larger return become more likely.


### Give the policy a small, readable task

We ask Fetch to move forward at target speed $v^*$, remain upright, and avoid needlessly large
controls:

$$
r_t =
\exp\!\left[-\frac{(v_{x,t}-v^*)^2}{2\sigma^2}\right]
+0.1\,\mathrm{upright}_t
-10^{-3}\lVert a_t\rVert^2.
$$

Falling ends the episode. There is no hand-written stepping pattern: cyclic locomotion must
emerge because it helps the return.



```python
def task_reward(forward_speed, upright, action, target_speed=3.0, speed_width=1.0):
    tracking = torch.exp(
        -(forward_speed - target_speed).square() / (2 * speed_width**2)
    )
    control_cost = 1e-3 * action.square().sum(dim=-1)
    return tracking + 0.1 * upright - control_cost


speed = torch.tensor([0.0, 2.0, 3.0, 4.0])
reward = task_reward(speed, torch.ones(4), torch.zeros(4, 10))
print("speed :", speed.tolist())
print("reward:", reward.round(decimals=3).tolist())
```

    speed : [0.0, 2.0, 3.0, 4.0]
    reward: [0.11100000143051147, 0.7070000171661377, 1.100000023841858, 0.7070000171661377]


The reward is highest at the requested speed. It says **what success means**, not how each leg
should move. That distinction is why RL is task-driven.


# Section 2: The policy and PPO

The motor controller is just a small MLP. Fetch supplies 101 observations. Four hidden layers
produce 20 numbers: a mean and scale for each of ten actions. Sampling makes the policy
exploratory; `tanh` keeps every actuator command between −1 and 1.

The class below is a PyTorch reading of the same 4×32 policy architecture used by Brax PPO.



```python
class PPOPolicy(nn.Module):
    def __init__(self, observation_dim=101, action_dim=10):
        super().__init__()
        layers = []
        width = observation_dim
        for _ in range(4):
            layers += [nn.Linear(width, 32), nn.SiLU()]
            width = 32
        self.trunk = nn.Sequential(*layers)
        self.distribution = nn.Linear(width, 2 * action_dim)

    def forward(self, observation):
        mean, raw_scale = self.distribution(self.trunk(observation)).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + 1e-3
        return mean, scale

    def sample(self, observation):
        mean, scale = self(observation)
        return torch.tanh(mean + scale * torch.randn_like(mean))
```


```python
policy = PPOPolicy()
observation = torch.randn(256, 101)
mean, scale = policy(observation)
action = policy.sample(observation)

print(f"policy parameters : {parameter_count(policy):,}")
print(f"observation       : {tuple(observation.shape)}")
print(f"mean and scale    : {tuple(mean.shape)}, {tuple(scale.shape)}")
print(
    f"bounded action    : {tuple(action.shape)} in [{action.min():.2f}, {action.max():.2f}]"
)
```

    policy parameters : 7,092
    observation       : (256, 101)
    mean and scale    : (256, 10), (256, 10)
    bounded action    : (256, 10) in [-0.97, 0.97]


PPO trains this policy with four repeated operations:

1. Run the current policy in many parallel Fetch environments.
2. Use a critic to estimate the **advantage** $\hat A_t$: was action $a_t$ better or worse than
   expected at observation $o_t$?
3. Increase the probability of positive-advantage actions and decrease the probability of
   negative-advantage actions.
4. Refuse to learn too much from one batch.

The last point is the “proximal” part. Compare the updated policy with the policy that collected
the rollout:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\mathrm{old}}(a_t\mid o_t)}.
$$

A ratio above 1 means the action became more likely; below 1 means less likely. PPO maximizes

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb E_t\!\left[
\min\!\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

Intuitively, the first term says “repeat actions that worked.” The clipped term says “but once
their probability has moved roughly $\epsilon$ from the old policy, stop rewarding an even larger
jump.” This conservative update is why PPO is usually easier to train than an unrestricted policy
gradient. The critic and its value loss are training machinery, so their code stays in Section 0.


### Train and inspect Part 1

One successful rollout can be luck. We therefore render **six independent environment resets**
from the same trained policy. The initial perturbations differ; the task and controller do not.
Parallel simulation, PPO bookkeeping, checkpoints, and rendering remain setup details.



```python
part1_result = run_part1(full=FULL_TRAINING)
part1_reports = [json.loads(path.read_text()) for path in part1_result["reports"]]
display(
    [
        {
            "seed": seed,
            "mean speed": round(report["mean_speed"], 2),
            "speed RMSE": round(report["speed_rmse"], 2),
            "return": round(report["return"], 1),
        }
        for seed, report in enumerate(part1_reports)
    ]
)
show_videos(part1_result["videos"])
```

Part 1 can learn locomotion quickly because the reward directly tests the desired function.
But reward alone does not say which successful gait belongs to the motion distribution we care
about. **Functional realism is not automatically distributional realism.**


# Part 2: Distributional realism from retargeted rodent motion

The data begin with **Coltrane**, a freely behaving Long–Evans rat from Aldarondo et al. (2024).
Six synchronized cameras recorded 3-D anatomical landmarks at 50 Hz while Coltrane moved around an
arena. We keep locomotion from all 38 Coltrane sessions and use only the behavioral pose here—the
simultaneously recorded neural activity is not an input to this model.

A rat and Fetch do not share a skeleton. The retargeting pipeline maps the rodent's trunk and limb
geometry onto Fetch's ten joints, solves for poses that preserve the limb trajectories, scales the
motion to Fetch's body size, recomputes its four foot positions and contacts, and rejects poor fits.
The result is **rodent-derived motion expressed on the same Fetch body used in Part 1**.

We use the accepted **1.75× temporal dilation**: a 0.20 m/s rodent command corresponds to about
2.44 Fetch units/s. This factor is an empirical retargeting choice, not a claim of exact dynamic
similarity. The same retimed clips will supply Part 3's physics-derived actions.

Part 2 removes reward and environment interaction. It asks:

> **Given recent retargeted motion and an intended displacement, what motion usually comes next?**

This is self-supervised learning because every target comes from the sequence itself, not a human
annotation.


# Section 3: Turn the future into its own supervision

Each retargeted Fetch clip contains 64 frames. One frame has 60 body features: root motion and
orientation, ten joint angles and velocities, four foot positions and velocities, and four
contact bits.

At time $t$, construct three pieces:

- $h_t$: retargeted motion before $t$;
- $w_t$: retargeted motion after $t$;
- $c_t$: future displacement and turn, measured from that same future.

Because $c_t$ is computed *after the motion happened*, it is a **hindsight command**. No one
manually labels a clip “walk this far.” The rodent-derived clip labels itself.



```python
motion_data = describe_motion_data()
```


    {'source': 'Coltrane rat locomotion retargeted to Fetch',
     'training clips': 1804,
     'body features': (1804, 64, 60),
     'hindsight commands': (1804, 3),
     'sampling': '50 Hz; 64 frames = 1.28 s',
     'timing': '1.75x temporal dilation',
     '0.20 m/s rodent command': '2.44 Fetch units/s'}


The learning objective is

$$
\max_\phi\;\log p_\phi(w_t\mid h_t,c_t).
$$

This is data-driven rather than task-driven. The model is never told to earn reward or keep a
body upright. It only learns the conditional distribution present in the clips.


# Section 4: Compress motion into causal tokens

Predicting all 60 numbers frame-by-frame is cumbersome. A causal convolutional autoencoder
compresses every four frames into one 16-dimensional token:

```text
64 frames × 60 features ──► 16 tokens × 16 values ──► 64 reconstructed frames
```

**Causal** means a token can use the present and past, never a future frame. That rule will
matter when the same encoder runs online inside a controller.



```python
class MotionAutoencoder(nn.Module):
    def __init__(self, feature_dim=60, hidden=192, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(feature_dim, hidden, 5),
            nn.SiLU(),
            CausalConv1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConv1d(hidden, latent_dim, 4, stride=2),
        )
        self.decoder = nn.Sequential(
            CausalConv1d(latent_dim, hidden, 3),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, hidden, 4, stride=2),
            nn.SiLU(),
            CausalConvTranspose1d(hidden, feature_dim, 4, stride=2),
        )

    def encode(self, features):
        return self.encoder(features.transpose(1, 2)).transpose(1, 2)

    def decode(self, tokens):
        return self.decoder(tokens.transpose(1, 2)).transpose(1, 2)

    def forward(self, features):
        return self.decode(self.encode(features))
```


```python
tokenizer = MotionAutoencoder()
frames = torch.randn(8, 64, 60)
tokens = tokenizer.encode(frames)
reconstruction = tokenizer.decode(tokens)

print(f"frames         : {tuple(frames.shape)}")
print(f"motion tokens  : {tuple(tokens.shape)}")
print(f"reconstruction : {tuple(reconstruction.shape)}")
print(f"parameters     : {parameter_count(tokenizer):,}")
```

    frames         : (8, 64, 60)
    motion tokens  : (8, 16, 16)
    reconstruction : (8, 64, 60)
    parameters     : 420,940


The bottleneck is not the final goal; it gives us a compact alphabet for motion. The next model
can now reason over 16-dimensional tokens instead of repeatedly predicting 60 raw features.


# Section 5: Predict the next token conditionally

A small Transformer reads four history tokens. A separate MLP embeds the three-number command.
Their representations meet only at the prediction head:

```text
four past tokens ──► Transformer ──┐
                                   ├──► next motion token
hindsight command ──► command MLP ─┘
```

Keeping the command path explicit lets us test the central claim: the *same* history should
lead to a different predicted future when the requested displacement changes.



```python
class ConditionalTransformer(nn.Module):
    def __init__(self, latent_dim=16, future_tokens=1, width=192, layers=4, heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.future_tokens = future_tokens
        self.input = nn.Linear(latent_dim, width)
        block = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.command = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64))
        self.output = nn.Sequential(
            nn.Linear(width + 64, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, future_tokens * latent_dim),
        )

    def context(self, history):
        hidden = self.input(history)
        hidden = (
            hidden
            + sinusoidal_positions(hidden.shape[1], hidden.shape[2], hidden.device)[
                None
            ]
        )
        return self.norm(self.transformer(hidden))[:, -1]

    def predict(self, history, command):
        hidden = torch.cat((self.context(history), self.command(command)), dim=-1)
        return self.output(hidden).view(-1, self.future_tokens, self.latent_dim)

    def log_prob(self, history, future, command, sigma):
        sigma = torch.as_tensor(sigma, dtype=history.dtype, device=history.device)
        residual = (future - self.predict(history, command)) / sigma
        return -0.5 * (
            residual.square() + 2 * sigma.log() + math.log(2 * math.pi)
        ).mean(dim=(-1, -2))
```


```python
predictor = ConditionalTransformer()
history = tokens[:, :4]
command = torch.zeros(8, 3)
next_token = predictor.predict(history, command)
likelihood = predictor.log_prob(history, next_token, command, sigma=0.1)

print(f"history       : {tuple(history.shape)}")
print(f"command       : {tuple(command.shape)}")
print(f"next token    : {tuple(next_token.shape)}")
print(f"log likelihood: {tuple(likelihood.shape)}")
print(f"parameters    : {parameter_count(predictor):,}")
```

    history       : (8, 4, 16)
    command       : (8, 3)
    next token    : (8, 1, 16)
    log likelihood: (8,)
    parameters    : 1,892,368


### What does the Gaussian likelihood mean?

The predictor outputs a mean $\mu_\phi(h_t,c_t)$. With a fixed standard deviation $\sigma$,

$$
\log p_\phi(w_t\mid h_t,c_t)
= -\frac{1}{2}\left[
\frac{\lVert w_t-\mu_\phi(h_t,c_t)\rVert^2}{\sigma^2}
+2\log\sigma+\text{constant}
\right].
$$

Higher likelihood therefore means smaller normalized token error. A useful validation is to
pair a future with several commands: its own command should score highest. Likelihood is a
learned data score, not a complete definition of physical or biological realism.


### Train and inspect Part 2

Training first reconstructs motion tokens, freezes the tokenizer, and then trains one-step
prediction while unrolling four predictions during training. Those loops stay in Section 0.

The video is rendered at the true 50 Hz timing of the 1.75×-retimed data. Its four rodent commands
(0.10–0.25 m/s) correspond to roughly **1.22–3.05 Fetch units/s**. We print both units below so a
playback-rate problem cannot be mistaken for a command-tracking problem.



```python
part2_result = run_part2(full=FULL_TRAINING)
show_report(
    part2_result["evaluation"],
    ["command_win_rate", "tracking_mae_mps", "prediction_skill_over_persistence"],
)
show_generation_speeds(part2_result["generation"])
show_video(part2_result["video"])
```

Part 2 now shows the motion at the same temporal scale that seeds Part 3. It can generate
command-dependent, rodent-derived Fetch motion, but it still moves a kinematic skeleton. It has
not proved that actuators can produce the motion while gravity and contacts push back.


# Part 3: A generative motor prior plus RL

Part 3 joins the two halves. We extend self-supervision from future **states** to future
**states and actions**, then let PPO adapt that frozen distribution to the locomotion task.

This is the brain–body–environment picture:

```text
generative prior: state + goal ──► intended motion + motor action
physics:          motor action ──► next body state
RL:               next state ──► task reward ──► policy update
```


# Section 6: Predict body state, then action

First replay every retargeted clip with a bounded feedback controller in the exact Fetch
simulator. Store what physics actually realizes as $s_t$, and the normalized actuator control
as $a_t$.

These controls are **physics-derived pseudo-labels**. They are not measured animal torques.
Data construction belongs to Section 0; the model is the important part:

$$
p_\theta(s_{t+1:t+H},a_{t:t+H-1}\mid s_{\leq t},a_{<t},g)
$$

is factorized as

$$
\underbrace{p_\theta(s_{t+1}\mid s_{\leq t},g)}_{\text{what body motion comes next?}}
\quad
\underbrace{p_\theta(a_t\mid s_t,\hat s_{t+1},a_{t-1},g)}_{\text{what action could produce it?}}.
$$

In this narrow world–action model, the “world” being generated is the body.



```python
action_data = describe_action_data()
```


    {'clips': 1784,
     'realized body states': (1784, 64, 60),
     'bounded controls': (1784, 63, 10),
     'one action': '10 actuator controls'}


The state model reuses Part 2's tokenizer and conditional Transformer. We add one small action
decoder. It sees the current body feature, the predicted motion token, the previous control,
a four-phase clock, and the goal.

Predicting a **correction to the previous action** gives the decoder an easy default: motor
commands are usually locally continuous.



```python
class FeedbackActionDecoder(nn.Module):
    def __init__(self, feature_dim=60, latent_dim=16, action_dim=10, hidden=192):
        super().__init__()
        input_dim = feature_dim + latent_dim + action_dim + 4 + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def forward(self, feature, plan, previous_control, phase, command):
        previous_control = previous_control.clamp(-0.98, 0.98)
        values = torch.cat((feature, plan, previous_control, phase, command), dim=-1)
        return torch.atanh(previous_control) + self.network(values)

    def distribution(self, feature, plan, previous_control, phase, command):
        mean = self(feature, plan, previous_control, phase, command)
        return mean, self.log_std.clamp(-5.0, 1.0)
```


```python
class BodyActionModel(nn.Module):
    def __init__(self, state_predictor, action_decoder):
        super().__init__()
        self.state_predictor = state_predictor
        self.action_decoder = action_decoder

    def forward(self, history, feature, previous_control, phase, command):
        next_plan = self.state_predictor.predict(history, command)[:, 0]
        action_mean, action_log_std = self.action_decoder.distribution(
            feature, next_plan, previous_control, phase, command
        )
        return next_plan, action_mean, action_log_std


action_decoder = FeedbackActionDecoder()
body_action_model = BodyActionModel(predictor, action_decoder)

feature = torch.randn(8, 60)
previous_control = torch.zeros(8, 10)
phase = F.one_hot(torch.arange(8) % 4, 4).float()
next_plan, action_mean, action_log_std = body_action_model(
    history, feature, previous_control, phase, command
)

print(f"predicted next state token : {tuple(next_plan.shape)}")
print(f"action mean                : {tuple(action_mean.shape)}")
print(f"shared action log-std      : {tuple(action_log_std.shape)}")
```

    predicted next state token : (8, 16)
    action mean                : (8, 10)
    shared action log-std      : (10,)


Pretraining has two likelihood terms:

$$
\mathcal L_{\mathrm{prior}}
=-\log p_\theta(s_{t+1}\mid h_t,g)
-\log p_\theta(a_t\mid s_t,\hat s_{t+1},a_{t-1},g).
$$

The second term is trained with both true and predicted plans, and both true and predicted
previous controls. That exposure matters: online, the model must consume its own imperfect
outputs rather than a perfect demonstration history.


### Pretrain and test the body–action prior

The offline gates ask whether state prediction beats persistence, the command and predicted
plan are informative, action prediction beats trivial controls, and a closed-loop action
rollout beats repeating the first control.



```python
part3_prior = pretrain_part3(full=FULL_TRAINING)
show_report(
    part3_prior["evaluation"],
    [
        "state_skill_over_persistence",
        "action_skill_over_previous",
        "passes_offline_gate",
    ],
)
```

# Section 7: Let PPO move, but charge it for leaving the prior

Freeze the whole body–action prior. PPO trains only a small residual network around its action
distribution. Its 194-number context contains the simulator observation, latest body feature,
predicted plan, action phase, goal, and prior action mean. The residual's last layer starts at
zero, so the initial policy is **exactly the pretrained prior**, not merely a randomly
initialized network with the same architecture.



```python
class ResidualPolicy(nn.Module):
    def __init__(self, context_dim, action_dim=10, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * action_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, prior_mean, prior_scale_logit):
        delta_mean, delta_scale = self.network(context).chunk(2, dim=-1)
        mean = prior_mean + 2.0 * torch.tanh(delta_mean)
        scale_logit = prior_scale_logit + torch.tanh(delta_scale)
        return mean, scale_logit


frozen_prior = body_action_model.requires_grad_(False)
residual = ResidualPolicy(context_dim=194)
context = torch.randn(8, 194)
prior_mean = torch.randn(8, 10)
prior_scale_logit = torch.randn(8, 10)
initial_mean, initial_scale = residual(context, prior_mean, prior_scale_logit)

print("starts at prior mean :", torch.allclose(initial_mean, prior_mean))
print("starts at prior scale:", torch.allclose(initial_scale, prior_scale_logit))
print(f"trainable residual parameters: {parameter_count(residual):,}")
```

    starts at prior mean : True
    starts at prior scale: True
    trainable residual parameters: 44,052


The task reward still asks Fetch to move at the commanded speed. A second term keeps the new
policy close to the frozen action distribution:

$$
J(\psi)=\mathbb E\sum_t\gamma^t\left[
r_t^{\mathrm{task}}
-\frac{\beta}{10}
D_{\mathrm{KL}}\!\left(
\pi_\psi(\cdot\mid h_t,g_t)\;\Vert\;
p_{\theta_0}(\cdot\mid h_t,g_t)
\right)
\right].
$$

Why does the code use prior log-likelihood plus policy entropy? Because

$$
\mathbb E_{a\sim\pi}[\log p_{\theta_0}(a)] + \mathcal H(\pi)
=-D_{\mathrm{KL}}(\pi\Vert p_{\theta_0}).
$$

The single visible knob is $\beta$:

- $\beta=0$: solve only the task;
- $\beta=0.10$: the accepted workshop balance;
- larger $\beta$: stay closer to the data prior, with less freedom to repair it.


### Fine-tune with RL and inspect a speed sweep

The prior remains frozen. PPO updates the residual and value network while commands vary from
1.5 to 4.0 Fetch units/s.



```python
part3_result = run_part3(beta=0.10, full=FULL_TRAINING)
show_report(part3_result["metrics"], ["training", "steps", "speeds"])
show_video(part3_result["video"])
```

Four-limb contact, stride, speed, uprightness, and action statistics are **validation only**.
They are not extra gait rewards. This preserves the scientific point: naturalness pressure
comes from the learned distribution rather than a hand-engineered checklist.

Also keep the limitation visible: the prior inherits artifacts from retargeted motion and is
not robust on every long rollout. RL can repair task failures, but it cannot turn low-quality
demonstrations into a perfect biological model.


---
# Recap

| part | information source | model | objective | realism |
|---|---|---|---|---|
| 1 · PPO locomotion | environment interaction | 4×32 policy MLP | maximize task return | functional |
| 2 · conditional motion | retargeted Coltrane locomotion | causal tokenizer + conditional Transformer | maximize future-motion likelihood | distributional |
| 3 · motor prior + PPO | states, controls, and environment | state predictor + action decoder + residual policy | task return minus prior KL | both |

Three ideas worth carrying out of the room:

1. **RL is task-driven.** Reward says what must work in the environment.
2. **SSL is data-driven.** The future supplies its own target and hindsight command.
3. **Pretraining and post-training play different roles.** The prior proposes data-like motor
   behavior; RL changes only what the task and physics require.

This is the full progression from body–environment interaction, through learned
brain–body statistics, to a brain–body–environment loop.

Further reading: [`workshop/README.md`](workshop/README.md). The reusable implementations live
under [`workshop/part1`](workshop/part1), [`workshop/part2`](workshop/part2), and
[`workshop/part3`](workshop/part3).
