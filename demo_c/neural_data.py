"""Pose/neural alignment and representation extraction for Demo C.

All arrays remain indexed on the genuine 80-ms token grid of one recording. Nothing
concatenates disjoint locomotion bouts or unit identities across sessions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from canvas.prepare import motion_features, quat2yaw
from demo_c.config import TASK, TRAIN_SEEDS
from demo_c.motor import CLIP, FPS, H, FrozenMotor
from demo_c.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ALDARONDO_ROOT", "/workspace/data/Aldarondo2024"))
CACHE = Path(__file__).resolve().parent / "out" / "neural_cache"
REGION = {"art": "DLS", "bud": "DLS", "coltrane": "DLS", "duke": "MC", "freddie": "MC", "gerry": "MC"}
DEFAULT_SESSIONS = (
    "coltrane/2021_08_05_1",  # outside Demo B's first-eight-session motion-training scope
    "coltrane/2021_08_06_1",
    "freddie/2022_05_16_1",  # different animal and MC region
    "freddie/2022_05_17_1",
)
LOCOMOTION = {"Amble", "Walk", "WalkFast"}
GOAL_TOKENS = 25  # 2.0-s future displacement acts as a recorded pseudo-goal
PREVIOUS_TOKENS = 8  # matches one high-level 0.64-s action
CACHE_VERSION = 4


def session_path(name: str) -> Path:
    path = DATA_ROOT / f"{name}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@torch.inference_mode()
def encode_continuous(motor: FrozenMotor, features: np.ndarray, chunk_frames: int = 32768):
    """Causally encode a continuous recording with overlap, preserving token indices."""
    n_tokens = len(features) // 4
    output = np.empty((n_tokens, 16), np.float32)
    chunk_tokens = chunk_frames // 4
    overlap_frames = CLIP
    for token_start in range(0, n_tokens, chunk_tokens):
        token_end = min(n_tokens, token_start + chunk_tokens)
        frame_start = token_start * 4
        read_start = max(0, frame_start - overlap_frames)
        read_end = token_end * 4
        x = torch.as_tensor(features[read_start:read_end], device=motor.device)[None]
        x = (x - motor.norms["mmean"]) / motor.norms["mstd"]
        mu = motor.motion.encode(x)[0][0]
        global_first = read_start // 4
        lo, hi = token_start - global_first, token_end - global_first
        output[token_start:token_end] = mu[lo:hi].float().cpu().numpy()
    return output


def _root_local_delta(xy0, yaw0, xy1, yaw1):
    delta = xy1 - xy0
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    local = np.stack((c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1]), -1)
    turn = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    return local, turn


def task_pseudo_goal(future_local):
    """Put food along the recorded future direction at an in-task distance.

    A rat typically travels far less than the task's 0.35--0.75 m food radius in
    two seconds. Feeding that raw displacement to a policy would probe it almost
    entirely outside its training distribution and unfairly collapse the goal-only
    baseline. Direction is the honest recorded quantity; distance is fixed at the
    midpoint of the frozen task range and bearing is clipped to its forward field.
    """
    bearing = np.arctan2(future_local[:, 1], future_local[:, 0])
    bearing = np.clip(bearing, -TASK.goal_bearing_max, TASK.goal_bearing_max)
    radius = 0.5 * (TASK.goal_radius_min + TASK.goal_radius_max)
    local = np.stack((np.cos(bearing), np.sin(bearing)), -1) * radius
    distance = np.full((len(local), 1), radius, np.float32)
    return local.astype(np.float32), distance


@torch.inference_mode()
def extract_representations(
    motor: FrozenMotor,
    latent: np.ndarray,
    xy: np.ndarray,
    yaw: np.ndarray,
    checkpoints: list[Path],
):
    latent_n = (latent - motor.norms["zmean"].cpu().numpy()) / motor.norms["zstd"].cpu().numpy()
    indices = np.arange(max(H - 1, PREVIOUS_TOKENS), len(latent) - GOAL_TOKENS, dtype=np.int64)
    previous = indices - PREVIOUS_TOKENS
    future = indices + GOAL_TOKENS

    prior_local, prior_turn = _root_local_delta(xy[previous], yaw[previous], xy[indices], yaw[indices])
    body_velocity = np.concatenate((prior_local, prior_turn[:, None]), -1) / TASK.step_seconds
    command = np.concatenate((prior_local, prior_turn[:, None]), -1).astype(np.float32)
    previous_action = motor.command_to_action(torch.as_tensor(command, device=motor.device)).cpu().numpy()
    recorded_future_local, _ = _root_local_delta(
        xy[indices], yaw[indices], xy[future], yaw[future]
    )
    goal_local, goal_distance = task_pseudo_goal(recorded_future_local)
    velocity = np.clip(body_velocity / np.array([0.35, 0.20, 1.2]), -2, 2)
    base = np.concatenate(
        (goal_local / TASK.goal_radius_max, goal_distance / TASK.goal_radius_max, velocity, previous_action), -1
    ).astype(np.float32)

    contexts = np.empty((len(indices), 192), np.float32)
    batch = 4096
    latent_t = torch.as_tensor(latent_n, device=motor.device)
    for begin in range(0, len(indices), batch):
        chosen = indices[begin:begin + batch]
        gather = chosen[:, None] - np.arange(H - 1, -1, -1)[None]
        history = latent_t[torch.as_tensor(gather, device=motor.device)]
        contexts[begin:begin + len(chosen)] = motor.context(history).cpu().numpy()

    representations = {
        "motion_latent": latent[indices].astype(np.float32),
        "predictive_context": contexts,
    }
    base_t = torch.as_tensor(base, device=motor.device)
    context_t = torch.as_tensor(contexts, device=motor.device)
    for checkpoint in checkpoints:
        policy, payload = load_policy(checkpoint, motor.device)
        obs = torch.cat((base_t, context_t), -1) if payload["variant"] == "wam" else base_t
        name = f"{payload['variant']}_policy_s{payload['seed']}"
        output = np.empty((len(indices), 128), np.float32)
        for begin in range(0, len(indices), batch):
            output[begin:begin + batch] = policy.features(obs[begin:begin + batch]).cpu().numpy()
        representations[name] = output
    return indices, base, latent_n[indices].astype(np.float32), representations


def build_cache(
    name: str,
    motor: FrozenMotor,
    checkpoints: list[Path],
    max_frames: int | None = None,
    rebuild: bool = False,
):
    """Build one atomic per-session cache (full continuous session by default)."""
    suffix = f"_n{max_frames}" if max_frames else ""
    target = CACHE / f"v{CACHE_VERSION}_{name.replace('/', '_')}{suffix}.npz"
    if target.exists() and not rebuild:
        return target
    path = session_path(name); CACHE.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as file:
        total = len(file["/pose/qpos"]); stop = min(total, max_frames) if max_frames else total
        qpos = file["/pose/qpos"][:stop].astype(np.float32)
        keypoints = file["/pose/keypoints"][:stop].astype(np.float32)
        spike_ds = file["/ephys/spike_counts"]
        active = np.asarray(spike_ds.attrs["active_units"], bool)
        spikes_frame = spike_ds[:stop, active].astype(np.uint16)
        behavior_ds = file["/behavior/motion_mapper"]
        behavior = behavior_ds[:stop]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in behavior_ds.attrs["names"]]
    if active.sum() < 5:
        raise ValueError(f"{name} has only {active.sum()} active units")

    print(f"[{name}] motion features: {stop:,} frames", flush=True)
    features = motion_features(qpos, keypoints)
    print(f"[{name}] causal encoding", flush=True)
    latent = encode_continuous(motor, features)
    n_tokens = len(latent); anchor = np.arange(n_tokens) * 4 + 3
    xy = qpos[anchor, :2].astype(np.float32); yaw = quat2yaw(qpos[anchor, 3:7]).astype(np.float32)
    spikes = spikes_frame[:n_tokens * 4].reshape(n_tokens, 4, -1).sum(1).astype(np.uint16)
    loco_frame = np.zeros(stop, bool)
    for label_id, label_name in enumerate(names):
        if label_name in LOCOMOTION:
            loco_frame |= behavior == label_id
    locomotion = loco_frame[:n_tokens * 4].reshape(n_tokens, 4).sum(1) >= 2
    speed = np.linalg.norm(features[anchor, :2], axis=-1).astype(np.float32)
    turn_rate = np.zeros(n_tokens, np.float32)
    turn_rate[1:] = ((np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi) * (FPS / 4)

    indices, base, latent_n, representations = extract_representations(motor, latent, xy, yaw, checkpoints)
    animal = name.split("/")[0]
    metadata = {
        "version": CACHE_VERSION,
        "session": name,
        "animal": animal,
        "region": REGION[animal],
        "source": str(path),
        "frames": stop,
        "tokens": len(indices),
        "active_units": int(active.sum()),
        "token_seconds": 4 / FPS,
        "goal_tokens": GOAL_TOKENS,
        "pseudo_goal": "recorded 2-s future bearing; clipped to task field; fixed midpoint radius",
        "checkpoints": [str(p.relative_to(ROOT)) for p in checkpoints],
    }
    arrays = {
        "metadata": np.array(json.dumps(metadata)),
        "token_index": indices.astype(np.int32),
        "xy": xy[indices],
        "yaw": yaw[indices],
        "spikes": spikes[indices],
        "locomotion": locomotion[indices],
        "speed": speed[indices],
        "turn_rate": turn_rate[indices],
        "base_observation": base.astype(np.float16),
        "kinematics": features[anchor[indices]].astype(np.float16),
        "latent_normalized": latent_n.astype(np.float16),
    }
    arrays.update({key: value.astype(np.float16) for key, value in representations.items()})
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(temporary, target)
    print(f"[{name}] cached {target} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return target


def default_checkpoints():
    return [
        ROOT / "demo_c" / "out" / "checkpoints" / f"{variant}_seed{seed}.pt"
        for variant in ("goal_only", "wam") for seed in TRAIN_SEEDS
    ]


def load_cache(path: Path):
    file = np.load(path, allow_pickle=False)
    return {key: file[key] for key in file.files}
