"""Broaden Demo B's action-conditioned transition without changing its architecture.

The frozen tokenizer constructs its own SSL targets from real 64-frame clips. Complete
sessions are assigned to train/validation; the four neural-evaluation sessions remain
untouched. Output is a drop-in ``FrozenMotor`` checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from canvas.prepare import motion_features, quat2yaw
from demo_c.motor import CLIP, H, K, FrozenMotor

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ALDARONDO_ROOT", "/workspace/data/Aldarondo2024"))
OUT = Path(__file__).resolve().parent / "out" / "world"
DATA_CACHE = OUT / "data"
DEFAULT_ASSET = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
TRAIN_SESSIONS = (
    "art/2020_12_22_1", "art/2020_12_22_2",
    "bud/2021_06_21_1", "bud/2021_06_23_1",
    "coltrane/2021_07_28_1", "coltrane/2021_07_29_1",
    "duke/2022_02_16_1", "duke/2022_02_17_1",
    "freddie/2022_05_19_1", "freddie/2022_05_20_1",
    "gerry/2022_05_30_1", "gerry/2022_05_31_1",
)
VALIDATION_SESSIONS = ("coltrane/2021_08_07_1", "freddie/2022_05_21_1")
MAX_CROPS_PER_SESSION = 1024
DATA_VERSION = 3
LOCOMOTION = {"Amble", "Walk", "WalkFast"}


def _check_sessions():
    missing = [name for name in TRAIN_SESSIONS + VALIDATION_SESSIONS if not (DATA_ROOT / f"{name}.h5").exists()]
    if missing:
        raise FileNotFoundError(f"frozen world-data sessions missing: {missing}")


def _command(xy, yaw, starts):
    f0 = starts + 32; f1 = starts + 63
    delta = xy[f1] - xy[f0]; c, s = np.cos(-yaw[f0]), np.sin(-yaw[f0])
    local = np.stack((c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1]), -1)
    turn = (yaw[f1] - yaw[f0] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate((local, turn[:, None]), -1).astype(np.float32)


def locomotion_starts(qpos, behavior, behavior_names, *, max_crops=None, seed_name=""):
    """Select real contiguous clips with the frozen world-data rule.

    Motion-mapper labels flicker at frame scale, so requiring an entirely labelled
    locomotion bout would throw away nearly all useful transitions. We instead require
    some locomotion evidence plus measured planar motion, without ever stitching clips.
    """
    loco = np.zeros(len(qpos), bool)
    for label_id, label_name in enumerate(behavior_names):
        if label_name in LOCOMOTION:
            loco |= behavior == label_id
    starts = np.arange(0, len(qpos) - CLIP + 1, CLIP // 2, dtype=np.int64)
    cumulative = np.concatenate(([0], np.cumsum(loco, dtype=np.int64)))
    fraction = (cumulative[starts + CLIP] - cumulative[starts]) / CLIP
    frame_speed = np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=-1) * 50
    speed_sum = np.concatenate(([0.0], np.cumsum(frame_speed, dtype=np.float64)))
    mean_speed = (speed_sum[starts + CLIP - 1] - speed_sum[starts]) / (CLIP - 1)
    starts = starts[(fraction >= 0.10) & (mean_speed >= 0.08)]
    if max_crops is not None and len(starts) > max_crops:
        seed = zlib.crc32(seed_name.encode())
        starts = np.sort(
            np.random.default_rng(seed).choice(starts, max_crops, replace=False)
        )
    return starts


@torch.inference_mode()
def extract_session(name, motor, rebuild=False):
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    target = DATA_CACHE / f"v{DATA_VERSION}_{name.replace('/', '_')}.npz"
    if target.exists() and not rebuild:
        file = np.load(target); return file["history"], file["future"], file["command"]
    path = DATA_ROOT / f"{name}.h5"
    with h5py.File(path, "r") as file:
        qpos = file["/pose/qpos"][:].astype(np.float32)
        kp_ds = file["/pose/keypoints"]
        keypoints = kp_ds[:].astype(np.float32)
        behavior_ds = file["/behavior/motion_mapper"]
        behavior = behavior_ds[:]
        behavior_names = [x.decode() if isinstance(x, bytes) else str(x) for x in behavior_ds.attrs["names"]]
    starts = locomotion_starts(
        qpos,
        behavior,
        behavior_names,
        max_crops=MAX_CROPS_PER_SESSION,
        seed_name=name,
    )
    if not len(starts):
        raise ValueError(f"{name}: no locomotion crops after the frozen selection rule")
    features = motion_features(qpos, keypoints); yaw = quat2yaw(qpos[:, 3:7]); xy = qpos[:, :2]
    histories, futures = [], []; offsets = np.arange(CLIP)[None]
    for begin in range(0, len(starts), 512):
        chosen = starts[begin:begin + 512]
        clip = torch.as_tensor(features[chosen[:, None] + offsets], device=motor.device)
        clip = (clip - motor.norms["mmean"]) / motor.norms["mstd"]
        latent = motor.motion.encode(clip)[0]
        latent = (latent - motor.norms["zmean"]) / motor.norms["zstd"]
        histories.append(latent[:, :H].cpu().numpy()); futures.append(latent[:, H:H + K].cpu().numpy())
    history = np.concatenate(histories).astype(np.float32); future = np.concatenate(futures).astype(np.float32)
    command = _command(xy, yaw, starts)
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, history=history.astype(np.float16), future=future.astype(np.float16), command=command)
    os.replace(temporary, target)
    print(f"{name}: {len(history)} real contiguous locomotion crops", flush=True)
    return history, future, command


def build_dataset(motor, sessions, rebuild=False):
    groups = [extract_session(name, motor, rebuild) for name in sessions]
    return tuple(np.concatenate([group[i] for group in groups]).astype(np.float32) for i in range(3))


@torch.inference_mode()
def score(model, history, future, command, motor, batch=1024):
    losses, counts = 0.0, 0
    for begin in range(0, len(history), batch):
        h = torch.as_tensor(history[begin:begin + batch], device=motor.device)
        f = torch.as_tensor(future[begin:begin + batch], device=motor.device)
        c = torch.as_tensor(command[begin:begin + batch], device=motor.device)
        c = (c - motor.norms["cmean"]) / motor.norms["cstd"]
        prediction = model.predict(h, c)
        losses += float(F.mse_loss(prediction, f, reduction="sum")); counts += f.numel()
    mse = losses / counts
    persistence = float(np.mean((np.repeat(history[:, -1:, :], K, axis=1) - future) ** 2))
    return {"mse": mse, "persistence_mse": persistence, "skill": 1 - mse / persistence}


def provenance():
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    return {"git_commit": run("git", "rev-parse", "HEAD"), "git_dirty": bool(run("git", "status", "--porcelain"))}


def train(steps=8000, seed=0, rebuild=False, tag=""):
    _check_sessions(); torch.manual_seed(seed); np.random.seed(seed)
    motor = FrozenMotor("cuda" if torch.cuda.is_available() else "cpu", DEFAULT_ASSET)
    print("building frozen session-level SSL dataset", flush=True)
    train_h, train_f, train_c = build_dataset(motor, TRAIN_SESSIONS, rebuild)
    val_h, val_f, val_c = build_dataset(motor, VALIDATION_SESSIONS, rebuild)
    from models import SimpleTrans  # Demo B standalone module, made importable by demo_c.motor
    model = SimpleTrans(d=192, layers=6, heads=4).to(motor.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_h_t = torch.as_tensor(train_h, device=motor.device)
    train_f_t = torch.as_tensor(train_f, device=motor.device)
    train_c_t = torch.as_tensor(train_c, device=motor.device)
    train_c_t = (train_c_t - motor.norms["cmean"]) / motor.norms["cstd"]
    generator = torch.Generator(device=motor.device).manual_seed(seed)
    curve, best, best_step, best_state = [], -float("inf"), None, None; start = time.perf_counter()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        idx = torch.randint(len(train_h_t), (256,), generator=generator, device=motor.device)
        model.train(); loss = model.loss(train_h_t[idx], train_f_t[idx], train_c_t[idx])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 0 or (step + 1) % 500 == 0 or step + 1 == steps:
            model.eval(); val = score(model, val_h, val_f, val_c, motor)
            item = {"step": step + 1, "train_mse": float(loss.detach()), "grad_norm": float(grad_norm), **val}
            curve.append(item)
            print(
                f"step {step + 1:5d}: train={item['train_mse']:.4f} val={val['mse']:.4f} "
                f"persistence={val['persistence_mse']:.4f} skill={val['skill']:+.1%}", flush=True,
            )
            if val["skill"] > best:
                best = val["skill"]
                best_step = step + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = torch.load(DEFAULT_ASSET, map_location="cpu", weights_only=False)
    payload["trans"] = best_state
    payload["model_cfg"] = {"d": 192, "layers": 6, "heads": 4}
    payload["world_training"] = {
        "train_sessions": TRAIN_SESSIONS, "validation_sessions": VALIDATION_SESSIONS,
        "train_crops": len(train_h), "validation_crops": len(val_h), "steps": steps,
        "seed": seed, "best_validation_step": best_step,
        "best_validation_skill": best, "provenance": provenance(),
    }
    safe_tag = "".join(c for c in tag if c.isalnum() or c in ("-", "_"))
    stem = f"world_seed{seed}" + (f"_{safe_tag}" if safe_tag else "")
    OUT.mkdir(parents=True, exist_ok=True); checkpoint = OUT / f"{stem}.pt"
    torch.save(payload, checkpoint)
    metrics = {
        "best_validation_step": best_step, "best_validation_skill": best,
        "curve": curve, "train_crops": len(train_h),
        "validation_crops": len(val_h), "elapsed_seconds": time.perf_counter() - start,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }
    (OUT / f"{stem}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2)); return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train(
        20 if args.smoke else args.steps,
        args.seed,
        args.rebuild,
        tag="smoke" if args.smoke else (f"steps{args.steps}" if args.steps != 8000 else ""),
    )


if __name__ == "__main__":
    main()
