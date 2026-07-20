"""Research-only trainer for the rejected compact 85-D representation.

Every target is a future portion of the same continuous recording.  Sessions,
not overlapping crops, are assigned to train/validation/test.  The transition's
MSE is calibrated as a fixed-variance Gaussian likelihood for Demo E.

Run from the repository root:

The workshop asset is restored with ``python -m demo_b.promote_coltrane``.
This file remains for controlled historical comparisons and refuses to
overwrite the canonical asset without an explicit flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import DEV, DM, FM, FPS, H, K, SL
from .dataset import (
    MAX_CROPS_PER_SESSION,
    MIN_LOCOMOTION_FRACTION,
    MIN_PLANAR_SPEED,
    CropSet,
    load_crop_set,
    session_path,
)
from .features import motion_features, quat_to_yaw
from .models import MotionVAE, SimpleTrans
from .splits import (
    ALL_SESSIONS,
    ANIMAL,
    TEST_SESSIONS,
    TRAIN_SESSIONS,
    VAL_SESSIONS,
    validate_split,
)


ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
METRICS = ROOT / "demo_b" / "assets" / "motor_metrics.json"
WEIGHTS = {"vxy": 5.0, "h": 2.0, "d6": 5.0, "q": 1.0, "qd": 1.0}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> tuple[list[dict], str]:
    rows = []
    combined = hashlib.sha256()
    for session in ALL_SESSIONS:
        path = session_path(session)
        digest = sha256(path)
        row = {
            "animal": ANIMAL,
            "session": session,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        rows.append(row)
        combined.update(json.dumps(row, sort_keys=True).encode())
    return rows, combined.hexdigest()


def motion_loss(reconstruction, target, mu, logvar):
    loss = target.new_zeros(())
    for key, (lo, hi) in SL.items():
        loss = loss + WEIGHTS[key] * F.smooth_l1_loss(
            reconstruction[..., lo:hi], target[..., lo:hi]
        )
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return loss + 1e-6 * kl, kl


def normalize_crops(crops: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((crops - mean) / std).to(DEV)


def train_tokenizer(features: torch.Tensor, *, steps: int, batch: int, hidden: int):
    model = MotionVAE(fm=FM, hid=hidden, dm=DM).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ema = [parameter.detach().clone() for parameter in model.parameters()]
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(len(features), (batch,), device=DEV)
        target = features[index]
        reconstructed, mu, logvar = model(target)
        loss, kl = motion_loss(reconstructed, target, mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(ema, model.parameters(), strict=True):
                average.mul_(0.99).add_(parameter, alpha=0.01)
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            print(
                f"[tokenizer] {step + 1:5d}/{steps} loss={loss.item():.4f} "
                f"kl={kl.item():.4f}",
                flush=True,
            )
    with torch.no_grad():
        for average, parameter in zip(ema, model.parameters(), strict=True):
            parameter.copy_(average)
    model.eval()
    return model, time.perf_counter() - started


@torch.inference_mode()
def encode_split(model, normalized: torch.Tensor, raw_command: np.ndarray):
    output = []
    for offset in range(0, len(normalized), 512):
        output.append(model.encode(normalized[offset : offset + 512])[0].float().cpu())
    return torch.cat(output), torch.from_numpy(raw_command)


def train_transition(train, validation, *, steps: int, batch: int, cfg: dict):
    model = SimpleTrans(**cfg).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    history, future, command = (value.to(DEV) for value in train)
    val_history, val_future, val_command = (value.to(DEV) for value in validation)
    best, best_val = None, math.inf
    started = time.perf_counter()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        index = torch.randint(len(history), (batch,), device=DEV)
        loss = model.loss(history[index], future[index], command[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            with torch.inference_mode():
                count = min(4096, len(val_history))
                value = F.mse_loss(
                    model.predict(val_history[:count], val_command[:count]),
                    val_future[:count],
                ).item()
            print(
                f"[transition] {step + 1:5d}/{steps} train={loss.item():.4f} "
                f"val={value:.4f}",
                flush=True,
            )
            if value < best_val:
                best_val = value
                best = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
    if best is None:
        raise AssertionError("transition produced no checkpoint")
    model.load_state_dict(best)
    model.eval()
    return model, best_val, time.perf_counter() - started


@torch.inference_mode()
def likelihood_metrics(model, split, sigma=None):
    history, future, command = (value.to(DEV) for value in split)
    prediction = model.predict(history, command)
    mse = F.mse_loss(prediction, future).item()
    first_residual = future[:, 0] - prediction[:, 0]
    if sigma is None:
        sigma = first_residual.square().mean().sqrt().clamp_min(1e-3)
    logp = model.log_prob_next(history, future[:, 0], command, sigma)
    persistence = F.mse_loss(history[:, -1], future[:, 0]).item()
    permutation = torch.randperm(len(future), device=DEV, generator=torch.Generator(device=DEV).manual_seed(13))
    shuffled = model.log_prob_next(history, future[permutation, 0], command, sigma)
    return {
        "mse": mse,
        "first_mse": first_residual.square().mean().item(),
        "persistence_first_mse": persistence,
        "skill_over_persistence": 1.0 - first_residual.square().mean().item() / persistence,
        "logp_mean": logp.mean().item(),
        "logp_shuffled_mean": shuffled.mean().item(),
        "logp_q01": torch.quantile(logp, 0.01).item(),
        "logp_q99": torch.quantile(logp, 0.99).item(),
        "sigma": float(sigma),
    }, sigma


def normalized_splits(encoded: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    train_latent, train_command = encoded["train"]
    zmean = train_latent.reshape(-1, DM).mean(0).numpy().astype(np.float32)
    zstd = (train_latent.reshape(-1, DM).std(0).numpy() + 1e-4).astype(np.float32)
    cmean = train_command.numpy().mean(0).astype(np.float32)
    cstd = (train_command.numpy().std(0) + 1e-4).astype(np.float32)
    output = {}
    for name, (latent, command) in encoded.items():
        output[name] = (
            (latent - torch.from_numpy(zmean))[:, :H] / torch.from_numpy(zstd),
            (latent - torch.from_numpy(zmean))[:, H : H + K] / torch.from_numpy(zstd),
            (command - torch.from_numpy(cmean)) / torch.from_numpy(cstd),
        )
    return output, zmean, zstd, cmean, cstd


def make_reset_bank(
    dataset: CropSet,
    normalized_features: torch.Tensor,
    split,
    *,
    limit: int = 512,
):
    keep = np.arange(len(dataset.features))
    if len(keep) > limit:
        keep = np.linspace(0, len(keep) - 1, limit, dtype=np.int64)
    history = split[0][keep].numpy()
    buffer = normalized_features[keep, 8:32].cpu().numpy()
    return {
        "history": history.astype(np.float32),
        "feature_buffer": buffer.astype(np.float32),
        "qpos": dataset.reset_qpos[keep].astype(np.float32),
        # Aldarondo qpos has no dynamically consistent qvel.  Demo E's reset
        # contract initializes physical velocity to zero explicitly.
        "qvel": np.zeros((len(keep), 73), np.float32),
        "clip_id": keep.astype(np.int32),
        "session_index": dataset.session_index[keep].astype(np.int16),
        "frame": (dataset.start[keep] + 31).astype(np.int32),
    }


def dataset_summary(dataset: CropSet) -> dict:
    duration = 31 / FPS
    velocity = dataset.command / np.asarray([duration, duration, duration])
    return {
        "sessions": list(dataset.sessions),
        "crops": int(len(dataset.features)),
        "forward_speed_mean": float(velocity[:, 0].mean()),
        "forward_speed_quantiles": np.quantile(velocity[:, 0], [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
        "planar_speed_mean": float(np.linalg.norm(velocity[:, :2], axis=-1).mean()),
        "session_rows": dataset.session_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--motion-steps", type=int, default=2500)
    parser.add_argument("--transition-steps", type=int, default=6000)
    parser.add_argument("--motion-hidden", type=int, default=128)
    parser.add_argument("--transition-width", type=int, default=192)
    parser.add_argument("--transition-layers", type=int, default=6)
    parser.add_argument("--max-crops-per-session", type=int, default=MAX_CROPS_PER_SESSION)
    parser.add_argument("--output", type=Path, default=DST)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--allow-canonical-overwrite",
        action="store_true",
        help="explicitly allow the rejected 85-D experiment to replace the workshop asset",
    )
    args = parser.parse_args()
    if args.output == DST and not args.allow_canonical_overwrite:
        raise SystemExit(
            "Refusing to overwrite the validated 281-D Coltrane asset with the "
            "rejected 85-D recipe. Use `python -m demo_b.promote_coltrane`, or "
            "choose a separate --output for a research comparison."
        )
    if args.smoke:
        args.motion_steps, args.transition_steps = 20, 20
        args.max_crops_per_session = min(args.max_crops_per_session, 64)
    validate_split()
    seed_everything(args.seed)
    if DEV != "cuda":
        print("WARNING: CUDA is unavailable; the workshop runtime target assumes a GPU", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    extracted = {
        "train": load_crop_set(TRAIN_SESSIONS, max_crops=args.max_crops_per_session),
        "val": load_crop_set(VAL_SESSIONS, max_crops=args.max_crops_per_session),
        "test": load_crop_set(TEST_SESSIONS, max_crops=args.max_crops_per_session),
    }
    train_frames = extracted["train"].features.reshape(-1, FM)
    mmean = train_frames.mean(0).astype(np.float32)
    mstd = (train_frames.std(0) + 1e-4).astype(np.float32)
    normalized = {
        name: normalize_crops(dataset.features, mmean, mstd)
        for name, dataset in extracted.items()
    }
    tokenizer, tokenizer_seconds = train_tokenizer(
        normalized["train"], steps=args.motion_steps, batch=128, hidden=args.motion_hidden
    )
    encoded = {
        name: encode_split(tokenizer, normalized[name], dataset.command)
        for name, dataset in extracted.items()
    }
    splits, zmean, zstd, cmean, cstd = normalized_splits(encoded)
    transition_cfg = {
        "d": args.transition_width,
        "layers": args.transition_layers,
        "heads": 4,
        "ff": args.transition_width * 4,
    }
    transition, best_val, transition_seconds = train_transition(
        splits["train"],
        splits["val"],
        steps=args.transition_steps,
        batch=256,
        cfg=transition_cfg,
    )
    metrics = {
        "best_val_full_mse": best_val,
        "dataset": {name: dataset_summary(value) for name, value in extracted.items()},
    }
    train_metrics, sigma = likelihood_metrics(transition, splits["train"])
    metrics["train"] = train_metrics
    metrics["val"] = likelihood_metrics(transition, splits["val"], sigma)[0]
    metrics["test"] = likelihood_metrics(transition, splits["test"], sigma)[0]
    logp_clip = np.asarray(
        [train_metrics["logp_q01"], train_metrics["logp_q99"]], np.float32
    )
    duration = 31 / FPS
    train_velocity = extracted["train"].command[:, [0, 2]] / duration
    command_support = np.quantile(train_velocity, [0.01, 0.99], axis=0).astype(np.float32)
    reset_banks = {
        name: make_reset_bank(extracted[name], normalized[name], splits[name])
        for name in ("train", "val", "test")
    }
    test_count = min(8192, len(splits["test"][0]))
    evaluation_bank = {
        "history": splits["test"][0][:test_count].numpy().astype(np.float32),
        "future": splits["test"][1][:test_count].numpy().astype(np.float32),
        "command_raw": extracted["test"].command[:test_count].astype(np.float32),
    }
    seed_qpos = extracted["train"].seed_qpos
    # Preserve the true finite difference at the crop boundary; recomputing
    # features on the sliced qpos would fabricate a zero-velocity first frame.
    seed_features = extracted["train"].features[0]
    print(f"hashing {len(ALL_SESSIONS)} {ANIMAL} source sessions for provenance", flush=True)
    manifest, source_digest = source_manifest()
    bundle = {
        "format_version": 3,
        "arch": "simple_gaussian",
        "feature_dim": FM,
        "animal": ANIMAL,
        "motion": {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()},
        "motion_cfg": {"fm": FM, "hid": args.motion_hidden, "dm": DM},
        "trans": {key: value.detach().cpu() for key, value in transition.state_dict().items()},
        "model_cfg": transition_cfg,
        "zmean": zmean,
        "zstd": zstd,
        "cmean": cmean,
        "cstd": cstd,
        "mmean": mmean,
        "mstd": mstd,
        "sigma": np.float32(sigma.cpu()),
        "logp_clip": logp_clip,
        "command_support_velocity": command_support,
        "reset_banks": reset_banks,
        "evaluation_bank": evaluation_bank,
        "seed_feat": seed_features,
        "seed_xy": seed_qpos[:, :2],
        "seed_yaw": quat_to_yaw(seed_qpos[:, 3:7]).astype(np.float32),
        "seed_name": f"{ANIMAL}/{TRAIN_SESSIONS[0]}",
        "source_sha256": source_digest,
        "source_manifest": manifest,
        "split_ids": {
            "train": list(TRAIN_SESSIONS),
            "validation": list(VAL_SESSIONS),
            "test": list(TEST_SESSIONS),
        },
        "dataset_config": {
            "animal": ANIMAL,
            "locomotion_labels": sorted(("Amble", "Walk", "WalkFast")),
            "minimum_locomotion_fraction": MIN_LOCOMOTION_FRACTION,
            "minimum_planar_speed": MIN_PLANAR_SPEED,
            "max_crops_per_session": args.max_crops_per_session,
            "session_safe": True,
        },
        "metrics": metrics,
    }
    output = args.output
    metrics_output = args.metrics or (
        METRICS if output == DST else output.with_name(f"{output.stem}_metrics.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    metrics.update(
        {
            "animal": ANIMAL,
            "source_sha256": source_digest,
            "tokenizer_seconds": tokenizer_seconds,
            "transition_seconds": transition_seconds,
            "total_training_seconds": tokenizer_seconds + transition_seconds,
            "motion_steps": args.motion_steps,
            "transition_steps": args.transition_steps,
            "motion_hidden": args.motion_hidden,
            "transition_config": transition_cfg,
            "command_support_velocity_q01_q99": command_support.tolist(),
            "asset": str(output),
            "asset_sha256": sha256(output),
        }
    )
    metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
