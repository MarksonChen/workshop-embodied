"""Train the lightweight conditional prior with Demo B's original 281-D tokenizer.

The tokenizer and strict geometric locomotion rule are held fixed.  This makes
``--animal freddie`` versus ``--animal coltrane`` a controlled comparison of
the transition data rather than another representation ablation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import DEV, DM, FULL_FM, FULL_SL, H, K
from .models import MotionVAE, SimpleTrans
from .splits import split_for
from .strict_locomotion import (
    CROP_STRIDE,
    GAIT_PAIRS,
    MAX_NECK_DRIFT_MM,
    MAX_TURN_DEGREES,
    MIN_GAIT_COORDINATION,
    MIN_SPEED,
    load_strict_crop_set,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "rl_standalone" / "assets" / "motor_standalone.pt"
MOTION_WEIGHTS = {
    "vxy": 5.0,
    "h": 2.0,
    "d6": 5.0,
    "q": 1.0,
    "qd": 1.0,
    "kp": 8.0,
    "kpd": 5.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def motion_loss(reconstruction, target, mu, logvar):
    loss = target.new_zeros(())
    for name, (begin, end) in FULL_SL.items():
        loss = loss + MOTION_WEIGHTS[name] * F.smooth_l1_loss(
            reconstruction[..., begin:end], target[..., begin:end]
        )
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return loss + 1e-6 * kl, kl


def train_tokenizer(
    normalized_features: torch.Tensor,
    *,
    steps: int,
    seed: int,
) -> tuple[MotionVAE, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MotionVAE(fm=FULL_FM, hid=256, dm=DM).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    averages = [parameter.detach().clone() for parameter in model.parameters()]
    generator = torch.Generator(device=DEV).manual_seed(seed)
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        index = torch.randint(
            len(normalized_features),
            (128,),
            device=DEV,
            generator=generator,
        )
        target = normalized_features[index]
        reconstruction, mu, logvar = model(target)
        loss, kl = motion_loss(reconstruction, target, mu, logvar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for average, parameter in zip(averages, model.parameters(), strict=True):
                average.mul_(0.99).add_(parameter, alpha=0.01)
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            print(
                f"[tokenizer] {step + 1:5d}/{steps} "
                f"loss={loss.item():.5f} kl={kl.item():.5f}",
                flush=True,
            )
    with torch.no_grad():
        for average, parameter in zip(averages, model.parameters(), strict=True):
            parameter.copy_(average)
    model.eval()
    return model, time.perf_counter() - started


@torch.inference_mode()
def encode_features(
    tokenizer: MotionVAE,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    encoded = []
    for offset in range(0, len(features), batch_size):
        batch = torch.from_numpy(
            (features[offset : offset + batch_size] - mean) / std
        ).to(DEV)
        encoded.append(tokenizer.encode(batch)[0].float().cpu())
    return torch.cat(encoded)


def build_encoded_data(
    animal: str,
    source_path: Path,
    cache_path: Path,
    *,
    rebuild: bool,
) -> tuple[dict, float]:
    train_sessions, val_sessions, test_sessions = split_for(animal)
    source_hash = sha256(source_path)
    expected = {
        "animal": animal,
        "source_sha256": source_hash,
        "split_ids": {
            "train": list(train_sessions),
            "val": list(val_sessions),
            "test": list(test_sessions),
        },
        "feature_dim": FULL_FM,
        "crop_stride": CROP_STRIDE,
    }
    if cache_path.exists() and not rebuild:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("contract") == expected:
            print(f"[cache] HIT {cache_path}", flush=True)
            return cached, 0.0
        print(f"[cache] STALE {cache_path}", flush=True)

    started = time.perf_counter()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    mean = np.asarray(source["mmean"], np.float32)
    std = np.asarray(source["mstd"], np.float32)
    if mean.shape != (FULL_FM,) or std.shape != (FULL_FM,):
        raise ValueError("source checkpoint is not the known-good 281-D tokenizer")
    first_weight = source["motion"]["enc.0.conv.weight"]
    tokenizer = MotionVAE(
        fm=FULL_FM,
        hid=int(first_weight.shape[0]),
        dm=int(source["motion"]["to_mu.conv.weight"].shape[0]),
    ).to(DEV)
    tokenizer.load_state_dict(source["motion"])
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    split_sessions = {
        "train": train_sessions,
        "val": val_sessions,
        "test": test_sessions,
    }
    output = {"contract": expected, "splits": {}}
    for name, sessions in split_sessions.items():
        dataset = load_strict_crop_set(animal, sessions)
        latent = encode_features(tokenizer, dataset.features, mean, std)
        output["splits"][name] = {
            "latent": latent,
            "command": dataset.command,
            "sessions_retained": list(dataset.sessions),
            "session_rows": dataset.session_rows,
        }
        print(
            f"[{name}] encoded {len(latent)} crops from "
            f"{len(dataset.sessions)}/{len(sessions)} sessions",
            flush=True,
        )
        if name == "train":
            output["seed"] = {
                "feat": dataset.seed_features,
                "xy": dataset.seed_xy,
                "yaw": dataset.seed_yaw,
                "name": dataset.seed_name,
            }
        del dataset
    output["mmean"] = mean
    output["mstd"] = std
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cache_path)
    seconds = time.perf_counter() - started
    print(f"[cache] wrote {cache_path} | extraction+encoding={seconds:.1f}s", flush=True)
    return output, seconds


def build_native_encoded_data(
    animal: str,
    cache_path: Path,
    *,
    maximum_train_sessions: int,
    tokenizer_steps: int,
    seed: int,
    rebuild: bool,
) -> tuple[dict, float]:
    """Fit the full tokenizer and encode a matched animal-native corpus."""
    train_sessions, val_sessions, test_sessions = split_for(animal)
    train_sessions = train_sessions[:maximum_train_sessions]
    expected = {
        "animal": animal,
        "tokenizer": "animal_native_full_281",
        "tokenizer_steps": tokenizer_steps,
        "split_ids": {
            "train": list(train_sessions),
            "val": list(val_sessions),
            "test": list(test_sessions),
        },
        "feature_dim": FULL_FM,
        "crop_stride": CROP_STRIDE,
    }
    if cache_path.exists() and not rebuild:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("contract") == expected:
            print(f"[native cache] HIT {cache_path}", flush=True)
            return cached, 0.0
        print(f"[native cache] STALE {cache_path}", flush=True)

    started = time.perf_counter()
    datasets = {
        "train": load_strict_crop_set(animal, train_sessions),
        "val": load_strict_crop_set(animal, val_sessions),
        "test": load_strict_crop_set(animal, test_sessions),
    }
    train_frames = datasets["train"].features.reshape(-1, FULL_FM)
    mean = train_frames.mean(0, dtype=np.float64).astype(np.float32)
    std = (train_frames.std(0, dtype=np.float64) + 1e-4).astype(np.float32)
    normalized_train = torch.from_numpy(
        (datasets["train"].features - mean) / std
    ).to(DEV)
    tokenizer, tokenizer_seconds = train_tokenizer(
        normalized_train,
        steps=tokenizer_steps,
        seed=seed,
    )
    del normalized_train

    output = {
        "contract": expected,
        "splits": {},
        "mmean": mean,
        "mstd": std,
        "motion": {
            key: value.detach().cpu().clone()
            for key, value in tokenizer.state_dict().items()
        },
        "motion_cfg": {"fm": FULL_FM, "hid": 256, "dm": DM},
        "tokenizer_seconds": tokenizer_seconds,
    }
    for name, dataset in datasets.items():
        latent = encode_features(tokenizer, dataset.features, mean, std)
        output["splits"][name] = {
            "latent": latent,
            "command": dataset.command,
            "sessions_retained": list(dataset.sessions),
            "session_rows": dataset.session_rows,
        }
        print(
            f"[{name}] native tokenizer encoded {len(latent)} crops from "
            f"{len(dataset.sessions)} sessions",
            flush=True,
        )
        if name == "train":
            output["seed"] = {
                "feat": dataset.seed_features,
                "xy": dataset.seed_xy,
                "yaw": dataset.seed_yaw,
                "name": dataset.seed_name,
            }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cache_path)
    seconds = time.perf_counter() - started
    print(
        f"[native cache] wrote {cache_path} | preprocessing+tokenizer={seconds:.1f}s",
        flush=True,
    )
    return output, seconds


def normalized_splits(data: dict):
    train_latent = data["splits"]["train"]["latent"].float()
    train_command = torch.from_numpy(data["splits"]["train"]["command"]).float()
    zmean = train_latent.reshape(-1, DM).mean(0)
    zstd = train_latent.reshape(-1, DM).std(0) + 1e-4
    cmean = train_command.mean(0)
    cstd = train_command.std(0) + 1e-6
    output = {}
    for name, values in data["splits"].items():
        latent = values["latent"].float()
        command = torch.from_numpy(values["command"]).float()
        normalized = (latent - zmean) / zstd
        output[name] = (
            normalized[:, :H],
            normalized[:, H : H + K],
            (command - cmean) / cstd,
        )
    return output, zmean, zstd, cmean, cstd


def limit_training_sessions(data: dict, maximum: int | None) -> dict:
    """Return a shallow data view containing only the first training sessions."""
    if maximum is None:
        return data
    rows = data["splits"]["train"]["session_rows"][:maximum]
    retained = data["splits"]["train"]["sessions_retained"][:maximum]
    count = sum(int(row["crops"]) for row in rows)
    output = dict(data)
    output["splits"] = dict(data["splits"])
    train = dict(data["splits"]["train"])
    train["latent"] = train["latent"][:count]
    train["command"] = train["command"][:count]
    train["session_rows"] = rows
    train["sessions_retained"] = retained
    output["splits"]["train"] = train
    output["contract"] = copy.deepcopy(data["contract"])
    output["contract"]["split_ids"]["train"] = retained
    animal = data["contract"]["animal"]
    if not any(output["seed"]["name"].startswith(f"{animal}/{name}") for name in retained):
        raise ValueError("cached longest seed is outside the requested training-session subset")
    print(
        f"[train subset] {len(retained)} sessions, {count} crops: {retained}",
        flush=True,
    )
    return output


@torch.inference_mode()
def _validation_mse(model: SimpleTrans, split, batch_size: int = 4096) -> float:
    history, future, command = split
    total, count = 0.0, 0
    model.eval()
    for offset in range(0, len(history), batch_size):
        stop = offset + batch_size
        prediction = model.predict(
            history[offset:stop].to(DEV), command[offset:stop].to(DEV)
        )
        error = F.mse_loss(prediction, future[offset:stop].to(DEV), reduction="sum")
        total += float(error)
        count += prediction.numel()
    return total / count


def train_transition(
    split,
    validation,
    *,
    steps: int,
    seed: int,
    selection: str,
    contrastive_weight: float = 0.0,
    contrastive_margin: float = 0.10,
    contrastive_negatives: int = 1,
    speed_contrastive_delta: float = 0.0,
):
    # Cache hits must not change model initialization.  Dataset construction
    # instantiates the frozen tokenizer on a cold run, which otherwise advances
    # the global RNG before this point.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = {"d": 192, "layers": 6, "heads": 4, "ff": 768}
    model = SimpleTrans(**config).to(DEV)
    history, future, command = (value.to(DEV) for value in split)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    generator = torch.Generator(device=DEV).manual_seed(seed)
    best_state, best_validation = None, math.inf
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * min((step + 1) / 400, 1.0)
        index = torch.randint(
            len(history), (256,), device=DEV, generator=generator
        )
        batch_history = history[index]
        batch_future = future[index]
        batch_command = command[index]
        if contrastive_weight > 0:
            context = model.context(batch_history)
            prediction = model.predict_from_context(context, batch_command)
            prediction_mse = F.mse_loss(prediction, batch_future)
            # Deterministic batch rolls supply real, distribution-matched
            # counterfactual commands without fabricating motion.  The hinge
            # prevents the history encoder from explaining the target while
            # silently ignoring its command.
            positive_error = (prediction - batch_future).square().mean((1, 2))
            negative_errors = []
            for negative_index in range(contrastive_negatives):
                shift = (negative_index + 1) * len(batch_command) // (
                    contrastive_negatives + 1
                )
                counterfactual = torch.roll(batch_command, shift, 0)
                counterfactual_prediction = model.predict_from_context(
                    context, counterfactual
                )
                negative_errors.append(
                    (counterfactual_prediction - batch_future)
                    .square()
                    .mean((1, 2))
                )
            if speed_contrastive_delta > 0:
                # Add symmetric local speed negatives to the real shuffled
                # commands. Delta is in normalized forward-displacement units.
                for sign in (-1.0, 1.0):
                    counterfactual = batch_command.clone()
                    counterfactual[:, 0] += sign * speed_contrastive_delta
                    counterfactual_prediction = model.predict_from_context(
                        context, counterfactual
                    )
                    negative_errors.append(
                        (counterfactual_prediction - batch_future)
                        .square()
                        .mean((1, 2))
                    )
            ranking = torch.stack(
                [
                    F.relu(
                        contrastive_margin + positive_error - negative_error
                    )
                    for negative_error in negative_errors
                ]
            ).mean()
            loss = prediction_mse + contrastive_weight * ranking
        else:
            prediction_mse = model.loss(
                batch_history, batch_future, batch_command
            )
            ranking = prediction_mse.new_zeros(())
            loss = prediction_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            validation_mse = _validation_mse(model, validation)
            print(
                f"[transition] {step + 1:5d}/{steps} "
                f"train={prediction_mse.item():.5f} "
                f"rank={ranking.item():.5f} val={validation_mse:.5f}",
                flush=True,
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise AssertionError("no transition checkpoint was selected")
    final_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(best_state if selection == "validation" else final_state)
    model.eval()
    return model, config, best_validation, time.perf_counter() - started


@torch.inference_mode()
def predict_split(model: SimpleTrans, split, batch_size: int = 4096) -> torch.Tensor:
    history, _, command = split
    output = []
    for offset in range(0, len(history), batch_size):
        stop = offset + batch_size
        output.append(
            model.predict(
                history[offset:stop].to(DEV), command[offset:stop].to(DEV)
            ).cpu()
        )
    return torch.cat(output)


def likelihood_metrics(model: SimpleTrans, split, sigma=None) -> tuple[dict, torch.Tensor]:
    history, future, command = split
    prediction = predict_split(model, split)
    residual = future[:, 0] - prediction[:, 0]
    if sigma is None:
        sigma = residual.square().mean().sqrt().clamp_min(1e-3)
    logp = -0.5 * (
        (residual / sigma).square()
        + 2 * sigma.log()
        + math.log(2 * math.pi)
    ).mean(-1)
    permutation = torch.randperm(
        len(future), generator=torch.Generator().manual_seed(13)
    )
    shuffled_residual = future[permutation, 0] - prediction[:, 0]
    shuffled_logp = -0.5 * (
        (shuffled_residual / sigma).square()
        + 2 * sigma.log()
        + math.log(2 * math.pi)
    ).mean(-1)
    persistence = F.mse_loss(history[:, -1], future[:, 0]).item()
    first_mse = residual.square().mean().item()
    return {
        "mse": F.mse_loss(prediction, future).item(),
        "first_mse": first_mse,
        "persistence_first_mse": persistence,
        "skill_over_persistence": 1.0 - first_mse / persistence,
        "logp_mean": logp.mean().item(),
        "logp_shuffled_mean": shuffled_logp.mean().item(),
        "logp_q01": torch.quantile(logp, 0.01).item(),
        "logp_q99": torch.quantile(logp, 0.99).item(),
        "sigma": float(sigma),
    }, sigma


@torch.inference_mode()
def speed_likelihood_audit(
    model: SimpleTrans,
    split,
    raw_command: np.ndarray,
    cmean: torch.Tensor,
    cstd: torch.Tensor,
    sigma: torch.Tensor,
) -> dict:
    history, future, _ = split
    history = history.to(DEV)
    future = future[:, 0].to(DEV)
    context = model.context(history)

    duration = 31 / 50
    planar = raw_command[:, :2]
    speed = np.linalg.norm(planar, axis=-1) / duration
    keep = speed > 0.02
    direction = planar / np.maximum(
        np.linalg.norm(planar, axis=-1, keepdims=True), 1e-6
    )
    centers = np.quantile(speed[keep], [0.10, 0.30, 0.50, 0.70, 0.90]).astype(np.float32)
    actual_bin = np.argmin(
        np.abs(speed[keep, None] - centers[None]), axis=1
    )

    cmean_device, cstd_device = cmean.to(DEV), cstd.to(DEV)

    def score(command_array: np.ndarray) -> np.ndarray:
        normalized = (
            torch.from_numpy(command_array).to(DEV) - cmean_device
        ) / cstd_device
        prediction = model.predict_from_context(context, normalized)[:, 0]
        logp = -0.5 * (
            ((future - prediction) / sigma.to(DEV)).square()
            + 2 * sigma.to(DEV).log()
            + math.log(2 * math.pi)
        ).mean(-1)
        return logp.cpu().numpy()

    columns = []
    for conditioned_speed in centers:
        command = raw_command.copy()
        command[:, :2] = direction * conditioned_speed * duration
        columns.append(score(command)[keep])
    scores = np.stack(columns, axis=1)
    matrix = np.stack(
        [scores[actual_bin == row].mean(0) for row in range(len(centers))]
    )
    row_argmax = matrix.argmax(1)

    q25, q75 = np.quantile(speed[keep], [0.25, 0.75])
    central = keep & (speed >= q25) & (speed <= q75)
    spacing = max(float(q75 - q25) / 6, 1e-3)
    offsets = spacing * np.asarray([-2, -1, 0, 1, 2], np.float32)
    relative = []
    for offset in offsets:
        command = raw_command.copy()
        requested = np.maximum(speed + offset, 1e-3)
        command[:, :2] = direction * requested[:, None] * duration
        relative.append(float(score(command)[central].mean()))
    relative = np.asarray(relative)
    return {
        "n_examples": int(keep.sum()),
        "speed_centers_m_per_s": centers.tolist(),
        "mean_logp_matrix_actual_rows_conditioned_columns": matrix.tolist(),
        "row_argmax_conditioned_bin": row_argmax.tolist(),
        "diagonal_wins": int(np.sum(row_argmax == np.arange(len(centers)))),
        "sample_top1_speed_bin_accuracy": float(
            np.mean(scores.argmax(1) == actual_bin)
        ),
        "chance_accuracy": 1 / len(centers),
        "relative_speed_offsets_m_per_s": offsets.tolist(),
        "relative_mean_logp": relative.tolist(),
        "relative_peak_offset_m_per_s": float(offsets[relative.argmax()]),
        "relative_peak_at_match": bool(relative.argmax() == len(offsets) // 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=("freddie", "coltrane"), default="freddie")
    parser.add_argument("--source-tokenizer", type=Path, default=SOURCE)
    parser.add_argument(
        "--fit-tokenizer",
        action="store_true",
        help="fit a fresh 281-D tokenizer on this animal instead of freezing the Coltrane tokenizer",
    )
    parser.add_argument("--tokenizer-steps", type=int, default=2500)
    parser.add_argument("--steps", type=int, default=16000)
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="hinge weight forcing correct commands to beat shuffled commands",
    )
    parser.add_argument("--contrastive-margin", type=float, default=0.10)
    parser.add_argument(
        "--contrastive-negatives",
        type=int,
        default=1,
        help="number of real shuffled-command negatives per training example",
    )
    parser.add_argument(
        "--speed-contrastive-delta",
        type=float,
        default=0.0,
        help="optional symmetric local-speed negatives, added to shuffled commands",
    )
    parser.add_argument(
        "--selection",
        choices=("validation", "final"),
        default="validation",
        help="validation is scientific held-out selection; final matches the original Demo B rollout recipe",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument(
        "--max-train-sessions",
        type=int,
        help="controlled comparison with the first N training sessions",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.contrastive_negatives < 1:
        parser.error("--contrastive-negatives must be at least 1")
    if args.smoke:
        args.steps = 20
        args.tokenizer_steps = 20
    output = args.output or (
        ROOT / "demo_b" / "out" / f"{args.animal}_281" / "motor.pt"
    )
    cache = output.parent / (
        "native_encoded_data.pt" if args.fit_tokenizer else "encoded_data.pt"
    )
    # Preserve the historical default path, but give explicitly named research
    # candidates their own sidecars instead of silently overwriting one shared
    # metrics file in the output directory.
    metrics_path = (
        output.parent / "metrics.json"
        if output.name == "motor.pt"
        else output.with_suffix(".json")
    )
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.fit_tokenizer:
        native_sessions = args.max_train_sessions or len(split_for(args.animal)[0])
        data, data_seconds = build_native_encoded_data(
            args.animal,
            cache,
            maximum_train_sessions=native_sessions,
            tokenizer_steps=args.tokenizer_steps,
            seed=args.seed,
            rebuild=args.rebuild_data,
        )
    else:
        data, data_seconds = build_encoded_data(
            args.animal,
            args.source_tokenizer,
            cache,
            rebuild=args.rebuild_data,
        )
        data = limit_training_sessions(data, args.max_train_sessions)
    splits, zmean, zstd, cmean, cstd = normalized_splits(data)
    transition, model_config, best_val, training_seconds = train_transition(
        splits["train"],
        splits["val"],
        steps=args.steps,
        seed=args.seed,
        selection=args.selection,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        contrastive_negatives=args.contrastive_negatives,
        speed_contrastive_delta=args.speed_contrastive_delta,
    )

    metrics = {
        "animal": args.animal,
        "feature_dim": FULL_FM,
        "best_validation_mse": best_val,
        "data_seconds": data_seconds,
        "training_seconds": training_seconds,
        "steps": args.steps,
        "checkpoint_selection": args.selection,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_margin": args.contrastive_margin,
        "contrastive_negatives": args.contrastive_negatives,
        "speed_contrastive_delta": args.speed_contrastive_delta,
        "tokenizer": "animal_native" if args.fit_tokenizer else "frozen_coltrane",
        "tokenizer_steps": args.tokenizer_steps if args.fit_tokenizer else 0,
        "tokenizer_seconds": float(data.get("tokenizer_seconds", 0.0)),
        "datasets": {
            name: {
                "crops": int(len(data["splits"][name]["latent"])),
                "sessions_retained": data["splits"][name]["sessions_retained"],
                "session_rows": data["splits"][name]["session_rows"],
            }
            for name in ("train", "val", "test")
        },
    }
    train_metrics, sigma = likelihood_metrics(transition, splits["train"])
    metrics["train"] = train_metrics
    metrics["val"] = likelihood_metrics(transition, splits["val"], sigma)[0]
    metrics["test"] = likelihood_metrics(transition, splits["test"], sigma)[0]
    metrics["speed_likelihood"] = speed_likelihood_audit(
        transition,
        splits["test"],
        data["splits"]["test"]["command"],
        cmean,
        cstd,
        sigma,
    )
    metrics["validation_speed_likelihood"] = speed_likelihood_audit(
        transition,
        splits["val"],
        data["splits"]["val"]["command"],
        cmean,
        cstd,
        sigma,
    )

    source = (
        None
        if args.fit_tokenizer
        else torch.load(args.source_tokenizer, map_location="cpu", weights_only=False)
    )
    duration = 31 / 50
    train_velocity = data["splits"]["train"]["command"][:, [0, 2]] / duration
    command_support = np.quantile(
        train_velocity, [0.01, 0.99], axis=0
    ).astype(np.float32)
    logp_clip = np.asarray(
        [train_metrics["logp_q01"], train_metrics["logp_q99"]], np.float32
    )
    seed = data["seed"]
    bundle = {
        "format_version": 4,
        "arch": "simple_gaussian_full_motion",
        "feature_dim": FULL_FM,
        "animal": args.animal,
        "motion": data["motion"] if args.fit_tokenizer else source["motion"],
        "motion_cfg": data.get("motion_cfg", {"fm": FULL_FM, "hid": 256, "dm": DM}),
        "trans": {
            key: value.detach().cpu()
            for key, value in transition.state_dict().items()
        },
        "model_cfg": model_config,
        "zmean": zmean.numpy().astype(np.float32),
        "zstd": zstd.numpy().astype(np.float32),
        "cmean": cmean.numpy().astype(np.float32),
        "cstd": cstd.numpy().astype(np.float32),
        "mmean": data["mmean"],
        "mstd": data["mstd"],
        "sigma": np.float32(sigma),
        "logp_clip": logp_clip,
        "command_support_velocity": command_support,
        "seed_feat": seed["feat"],
        "seed_xy": seed["xy"],
        "seed_yaw": seed["yaw"],
        "seed_name": seed["name"],
        "tokenizer_source": (
            f"trained_from_{args.animal}" if args.fit_tokenizer else str(args.source_tokenizer)
        ),
        "tokenizer_source_sha256": (
            None if args.fit_tokenizer else sha256(args.source_tokenizer)
        ),
        "split_ids": data["contract"]["split_ids"],
        "dataset_config": {
            "representation": "full_281",
            "minimum_speed": MIN_SPEED,
            "minimum_gait_coordination": MIN_GAIT_COORDINATION,
            "maximum_turn_degrees": MAX_TURN_DEGREES,
            "maximum_neck_drift_mm": MAX_NECK_DRIFT_MM,
            "gait_pairs": [list(pair) for pair in GAIT_PAIRS],
            "crop_stride": CROP_STRIDE,
            "session_safe": True,
        },
        "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    metrics["asset"] = str(output)
    metrics["asset_sha256"] = sha256(output)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
