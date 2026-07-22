from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core.artifacts import sha256
from .config import FEATURE_CONTRACT_VERSION, FEATURE_DIM, OUT, PriorConfig
from .data import load_manifest, load_split
from .data.contract import DYNAMIC_ROOT
from .core.model import joint_limit_loss
from .evaluate import rollout_report
from .core.model import dataset_command_calibration
from .core.model import ConditionalTransformer, MotionAutoencoder
from .core.model import encode_in_batches, predictor_windows


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(values: torch.Tensor, size: int, generator: torch.Generator):
    index = torch.randint(
        len(values), (size,), device=values.device, generator=generator
    )
    return values[index]


@torch.inference_mode()
def prediction_mse(
    model: ConditionalTransformer,
    history: torch.Tensor,
    future: torch.Tensor,
    command: torch.Tensor,
) -> float:
    return F.mse_loss(model.predict(history, command), future).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-tokens", type=int, default=PriorConfig.history_tokens
    )
    parser.add_argument("--latent-dim", type=int, default=PriorConfig.latent_dim)
    parser.add_argument(
        "--learning-rate", type=float, default=PriorConfig.learning_rate
    )
    parser.add_argument(
        "--joint-limit-penalty",
        type=float,
        default=PriorConfig.joint_limit_penalty,
    )
    parser.add_argument(
        "--training-rollout-tokens",
        type=int,
        default=PriorConfig.training_rollout_tokens,
    )
    parser.add_argument(
        "--tokenizer-steps", type=int, default=PriorConfig.tokenizer_steps
    )
    parser.add_argument(
        "--predictor-steps", type=int, default=PriorConfig.predictor_steps
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = PriorConfig(
        history_tokens=args.history_tokens,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        joint_limit_penalty=args.joint_limit_penalty,
        training_rollout_tokens=args.training_rollout_tokens,
        tokenizer_steps=20 if args.smoke else args.tokenizer_steps,
        predictor_steps=20 if args.smoke else args.predictor_steps,
    )
    seed_everything(args.seed)
    manifest = load_manifest(args.dataset_root)
    train = load_split("train", args.dataset_root)
    validation = load_split("validation", args.dataset_root)
    command_calibration = dataset_command_calibration(
        manifest, train.command, train.source_speed_mps
    )
    print(
        f"dataset schema {manifest['schema_version']} | train={len(train.features):,} "
        f"validation={len(validation.features):,} | device={DEVICE}",
        flush=True,
    )

    mean = train.features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    train_features = torch.from_numpy((train.features - mean) / std).to(DEVICE)
    validation_features = torch.from_numpy((validation.features - mean) / std).to(
        DEVICE
    )
    feature_mean = torch.from_numpy(mean).to(DEVICE)
    feature_std = torch.from_numpy(std).to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    tokenizer = MotionAutoencoder(
        FEATURE_DIM, config.hidden_channels, config.latent_dim
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=config.learning_rate)
    started = time.perf_counter()
    tokenizer.train()
    for step in range(config.tokenizer_steps):
        target = batches(train_features, config.tokenizer_batch_size, generator)
        reconstruction = tokenizer(target)
        reconstruction_loss = F.smooth_l1_loss(reconstruction, target)
        safety_loss = joint_limit_loss(reconstruction, feature_mean, feature_std)
        loss = reconstruction_loss + config.joint_limit_penalty * safety_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == config.tokenizer_steps:
            print(
                f"[tokenizer] {step + 1:4d}/{config.tokenizer_steps} "
                f"loss={loss.item():.5f} safety={safety_loss.item():.6f}",
                flush=True,
            )

    tokenizer.eval()
    with torch.inference_mode():
        tokenizer_train_loss = F.smooth_l1_loss(
            tokenizer(train_features), train_features
        ).item()
        tokenizer_validation_loss = F.smooth_l1_loss(
            tokenizer(validation_features), validation_features
        ).item()
    train_tokens = encode_in_batches(tokenizer, train_features)
    validation_tokens = encode_in_batches(tokenizer, validation_features)
    token_mean = train_tokens.mean(dim=(0, 1))
    token_std = train_tokens.std(dim=(0, 1)).clamp_min(1e-4)
    train_tokens = (train_tokens - token_mean) / token_std
    validation_tokens = (validation_tokens - token_mean) / token_std
    tokenizer.requires_grad_(False)
    train_history, train_future, train_command, train_anchors = predictor_windows(
        train_tokens,
        train,
        config,
        target_tokens=config.training_rollout_tokens,
    )
    validation_history, validation_future, validation_command, validation_anchors = (
        predictor_windows(validation_tokens, validation, config)
    )
    command_mean = train_command.mean(dim=0)
    command_std = train_command.std(dim=0).clamp_min(1e-4)
    train_command = (train_command - command_mean) / command_std
    validation_command = (validation_command - command_mean) / command_std
    validation_persistence_mse = F.mse_loss(
        validation_history[:, -1:, :].expand_as(validation_future),
        validation_future,
    ).item()
    print(
        f"predictor windows: train={len(train_history):,} validation={len(validation_history):,} "
        f"anchors={train_anchors.tolist()}",
        flush=True,
    )

    predictor = ConditionalTransformer(
        latent_dim=config.latent_dim,
        future_tokens=config.future_tokens,
        width=config.hidden_channels,
        layers=config.transformer_layers,
        heads=config.transformer_heads,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.learning_rate)
    best_rollout_objective = float("inf")
    best_validation_mse = float("inf")
    best_predictor_step = 0
    best_predictor_state = None
    predictor.train()
    for step in range(config.predictor_steps):
        index = torch.randint(
            len(train_history),
            (config.predictor_batch_size,),
            device=DEVICE,
            generator=generator,
        )
        history = train_history[index]
        future = train_future[index]
        rollout_history = history
        rollout_predictions = []
        for _ in range(config.training_rollout_tokens):
            next_token = predictor.predict(rollout_history, train_command[index])[:, :1]
            rollout_predictions.append(next_token)
            rollout_history = torch.cat((rollout_history, next_token), dim=1)[
                :, -config.history_tokens :
            ]
        prediction = torch.cat(rollout_predictions, dim=1)
        prediction_loss = F.mse_loss(prediction, future)
        decoded_stream = tokenizer.decode(
            torch.cat((history, prediction), dim=1) * token_std + token_mean
        )
        decoded_prediction = decoded_stream[
            :, -config.training_rollout_tokens * config.downsample :
        ]
        safety_loss = joint_limit_loss(decoded_prediction, feature_mean, feature_std)
        loss = prediction_loss + config.joint_limit_penalty * safety_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        validate = (
            completed % config.predictor_validation_interval == 0
            or completed == config.predictor_steps
        )
        validation_at_step = None
        rollout_at_step = None
        if validate:
            predictor.eval()
            validation_at_step = prediction_mse(
                predictor, validation_history, validation_future, validation_command
            )
            selection_checkpoint = {
                "config": asdict(config),
                "feature_mean": mean,
                "feature_std": std,
                "token_mean": token_mean.detach().cpu().numpy(),
                "token_std": token_std.detach().cpu().numpy(),
                "command_mean": command_mean.detach().cpu().numpy(),
                "command_std": command_std.detach().cpu().numpy(),
                "command_calibration": command_calibration,
            }
            rollout_at_step = rollout_report(
                selection_checkpoint,
                config,
                tokenizer,
                predictor,
                train,
                validation,
            )["objective"]
            if (
                validation_at_step < validation_persistence_mse
                and rollout_at_step < best_rollout_objective
            ) or (args.smoke and completed == config.predictor_steps):
                best_rollout_objective = rollout_at_step
                best_validation_mse = validation_at_step
                best_predictor_step = completed
                best_predictor_state = copy.deepcopy(predictor.state_dict())
            predictor.train()
        if step == 0 or completed % 250 == 0 or completed == config.predictor_steps:
            suffix = (
                f" val={validation_at_step:.5f} rollout={rollout_at_step:.4f}"
                if validation_at_step is not None
                else ""
            )
            print(
                f"[predictor] {completed:4d}/{config.predictor_steps} "
                f"loss={loss.item():.5f} safety={safety_loss.item():.6f}{suffix}",
                flush=True,
            )

    if best_predictor_state is None:
        raise RuntimeError("predictor validation never ran")
    predictor.load_state_dict(best_predictor_state)
    predictor.eval()
    with torch.inference_mode():
        prediction = predictor.predict(validation_history, validation_command)
        validation_mse = F.mse_loss(prediction, validation_future).item()
        persistence_mse = validation_persistence_mse
        sigma = float(
            torch.sqrt(F.mse_loss(prediction, validation_future)).clamp_min(1e-3)
        )
        own = ((prediction - validation_future) ** 2).mean(dim=(1, 2))
        shuffled_prediction = predictor.predict(
            validation_history, validation_command.flip(0)
        )
        shuffled = ((shuffled_prediction - validation_future) ** 2).mean(dim=(1, 2))
        command_win_rate = float((own < shuffled).float().mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "workshop-part2-prior-v1",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "config": asdict(config),
        "seed": args.seed,
        "dataset_manifest_sha256": sha256(args.dataset_root / "manifest.json"),
        "dataset_repository_id": manifest["repository_id"],
        "dataset_variant": manifest.get("variant", "kinematic-v1"),
        "dynamic_scaling": manifest.get("dynamic_scaling"),
        "command_calibration": command_calibration,
        "feature_mean": mean,
        "feature_std": std,
        "token_mean": token_mean.cpu().numpy(),
        "token_std": token_std.cpu().numpy(),
        "command_mean": command_mean.cpu().numpy(),
        "command_std": command_std.cpu().numpy(),
        "sigma": sigma,
        "tokenizer": tokenizer.state_dict(),
        "predictor": predictor.state_dict(),
        "metrics": {
            "tokenizer_train_smooth_l1": tokenizer_train_loss,
            "tokenizer_validation_smooth_l1": tokenizer_validation_loss,
            "validation_mse": validation_mse,
            "persistence_mse": persistence_mse,
            "command_vs_reversed_win_rate": command_win_rate,
            "train_predictor_windows": len(train_history),
            "validation_predictor_windows": len(validation_history),
            "predictor_anchor_tokens": train_anchors.tolist(),
            "best_predictor_step": best_predictor_step,
            "best_validation_selection_mse": best_validation_mse,
            "best_rollout_objective": best_rollout_objective,
            "training_seconds": time.perf_counter() - started,
        },
    }
    torch.save(checkpoint, args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(checkpoint["metrics"], indent=2) + "\n")
    print(
        f"validation mse={validation_mse:.5f} | persistence={persistence_mse:.5f} | "
        f"command win={command_win_rate:.1%} | best step={best_predictor_step} | "
        f"rollout={best_rollout_objective:.4f} | sigma={sigma:.4f}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
