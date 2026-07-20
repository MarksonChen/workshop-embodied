"""Train Demo F exclusively from the standalone retargeted dataset release."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import OUT, PriorConfig
from .dataset import load_manifest, load_split
from .dataset.contract import DEFAULT_ROOT
from .features import FEATURE_DIM
from .models import ConditionalTransformer, MotionAutoencoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(values: torch.Tensor, size: int, generator: torch.Generator):
    index = torch.randint(len(values), (size,), device=values.device, generator=generator)
    return values[index]


@torch.inference_mode()
def encode(model, values: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    return torch.cat(
        [model.encode(values[offset:offset + batch_size]) for offset in range(0, len(values), batch_size)]
    )


@torch.inference_mode()
def prediction_mse(
    model: ConditionalTransformer,
    tokens: torch.Tensor,
    command: torch.Tensor,
    config: PriorConfig,
) -> float:
    history = tokens[:, :config.history_tokens]
    future = tokens[:, config.history_tokens:config.history_tokens + config.future_tokens]
    return F.mse_loss(model.predict(history, command), future).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer-steps", type=int, default=PriorConfig.tokenizer_steps)
    parser.add_argument("--predictor-steps", type=int, default=PriorConfig.predictor_steps)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = PriorConfig(
        tokenizer_steps=20 if args.smoke else args.tokenizer_steps,
        predictor_steps=20 if args.smoke else args.predictor_steps,
    )
    seed_everything(args.seed)
    manifest = load_manifest(args.dataset_root)
    train = load_split("train", args.dataset_root)
    validation = load_split("validation", args.dataset_root)
    print(
        f"dataset schema {manifest['schema_version']} | train={len(train.features):,} "
        f"validation={len(validation.features):,} | device={DEVICE}",
        flush=True,
    )

    mean = train.features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train.features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    train_features = torch.from_numpy((train.features - mean) / std).to(DEVICE)
    validation_features = torch.from_numpy((validation.features - mean) / std).to(DEVICE)
    train_command = torch.from_numpy(train.command).to(DEVICE)
    validation_command = torch.from_numpy(validation.command).to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    tokenizer = MotionAutoencoder(
        FEATURE_DIM, config.hidden_channels, config.latent_dim
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=config.learning_rate)
    started = time.perf_counter()
    tokenizer.train()
    for step in range(config.tokenizer_steps):
        target = batches(train_features, config.tokenizer_batch_size, generator)
        loss = F.smooth_l1_loss(tokenizer(target), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == config.tokenizer_steps:
            print(f"[tokenizer] {step + 1:4d}/{config.tokenizer_steps} loss={loss.item():.5f}", flush=True)

    tokenizer.eval()
    with torch.inference_mode():
        tokenizer_train_loss = F.smooth_l1_loss(
            tokenizer(train_features), train_features
        ).item()
        tokenizer_validation_loss = F.smooth_l1_loss(
            tokenizer(validation_features), validation_features
        ).item()
    train_tokens = encode(tokenizer, train_features)
    validation_tokens = encode(tokenizer, validation_features)
    token_mean = train_tokens.mean(dim=(0, 1))
    token_std = train_tokens.std(dim=(0, 1)).clamp_min(1e-4)
    train_tokens = (train_tokens - token_mean) / token_std
    validation_tokens = (validation_tokens - token_mean) / token_std

    predictor = ConditionalTransformer(
        latent_dim=config.latent_dim,
        future_tokens=config.future_tokens,
        width=config.hidden_channels,
        layers=config.transformer_layers,
        heads=config.transformer_heads,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.learning_rate)
    best_validation_mse = float("inf")
    best_predictor_step = 0
    best_predictor_state = None
    predictor.train()
    for step in range(config.predictor_steps):
        index = torch.randint(
            len(train_tokens),
            (config.predictor_batch_size,),
            device=DEVICE,
            generator=generator,
        )
        history = train_tokens[index, :config.history_tokens]
        future = train_tokens[
            index,
            config.history_tokens:config.history_tokens + config.future_tokens,
        ]
        prediction = predictor.predict(history, train_command[index])
        loss = F.mse_loss(prediction, future)
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
        if validate:
            predictor.eval()
            validation_at_step = prediction_mse(
                predictor, validation_tokens, validation_command, config
            )
            if validation_at_step < best_validation_mse:
                best_validation_mse = validation_at_step
                best_predictor_step = completed
                best_predictor_state = copy.deepcopy(predictor.state_dict())
            predictor.train()
        if step == 0 or completed % 250 == 0 or completed == config.predictor_steps:
            suffix = (
                f" val={validation_at_step:.5f}"
                if validation_at_step is not None
                else ""
            )
            print(
                f"[predictor] {completed:4d}/{config.predictor_steps} "
                f"loss={loss.item():.5f}{suffix}",
                flush=True,
            )

    if best_predictor_state is None:
        raise RuntimeError("predictor validation never ran")
    predictor.load_state_dict(best_predictor_state)
    predictor.eval()
    with torch.inference_mode():
        history = validation_tokens[:, :config.history_tokens]
        future = validation_tokens[
            :, config.history_tokens:config.history_tokens + config.future_tokens
        ]
        prediction = predictor.predict(history, validation_command)
        validation_mse = F.mse_loss(prediction, future).item()
        persistence_mse = F.mse_loss(
            history[:, -1:, :].expand_as(future), future
        ).item()
        sigma = float(torch.sqrt(F.mse_loss(prediction, future)).clamp_min(1e-3))
        own = ((prediction - future) ** 2).mean(dim=(1, 2))
        shuffled_prediction = predictor.predict(history, validation_command.flip(0))
        shuffled = ((shuffled_prediction - future) ** 2).mean(dim=(1, 2))
        command_win_rate = float((own < shuffled).float().mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "demo-f-prior-v1",
        "config": asdict(config),
        "seed": args.seed,
        "dataset_manifest_sha256": sha256(args.dataset_root / "manifest.json"),
        "dataset_repository_id": manifest["repository_id"],
        "feature_mean": mean,
        "feature_std": std,
        "token_mean": token_mean.cpu().numpy(),
        "token_std": token_std.cpu().numpy(),
        "sigma": sigma,
        "tokenizer": tokenizer.state_dict(),
        "predictor": predictor.state_dict(),
        "metrics": {
            "tokenizer_train_smooth_l1": tokenizer_train_loss,
            "tokenizer_validation_smooth_l1": tokenizer_validation_loss,
            "validation_mse": validation_mse,
            "persistence_mse": persistence_mse,
            "command_vs_reversed_win_rate": command_win_rate,
            "best_predictor_step": best_predictor_step,
            "training_seconds": time.perf_counter() - started,
        },
    }
    torch.save(checkpoint, args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(checkpoint["metrics"], indent=2) + "\n")
    print(
        f"validation mse={validation_mse:.5f} | persistence={persistence_mse:.5f} | "
        f"command win={command_win_rate:.1%} | best step={best_predictor_step} | "
        f"sigma={sigma:.4f}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
