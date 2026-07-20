"""Export the accepted Demo F Gaussian prior for pure-JAX Demo G inference.

The exported archive contains only the causal encoder, conditional predictor,
normalization constants, and likelihood calibration.  Demo G never imports
PyTorch inside its compiled PPO environment.

    uv run --extra workshop python -m demo_f.export_jax
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import OUT
from .dataset import load_split
from .dataset.contract import DYNAMIC_ROOT
from .generate import checkpoint_command_scale, load_prior, sha256
from .windows import encode_in_batches, predictor_windows


@torch.inference_mode()
def export_prior(checkpoint_path: Path, output: Path, dataset_root: Path) -> dict:
    checkpoint, config, tokenizer, predictor = load_prior(checkpoint_path)
    if checkpoint.get("schema") != "demo-f-prior-v2":
        raise ValueError("Demo G requires the aligned next-token Demo F v2 prior")

    train = load_split("train", dataset_root)
    validation = load_split("validation", dataset_root)
    device = next(tokenizer.parameters()).device
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=device)
    features = (
        torch.as_tensor(validation.features, device=device) - feature_mean
    ) / feature_std
    tokens = encode_in_batches(tokenizer, features)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=device)
    token_std = torch.as_tensor(checkpoint["token_std"], device=device)
    history, future, command, _ = predictor_windows(
        (tokens - token_mean) / token_std, validation, config
    )
    command_mean = torch.as_tensor(checkpoint["command_mean"], device=device)
    command_std = torch.as_tensor(checkpoint["command_std"], device=device)
    logp = predictor.log_prob(
        history,
        future,
        (command - command_mean) / command_std,
        checkpoint["sigma"],
    )
    quantiles = torch.quantile(
        logp, torch.as_tensor((0.01, 0.05, 0.50, 0.95, 0.99), device=device)
    ).cpu().numpy()

    metadata = {
        "schema": "demo-f-jax-prior-v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "dataset_repository_id": checkpoint["dataset_repository_id"],
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "config": asdict(config),
        "validation_logp_quantiles": {
            name: float(value)
            for name, value in zip(
                ("q01", "q05", "q50", "q95", "q99"), quantiles, strict=True
            )
        },
        "dataset_variant": checkpoint.get("dataset_variant", "kinematic-v1"),
        "dynamic_scaling": checkpoint.get("dynamic_scaling"),
        "command_scale_fetch_displacement_per_mps": checkpoint_command_scale(
            checkpoint, train
        ),
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "feature_mean": np.asarray(checkpoint["feature_mean"], np.float32),
        "feature_std": np.asarray(checkpoint["feature_std"], np.float32),
        "token_mean": np.asarray(checkpoint["token_mean"], np.float32),
        "token_std": np.asarray(checkpoint["token_std"], np.float32),
        "command_mean": np.asarray(checkpoint["command_mean"], np.float32),
        "command_std": np.asarray(checkpoint["command_std"], np.float32),
        "sigma": np.asarray(checkpoint["sigma"], np.float32),
        "validation_logp_quantiles": quantiles.astype(np.float32),
    }
    arrays.update(
        {
            f"tokenizer::{name}": value.detach().cpu().numpy().astype(np.float32)
            for name, value in tokenizer.state_dict().items()
            if name.startswith("encoder.")
        }
    )
    arrays.update(
        {
            f"predictor::{name}": value.detach().cpu().numpy().astype(np.float32)
            for name, value in predictor.state_dict().items()
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--dataset-root", type=Path, default=DYNAMIC_ROOT)
    parser.add_argument("--output", type=Path, default=OUT / "prior_jax.npz")
    args = parser.parse_args()
    metadata = export_prior(args.checkpoint, args.output, args.dataset_root)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
