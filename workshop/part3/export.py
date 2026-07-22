from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch

from workshop.part2.core.artifacts import sha256
from workshop.part2.core.motion import hindsight_command
from workshop.part2.config import FEATURE_CONTRACT_VERSION
from workshop.part2.core.model import encode_in_batches
from workshop.part3.config import BUFFER_FRAMES, OUT, PHASE_DIM
from workshop.part3.data.contract import DEFAULT_ROOT
from workshop.part3.data.loader import load_split
from workshop.part3.evaluate_prior import DEVICE, load_models
from workshop.part3.core.prior import load_prior


@torch.inference_mode()
def export_prior(checkpoint_path: Path, output: Path, dataset_root: Path) -> dict:
    checkpoint, config, tokenizer, predictor, decoder = load_models(checkpoint_path)
    config.validate_online_contract()
    metadata = {
        "schema": "workshop-part3-jax-prior-v1",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "dataset_variant": checkpoint["dataset_variant"],
        "config": checkpoint["config"],
        "temporal_contract": (
            f"{BUFFER_FRAMES} raw frames -> {config.history_tokens} history tokens "
            "-> one next-token plan; one normalized control per frame"
        ),
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **{
            name: np.asarray(checkpoint[name], np.float32)
            for name in (
                "feature_mean",
                "feature_std",
                "token_mean",
                "token_std",
                "command_mean",
                "command_std",
                "state_sigma",
            )
        },
    }
    groups = {
        "tokenizer": {
            name: value
            for name, value in tokenizer.state_dict().items()
            if name.startswith("encoder.")
        },
        "predictor": predictor.state_dict(),
        "action_decoder": decoder.state_dict(),
    }
    for group, values in groups.items():
        arrays.update(
            {
                f"{group}::{name}": value.detach().cpu().numpy().astype(np.float32)
                for name, value in values.items()
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    validation = load_split(
        "validation",
        dataset_root,
        expected_variant=checkpoint["dataset_variant"],
    )
    raw_features = validation.features[0, :BUFFER_FRAMES]
    command = hindsight_command(
        validation.root_position[:1],
        validation.root_quaternion[:1],
        start=15,
        future=46,
    )[0]
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=DEVICE)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=DEVICE)
    normalized = (
        torch.as_tensor(raw_features[None], device=DEVICE) - feature_mean
    ) / feature_std
    token = encode_in_batches(tokenizer, normalized)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=DEVICE)
    token_std = torch.as_tensor(checkpoint["token_std"], device=DEVICE)
    history = (token - token_mean) / token_std
    command_tensor = torch.as_tensor(command[None], device=DEVICE)
    command_normalized = (
        command_tensor - torch.as_tensor(checkpoint["command_mean"], device=DEVICE)
    ) / torch.as_tensor(checkpoint["command_std"], device=DEVICE)
    torch_plan = predictor.predict(history, command_normalized)[0, 0]
    torch_mean = decoder(
        normalized[0, -1],
        torch_plan,
        torch.as_tensor(validation.normalized_control[0, 14], device=DEVICE),
        torch.eye(PHASE_DIM, device=DEVICE)[0],
        command_normalized[0],
    )
    prior = load_prior(output)
    jax_features = jnp.asarray(raw_features)
    jax_history = prior.encode(jax_features)
    jax_plan = prior.predict_plan(
        jax_history[-config.history_tokens :], jnp.asarray(command)
    )
    jax_mean = prior.action_mean(
        jax_features[-1],
        jax_plan,
        jnp.asarray(validation.normalized_control[0, 14]),
        jnp.eye(PHASE_DIM)[0],
        jnp.asarray(command),
    )
    plan_error = float(np.max(np.abs(np.asarray(jax_plan) - torch_plan.cpu().numpy())))
    action_error = float(
        np.max(np.abs(np.asarray(jax_mean) - torch_mean.cpu().numpy()))
    )

    if plan_error > 1e-3 or action_error > 5e-4:
        raise ValueError(
            f"PyTorch/JAX parity failed: plan={plan_error}, action={action_error}"
        )
    metadata["parity"] = {
        "maximum_plan_error": plan_error,
        "maximum_action_mean_error": action_error,
    }

    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(output, **arrays)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior.pt")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=OUT / "prior_jax.npz")
    args = parser.parse_args()
    metadata = export_prior(args.checkpoint, args.output, args.dataset_root)
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
