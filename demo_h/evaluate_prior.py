"""Held-out offline gates for the frozen Demo H state/action prior."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from demo_f.artifacts import sha256
from demo_f.config import (
    FEATURE_CONTRACT_VERSION,
    LEGACY_FEATURE_CONTRACT_VERSION,
)
from demo_f.models import ConditionalTransformer, MotionAutoencoder
from demo_f.windows import encode_in_batches
from demo_h.config import ACTION_PHASES, OUT, PriorConfig
from demo_h.dataset.contract import DEFAULT_ROOT
from demo_h.dataset.loader import load_manifest, load_split
from demo_h.models import FeedbackActionDecoder
from demo_h.train_prior import (
    DEVICE,
    _action_metrics,
    _closed_loop_action_metrics,
    _state_mse,
)
from demo_h.windows import state_action_windows


def load_models(path: Path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if checkpoint.get("schema") != "demo-h-prior-v1":
        raise ValueError(f"unsupported checkpoint {checkpoint.get('schema')!r}")
    feature_contract = checkpoint.get(
        "feature_contract_version", LEGACY_FEATURE_CONTRACT_VERSION
    )
    if feature_contract != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"checkpoint feature contract {feature_contract!r}; "
            f"expected {FEATURE_CONTRACT_VERSION!r}"
        )
    names = {field.name for field in fields(PriorConfig)}
    config = PriorConfig(
        **{k: v for k, v in checkpoint["config"].items() if k in names}
    )
    config.validate_online_contract()
    tokenizer = MotionAutoencoder(
        config.feature_dim, config.hidden, config.latent_dim
    ).to(DEVICE)
    predictor = ConditionalTransformer(
        latent_dim=config.latent_dim,
        future_tokens=1,
        width=config.hidden,
        layers=config.transformer_layers,
        heads=config.transformer_heads,
    ).to(DEVICE)
    decoder = FeedbackActionDecoder(
        config.feature_dim,
        config.latent_dim,
        config.action_dim,
        config.hidden,
        config.action_parameterization,
        config.previous_mean_coefficient,
    ).to(DEVICE)
    tokenizer.load_state_dict(checkpoint["tokenizer"])
    predictor.load_state_dict(checkpoint["predictor"])
    decoder.load_state_dict(checkpoint["action_decoder"])
    tokenizer.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    return checkpoint, config, tokenizer, predictor, decoder


@torch.inference_mode()
def evaluate(checkpoint_path: Path, dataset_root: Path, split: str) -> dict:
    checkpoint, config, tokenizer, predictor, decoder = load_models(checkpoint_path)
    dataset_variant = checkpoint["dataset_variant"]
    load_manifest(dataset_root, expected_variant=dataset_variant)
    manifest_hash = sha256(Path(dataset_root) / "manifest.json")
    if checkpoint["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("checkpoint was not trained from this dataset manifest")
    dataset = load_split(split, dataset_root, expected_variant=dataset_variant)
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], device=DEVICE)
    feature_std = torch.as_tensor(checkpoint["feature_std"], device=DEVICE)
    features = (
        torch.as_tensor(dataset.features, device=DEVICE) - feature_mean
    ) / feature_std
    controls = torch.as_tensor(dataset.normalized_control, device=DEVICE)
    tokens = encode_in_batches(tokenizer, features)
    token_mean = torch.as_tensor(checkpoint["token_mean"], device=DEVICE)
    token_std = torch.as_tensor(checkpoint["token_std"], device=DEVICE)
    tokens = (tokens - token_mean) / token_std
    windows = state_action_windows(tokens, features, controls, dataset, config)
    command_mean = torch.as_tensor(checkpoint["command_mean"], device=DEVICE)
    command_std = torch.as_tensor(checkpoint["command_std"], device=DEVICE)
    state_mse = _state_mse(predictor, windows, command_mean, command_std)
    persistence_mse = F.mse_loss(windows.history[:, -1:], windows.future[:, :1]).item()
    normalized_command = (windows.command - command_mean) / command_std
    prediction = predictor.predict(windows.history, normalized_command)
    shuffled_prediction = predictor.predict(
        windows.history,
        torch.roll(normalized_command, len(windows.anchors), dims=0),
    )
    shuffled_command_mse = F.mse_loss(shuffled_prediction, windows.future[:, :1]).item()
    action_prediction = predictor.predict(
        windows.action_history,
        (windows.action_anchor_command - command_mean) / command_std,
    )
    predicted_plan = action_prediction[:, 0].repeat_interleave(ACTION_PHASES, dim=0)
    action = _action_metrics(
        decoder, windows, predicted_plan, command_mean, command_std
    )
    action.update(
        _closed_loop_action_metrics(
            decoder, windows, predicted_plan, command_mean, command_std
        )
    )
    report = {
        "schema": "demo-h-offline-evaluation-v1",
        "split": split,
        "clips": len(dataset.features),
        "state_windows": len(windows.history),
        "action_targets": len(windows.target_control),
        "state_mse": state_mse,
        "state_persistence_mse": persistence_mse,
        "state_skill_over_persistence": 1.0 - state_mse / persistence_mse,
        "state_shuffled_command_mse": shuffled_command_mse,
        "matching_command_win": float(
            (
                (prediction - windows.future[:, :1]).square().mean(dim=(1, 2))
                < (shuffled_prediction - windows.future[:, :1])
                .square()
                .mean(dim=(1, 2))
            )
            .float()
            .mean()
        ),
        **{f"action_{name}": value for name, value in action.items()},
    }
    # At 50 Hz, copying the immediately preceding control is an unusually
    # strong one-step baseline.  It can beat a conditional predictor by a few
    # percent while becoming useless over a rollout.  The workshop claim is
    # therefore gated on command-conditioned state skill, plan dependence,
    # non-trivial control prediction, and closed-loop skill—not on beating the
    # previous-control baseline at exactly one frame.
    gate_criteria = {
        "state_beats_persistence": report["state_skill_over_persistence"] > 0.0,
        "command_is_informative": report["matching_command_win"] > 0.5,
        "action_beats_zero": report["action_mse"] < report["action_zero_control_mse"],
        "plan_is_informative": report["action_shuffled_plan_mse"]
        > report["action_mse"],
        "closed_loop_beats_repeated_control": (
            report["action_closed_loop_skill_over_repeated_initial"] > 0.0
        ),
    }
    report["offline_gate_criteria"] = gate_criteria
    report["passes_offline_gate"] = bool(all(gate_criteria.values()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=OUT / "prior_retime_1p75.pt")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.dataset_root, args.split)
    output = args.output or args.checkpoint.with_name(f"evaluation_{args.split}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
