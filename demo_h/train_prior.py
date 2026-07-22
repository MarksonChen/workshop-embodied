"""Train the Demo F-style future planner plus a physical control decoder."""

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

from demo_f.artifacts import sha256
from demo_f.config import FEATURE_CONTRACT_VERSION
from demo_f.models import ConditionalTransformer, MotionAutoencoder
from demo_f.losses import joint_limit_loss
from demo_f.windows import encode_in_batches
from demo_h.config import ACTION_PHASES, BUFFER_FRAMES, OUT, PriorConfig
from demo_h.dataset.contract import DATASET_VARIANT, DEFAULT_ROOT
from demo_h.dataset.loader import load_manifest, load_split
from demo_h.models import FeedbackActionDecoder, pre_tanh, tanh_gaussian_nll
from demo_h.windows import StateActionWindows, state_action_windows


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batched_loss(model, values: torch.Tensor, batch_size: int = 512) -> float:
    losses = []
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            batch = values[offset : offset + batch_size]
            losses.append(F.smooth_l1_loss(model(batch), batch).item() * len(batch))
    return sum(losses) / len(values)


def _predict_rollout(model, history, command, steps: int = 4):
    predictions = []
    rolling = history
    for _ in range(steps):
        prediction = model.predict(rolling, command)[:, :1]
        predictions.append(prediction)
        rolling = torch.cat((rolling, prediction), dim=1)[:, -history.shape[1] :]
    return torch.cat(predictions, dim=1)


@torch.inference_mode()
def _rolling_buffer_histories(
    tokenizer,
    normalized_features: torch.Tensor,
    anchors: np.ndarray,
    token_mean: torch.Tensor,
    token_std: torch.Tensor,
) -> torch.Tensor:
    """Encode the exact 16-frame buffers that the deployed prior will see."""

    frames = torch.stack(
        [
            normalized_features[
                :,
                int(anchor * ACTION_PHASES - BUFFER_FRAMES) : int(
                    anchor * ACTION_PHASES
                ),
            ]
            for anchor in anchors
        ],
        dim=1,
    )
    clips, windows = frames.shape[:2]
    tokens = encode_in_batches(tokenizer, frames.flatten(0, 1))
    tokens = (tokens - token_mean) / token_std
    return tokens.view(clips, windows, tokens.shape[-2], tokens.shape[-1]).flatten(0, 1)


def _state_mse(model, windows, command_mean, command_std, batch_size=2048):
    squared, count = 0.0, 0
    with torch.inference_mode():
        for offset in range(0, len(windows.history), batch_size):
            history = windows.history[offset : offset + batch_size]
            future = windows.future[offset : offset + batch_size]
            command = (
                windows.command[offset : offset + batch_size] - command_mean
            ) / command_std
            prediction = model.predict(history, command)
            squared += F.mse_loss(prediction, future[:, :1], reduction="sum").item()
            count += future[:, :1].numel()
    return squared / count


def _action_metrics(
    decoder,
    windows: StateActionWindows,
    predicted_plan: torch.Tensor,
    command_mean,
    command_std,
    batch_size=4096,
):
    sums = {name: 0.0 for name in ("squared", "tanh_nll", "shuffled_squared")}
    count = 0
    # Flattening is clip-major, then anchor, then action phase. Shift by one
    # complete clip so the negative retains phase/anchor but changes behavior.
    permutation = torch.roll(
        torch.arange(len(predicted_plan), device=DEVICE),
        len(windows.action_anchors) * ACTION_PHASES,
    )
    with torch.inference_mode():
        for offset in range(0, len(predicted_plan), batch_size):
            stop = min(offset + batch_size, len(predicted_plan))
            command = (windows.action_command[offset:stop] - command_mean) / command_std
            args = (
                windows.current_feature[offset:stop],
                predicted_plan[offset:stop],
                windows.previous_control[offset:stop],
                windows.phase[offset:stop],
                command,
            )
            target = windows.target_control[offset:stop]
            mean, log_std = decoder.distribution(*args)
            prediction = torch.tanh(mean)
            sums["squared"] += F.mse_loss(prediction, target, reduction="sum").item()
            sums["tanh_nll"] += tanh_gaussian_nll(mean, log_std, target).sum().item()
            shuffled_mean = decoder(
                windows.current_feature[offset:stop],
                predicted_plan[permutation[offset:stop]],
                windows.previous_control[offset:stop],
                windows.phase[offset:stop],
                command,
            )
            sums["shuffled_squared"] += F.mse_loss(
                torch.tanh(shuffled_mean), target, reduction="sum"
            ).item()
            count += target.numel()
    target = windows.target_control
    previous_mse = F.mse_loss(windows.previous_control, target).item()
    zero_mse = target.square().mean().item()
    mean_control = target.mean(dim=0, keepdim=True)
    mean_mse = F.mse_loss(mean_control.expand_as(target), target).item()
    return {
        "mse": sums["squared"] / count,
        "tanh_nll_per_dimension": sums["tanh_nll"] / count,
        "shuffled_plan_mse": sums["shuffled_squared"] / count,
        "previous_control_mse": previous_mse,
        "zero_control_mse": zero_mse,
        "mean_control_mse": mean_mse,
        "skill_over_previous": 1.0 - (sums["squared"] / count) / previous_mse,
    }


def _calibrate_action_std(
    decoder,
    windows: StateActionWindows,
    predicted_plan: torch.Tensor,
    command_mean,
    command_std,
    batch_size: int = 4096,
) -> None:
    squared = torch.zeros_like(decoder.log_std)
    count = 0
    with torch.inference_mode():
        for offset in range(0, len(predicted_plan), batch_size):
            stop = min(offset + batch_size, len(predicted_plan))
            command = (windows.action_command[offset:stop] - command_mean) / command_std
            mean = decoder(
                windows.current_feature[offset:stop],
                predicted_plan[offset:stop],
                windows.previous_control[offset:stop],
                windows.phase[offset:stop],
                command,
            )
            residual = pre_tanh(windows.target_control[offset:stop]) - mean
            squared += residual.square().sum(dim=0)
            count += len(residual)
        std = torch.sqrt(squared / count).clamp(0.01, 1.0)
        decoder.log_std.copy_(std.log())


def _closed_loop_action_metrics(
    decoder,
    windows: StateActionWindows,
    predicted_plan: torch.Tensor,
    command_mean,
    command_std,
) -> dict:
    """Roll predicted controls through the decoder's previous-action input."""

    sequence_steps = len(windows.action_anchors) * ACTION_PHASES
    sequence_count = len(windows.target_control) // sequence_steps
    feature = windows.current_feature.view(sequence_count, sequence_steps, -1)
    plan = predicted_plan.view(sequence_count, sequence_steps, -1)
    previous_targets = windows.previous_control.view(sequence_count, sequence_steps, -1)
    phase = windows.phase.view(sequence_count, sequence_steps, -1)
    command = ((windows.action_command - command_mean) / command_std).view(
        sequence_count, sequence_steps, -1
    )
    target = windows.target_control.view(sequence_count, sequence_steps, -1)
    previous = previous_targets[:, 0]
    predictions = []
    with torch.inference_mode():
        for index in range(sequence_steps):
            mean = decoder(
                feature[:, index],
                plan[:, index],
                previous,
                phase[:, index],
                command[:, index],
            )
            previous = torch.tanh(mean)
            predictions.append(previous)
    prediction = torch.stack(predictions, dim=1)
    repeated = previous_targets[:, :1].expand_as(target)
    model_mse = F.mse_loss(prediction, target).item()
    repeated_mse = F.mse_loss(repeated, target).item()
    return {
        "closed_loop_mse": model_mse,
        "repeated_initial_control_mse": repeated_mse,
        "closed_loop_skill_over_repeated_initial": 1.0 - model_mse / repeated_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-variant", default=DATASET_VARIANT)
    parser.add_argument("--output", type=Path, default=OUT / "prior_retime_1p75.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-steps", type=int, default=PriorConfig.tokenizer_steps
    )
    parser.add_argument(
        "--predictor-steps", type=int, default=PriorConfig.predictor_steps
    )
    parser.add_argument("--action-steps", type=int, default=PriorConfig.action_steps)
    parser.add_argument(
        "--history-encoding",
        choices=("full_clip", "rolling_buffer"),
        default="full_clip",
        help="Planner history construction; rolling_buffer matches deployment exactly",
    )
    parser.add_argument(
        "--action-parameterization",
        choices=("previous_control_residual", "leaky_previous", "direct"),
        default="previous_control_residual",
    )
    parser.add_argument("--previous-mean-coefficient", type=float, default=1.0)
    parser.add_argument(
        "--predicted-plan-probability",
        type=float,
        default=PriorConfig.predicted_plan_probability,
    )
    parser.add_argument(
        "--predicted-previous-control-probability",
        type=float,
        default=PriorConfig.predicted_previous_control_probability,
    )
    parser.add_argument(
        "--plan-noise-std", type=float, default=PriorConfig.plan_noise_std
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = PriorConfig(
        tokenizer_steps=20 if args.smoke else args.tokenizer_steps,
        predictor_steps=20 if args.smoke else args.predictor_steps,
        action_steps=20 if args.smoke else args.action_steps,
        action_parameterization=args.action_parameterization,
        previous_mean_coefficient=args.previous_mean_coefficient,
        predicted_plan_probability=args.predicted_plan_probability,
        predicted_previous_control_probability=(
            args.predicted_previous_control_probability
        ),
        plan_noise_std=args.plan_noise_std,
    )
    config.validate_online_contract()
    seed_everything(args.seed)
    manifest = load_manifest(args.dataset_root, expected_variant=args.dataset_variant)
    train = load_split(
        "train", args.dataset_root, expected_variant=args.dataset_variant
    )
    validation = load_split(
        "validation", args.dataset_root, expected_variant=args.dataset_variant
    )
    print(
        f"Demo H data train={len(train.features):,} validation={len(validation.features):,} "
        f"device={DEVICE}",
        flush=True,
    )
    feature_mean_np = train.features.mean(axis=(0, 1), dtype=np.float64).astype(
        np.float32
    )
    std = np.maximum(
        train.features.std(axis=(0, 1), dtype=np.float64).astype(np.float32), 1e-4
    )
    train_features = torch.from_numpy((train.features - feature_mean_np) / std).to(
        DEVICE
    )
    validation_features = torch.from_numpy(
        (validation.features - feature_mean_np) / std
    ).to(DEVICE)
    train_controls = torch.from_numpy(train.normalized_control).to(DEVICE)
    validation_controls = torch.from_numpy(validation.normalized_control).to(DEVICE)
    feature_mean = torch.from_numpy(feature_mean_np).to(DEVICE)
    feature_std = torch.from_numpy(std).to(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

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
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=config.learning_rate)
    tokenizer.train()
    for step in range(config.tokenizer_steps):
        index = torch.randint(
            len(train_features), (128,), device=DEVICE, generator=generator
        )
        target = train_features[index]
        reconstruction = tokenizer(target)
        reconstruction_loss = F.smooth_l1_loss(reconstruction, target)
        safety = joint_limit_loss(reconstruction, feature_mean, feature_std)
        loss = reconstruction_loss + 10.0 * safety
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == config.tokenizer_steps:
            print(
                f"[tokenizer] {step + 1}/{config.tokenizer_steps} loss={loss.item():.5f}",
                flush=True,
            )
    tokenizer.eval()
    tokenizer_train_loss = _batched_loss(tokenizer, train_features)
    tokenizer_validation_loss = _batched_loss(tokenizer, validation_features)
    with torch.inference_mode():
        train_tokens = encode_in_batches(tokenizer, train_features)
        validation_tokens = encode_in_batches(tokenizer, validation_features)
    token_mean = train_tokens.mean(dim=(0, 1))
    token_std = train_tokens.std(dim=(0, 1)).clamp_min(1e-4)
    train_tokens = (train_tokens - token_mean) / token_std
    validation_tokens = (validation_tokens - token_mean) / token_std
    tokenizer.requires_grad_(False)
    train_windows = state_action_windows(
        train_tokens, train_features, train_controls, train, config
    )
    validation_windows = state_action_windows(
        validation_tokens, validation_features, validation_controls, validation, config
    )
    if args.history_encoding == "rolling_buffer":
        train_windows.history = _rolling_buffer_histories(
            tokenizer,
            train_features,
            train_windows.anchors,
            token_mean,
            token_std,
        )
        validation_windows.history = _rolling_buffer_histories(
            tokenizer,
            validation_features,
            validation_windows.anchors,
            token_mean,
            token_std,
        )
        train_windows.action_history = _rolling_buffer_histories(
            tokenizer,
            train_features,
            train_windows.action_anchors,
            token_mean,
            token_std,
        )
        validation_windows.action_history = _rolling_buffer_histories(
            tokenizer,
            validation_features,
            validation_windows.action_anchors,
            token_mean,
            token_std,
        )
    command_mean = train_windows.command.mean(dim=0)
    command_std = train_windows.command.std(dim=0).clamp_min(1e-4)
    print(
        f"state windows={len(train_windows.history):,}; "
        f"action targets={len(train_windows.target_control):,}; "
        f"planner_anchors={train_windows.anchors.tolist()}; "
        f"action_anchors={train_windows.action_anchors.tolist()}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.learning_rate)
    best_predictor = None
    best_state_mse = float("inf")
    predictor.train()
    for step in range(config.predictor_steps):
        index = torch.randint(
            len(train_windows.history),
            (config.batch_size,),
            device=DEVICE,
            generator=generator,
        )
        history = train_windows.history[index]
        command = (train_windows.command[index] - command_mean) / command_std
        prediction = _predict_rollout(predictor, history, command, ACTION_PHASES)
        loss = F.mse_loss(prediction, train_windows.future[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % 250 == 0 or completed == config.predictor_steps:
            predictor.eval()
            validation_mse = _state_mse(
                predictor, validation_windows, command_mean, command_std
            )
            if validation_mse < best_state_mse:
                best_state_mse = validation_mse
                best_predictor = copy.deepcopy(predictor.state_dict())
            predictor.train()
            print(
                f"[planner] {completed}/{config.predictor_steps} "
                f"loss={loss.item():.5f} val={validation_mse:.5f}",
                flush=True,
            )
    if best_predictor is None:
        best_predictor = copy.deepcopy(predictor.state_dict())
    predictor.load_state_dict(best_predictor)
    predictor.eval()
    state_validation_mse = _state_mse(
        predictor, validation_windows, command_mean, command_std
    )
    persistence_mse = F.mse_loss(
        validation_windows.history[:, -1:], validation_windows.future[:, :1]
    ).item()
    with torch.inference_mode():
        train_predicted_plan = predictor.predict(
            train_windows.action_history,
            (train_windows.action_anchor_command - command_mean) / command_std,
        )[:, 0].repeat_interleave(ACTION_PHASES, dim=0)
        validation_predicted_plan = predictor.predict(
            validation_windows.action_history,
            (validation_windows.action_anchor_command - command_mean) / command_std,
        )[:, 0].repeat_interleave(ACTION_PHASES, dim=0)
    if train_predicted_plan.shape != train_windows.true_plan.shape:
        raise AssertionError(
            (train_predicted_plan.shape, train_windows.true_plan.shape)
        )

    decoder = FeedbackActionDecoder(
        config.feature_dim,
        config.latent_dim,
        config.action_dim,
        config.hidden,
        config.action_parameterization,
        config.previous_mean_coefficient,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.learning_rate)
    initial_metrics = _action_metrics(
        decoder,
        validation_windows,
        validation_predicted_plan,
        command_mean,
        command_std,
    )
    initial_closed_loop = _closed_loop_action_metrics(
        decoder,
        validation_windows,
        validation_predicted_plan,
        command_mean,
        command_std,
    )
    best_decoder = copy.deepcopy(decoder.state_dict())
    # One-step control persistence is exceptionally strong at 50 Hz.  It is a
    # useful guardrail but a poor selector: copying one action forever cannot
    # drive the body. Give the causal multi-step rollout four times its weight.
    best_action_score = (
        0.25 * initial_metrics["mse"] / initial_metrics["previous_control_mse"]
        + initial_closed_loop["closed_loop_mse"]
        / initial_closed_loop["repeated_initial_control_mse"]
    )
    sequence_steps = len(train_windows.action_anchors) * ACTION_PHASES
    if len(train_windows.target_control) % sequence_steps:
        raise AssertionError("action windows are not complete clip-major sequences")
    sequence_count = len(train_windows.target_control) // sequence_steps
    sequence = {
        "feature": train_windows.current_feature.view(
            sequence_count, sequence_steps, -1
        ),
        "true_plan": train_windows.true_plan.view(sequence_count, sequence_steps, -1),
        "predicted_plan": train_predicted_plan.view(sequence_count, sequence_steps, -1),
        "previous": train_windows.previous_control.view(
            sequence_count, sequence_steps, -1
        ),
        "phase": train_windows.phase.view(sequence_count, sequence_steps, -1),
        "command": ((train_windows.action_command - command_mean) / command_std).view(
            sequence_count, sequence_steps, -1
        ),
        "target": train_windows.target_control.view(sequence_count, sequence_steps, -1),
    }
    decoder.train()
    for step in range(config.action_steps):
        clip_index = torch.randint(
            sequence_count,
            (max(config.batch_size // ACTION_PHASES, 1),),
            device=DEVICE,
            generator=generator,
        )
        previous = sequence["previous"][clip_index, 0]
        losses = []
        for sequence_index in range(sequence_steps):
            true_plan = sequence["true_plan"][clip_index, sequence_index]
            noisy_true = true_plan + config.plan_noise_std * torch.randn(
                true_plan.shape, device=DEVICE, generator=generator
            )
            choose_prediction = (
                torch.rand((len(clip_index), 1), device=DEVICE, generator=generator)
                < config.predicted_plan_probability
            )
            plan = torch.where(
                choose_prediction,
                sequence["predicted_plan"][clip_index, sequence_index],
                noisy_true,
            )
            target = sequence["target"][clip_index, sequence_index]
            action_mean = decoder(
                sequence["feature"][clip_index, sequence_index],
                plan,
                previous,
                sequence["phase"][clip_index, sequence_index],
                sequence["command"][clip_index, sequence_index],
            )
            teacher_mean = decoder(
                sequence["feature"][clip_index, sequence_index],
                plan,
                sequence["previous"][clip_index, sequence_index],
                sequence["phase"][clip_index, sequence_index],
                sequence["command"][clip_index, sequence_index],
            )
            losses.append(
                0.5 * F.mse_loss(action_mean, pre_tanh(target))
                + 0.5 * F.mse_loss(teacher_mean, pre_tanh(target))
            )
            predicted_control = torch.tanh(action_mean).detach()
            use_prediction = (
                torch.rand((len(clip_index), 1), device=DEVICE, generator=generator)
                < config.predicted_previous_control_probability
            )
            previous = torch.where(use_prediction, predicted_control, target)
        # Fixed-variance pre-tanh MSE is Gaussian maximum likelihood. Calibrate
        # diagonal variance after selecting the best mean.
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0 or completed == config.action_steps:
            decoder.eval()
            metrics = _action_metrics(
                decoder,
                validation_windows,
                validation_predicted_plan,
                command_mean,
                command_std,
            )
            metrics.update(
                _closed_loop_action_metrics(
                    decoder,
                    validation_windows,
                    validation_predicted_plan,
                    command_mean,
                    command_std,
                )
            )
            score = (
                0.25 * metrics["mse"] / metrics["previous_control_mse"]
                + metrics["closed_loop_mse"] / metrics["repeated_initial_control_mse"]
            )
            if score < best_action_score:
                best_action_score = score
                best_decoder = copy.deepcopy(decoder.state_dict())
            decoder.train()
            print(
                f"[action] {completed}/{config.action_steps} loss={loss.item():.5f} "
                f"val_mse={metrics['mse']:.6f} closed={metrics['closed_loop_mse']:.6f} "
                f"skill={metrics['closed_loop_skill_over_repeated_initial']:.1%}",
                flush=True,
            )
    decoder.load_state_dict(best_decoder)
    decoder.eval()
    _calibrate_action_std(
        decoder,
        validation_windows,
        validation_predicted_plan,
        command_mean,
        command_std,
    )
    action_metrics = _action_metrics(
        decoder,
        validation_windows,
        validation_predicted_plan,
        command_mean,
        command_std,
    )
    action_metrics.update(
        _closed_loop_action_metrics(
            decoder,
            validation_windows,
            validation_predicted_plan,
            command_mean,
            command_std,
        )
    )
    state_sigma = torch.sqrt(
        torch.tensor(state_validation_mse, device=DEVICE)
    ).clamp_min(1e-3)
    metrics = {
        "tokenizer_train_smooth_l1": tokenizer_train_loss,
        "tokenizer_validation_smooth_l1": tokenizer_validation_loss,
        "state_validation_mse": state_validation_mse,
        "state_persistence_mse": persistence_mse,
        "state_skill_over_persistence": 1.0 - state_validation_mse / persistence_mse,
        **{f"action_{name}": value for name, value in action_metrics.items()},
        "state_sigma": float(state_sigma),
        "action_log_std": decoder.log_std.detach().cpu().tolist(),
        "history_encoding": args.history_encoding,
        "training_seconds": time.perf_counter() - started,
    }
    checkpoint = {
        "schema": "demo-h-prior-v1",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "config": asdict(config),
        "seed": args.seed,
        "dataset_manifest_sha256": sha256(args.dataset_root / "manifest.json"),
        "dataset_variant": manifest["variant"],
        "history_encoding": args.history_encoding,
        "feature_mean": feature_mean_np,
        "feature_std": std,
        "token_mean": token_mean.cpu().numpy(),
        "token_std": token_std.cpu().numpy(),
        "command_mean": command_mean.cpu().numpy(),
        "command_std": command_std.cpu().numpy(),
        "state_sigma": float(state_sigma),
        "tokenizer": tokenizer.state_dict(),
        "predictor": predictor.state_dict(),
        "action_decoder": decoder.state_dict(),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
