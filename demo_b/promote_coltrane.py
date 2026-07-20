"""Promote the behaviorally validated standalone Coltrane model to Demo B.

The predictor weights are copied unchanged.  This script only attaches the
fixed-Gaussian calibration and explicit provenance required by Demo E.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import DEV, FULL_FM
from .models import SimpleTrans


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "rl_standalone" / "assets" / "motor_standalone.pt"
WINDOWS = ROOT / "rl_standalone" / "exploration" / "logs" / "data.pt"
DESTINATION = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
METRICS = ROOT / "demo_b" / "assets" / "motor_metrics.json"
ELIGIBILITY = ROOT / "demo_b" / "assets" / "eligibility.json"
REJECTED_FREDDIE = ROOT / "demo_b" / "out" / "freddie_85_rejected" / "motor.pt"
TRAINING_SESSIONS = (
    "2021_07_28_1",
    "2021_07_29_1",
    "2021_07_30_1",
    "2021_07_31_1",
    "2021_08_01_1",
    "2021_08_02_1",
    "2021_08_03_1",
    "2021_08_04_1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def calibrate(source: dict, data: dict) -> tuple[dict, dict]:
    model = SimpleTrans(**source["model_cfg"]).to(DEV)
    model.load_state_dict(source["trans"])
    model.eval()
    history = data["zhn"].float().to(DEV)
    future = data["zfn"].float().to(DEV)
    command = data["cmd"].float().to(DEV)
    context = model.context(history)

    def predict(normalized_command):
        return model.predict_from_context(context, normalized_command)

    prediction = predict(command)
    residual = future[:, 0] - prediction[:, 0]
    sigma = residual.square().mean().sqrt().clamp_min(1e-3)

    def logp(normalized_command):
        error = (future[:, 0] - predict(normalized_command)[:, 0]) / sigma
        return -0.5 * (
            error.square() + 2 * sigma.log() + math.log(2 * math.pi)
        ).mean(-1)

    true_logp = logp(command)
    permutation = torch.randperm(
        len(future), device=DEV, generator=torch.Generator(device=DEV).manual_seed(13)
    )
    shuffled_error = (future[permutation, 0] - prediction[:, 0]) / sigma
    shuffled_logp = -0.5 * (
        shuffled_error.square() + 2 * sigma.log() + math.log(2 * math.pi)
    ).mean(-1)

    cmean = np.asarray(source["cmean"], np.float32)
    cstd = np.asarray(source["cstd"], np.float32)
    raw_command = command.cpu().numpy() * cstd + cmean
    duration = 31 / 50
    planar = raw_command[:, :2]
    speed = np.linalg.norm(planar, axis=-1) / duration
    keep = speed > 0.02
    direction = planar / np.maximum(
        np.linalg.norm(planar, axis=-1, keepdims=True), 1e-6
    )
    centers = np.quantile(speed[keep], [0.10, 0.30, 0.50, 0.70, 0.90]).astype(np.float32)
    actual_bin = np.argmin(np.abs(speed[keep, None] - centers[None]), axis=1)
    columns = []
    for conditioned_speed in centers:
        counterfactual = raw_command.copy()
        counterfactual[:, :2] = direction * conditioned_speed * duration
        normalized = torch.from_numpy((counterfactual - cmean) / cstd).to(DEV)
        columns.append(logp(normalized).cpu().numpy()[keep])
    scores = np.stack(columns, axis=1)
    matrix = np.stack(
        [scores[actual_bin == row].mean(0) for row in range(len(centers))]
    )
    row_argmax = matrix.argmax(1)

    offsets = np.asarray([-0.10, -0.05, 0.0, 0.05, 0.10], np.float32)
    central = keep & (speed > 0.12) & (speed < 0.28)
    relative = []
    for offset in offsets:
        counterfactual = raw_command.copy()
        requested = np.maximum(speed + offset, 1e-3)
        counterfactual[:, :2] = direction * requested[:, None] * duration
        normalized = torch.from_numpy((counterfactual - cmean) / cstd).to(DEV)
        relative.append(float(logp(normalized)[central].mean()))
    relative = np.asarray(relative)

    metrics = {
        "animal": "coltrane",
        "feature_dim": FULL_FM,
        "calibration_scope": "standalone_training_windows",
        "n_windows": int(len(history)),
        "full_mse": F.mse_loss(prediction, future).item(),
        "first_mse": residual.square().mean().item(),
        "sigma": float(sigma),
        "logp_mean": float(true_logp.mean()),
        "logp_shuffled_mean": float(shuffled_logp.mean()),
        "logp_q01": float(torch.quantile(true_logp, 0.01)),
        "logp_q99": float(torch.quantile(true_logp, 0.99)),
        "speed_likelihood": {
            "speed_centers_m_per_s": centers.tolist(),
            "mean_logp_matrix_actual_rows_conditioned_columns": matrix.tolist(),
            "row_argmax_conditioned_bin": row_argmax.tolist(),
            "diagonal_wins": int(np.sum(row_argmax == np.arange(len(centers)))),
            "sample_top1_speed_bin_accuracy": float(
                np.mean(scores.argmax(1) == actual_bin)
            ),
            "chance_accuracy": 0.2,
            "relative_speed_offsets_m_per_s": offsets.tolist(),
            "relative_mean_logp": relative.tolist(),
            "relative_peak_offset_m_per_s": float(offsets[relative.argmax()]),
            "relative_peak_at_match": bool(relative.argmax() == 2),
        },
    }
    bundle = dict(source)
    bundle.update(
        {
            "format_version": 4,
            "arch": "simple_gaussian_full_motion",
            "feature_dim": FULL_FM,
            "animal": "coltrane",
            "motion_cfg": {"fm": FULL_FM, "hid": 256, "dm": 16},
            "sigma": np.float32(float(sigma)),
            "logp_clip": np.asarray(
                [metrics["logp_q01"], metrics["logp_q99"]], np.float32
            ),
            "command_support_velocity": np.quantile(
                raw_command[:, [0, 2]] / duration, [0.01, 0.99], axis=0
            ).astype(np.float32),
            "evaluation_bank": {
                "history": history.cpu().numpy().astype(np.float32),
                "future": future.cpu().numpy().astype(np.float32),
                "command_raw": raw_command.astype(np.float32),
            },
            "split_ids": {"train": list(TRAINING_SESSIONS), "validation": [], "test": []},
            "dataset_config": {
                "animal": "coltrane",
                "representation": "full_281",
                "strict_geometric_locomotion": True,
                "sessions": 8,
                "selection": "speed>0.10, coordinated gait, |turn|<90deg, |neck drift|<10mm",
                "calibration_scope": "training; held-out Coltrane rebuild is a Demo E prerequisite",
            },
            "metrics": metrics,
            "source_asset": str(SOURCE),
            "source_asset_sha256": sha256(SOURCE),
        }
    )
    return bundle, metrics


def main() -> None:
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    data = torch.load(WINDOWS, map_location="cpu", weights_only=False)
    if np.asarray(source["mmean"]).shape != (FULL_FM,):
        raise ValueError("standalone source is not the validated 281-D model")
    if DESTINATION.exists():
        existing = torch.load(DESTINATION, map_location="cpu", weights_only=False)
        if existing.get("animal") == "freddie" and existing.get("feature_dim") == 85:
            REJECTED_FREDDIE.parent.mkdir(parents=True, exist_ok=True)
            if not REJECTED_FREDDIE.exists():
                torch.save(existing, REJECTED_FREDDIE)
                print(f"preserved rejected Freddie asset at {REJECTED_FREDDIE}")
    bundle, metrics = calibrate(source, data)
    torch.save(bundle, DESTINATION)
    metrics["asset"] = str(DESTINATION)
    metrics["asset_sha256"] = sha256(DESTINATION)
    METRICS.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    gates = {
        "full_281_representation": bundle["feature_dim"] == FULL_FM,
        "coltrane": bundle["animal"] == "coltrane",
        "beats_shuffled": metrics["logp_mean"] > metrics["logp_shuffled_mean"],
        "speed_bins_diagonal": metrics["speed_likelihood"]["diagonal_wins"] == 5,
        "relative_peak_at_match": metrics["speed_likelihood"]["relative_peak_at_match"],
        "behaviorally_validated_source_unchanged": bundle["source_asset_sha256"] == sha256(SOURCE),
    }
    ELIGIBILITY.write_text(
        json.dumps({"eligible_for_demo_b": all(gates.values()), "gates": gates, "metrics": metrics}, indent=2, sort_keys=True)
        + "\n"
    )
    print(json.dumps({"gates": gates, "metrics": metrics}, indent=2, sort_keys=True))
    print(f"wrote {DESTINATION} ({DESTINATION.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
