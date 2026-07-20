"""Audit the frozen standalone Coltrane likelihood on unseen sessions.

This command never trains or edits model weights.  It encodes strict
locomotion crops from session-disjoint validation/test recordings with the
known-good tokenizer, scores them with the frozen transition, and writes the
eligibility evidence consumed by Demo E.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .constants import DEV, DM, FULL_FM, H, K
from .models import load_motor
from .splits import (
    COLTRANE_PRIOR_TEST_SESSIONS,
    COLTRANE_PRIOR_TRAIN_SESSIONS,
    COLTRANE_PRIOR_VAL_SESSIONS,
    validate_coltrane_prior_split,
)
from .strict_locomotion import load_strict_crop_set
from .train_full_prior import (
    encode_features,
    likelihood_metrics,
    sha256,
    speed_likelihood_audit,
)


ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
CACHE = ROOT / "demo_b" / "out" / "cache" / "coltrane_heldout.pt"
REPORT = ROOT / "demo_b" / "assets" / "coltrane_heldout.json"


def _contract(asset: Path) -> dict:
    return {
        "asset_sha256": sha256(asset),
        "animal": "coltrane",
        "feature_dim": FULL_FM,
        "train_sessions": list(COLTRANE_PRIOR_TRAIN_SESSIONS),
        "validation_sessions": list(COLTRANE_PRIOR_VAL_SESSIONS),
        "test_sessions": list(COLTRANE_PRIOR_TEST_SESSIONS),
        "selection": "frozen_strict_geometric_locomotion",
    }


@torch.inference_mode()
def build_heldout(asset: Path, cache: Path, *, rebuild: bool) -> dict:
    expected = _contract(asset)
    if cache.exists() and not rebuild:
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("contract") == expected:
            print(f"[cache] HIT {cache}", flush=True)
            return saved
        print(f"[cache] STALE {cache}", flush=True)

    motion, _, norms, _ = load_motor(asset)
    if norms["mmean"].shape != (FULL_FM,):
        raise ValueError("held-out calibration requires the 281-D Coltrane tokenizer")
    output = {"contract": expected, "splits": {}}
    for split, sessions in (
        ("validation", COLTRANE_PRIOR_VAL_SESSIONS),
        ("test", COLTRANE_PRIOR_TEST_SESSIONS),
    ):
        data = load_strict_crop_set("coltrane", sessions)
        latent = encode_features(
            motion, data.features, norms["mmean"], norms["mstd"]
        )
        output["splits"][split] = {
            "latent": latent,
            "command": data.command,
            "sessions_retained": list(data.sessions),
            "session_rows": data.session_rows,
        }
        print(
            f"[{split}] {len(latent)} crops from {len(data.sessions)} sessions",
            flush=True,
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, cache)
    print(f"[cache] wrote {cache}", flush=True)
    return output


def _normalized_split(values: dict, bundle: dict):
    latent = values["latent"].float()
    zmean = torch.as_tensor(bundle["zmean"]).reshape(1, 1, DM)
    zstd = torch.as_tensor(bundle["zstd"]).reshape(1, 1, DM)
    command = torch.from_numpy(values["command"]).float()
    cmean = torch.as_tensor(bundle["cmean"])
    cstd = torch.as_tensor(bundle["cstd"])
    normalized = (latent - zmean) / zstd
    return (
        normalized[:, :H],
        normalized[:, H : H + K],
        (command - cmean) / cstd,
    )


@torch.inference_mode()
def evaluate(asset: Path = ASSET, cache: Path = CACHE, *, rebuild: bool = False) -> dict:
    validate_coltrane_prior_split()
    bundle = torch.load(asset, map_location="cpu", weights_only=False)
    if (
        bundle.get("format_version") != 4
        or bundle.get("animal") != "coltrane"
        or bundle.get("feature_dim") != FULL_FM
    ):
        raise ValueError("expected the canonical format-v4 281-D Coltrane asset")
    _, transition, _, _ = load_motor(asset)
    heldout = build_heldout(asset, cache, rebuild=rebuild)
    sigma = torch.as_tensor(bundle["sigma"])
    cmean = torch.as_tensor(bundle["cmean"])
    cstd = torch.as_tensor(bundle["cstd"])
    metrics = {}
    normalized = {}
    for name in ("validation", "test"):
        normalized[name] = _normalized_split(heldout["splits"][name], bundle)
        metrics[name] = likelihood_metrics(
            transition, normalized[name], sigma
        )[0]
    speed = speed_likelihood_audit(
        transition,
        normalized["test"],
        np.asarray(heldout["splits"]["test"]["command"], np.float32),
        cmean,
        cstd,
        sigma,
    )
    metrics["test_speed_likelihood"] = speed
    sessions_retained = {
        name: heldout["splits"][name]["sessions_retained"]
        for name in ("validation", "test")
    }
    gates = {
        "canonical_281_coltrane": True,
        "session_disjoint": not (
            set(COLTRANE_PRIOR_TRAIN_SESSIONS)
            & (set(COLTRANE_PRIOR_VAL_SESSIONS) | set(COLTRANE_PRIOR_TEST_SESSIONS))
        ),
        "all_heldout_sessions_retained": sessions_retained
        == {
            "validation": list(COLTRANE_PRIOR_VAL_SESSIONS),
            "test": list(COLTRANE_PRIOR_TEST_SESSIONS),
        },
        "validation_beats_persistence": metrics["validation"]["skill_over_persistence"] > 0,
        "test_beats_persistence": metrics["test"]["skill_over_persistence"] > 0,
        "validation_beats_shuffled": metrics["validation"]["logp_mean"]
        > metrics["validation"]["logp_shuffled_mean"],
        "test_beats_shuffled": metrics["test"]["logp_mean"]
        > metrics["test"]["logp_shuffled_mean"],
        "test_speed_bins_diagonal": speed["diagonal_wins"] == 5,
        "test_speed_curve_peaks_at_match": speed["relative_peak_at_match"],
    }
    return {
        "eligible_for_demo_e": all(gates.values()),
        "gates": gates,
        "contract": heldout["contract"],
        "calibration": {
            "sigma_source": "standalone training windows only",
            "sigma": float(sigma),
            "heldout_sessions_used_for_sigma": False,
        },
        "sessions_retained": sessions_retained,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, default=ASSET)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.asset, args.cache, rebuild=args.rebuild)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["eligible_for_demo_e"]:
        raise SystemExit("frozen Coltrane prior failed a held-out eligibility gate")


if __name__ == "__main__":
    main()
