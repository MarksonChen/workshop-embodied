"""Export the accepted 281-D Coltrane likelihood for MJX inference.

The PyTorch checkpoint is unchanged.  This compact artifact contains only
framework-neutral weights, normalization statistics, and robust
skeleton-marker sites.  Demo E now preserves RodentJoystick's native reset, so
the old 40+ MB mocap reset banks are neither needed nor exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .constants import FULL_FM
from .marker_bridge import KEYPOINT_BODIES, KEYPOINT_NAMES, calibrate_offsets
from .splits import (
    COLTRANE_PRIOR_TEST_SESSIONS,
    COLTRANE_PRIOR_TRAIN_SESSIONS,
    COLTRANE_PRIOR_VAL_SESSIONS,
    validate_coltrane_prior_split,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "demo_b" / "assets" / "motor_standalone.pt"
DESTINATION = ROOT / "demo_b" / "assets" / "motor_prior_jax.npz"
DATA_ROOT = Path("/workspace/data/Aldarondo2024")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def export(
    source: Path = SOURCE,
    destination: Path = DESTINATION,
    *,
    data_root: Path = DATA_ROOT,
    marker_samples_per_session: int = 128,
) -> Path:
    validate_coltrane_prior_split()
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    if (
        bundle.get("format_version") != 4
        or bundle.get("animal") != "coltrane"
        or bundle.get("feature_dim") != FULL_FM
    ):
        raise ValueError("expected the accepted format-v4, 281-D Coltrane checkpoint")

    calibration = calibrate_offsets(
        data_root,
        COLTRANE_PRIOR_TRAIN_SESSIONS,
        samples_per_session=marker_samples_per_session,
    )
    mean_rmse_mm = float(calibration.rmse_m.mean() * 1000.0)
    max_rmse_mm = float(calibration.rmse_m.max() * 1000.0)
    print(
        f"marker bridge: {calibration.samples} frames, "
        f"mean RMSE={mean_rmse_mm:.2f} mm, max={max_rmse_mm:.2f} mm"
    )
    if mean_rmse_mm > 10.0 or max_rmse_mm > 15.0:
        raise ValueError("skeleton-marker calibration is too inaccurate for Demo E")

    arrays: dict[str, np.ndarray] = {}
    encoder_keys = (
        "enc.0.conv.weight", "enc.0.conv.bias",
        "enc.2.conv.weight", "enc.2.conv.bias",
        "enc.4.conv.weight", "enc.4.conv.bias",
        "to_mu.conv.weight", "to_mu.conv.bias",
    )
    for key in encoder_keys:
        arrays[f"motion/{key}"] = _array(bundle["motion"][key]).astype(np.float32)
    for key, value in bundle["trans"].items():
        arrays[f"trans/{key}"] = _array(value).astype(np.float32)
    for key in (
        "zmean", "zstd", "cmean", "cstd", "mmean", "mstd", "sigma",
        "logp_clip", "command_support_velocity",
    ):
        arrays[f"norm/{key}"] = _array(bundle[key]).astype(np.float32)
    arrays["norm/keypoint_offsets"] = calibration.offsets
    arrays["norm/keypoint_rmse_m"] = calibration.rmse_m
    metadata = {
        "format_version": 6,
        "source_format_version": int(bundle["format_version"]),
        "feature_dim": FULL_FM,
        "animal": "coltrane",
        "motion_cfg": bundle["motion_cfg"],
        "model_cfg": bundle["model_cfg"],
        "source_asset_sha256": sha256(source),
        "split_ids": {
            "train": list(COLTRANE_PRIOR_TRAIN_SESSIONS),
            "validation": list(COLTRANE_PRIOR_VAL_SESSIONS),
            "test": list(COLTRANE_PRIOR_TEST_SESSIONS),
        },
        "keypoint_names": list(KEYPOINT_NAMES),
        "keypoint_bodies": list(KEYPOINT_BODIES),
        "marker_calibration": {
            "samples": calibration.samples,
            "mean_rmse_mm": mean_rmse_mm,
            "max_rmse_mm": max_rmse_mm,
            "source": "body-local robust median on Coltrane training sessions",
        },
        "environment_reset": "not included; Demo E uses native RodentJoystick reset",
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256 {sha256(destination)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--marker-samples-per-session", type=int, default=128)
    args = parser.parse_args()
    export(
        args.source,
        args.destination,
        data_root=args.data_root,
        marker_samples_per_session=args.marker_samples_per_session,
    )


if __name__ == "__main__":
    main()
