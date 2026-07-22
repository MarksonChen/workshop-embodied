"""Runtime-neutral provenance checks for the matched Demo H beta sweep."""

from __future__ import annotations

import json
from pathlib import Path

from demo_h.artifacts import load_policy_checkpoint


SWEEP_ID = "demo-j-beta-v1"
TRAINING_SPEED_RANGE = (1.5, 4.0)
TRAINING_TIMESTEPS = 30_000_000
TRAINING_ENVS = 2_048


def _sidecar(path: Path) -> dict:
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing checkpoint report {sidecar}")
    return json.loads(sidecar.read_text())


def checkpoint_contract(path: Path, *, require_sweep: bool) -> dict:
    """Validate a training report and, for the locked sweep, its checkpoint."""

    report = _sidecar(path)
    expected = {
        "num_timesteps": TRAINING_TIMESTEPS,
        "num_envs": TRAINING_ENVS,
        "task_speed_training_range": list(TRAINING_SPEED_RANGE),
    }
    actual = {name: report.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"unmatched Demo H run {path}: {actual} != {expected}")
    if require_sweep and report.get("sweep_id") != SWEEP_ID:
        raise ValueError(
            f"checkpoint {path} belongs to {report.get('sweep_id')!r}, not {SWEEP_ID!r}"
        )
    beta = float(report["beta"])
    arm = report["arm"]
    if (arm == "h1") != (beta == 0.0):
        raise ValueError(f"arm/beta mismatch in {path}: {arm}, {beta}")
    if require_sweep:
        _, envelope = load_policy_checkpoint(
            path,
            expected_arm=arm,
            expected_prior_sha256=report["prior"]["sha256"],
        )
        expected_training = {
            "sweep_id": SWEEP_ID,
            "beta": beta,
            "seed": int(report["seed"]),
            "num_timesteps": TRAINING_TIMESTEPS,
            "num_envs": TRAINING_ENVS,
            "task_speed_training_range": list(TRAINING_SPEED_RANGE),
        }
        if envelope.get("training") != expected_training:
            raise ValueError(
                f"checkpoint training envelope mismatch in {path}: "
                f"{envelope.get('training')} != {expected_training}"
            )
    return report
