"""Artifact provenance and fail-closed guards for Demo E."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import (
    MIMIC_CHECKPOINT,
    OUT,
    PIPELINE_VERSION,
    PRIOR_ASSET,
    PRIOR_READY_FOR_RL,
    ROOT,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def validate_metadata(metadata: dict) -> None:
    demo = metadata.get("demo_e", {})
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("checkpoint belongs to a different Demo E pipeline")
    if demo.get("trainable_policy") != "high_level_intention_policy":
        raise ValueError("only the high-level intention policy may be trainable")
    if not demo.get("policy_from_scratch", False):
        raise ValueError("report policies must start from scratch")
    if not demo.get("decoder_frozen", False):
        raise ValueError("the shared imitation decoder must remain frozen")
    expected = str(MIMIC_CHECKPOINT.resolve())
    if demo.get("decoder_checkpoint") != expected:
        raise ValueError("checkpoint uses a different low-level motor decoder")


def run_metadata(arm: str, seed: int, beta: float) -> dict:
    decoder_metadata = MIMIC_CHECKPOINT / "PPONetwork_110" / "config" / "metadata"
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "demo_e": {
            "arm": arm,
            "trainable_policy": "high_level_intention_policy",
            "policy_from_scratch": True,
            "parent_policy": None,
            "decoder_frozen": True,
            "decoder_checkpoint": str(MIMIC_CHECKPOINT.resolve()),
            "decoder_metadata_sha256": sha256(decoder_metadata),
            "prior_frozen": True,
            "prior_asset": str(PRIOR_ASSET.resolve()),
            "prior_sha256": sha256(PRIOR_ASSET),
            "prior_ready_for_rl": PRIOR_READY_FOR_RL,
            "beta": float(beta),
        },
        "seed": seed,
        "git": git_state(),
    }
    validate_metadata(metadata)
    return metadata


def write_pointer(name: str, run: Path, **extra) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"latest_{name}.json"
    path.write_text(
        json.dumps({"run": str(run.resolve()), **extra}, indent=2, sort_keys=True)
        + "\n"
    )
    return path


def read_pointer(name: str) -> Path:
    path = OUT / f"latest_{name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return Path(json.loads(path.read_text())["run"])
