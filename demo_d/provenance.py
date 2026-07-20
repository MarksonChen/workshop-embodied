"""Small, testable guards for Demo D's from-scratch claim."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from demo_d.config import OUT, PIPELINE_VERSION, REFERENCE_SHA256


PUBLISHED_CHECKPOINT_FRAGMENT = "feedforward_260210_013247_285744"


def validate_scratch_metadata(config: Mapping[str, Any]) -> None:
    """Raise if a low-level config is not a Demo D random-init checkpoint."""
    marker = config.get("demo_d")
    if not isinstance(marker, Mapping):
        raise ValueError("checkpoint has no Demo D provenance marker")
    if marker.get("from_scratch") is not True:
        raise ValueError("low-level policy was not marked as random initialization")
    if int(marker.get("pipeline_version", -1)) != PIPELINE_VERSION:
        raise ValueError("checkpoint was produced by a different Demo D pipeline")
    if marker.get("reference_sha256") != REFERENCE_SHA256:
        raise ValueError("checkpoint used a different reference dataset")
    if marker.get("parent_checkpoint") not in (None, "null"):
        raise ValueError("from-scratch checkpoint unexpectedly names a parent")
    serialized = json.dumps(config, sort_keys=True)
    if PUBLISHED_CHECKPOINT_FRAGMENT in serialized:
        raise ValueError("published MIMIC weights are forbidden in Demo D stage 1")


def read_pointer(name: str) -> Path:
    path = OUT / f"latest_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; pass an explicit checkpoint or train stage {name} first"
        )
    payload = json.loads(path.read_text())
    target = Path(payload["checkpoint"]).resolve()
    if not target.exists():
        raise FileNotFoundError(f"pointer target does not exist: {target}")
    return target


def write_pointer(name: str, checkpoint: Path, **extra: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"latest_{name}.json"
    payload = {"checkpoint": str(checkpoint.resolve()), **extra}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    return path
