"""Fail-closed validation for the standalone retargeted dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..config import JOINT_LIMIT
from .contract import (
    DEFAULT_ROOT,
    DTYPES,
    FIELDS,
    SCHEMA_VERSION,
    SPLIT_SESSIONS,
    validate_split_contract,
)
from ..kinematics import fetch_feet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(
    root: Path,
    *,
    hashes: bool = True,
    require_complete: bool = False,
) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema {manifest.get('schema_version')!r} != supported {SCHEMA_VERSION!r}"
        )
    if require_complete and not manifest.get("complete_release", False):
        raise ValueError("partial release cannot be published or used for canonical training")
    validate_split_contract()
    frozen_splits = {key: list(value) for key, value in SPLIT_SESSIONS.items()}
    if manifest["splits"] != frozen_splits:
        raise ValueError("session split differs from the frozen public contract")

    totals = {split: 0 for split in SPLIT_SESSIONS}
    seen_sessions = set()
    speed_values, rmse_values = [], []
    for row in manifest["sessions"]:
        session = row["session"]
        if session in seen_sessions:
            raise ValueError(f"duplicate session row: {session}")
        seen_sessions.add(session)
        if session not in SPLIT_SESSIONS[row["split"]]:
            raise ValueError(f"{session}: incorrect split {row['split']!r}")
        path = root / row["shard"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashes and sha256(path) != row["shard_sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        with np.load(path) as shard:
            arrays = {key: shard[key] for key in shard.files}
        expected_keys = set(FIELDS)
        if set(arrays) != expected_keys:
            raise ValueError(f"{path}: fields {set(arrays)} != {expected_keys}")
        count = len(arrays["joint_angles"])
        for name, shape in FIELDS.items():
            expected = tuple(count if value == "clips" else value for value in shape)
            if arrays[name].shape != expected:
                raise ValueError(f"{path}:{name} {arrays[name].shape} != {expected}")
        for name, values in arrays.items():
            if values.dtype != np.dtype(DTYPES[name]):
                raise ValueError(
                    f"{path}:{name} dtype {values.dtype} != {DTYPES[name]}"
                )
            if name != "contacts" and not np.isfinite(values).all():
                raise ValueError(f"{path}:{name} contains non-finite values")
        if np.abs(arrays["joint_angles"]).max() > JOINT_LIMIT + 1e-5:
            raise ValueError(f"{path}: joint limit violation")
        gates = manifest["quality_gates"]
        if arrays["ik_foot_rmse"].max() > gates["ik_foot_rmse_max"] + 1e-6:
            raise ValueError(f"{path}: retained clip exceeds IK-error gate")
        if arrays["minimum_foot_height"].min() < gates["minimum_foot_height_min"] - 1e-6:
            raise ValueError(f"{path}: retained clip exceeds foot-height gate")
        if arrays["joint_limit_fraction"].max() > gates["joint_limit_fraction_max"] + 1e-6:
            raise ValueError(f"{path}: retained clip exceeds saturation gate")
        if arrays["source_path_speed_mps"].min() <= manifest["selection"]["minimum_speed_mps"]:
            raise ValueError(f"{path}: retained clip fails locomotion speed screen")
        recomputed_feet = np.asarray(fetch_feet(jnp.asarray(arrays["joint_angles"])))
        np.testing.assert_allclose(recomputed_feet, arrays["feet_local"], atol=2e-5)
        if count != row["released_clips"]:
            raise ValueError(f"{path}: manifest count mismatch")
        totals[row["split"]] += count
        speed_values.append(arrays["source_speed_mps"])
        rmse_values.append(arrays["ik_foot_rmse"])
    if totals != manifest["counts"]:
        raise ValueError(f"split totals {totals} != manifest {manifest['counts']}")
    if manifest.get("complete_release", False):
        expected_sessions = set().union(*map(set, SPLIT_SESSIONS.values()))
        if seen_sessions != expected_sessions:
            raise ValueError("complete release does not contain all 38 frozen sessions")
    strict_total = sum(row["strict_blocks"] for row in manifest["sessions"])
    global_pass_rate = sum(totals.values()) / strict_total
    if not np.isclose(global_pass_rate, manifest["global_pass_rate"], atol=1e-12):
        raise ValueError("manifest global pass rate does not match session counts")
    if require_complete and global_pass_rate < manifest["minimum_global_pass_rate"]:
        raise ValueError("complete release falls below its global quality floor")
    return {
        "root": str(root),
        "clips": sum(totals.values()),
        "counts": totals,
        "source_speed_range_mps": [
            float(np.concatenate(speed_values).min()),
            float(np.concatenate(speed_values).max()),
        ],
        "mean_ik_foot_rmse": float(np.concatenate(rmse_values).mean()),
        "hashes_checked": hashes,
        "complete_release": bool(manifest.get("complete_release", False)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    report = validate_release(args.root, hashes=not args.skip_hashes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
