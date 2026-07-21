"""Deterministically realize the independent Demo H controls in modern MJX.

The legacy and modern contact solvers accumulate different global root paths
under the same controls.  Demo J therefore stores what its *training engine*
actually realizes, just as Demo H stored what legacy Brax realized.  This is a
physics replay, not an ML reconstruction and not inverse dynamics.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from demo_f.features import trajectory_features
from demo_f.kinematics import fetch_feet_numpy
from demo_j.dataset import ReferenceSet, load_reference_set, take_references
from demo_j.env import FetchTracking
from demo_j.fetch_mjx import (
    XML_PATH,
    foot_site_indices,
    joint_qpos_addresses,
)


SCHEMA = "demo-j-modern-mjx-replay-v1"
DEFAULT_ROOT = Path(__file__).resolve().parent / "out" / "mjx_reference"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_contract_hash(base_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode())
    digest.update(base_hash.encode())
    digest.update(XML_PATH.read_bytes())
    return digest.hexdigest()


def project_reference(
    reference: ReferenceSet,
    *,
    batch_size: int = 256,
) -> ReferenceSet:
    """Replay every paired control and return the exact modern-MJX states."""

    environment = FetchTracking(reference, random_start=False)
    initialize = jax.jit(jax.vmap(environment.pipeline_init))
    advance = jax.jit(jax.vmap(environment.pipeline_step))
    qpos_parts, qvel_parts, contact_parts = [], [], []
    sites = jnp.asarray(foot_site_indices())

    for start in range(0, reference.clips, batch_size):
        stop = min(start + batch_size, reference.clips)
        batch = take_references(reference, np.arange(start, stop))
        state = initialize(
            jnp.asarray(batch.qpos[:, 0]), jnp.asarray(batch.qvel[:, 0])
        )
        initial_qpos = state.qpos
        initial_qvel = state.qvel
        initial_contact = state.site_xpos[:, sites, 2] <= 0.025

        def step(state, action):
            state = advance(state, action)
            contact = state.site_xpos[:, sites, 2] <= 0.025
            return state, (state.qpos, state.qvel, contact)

        _, (qpos, qvel, contact) = jax.lax.scan(
            step,
            state,
            jnp.asarray(batch.teacher_action).swapaxes(0, 1),
        )
        qpos_parts.append(
            np.concatenate(
                (np.asarray(initial_qpos)[:, None], np.asarray(qpos).swapaxes(0, 1)),
                axis=1,
            )
        )
        qvel_parts.append(
            np.concatenate(
                (np.asarray(initial_qvel)[:, None], np.asarray(qvel).swapaxes(0, 1)),
                axis=1,
            )
        )
        contact_parts.append(
            np.concatenate(
                (
                    np.asarray(initial_contact)[:, None],
                    np.asarray(contact).swapaxes(0, 1),
                ),
                axis=1,
            )
        )

    qpos = np.concatenate(qpos_parts).astype(np.float32)
    qvel = np.concatenate(qvel_parts).astype(np.float32)
    contacts = np.concatenate(contact_parts).astype(np.uint8)
    angles = qpos[..., joint_qpos_addresses()]
    feet = fetch_feet_numpy(angles)
    features = trajectory_features(
        qpos[..., :3], qpos[..., 3:7], angles, feet, contacts
    )
    if not (
        np.isfinite(qpos).all()
        and np.isfinite(qvel).all()
        and np.isfinite(features).all()
    ):
        raise ValueError("modern MJX projection contains non-finite values")
    return replace(
        reference,
        qpos=qpos,
        qvel=qvel,
        features=features,
        contacts=contacts,
        root_position=qpos[..., :3],
        root_quaternion=qpos[..., 3:7],
        joint_angles=angles,
        manifest_sha256=_combined_contract_hash(reference.manifest_sha256),
    )


def save_projected_reference(
    reference: ReferenceSet,
    root: Path = DEFAULT_ROOT,
) -> dict[str, object]:
    """Save only replaced physics arrays; provenance stays in the base release."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{reference.split}.npz"
    np.savez_compressed(
        archive,
        qpos=reference.qpos,
        qvel=reference.qvel,
        features=reference.features,
        contacts=reference.contacts,
    )
    return {
        "split": reference.split,
        "clips": reference.clips,
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "contract_sha256": reference.manifest_sha256,
    }


def load_projected_reference(
    split: str,
    root: Path = DEFAULT_ROOT,
) -> ReferenceSet:
    """Join cached modern states to independently verified base provenance."""

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unexpected projection schema {manifest.get('schema')!r}")
    base = load_reference_set(split)
    row = next((row for row in manifest["splits"] if row["split"] == split), None)
    if row is None:
        raise ValueError(f"projection has no {split!r} split")
    expected_contract = _combined_contract_hash(base.manifest_sha256)
    if row["contract_sha256"] != expected_contract:
        raise ValueError("base data/XML contract does not match cached projection")
    archive = root / row["archive"]
    if _sha256(archive) != row["archive_sha256"]:
        raise ValueError("projected archive hash mismatch")
    with np.load(archive) as values:
        qpos = np.asarray(values["qpos"], np.float32)
        qvel = np.asarray(values["qvel"], np.float32)
        features = np.asarray(values["features"], np.float32)
        contacts = np.asarray(values["contacts"], np.uint8)
    if qpos.shape[:2] != (base.clips, base.frames):
        raise ValueError(qpos.shape)
    angles = qpos[..., joint_qpos_addresses()]
    return replace(
        base,
        qpos=qpos,
        qvel=qvel,
        features=features,
        contacts=contacts,
        root_position=qpos[..., :3],
        root_quaternion=qpos[..., 3:7],
        joint_angles=angles,
        manifest_sha256=expected_contract,
    )


def build(root: Path = DEFAULT_ROOT, batch_size: int = 256) -> dict[str, object]:
    """Build the complete immutable local training/validation/test cache."""

    rows = []
    for split in ("train", "validation", "test"):
        projected = project_reference(load_reference_set(split), batch_size=batch_size)
        rows.append(save_projected_reference(projected, root))
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic open-loop replay; no learned reconstruction",
        "source": "Demo H accepted 1.75x projection and paired controls",
        "xml": str(XML_PATH),
        "xml_sha256": _sha256(XML_PATH),
        "control_semantics": "u[t] produces saved state[t+1]",
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "brax": __import__("brax").__version__,
            "backend": jax.default_backend(),
            "devices": sorted({device.device_kind for device in jax.devices()}),
        },
        "splits": rows,
    }
    root = Path(root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
