"""Reference-data loading and the immutable Demo D split."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import h5py
import numpy as np

from demo_d.config import (
    ALL_CLIPS,
    REFERENCE_DATA,
    REFERENCE_HF_FILE,
    REFERENCE_SHA256,
    TRAIN_CLIPS,
    VAL_CLIPS,
)


def ensure_reference_data(path: Path = REFERENCE_DATA) -> Path:
    """Fetch the public mocap clips when absent and verify their exact bytes."""
    if not path.exists():
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id="talmolab/MIMIC-MJX",
                repo_type="dataset",
                filename=REFERENCE_HF_FILE,
                # REFERENCE_HF_FILE already starts with ``data/``; use the
                # outer data directory so it resolves to REFERENCE_DATA.
                local_dir=path.parents[2],
            )
        )
        if downloaded.resolve() != path.resolve():
            raise RuntimeError(f"dataset downloaded to unexpected path: {downloaded}")
    hasher = hashlib.sha256()
    with path.open("rb") as fid:
        for block in iter(lambda: fid.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != REFERENCE_SHA256:
        raise ValueError(f"reference SHA-256 mismatch: expected {REFERENCE_SHA256}, got {digest}")
    return path


def joint_names(path: Path = REFERENCE_DATA) -> list[str]:
    """Return the 67 actuated names, stripping the seven free-root entries."""
    with h5py.File(path, "r") as fid:
        names = [x.decode() for x in fid["names_qpos"][()]]
    n_root = sum(name == "root" for name in names)
    if n_root != 7:
        raise ValueError(f"expected seven root coordinates, found {n_root}")
    return names[n_root:]


def load_clips(clip_ids: tuple[int, ...], path: Path = REFERENCE_DATA):
    """Load only the requested public clip IDs with correct joint metadata."""
    from vnl_playground import registry

    ensure_reference_data(path)
    return registry.load_reference_clips(
        "RodentImitation",
        data_path=str(path),
        n_frames_per_clip=250,
        # A NumPy array is required here: current JAX rejects Python-list
        # advanced indexing inside the upstream legacy loader.
        keep_clips_idx=np.asarray(clip_ids, dtype=np.int32),
        joint_names=joint_names(path),
    )


def split_loaded_clips(clips):
    """Split a container loaded in ``ALL_CLIPS`` order without reshuffling."""
    n_train = len(TRAIN_CLIPS)
    if int(clips.qpos.shape[0]) != len(ALL_CLIPS):
        raise ValueError(f"expected {len(ALL_CLIPS)} clips, got {clips.qpos.shape[0]}")

    def subset(indices):
        out = copy.copy(clips)
        out._data_arrays = {
            key: clips._data_arrays[key][indices]
            for key in clips._data_array_keys
            if key in clips._data_arrays
        }
        out._clip_idx = clips._clip_idx[indices]
        if clips.clip_names is not None:
            out.clip_names = clips.clip_names[indices]
        return out

    return subset(np.arange(n_train)), subset(np.arange(n_train, len(ALL_CLIPS)))


def validate_split() -> None:
    if len(TRAIN_CLIPS) != 48 or len(VAL_CLIPS) != 16:
        raise AssertionError("Demo D split size changed")
    if set(TRAIN_CLIPS) & set(VAL_CLIPS):
        raise AssertionError("Demo D train and validation clips overlap")
    if tuple(ALL_CLIPS) != tuple(TRAIN_CLIPS) + tuple(VAL_CLIPS):
        raise AssertionError("fixed split ordering changed")
