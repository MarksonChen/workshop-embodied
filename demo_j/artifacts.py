"""Small, shared artifact helpers for Demo J.

All generated files live below :data:`OUTPUT_ROOT`; modules should not each
reimplement hashing, pickle validation, or output-path discovery.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = PACKAGE_ROOT / "out"
ALIGNED_OUTPUT_ROOT = OUTPUT_ROOT / "aligned"

# Accepted checkpoints predate the package-layer refactor. Pickle stores the
# defining module of NamedTuple parameter containers, so remap those stable
# types while loading instead of keeping misleading top-level shim modules.
_LEGACY_MODULES = {
    "demo_j.config": "demo_j.control.config",
    "demo_j.policy": "demo_j.control.policy",
    "demo_j.snn": "demo_j.control.snn",
}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path: str | Path, schemas: Iterable[str]) -> dict:
    path = Path(path)
    with path.open("rb") as stream:
        payload = _CheckpointUnpickler(stream).load()
    allowed = frozenset(schemas)
    if payload.get("schema") not in allowed:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema')!r}; "
            f"expected one of {sorted(allowed)}"
        )
    return payload


def save_pickle(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
