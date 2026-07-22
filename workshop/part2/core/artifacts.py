from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_derived_release_output(
    parent_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool,
    expected_manifest: dict[str, object],
) -> tuple[Path, Path]:
    parent = Path(parent_root).resolve()
    output = Path(output_root).resolve()
    workshop_root = Path(__file__).resolve().parents[2]
    overlaps = output == parent or output in parent.parents or parent in output.parents
    protected = {Path("/").resolve(), Path.home().resolve(), workshop_root}
    if overlaps or output in protected or len(output.parts) < 3:
        raise ValueError(f"refusing unsafe output target {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output target is not a directory: {output}")
    if not output.exists():
        return parent, output
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        mismatch = {
            name: (existing.get(name), expected)
            for name, expected in expected_manifest.items()
            if existing.get(name) != expected
        }
        if mismatch:
            raise ValueError(f"unrecognized release at {output}: {mismatch}")
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
    elif any(output.iterdir()):
        raise ValueError(f"refusing non-release directory {output}")
    return parent, output
