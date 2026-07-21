"""Small artifact helpers shared by Demos F and H."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading the artifact in memory."""

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
    """Validate a derived-dataset target before any shard is written."""

    parent = Path(parent_root).resolve()
    output = Path(output_root).resolve()
    repository = Path(__file__).resolve().parents[1]
    source_overlap = (
        output == parent or output in parent.parents or parent in output.parents
    )
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        repository,
        repository / "demo_f",
        repository / "demo_h",
    }
    if source_overlap or output in protected or len(output.parts) < 3:
        raise ValueError(f"refusing broad or source output target {output}")
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
            raise ValueError(
                f"refusing to overwrite unrecognized release {output}: {mismatch}"
            )
        if not overwrite:
            raise FileExistsError(
                f"{output} already contains a release; pass --overwrite explicitly"
            )
    elif any(output.iterdir()):
        raise ValueError(
            f"refusing existing non-release directory {output}; "
            "a recognized manifest is required"
        )
    return parent, output
