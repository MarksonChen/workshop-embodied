import json

import pytest

from demo_f.artifacts import guard_derived_release_output


def test_derived_output_rejects_source_and_unrecognized_directories(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    with pytest.raises(ValueError, match="source output"):
        guard_derived_release_output(
            parent, parent, overwrite=True, expected_manifest={"schema": "demo"}
        )

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("user data")
    with pytest.raises(ValueError, match="non-release"):
        guard_derived_release_output(
            parent,
            unrelated,
            overwrite=True,
            expected_manifest={"schema": "demo"},
        )


def test_derived_output_requires_overwrite_and_recognized_manifest(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    output = tmp_path / "derived"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"schema": "demo"}))
    with pytest.raises(FileExistsError, match="--overwrite"):
        guard_derived_release_output(
            parent,
            output,
            overwrite=False,
            expected_manifest={"schema": "demo"},
        )
    assert guard_derived_release_output(
        parent,
        output,
        overwrite=True,
        expected_manifest={"schema": "demo"},
    ) == (parent.resolve(), output.resolve())
