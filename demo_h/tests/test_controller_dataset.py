import json

import numpy as np
import pytest

from demo_f.config import FEATURE_CONTRACT_VERSION
from demo_h.dataset.contract import (
    CONTROLLER_DATASET_VARIANT,
    DTYPES,
    FIELDS,
    SCHEMA_VERSION,
)
from demo_h.dataset.controller_rollouts import (
    _pad_body_rows,
    _validate_assignments,
)
from demo_h.dataset.loader import load_manifest


def test_modern_link_state_is_padded_without_inventing_legacy_bodies():
    values = np.arange(2 * 11 * 4, dtype=np.float32).reshape(2, 11, 4)
    padded = _pad_body_rows(values, quaternion=True)
    np.testing.assert_array_equal(padded[:, :11], values)
    np.testing.assert_array_equal(padded[:, 11:, 0], 1.0)
    np.testing.assert_array_equal(padded[:, 11:, 1:], 0.0)


def test_held_out_controller_cannot_enter_prior_fitting():
    _validate_assignments([{"seed": 0}, {"seed": 2}], {"seed": 1}, 1)
    with pytest.raises(ValueError, match="entered train/validation"):
        _validate_assignments([{"seed": 0}, {"seed": 1}], {"seed": 1}, 1)
    with pytest.raises(ValueError, match="not held-out seed"):
        _validate_assignments([{"seed": 0}, {"seed": 2}], {"seed": 3}, 1)


def test_loader_requires_the_experiment_variant_explicitly(tmp_path):
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete_release": True,
        "variant": CONTROLLER_DATASET_VARIANT,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "fields": {name: list(shape) for name, shape in FIELDS.items()},
        "dtypes": DTYPES,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected Demo H dataset variant"):
        load_manifest(tmp_path)
    loaded = load_manifest(
        tmp_path, expected_variant=CONTROLLER_DATASET_VARIANT
    )
    assert loaded["variant"] == CONTROLLER_DATASET_VARIANT
