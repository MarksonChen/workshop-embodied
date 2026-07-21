import numpy as np

from demo_h.dataset.commands import hindsight_command
from demo_h.dataset.contract import DTYPES, FIELDS, expected_shape


def test_every_field_has_a_dtype_and_clip_shape():
    assert set(FIELDS) == set(DTYPES)
    assert expected_shape("normalized_control", 7) == (7, 63, 10)
    assert expected_shape("realized_features", 7) == (7, 64, 60)


def test_hindsight_command_is_egocentric_and_time_aligned():
    root = np.zeros((1, 64, 3), np.float32)
    root[0, :, 0] = np.arange(64) * 0.02
    quaternion = np.zeros((1, 64, 4), np.float32)
    quaternion[..., 0] = 1.0
    command = hindsight_command(root, quaternion)
    np.testing.assert_allclose(command, [[0.62, 0.0, 0.0]], atol=1e-6)
