import numpy as np

from demo_f.commands import hindsight_command
from demo_f.config import PriorConfig as DemoFConfig
from demo_f.windows import command_frames as demo_f_command_frames
from demo_h.config import PriorConfig as DemoHConfig
from demo_h.dataset.contract import DTYPES, FIELDS, expected_shape
from demo_h.windows import command_frames as demo_h_command_frames


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


def test_state_and_action_models_keep_their_explicit_one_frame_anchor_difference():
    anchor = np.asarray((4,))
    f_start, f_future = demo_f_command_frames(anchor, DemoFConfig())
    h_start, h_future = demo_h_command_frames(anchor, DemoHConfig())
    assert (int(f_start[0]), int(f_future[0])) == (16, 47)
    assert (int(h_start[0]), int(h_future[0])) == (15, 46)
