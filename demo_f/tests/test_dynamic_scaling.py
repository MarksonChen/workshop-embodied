import numpy as np

from demo_f.config import FPS
from demo_f.dataset.retime import (
    CLIP_FRAMES,
    DYNAMIC_JOINT_LIMIT_FRACTION_MAX,
    TIME_SCALE,
    TARGET_COMMAND_X,
    TARGET_SPEED_FETCH,
    crop_starts,
    retime_clip,
)


def synthetic_clip(speed=2.0):
    time = np.arange(CLIP_FRAMES, dtype=np.float32) / FPS
    root = np.zeros((CLIP_FRAMES, 3), np.float32)
    root[:, 0] = speed * time
    root[:, 2] = 1.375
    quaternion = np.zeros((CLIP_FRAMES, 4), np.float32)
    quaternion[:, 0] = 1.0
    angles = np.zeros((CLIP_FRAMES, 10), np.float32)
    contacts = np.zeros((CLIP_FRAMES, 4), np.uint8)
    contacts[::2, :2] = 1
    contacts[1::2, 2:] = 1
    return root, quaternion, angles, contacts


def test_dynamic_contract_has_disjoint_crops_and_froude_speed():
    starts = crop_starts()
    assert len(starts) == 4
    assert np.all(np.diff(starts) >= CLIP_FRAMES)
    np.testing.assert_allclose(TARGET_SPEED_FETCH, 0.20 * TIME_SCALE)
    np.testing.assert_allclose(TARGET_COMMAND_X, TARGET_SPEED_FETCH * 0.62)
    assert DYNAMIC_JOINT_LIMIT_FRACTION_MAX == 0.01


def test_retime_slows_velocity_and_uses_nearest_contacts():
    root, quaternion, angles, contacts = synthetic_clip()
    clip = retime_clip(root, quaternion, angles, contacts, int(crop_starts()[0]))
    realized = np.diff(clip["root_position"][:, 0]).mean() * FPS
    np.testing.assert_allclose(realized, 2.0 / TIME_SCALE, rtol=2e-5)
    expected_index = np.rint(clip["source_time"]).astype(int)
    np.testing.assert_array_equal(clip["contacts"], contacts[expected_index])
    assert clip["joint_angles"].shape == (CLIP_FRAMES, 10)
    assert clip["feet_local"].shape == (CLIP_FRAMES, 4, 3)
