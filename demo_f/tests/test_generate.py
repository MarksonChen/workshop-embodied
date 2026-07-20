import numpy as np

from demo_f.features import SL
from demo_f.generate import (
    COMMAND_HORIZON_SECONDS,
    command_scale,
    dataset_command_calibration,
    integrate_root,
)


def test_command_calibration_is_robust_to_nonstraight_outlier():
    command = np.asarray(
        [[1.0, 0.0, 0.0]] * 24 + [[100.0, 2.0, 1.0]], dtype=np.float32
    )
    speed = np.asarray([0.1] * 25, dtype=np.float32)
    assert np.isclose(command_scale(command, speed), 10.0)


def test_dynamic_release_uses_declared_froude_velocity_scale():
    calibration = dataset_command_calibration(
        {"dynamic_scaling": {"velocity_scale": 4.5}},
        np.zeros((1, 3), np.float32),
        np.zeros(1, np.float32),
    )
    assert calibration["method"].startswith("declared Froude")
    assert np.isclose(
        calibration["fetch_displacement_per_mps"],
        4.5 * COMMAND_HORIZON_SECONDS,
    )


def test_integrate_root_uses_local_velocity_and_bounds_joints():
    features = np.zeros((4, 60), np.float32)
    features[:, SL["root_velocity"][0]] = 1.0
    features[:, SL["root_height"][0]] = 1.375
    features[:, SL["rotation_delta_6d"][0] + 0] = 1.0
    features[:, SL["rotation_delta_6d"][0] + 3] = 1.0
    features[:, slice(*SL["joint_angles"])] = 100.0
    angles, root, quaternion = integrate_root(features)
    np.testing.assert_allclose(root[:, 0], [0.0, 0.02, 0.04, 0.06], atol=1e-6)
    np.testing.assert_allclose(root[:, 2], 1.375)
    np.testing.assert_allclose(quaternion[:, 0], 1.0)
    assert np.max(angles) <= np.pi / 3 + 1e-6
